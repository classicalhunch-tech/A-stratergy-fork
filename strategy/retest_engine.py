"""
strategy/retest_engine.py

Shared PENDING_RETEST -> {STILL_PENDING, EXPIRED, INVALIDATED, TRIGGERED}
state machine.

WHY THIS FILE EXISTS
---------------------
This logic used to live inline inside strategy/backtest.py's
run_backtest() replay loop (the "C. MANAGE PENDING SIGNALS" section).

Phase 3's roadmap requires a Strategy Adapter that feeds the
PaperTradeEngine only already-TRIGGERED signals, and explicitly
forbids the adapter from containing its own copy of "when does a
signal trigger" logic (that would be a second signal engine).

Since generate_flip_zone_signals() only ever emits PENDING_RETEST
signals (when run_retest_simulation=False), something has to run
this exact state machine bar-by-bar in both:

    - strategy/backtest.py   (replay over an already-known dataframe)
    - Phase 3 Strategy Adapter (live/incremental, one candle at a time)

Extracting it here means both callers share one implementation, and
neither duplicates it.

BEHAVIOR GUARANTEE
-------------------
Every line of decision logic below is copied unchanged from the
original inline block in strategy/backtest.py. Nothing about
thresholds, ordering, or the EPSILON tolerance has been altered.
Only the surrounding bookkeeping (which list a signal lands in,
which counter increments) has moved to the caller, via the returned
PendingOutcome.

The single addition is the optional strict_entry_fill flag. It
defaults to False, which keeps the original behaviour exactly. When
True, a setup only triggers if price actually trades through the
signal's entry price (limit-order realism), instead of triggering as
soon as price touches the zone edge.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import pandas as pd

from strategy.signals import (
    SignalType,
    SignalStatus,
    TradeSignal,
    _stable_cache_key,
)
from strategy.zones import Zone


EPSILON = 1e-8


# ============================================================
# OUTCOME TYPES
# ============================================================

class PendingOutcomeType(Enum):
    STILL_PENDING = "STILL_PENDING"
    EXPIRED = "EXPIRED"
    INVALIDATED = "INVALIDATED"
    TRIGGERED = "TRIGGERED"


@dataclass
class PendingOutcome:
    """
    Result of advancing one pending signal by exactly one bar.

    For TRIGGERED outcomes:
        planned_entry  -- the signal's entry price BEFORE the actual
                           fill was applied (mirrors backtest.py's
                           local `planned_entry` variable, used as
                           TradeResult.entry_price).
        fill_price     -- the actual OHLC-compatible fill.
        initial_risk   -- abs(fill_price - stop_loss).

    For all other outcomes these fields are None.
    """

    outcome: PendingOutcomeType
    planned_entry: Optional[float] = None
    fill_price: Optional[float] = None
    initial_risk: Optional[float] = None


# ============================================================
# EXECUTION HELPERS (copied unchanged from backtest.py)
# ============================================================

def _gap_aware_fill(
    signal_type: SignalType,
    desired_entry: float,
    bar_open: float,
) -> float:
    """
    Calculate an OHLC-compatible fill.

    LONG:
        Gap/open below desired entry -> fill at open.
        Otherwise -> fill at desired entry.

    SHORT:
        Gap/open above desired entry -> fill at open.
        Otherwise -> fill at desired entry.
    """

    if signal_type == SignalType.LONG:
        if bar_open <= desired_entry:
            return float(bar_open)
        return float(desired_entry)

    if signal_type == SignalType.SHORT:
        if bar_open >= desired_entry:
            return float(bar_open)
        return float(desired_entry)

    raise ValueError(f"Unsupported signal type: {signal_type}")


def _long_retest_touched(
    bar_low: float,
    zone_top: float,
) -> bool:
    """LONG retest condition: price reaches the zone's top boundary."""
    return bar_low <= (zone_top + EPSILON)


def _short_retest_touched(
    bar_high: float,
    zone_bottom: float,
) -> bool:
    """SHORT retest condition: price reaches the zone's bottom boundary."""
    return bar_high >= (zone_bottom - EPSILON)


# ============================================================
# MAIN STATE-MACHINE STEP
# ============================================================

def advance_pending_signal(
    signal: TradeSignal,
    setup_idx: int,
    current_index: int,
    current_time: pd.Timestamp,
    current_open: float,
    current_high: float,
    current_low: float,
    visible_zones: List[Zone],
    max_bars_to_retest: int,
    strict_entry_fill: bool = False,
) -> PendingOutcome:
    """
    Advance one PENDING_RETEST signal by exactly one bar.

    Mutates `signal.status` (and, on TRIGGERED, `signal.entry_price` /
    `signal.risk_reward_ratio` via recalculate_risk_reward()) exactly
    as the original inline backtest.py loop did. The caller is
    responsible for counters and list membership based on the
    returned PendingOutcome.outcome.

    strict_entry_fill:
        False (default) -- original behaviour: the signal triggers when
        price touches the zone edge (LONG: zone top, SHORT: zone
        bottom), and the fill is recorded at the entry price.
        True -- limit-order realism: the signal triggers only if price
        actually trades through the entry price (LONG: low <= entry,
        SHORT: high >= entry). The fill stays gap-aware.
    """

    bars_elapsed = current_index - setup_idx

    # ------------------------------------------------
    # EXPIRATION
    # ------------------------------------------------

    if bars_elapsed > max_bars_to_retest:
        signal.status = SignalStatus.EXPIRED
        return PendingOutcome(PendingOutcomeType.EXPIRED)

    # ------------------------------------------------
    # INVALIDATION
    #
    # Stop is checked before entry trigger.
    # ------------------------------------------------

    if signal.signal_type == SignalType.LONG:
        invalidated = current_low <= signal.stop_loss + EPSILON

    elif signal.signal_type == SignalType.SHORT:
        invalidated = current_high >= signal.stop_loss - EPSILON

    else:
        signal.status = SignalStatus.INVALIDATED
        return PendingOutcome(PendingOutcomeType.INVALIDATED)

    if invalidated:
        signal.status = SignalStatus.INVALIDATED
        return PendingOutcome(PendingOutcomeType.INVALIDATED)

    # ------------------------------------------------
    # LOCATE ORIGINAL ZONE
    # ------------------------------------------------

    matching_visible_zone = None

    zone_stable_key = getattr(signal, "zone_stable_key", None)

    if zone_stable_key is not None:
        for zone in visible_zones:
            if _stable_cache_key(zone) != zone_stable_key:
                continue
            matching_visible_zone = zone
            break

    # Compatibility fallback when an older signal has no stable key
    # but does have an explicit zone_id.
    if matching_visible_zone is None:
        signal_zone_id = getattr(signal, "zone_id", None)

        if signal_zone_id is not None:
            for zone in visible_zones:
                if getattr(zone, "zone_id", None) == signal_zone_id:
                    matching_visible_zone = zone
                    break

    if matching_visible_zone is None:
        # Keep waiting until expiry.
        return PendingOutcome(PendingOutcomeType.STILL_PENDING)

    # ------------------------------------------------
    # EXCLUDE MITIGATED ZONES
    # ------------------------------------------------

    zone_mitigated = bool(
        getattr(matching_visible_zone, "mitigated", False)
        or getattr(matching_visible_zone, "is_mitigated", False)
    )

    zone_mitigated_at = getattr(matching_visible_zone, "mitigated_at", None)

    # Only exclude a zone that was ALREADY mitigated on a strictly
    # earlier bar -- see backtest.py's original comment: requiring a
    # STRICTLY earlier mitigation bar lets the coincident retest/
    # mitigation bar still trigger correctly, while still excluding
    # zones genuinely used up before this bar began.
    zone_already_mitigated = bool(
        zone_mitigated
        and zone_mitigated_at is not None
        and zone_mitigated_at < current_time
    )

    if zone_already_mitigated:
        signal.status = SignalStatus.INVALIDATED
        return PendingOutcome(PendingOutcomeType.INVALIDATED)

    zone_top = float(matching_visible_zone.price_top)
    zone_bottom = float(matching_visible_zone.price_bottom)

    if zone_top <= zone_bottom:
        signal.status = SignalStatus.INVALIDATED
        return PendingOutcome(PendingOutcomeType.INVALIDATED)

    # ------------------------------------------------
    # AUTHORITATIVE RETEST
    # ------------------------------------------------

    if strict_entry_fill:
        # Limit-order realism: the setup only fills if price actually
        # trades through the entry price, not merely the zone edge.
        planned_limit = float(signal.entry_price)

        if signal.signal_type == SignalType.LONG:
            entry_touched = current_low <= planned_limit + EPSILON
        else:
            entry_touched = current_high >= planned_limit - EPSILON

    elif signal.signal_type == SignalType.LONG:
        entry_touched = _long_retest_touched(current_low, zone_top)
    else:
        entry_touched = _short_retest_touched(current_high, zone_bottom)

    if not entry_touched:
        return PendingOutcome(PendingOutcomeType.STILL_PENDING)

    # ------------------------------------------------
    # TRIGGER
    # ------------------------------------------------

    signal.status = SignalStatus.TRIGGERED
    signal.triggered_at_bar = int(current_index)
    signal.triggered_at_time = current_time
    signal.bars_to_trigger = int(bars_elapsed)

    # ------------------------------------------------
    # GAP-AWARE ACTUAL FILL
    # ------------------------------------------------

    planned_entry = float(signal.entry_price)

    fill_price = _gap_aware_fill(
        signal.signal_type,
        planned_entry,
        current_open,
    )

    initial_risk = abs(fill_price - float(signal.stop_loss))

    if initial_risk <= EPSILON:
        signal.status = SignalStatus.INVALIDATED
        return PendingOutcome(PendingOutcomeType.INVALIDATED)

    # ------------------------------------------------
    # SYNCHRONIZE SIGNAL
    # ------------------------------------------------

    signal.entry_price = float(fill_price)
    signal.recalculate_risk_reward()

    return PendingOutcome(
        outcome=PendingOutcomeType.TRIGGERED,
        planned_entry=planned_entry,
        fill_price=float(fill_price),
        initial_risk=float(initial_risk),
    )


__all__ = [
    "PendingOutcome",
    "PendingOutcomeType",
    "advance_pending_signal",
]
