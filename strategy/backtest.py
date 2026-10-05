"""
strategy/retest_engine.py

Shared PENDING_RETEST -> {STILL_PENDING, EXPIRED, INVALIDATED, TRIGGERED}
state machine.
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


class PendingOutcomeType(Enum):
    STILL_PENDING = "STILL_PENDING"
    EXPIRED = "EXPIRED"
    INVALIDATED = "INVALIDATED"
    TRIGGERED = "TRIGGERED"


@dataclass
class PendingOutcome:
    outcome: PendingOutcomeType
    planned_entry: Optional[float] = None
    fill_price: Optional[float] = None
    initial_risk: Optional[float] = None


def _gap_aware_fill(
    signal_type: SignalType,
    desired_entry: float,
    bar_open: float,
) -> float:

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
    return bar_low <= (zone_top + EPSILON)


def _short_retest_touched(
    bar_high: float,
    zone_bottom: float,
) -> bool:
    return bar_high >= (zone_bottom - EPSILON)


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

    bars_elapsed = current_index - setup_idx

    # EXPIRATION
    if bars_elapsed > max_bars_to_retest:
        signal.status = SignalStatus.EXPIRED
        return PendingOutcome(PendingOutcomeType.EXPIRED)

    # INVALIDATION
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

    # LOCATE ORIGINAL ZONE
    matching_visible_zone = None
    zone_stable_key = getattr(signal, "zone_stable_key", None)

    if zone_stable_key is not None:
        for zone in visible_zones:
            if _stable_cache_key(zone) != zone_stable_key:
                continue
            matching_visible_zone = zone
            break

    if matching_visible_zone is None:
        signal_zone_id = getattr(signal, "zone_id", None)
        if signal_zone_id is not None:
            for zone in visible_zones:
                if getattr(zone, "zone_id", None) == signal_zone_id:
                    matching_visible_zone = zone
                    break

    if matching_visible_zone is None:
        return PendingOutcome(PendingOutcomeType.STILL_PENDING)

    # EXCLUDE MITIGATED ZONES
    zone_mitigated = bool(
        getattr(matching_visible_zone, "mitigated", False)
        or getattr(matching_visible_zone, "is_mitigated", False)
    )

    zone_mitigated_at = getattr(matching_visible_zone, "mitigated_at", None)

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

    # AUTHORITATIVE RETEST
    if strict_entry_fill:
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

    # TRIGGER
    signal.status = SignalStatus.TRIGGERED
    signal.triggered_at_bar = int(current_index)
    signal.triggered_at_time = current_time
    signal.bars_to_trigger = int(bars_elapsed)

    # GAP-AWARE ACTUAL FILL
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

    # SYNCHRONIZE SIGNAL
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
