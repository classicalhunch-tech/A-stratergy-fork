"""
strategy/backtest.py

Deterministic Event-Driven Backtest Engine
------------------------------------------

Walks historical OHLC data bar-by-bar while enforcing:

    - strict chronological processing
    - no future-bar execution visibility
    - pending -> triggered -> WIN/LOSS
    - pending -> INVALIDATED
    - pending -> EXPIRED
    - gap-aware entry fills
    - conservative SL/TP handling
    - actual-fill risk calculation
    - duplicate setup protection
    - OPEN trades at dataset end

Performance architecture
------------------------

The expensive structural pipeline is computed once:

    find_swings
        ->
    analyze_structure
        ->
    find_zones
        ->
    detect_liquidity_levels
        ->
    update_liquidity_sweeps
        ->
    _prepare_full_context

During replay, only events whose timestamps have actually been
reached are exposed to the signal engine.

The signal engine receives:

    precomputed_context
    +
    upto_index=i

so OHLC access remains bounded to the current replay candle.

Important replay rule
---------------------

New signal discovery is driven ONLY by structure breaks that occur
on the current replay bar.

This prevents an already-processed historical structure break from
being rediscovered on every later bar and accidentally consuming a
zone that should remain available for a newer setup.

Retest execution
----------------

generate_flip_zone_signals() is called with:

    run_retest_simulation=False

The signal engine therefore creates fresh PENDING_RETEST setups.

This backtest engine owns the forward-time state machine:

    PENDING
       |
       +--> TRIGGERED
       |
       +--> INVALIDATED
       |
       +--> EXPIRED

As of this version, that pending-signal state machine (expiration,
invalidation, zone re-matching, mitigation exclusion, authoritative
retest, gap-aware fill) lives in strategy/retest_engine.py so it can
be shared with Phase 3's Strategy Adapter instead of being duplicated.
This file now calls advance_pending_signal() per pending signal per
bar; the decision logic itself is unchanged from the original inline
version.

Zone identity
-------------

Pending signals store zone_id and zone_stable_key.

zone_stable_key is preferred because detector objects may be recreated
during replay. Python object identity is therefore deliberately not
used for zone matching.

Mitigation
----------

Zones are detected once.

Their mitigation state is replayed chronologically once before the
main execution replay. _visible_zones() then exposes only mitigation
state whose timestamp has actually been reached.

Execution
---------

LONG retest:

    current_low <= zone_top

SHORT retest:

    current_high >= zone_bottom

Once a retest occurs, the planned entry is converted to an
OHLC-compatible actual fill through _gap_aware_fill() (now inside
retest_engine.py).

Initial risk is always calculated from the actual fill.
"""

from dataclasses import dataclass, field, replace
from typing import Callable, List, Optional, Tuple, Set

import pandas as pd

from strategy.signals import (
    generate_flip_zone_signals,
    SignalType,
    SignalStatus,
    TradeSignal,
    _prepare_full_context,
    _stable_cache_key,
)

from strategy.retest_engine import (
    advance_pending_signal,
    PendingOutcomeType,
)

from strategy.swings import find_swings
from strategy.structure import analyze_structure

from strategy.zones import (
    Zone,
    find_zones,
    update_mitigation,
)

from strategy.liquidity import (
    LiquidityLevel,
    LiquidityStatus,
    detect_liquidity_levels,
    update_liquidity_sweeps,
)


EPSILON = 1e-8


# ============================================================
# RESULT OBJECTS
# ============================================================

@dataclass
class TradeResult:
    """
    Represents one complete or still-open trade.
    """

    signal: TradeSignal
    direction: SignalType

    setup_time: Optional[pd.Timestamp]

    entry_time: Optional[pd.Timestamp] = None
    exit_time: Optional[pd.Timestamp] = None

    # Planned entry from the signal.
    entry_price: float = 0.0

    # Actual simulated fill.
    fill_price: float = 0.0

    stop_loss: float = 0.0
    take_profit: float = 0.0

    exit_price: float = 0.0

    # Actual initial risk based on actual fill.
    initial_risk: float = 0.0

    # WIN / LOSS / OPEN.
    result_status: str = "OPEN"

    # Realized R.
    r_multiple: float = 0.0

    # Number of completed bars after entry.
    bars_held: int = 0

    session: str = "UNKNOWN"


@dataclass
class BacktestResult:
    """
    Summary returned by run_backtest().
    """

    trades: List[TradeResult] = field(
        default_factory=list
    )

    total_signals_generated: int = 0
    total_trades_triggered: int = 0
    total_invalidated: int = 0
    total_expired: int = 0
    total_mtf_rejected: int = 0

    win_rate: float = 0.0
    expectancy: float = 0.0

    errors: List[str] = field(
        default_factory=list
    )


# ============================================================
# EXECUTION HELPERS
# ============================================================
#
# NOTE: _gap_aware_fill, _long_retest_touched, and
# _short_retest_touched now live in strategy/retest_engine.py.
# They are intentionally NOT re-defined here to avoid two copies
# of the same logic drifting apart. Anything in this file that
# still needs them should import from retest_engine.


def _calculate_r_multiple(
    direction: SignalType,
    fill_price: float,
    exit_price: float,
    initial_risk: float,
) -> float:
    """
    Calculate realized R multiple from actual fill.
    """

    if initial_risk <= EPSILON:
        return 0.0

    if direction == SignalType.LONG:
        return (
            exit_price - fill_price
        ) / initial_risk

    if direction == SignalType.SHORT:
        return (
            fill_price - exit_price
        ) / initial_risk

    return 0.0


def _make_setup_key(
    signal: TradeSignal,
    setup_idx: int,
) -> tuple:
    """
    Build deterministic duplicate-setup identity.

    setup_idx is the structure-break bar that starts the
    pending retest lifecycle.
    """

    return (
        int(setup_idx),
        getattr(
            signal,
            "setup_timestamp",
            None,
        ),
        signal.signal_type,
        getattr(
            signal,
            "zone_id",
            None,
        ),
        getattr(
            signal,
            "zone_stable_key",
            None,
        ),
        float(signal.entry_price),
        float(signal.stop_loss),
        float(signal.take_profit),
    )


# ============================================================
# INPUT VALIDATION
# ============================================================

def _validate_dataframe(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Validate and normalize backtest input.
    """

    if not isinstance(
        df,
        pd.DataFrame,
    ):
        raise ValueError(
            "Backtest input must be a pandas DataFrame."
        )

    result = df.copy()

    result.columns = [
        str(column).strip().lower()
        for column in result.columns
    ]

    required = {
        "open",
        "high",
        "low",
        "close",
    }

    missing = (
        required
        - set(result.columns)
    )

    if missing:
        raise ValueError(
            "Backtest dataframe missing required "
            f"columns: {sorted(missing)}"
        )

    if not isinstance(
        result.index,
        pd.DatetimeIndex,
    ):
        raise ValueError(
            "Backtest dataframe must be indexed "
            "by timestamp (DatetimeIndex)."
        )

    if result.index.has_duplicates:
        raise ValueError(
            "Backtest dataframe contains duplicate timestamps."
        )

    if result.empty:
        return result

    result = result.sort_index(
        kind="stable"
    )

    # Convert OHLC values to numeric.
    for column in (
        "open",
        "high",
        "low",
        "close",
    ):
        result[column] = pd.to_numeric(
            result[column],
            errors="coerce",
        )

    if result[
        [
            "open",
            "high",
            "low",
            "close",
        ]
    ].isna().any().any():
        raise ValueError(
            "Backtest dataframe contains invalid "
            "OHLC values."
        )

    # Basic candle integrity.
    invalid_high = (
        result["high"]
        < result[
            [
                "open",
                "close",
                "low",
            ]
        ].max(axis=1)
    )

    invalid_low = (
        result["low"]
        > result[
            [
                "open",
                "close",
                "high",
            ]
        ].min(axis=1)
    )

    if invalid_high.any() or invalid_low.any():
        raise ValueError(
            "Backtest dataframe contains invalid "
            "OHLC candle ranges."
        )

    return result


# ============================================================
# VISIBLE STRUCTURE HELPERS
# ============================================================

def _visible_structure_breaks(
    all_breaks,
    current_time: pd.Timestamp,
):
    """
    Return only structure breaks whose event timestamp
    has actually been reached.
    """

    visible = []

    for structure_break in all_breaks:

        broken_at = getattr(
            structure_break,
            "broken_at",
            None,
        )

        if broken_at is None:
            continue

        try:
            broken_at = pd.Timestamp(
                broken_at
            )
        except (
            TypeError,
            ValueError,
        ):
            continue

        if broken_at <= current_time:
            visible.append(
                structure_break
            )

    return visible


def _current_structure_breaks(
    all_breaks,
    current_time: pd.Timestamp,
):
    """
    Return ONLY structure breaks born on the current replay bar.

    This is critical for causal signal discovery.

    Passing every historical structure break into the signal engine
    on every replay bar can cause old setups to be rediscovered and
    can incorrectly consume zones that should remain available.
    """

    current = []

    for structure_break in all_breaks:

        broken_at = getattr(
            structure_break,
            "broken_at",
            None,
        )

        if broken_at is None:
            continue

        try:
            broken_at = pd.Timestamp(
                broken_at
            )
        except (
            TypeError,
            ValueError,
        ):
            continue

        if broken_at == current_time:
            current.append(
                structure_break
            )

    return current


# ============================================================
# VISIBLE ZONE HELPERS
# ============================================================

def _visible_zones(
    all_zones: List[Zone],
    current_time: pd.Timestamp,
) -> List[Zone]:
    """
    Return zones visible at current_time.

    Zone formation becomes visible at formed_from_break_at.

    Future mitigation remains hidden until mitigated_at
    has been reached.
    """

    visible_zones = []

    for zone in all_zones:

        formed_at = getattr(
            zone,
            "formed_from_break_at",
            None,
        )

        if formed_at is None:
            continue

        try:
            formed_at = pd.Timestamp(
                formed_at
            )
        except (
            TypeError,
            ValueError,
        ):
            continue

        if formed_at > current_time:
            continue

        mitigated = getattr(
            zone,
            "mitigated",
            False,
        )

        # Compatibility with schemas that use is_mitigated.
        is_mitigated = getattr(
            zone,
            "is_mitigated",
            False,
        )

        mitigated = bool(
            mitigated
            or is_mitigated
        )

        mitigated_at = getattr(
            zone,
            "mitigated_at",
            None,
        )

        if mitigated_at is not None:
            try:
                mitigated_at = pd.Timestamp(
                    mitigated_at
                )
            except (
                TypeError,
                ValueError,
            ):
                mitigated_at = None

        mitigation_reached = (
            mitigated
            and mitigated_at is not None
            and mitigated_at <= current_time
        )

        visible_zones.append(
            replace(
                zone,
                mitigated=(
                    mitigation_reached
                ),
                mitigated_at=(
                    mitigated_at
                    if mitigation_reached
                    else None
                ),
            )
        )

    return visible_zones


# ============================================================
# VISIBLE LIQUIDITY HELPERS
# ============================================================

def _visible_liquidity_levels(
    all_levels: List[LiquidityLevel],
    current_time: pd.Timestamp,
) -> List[LiquidityLevel]:
    """
    Return liquidity levels visible at current_time.

    Formation becomes visible at formed_at.

    A sweep becomes visible only after swept_at.
    """

    visible_levels = []

    for level in all_levels:

        formed_at = getattr(
            level,
            "formed_at",
            None,
        )

        if formed_at is None:
            continue

        try:
            formed_at = pd.Timestamp(
                formed_at
            )
        except (
            TypeError,
            ValueError,
        ):
            continue

        if formed_at > current_time:
            continue

        status = getattr(
            level,
            "status",
            None,
        )

        swept_at = getattr(
            level,
            "swept_at",
            None,
        )

        if swept_at is not None:
            try:
                swept_at = pd.Timestamp(
                    swept_at
                )
            except (
                TypeError,
                ValueError,
            ):
                swept_at = None

        sweep_reached = (
            status == LiquidityStatus.SWEPT
            and swept_at is not None
            and swept_at <= current_time
        )

        # Recreate the detector object with only the state
        # that is causally visible at this bar.
        visible_levels.append(
            LiquidityLevel(
                liquidity_type=(
                    level.liquidity_type
                ),
                price=float(
                    level.price
                ),
                formed_at=formed_at,
                swing=level.swing,
                status=(
                    LiquidityStatus.SWEPT
                    if sweep_reached
                    else LiquidityStatus.ACTIVE
                ),
                sweep_bar_index=(
                    level.sweep_bar_index
                    if sweep_reached
                    else None
                ),
                swept_at=(
                    swept_at
                    if sweep_reached
                    else None
                ),
            )
        )

    return visible_levels


# ============================================================
# MITIGATION REPLAY
# ============================================================

def _replay_zone_mitigation(
    df: pd.DataFrame,
    zones: List[Zone],
) -> None:
    """
    Replay zone mitigation chronologically once.
    """

    for (
        timestamp,
        open_price,
        high,
        low,
        close,
    ) in df[
        [
            "open",
            "high",
            "low",
            "close",
        ]
    ].itertuples(
        index=True,
        name=None,
    ):

        candle = pd.Series(
            {
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
            },
            name=timestamp,
        )

        update_mitigation(
            zones,
            candle,
            timestamp,
        )


# ============================================================
# TRADE FINALIZATION HELPERS
# ============================================================

def _close_trade(
    trade: TradeResult,
    exit_time: pd.Timestamp,
    exit_price: float,
    result_status: str,
) -> None:
    """
    Finalize a trade after SL/TP has been reached.
    """

    trade.result_status = result_status
    trade.exit_price = float(
        exit_price
    )
    trade.exit_time = exit_time

    trade.r_multiple = (
        _calculate_r_multiple(
            trade.direction,
            trade.fill_price,
            trade.exit_price,
            trade.initial_risk,
        )
    )


# ============================================================
# MAIN BACKTEST ENGINE
# ============================================================

def run_backtest(
    df: pd.DataFrame,
    max_bars_to_retest: int = 20,
    reward_multiple: float = 2.0,
    stop_buffer: float = 0.0,
    entry_mode: str = "midpoint",
    mtf_filter_fn: Optional[Callable[[TradeSignal, pd.Timestamp], bool]] = None,
) -> BacktestResult:
    """
    Run deterministic event-driven backtest.

    Parameters
    ----------
    df:
        Historical OHLC dataframe indexed by DatetimeIndex.

    max_bars_to_retest:
        Maximum number of bars a setup may remain pending.

    reward_multiple:
        TP distance expressed as a multiple of risk.

    stop_buffer:
        Additional stop-loss distance.

    entry_mode:
        "midpoint" or "extreme".

    mtf_filter_fn:
        Optional callable(signal, current_time) -> bool. When
        provided, a freshly discovered signal is only registered
        as pending if this returns True; otherwise it is dropped
        and counted in total_mtf_rejected.
    """

    # ========================================================
    # 1. INPUT VALIDATION
    # ========================================================

    if df is None:
        return BacktestResult()

    df = _validate_dataframe(df)

    if df.empty or len(df) < 5:
        return BacktestResult()

    if max_bars_to_retest <= 0:
        raise ValueError(
            "max_bars_to_retest must be > 0"
        )

    if reward_multiple <= 0:
        raise ValueError(
            "reward_multiple must be > 0"
        )

    if stop_buffer < 0:
        raise ValueError(
            "stop_buffer must be >= 0"
        )

    if entry_mode not in {
        "midpoint",
        "extreme",
    }:
        raise ValueError(
            "entry_mode must be "
            "'midpoint' or 'extreme'"
        )

    # ========================================================
    # 2. STRUCTURAL PIPELINE
    # ========================================================

    all_swings = find_swings(
        df
    )

    (
        all_structure_breaks,
        _,
        _,
    ) = analyze_structure(
        df,
        all_swings,
    )

    all_zones = find_zones(
        df,
        all_structure_breaks,
    )

    all_liquidity_levels = (
        detect_liquidity_levels(
            df,
            all_swings,
        )
    )

    all_liquidity_levels = (
        update_liquidity_sweeps(
            df,
            all_liquidity_levels,
        )
    )

    # ========================================================
    # 3. PREPARE SIGNAL CONTEXT ONCE
    # ========================================================

    signal_context = (
        _prepare_full_context(
            df
        )
    )

    # ========================================================
    # 4. REPLAY MITIGATION ONCE
    # ========================================================

    _replay_zone_mitigation(
        df,
        all_zones,
    )

    # ========================================================
    # 5. ENGINE STATE
    # ========================================================

    completed_trades: List[
        TradeResult
    ] = []

    pending_signals: List[
        Tuple[TradeSignal, int]
    ] = []

    open_trades: List[
        Tuple[TradeResult, int]
    ] = []

    registered_setup_keys: Set[
        tuple
    ] = set()

    total_signals = 0
    total_triggered = 0
    total_invalidated = 0
    total_expired = 0
    total_mtf_rejected = 0

    errors: List[str] = []

    # ========================================================
    # 6. BAR-BY-BAR REPLAY
    # ========================================================

    for i in range(len(df)):

        current_time = pd.Timestamp(
            df.index[i]
        )

        row = df.iloc[i]

        current_open = float(
            row["open"]
        )

        current_high = float(
            row["high"]
        )

        current_low = float(
            row["low"]
        )

        # ====================================================
        # A. VISIBLE ZONES
        # ====================================================

        current_visible_zones = (
            _visible_zones(
                all_zones,
                current_time,
            )
        )

        # ====================================================
        # B. MANAGE OPEN TRADES
        # ====================================================

        still_open: List[
            Tuple[TradeResult, int]
        ] = []

        for (
            trade,
            entry_idx,
        ) in open_trades:

            # Do not evaluate SL/TP on the trigger candle.
            if i <= entry_idx:

                still_open.append(
                    (
                        trade,
                        entry_idx,
                    )
                )

                continue

            trade.bars_held += 1

            if (
                trade.direction
                == SignalType.LONG
            ):

                hit_sl = (
                    current_low
                    <= trade.stop_loss
                    + EPSILON
                )

                hit_tp = (
                    current_high
                    >= trade.take_profit
                    - EPSILON
                )

            elif (
                trade.direction
                == SignalType.SHORT
            ):

                hit_sl = (
                    current_high
                    >= trade.stop_loss
                    - EPSILON
                )

                hit_tp = (
                    current_low
                    <= trade.take_profit
                    + EPSILON
                )

            else:

                errors.append(
                    f"{i}: unsupported trade "
                    f"direction {trade.direction}"
                )

                still_open.append(
                    (
                        trade,
                        entry_idx,
                    )
                )

                continue

            # ------------------------------------------------
            # CONSERVATIVE SAME-BAR COLLISION
            #
            # If both SL and TP are touched in the same candle,
            # SL wins because OHLC does not reveal intrabar order.
            # ------------------------------------------------

            if hit_sl:

                _close_trade(
                    trade=trade,
                    exit_time=current_time,
                    exit_price=trade.stop_loss,
                    result_status="LOSS",
                )

                completed_trades.append(
                    trade
                )

            elif hit_tp:

                _close_trade(
                    trade=trade,
                    exit_time=current_time,
                    exit_price=trade.take_profit,
                    result_status="WIN",
                )

                completed_trades.append(
                    trade
                )

            else:

                still_open.append(
                    (
                        trade,
                        entry_idx,
                    )
                )

        open_trades = still_open

        # ====================================================
        # C. MANAGE PENDING SIGNALS
        #
        # Delegates the actual expire/invalidate/re-match/retest/
        # fill decision logic to strategy.retest_engine so Phase 3's
        # Strategy Adapter can share the identical state machine
        # instead of re-implementing it. Behavior here is unchanged
        # from the original inline version -- only the bookkeeping
        # (which list a signal lands in, which counter increments)
        # stays local to this loop.
        # ====================================================

        still_pending: List[
            Tuple[TradeSignal, int]
        ] = []

        for (
            signal,
            setup_idx,
        ) in pending_signals:

            outcome = advance_pending_signal(
                signal=signal,
                setup_idx=setup_idx,
                current_index=i,
                current_time=current_time,
                current_open=current_open,
                current_high=current_high,
                current_low=current_low,
                visible_zones=current_visible_zones,
                max_bars_to_retest=max_bars_to_retest,
            )

            if (
                outcome.outcome
                == PendingOutcomeType.EXPIRED
            ):

                total_expired += 1
                continue

            if (
                outcome.outcome
                == PendingOutcomeType.INVALIDATED
            ):

                total_invalidated += 1
                continue

            if (
                outcome.outcome
                == PendingOutcomeType.STILL_PENDING
            ):

                still_pending.append(
                    (
                        signal,
                        setup_idx,
                    )
                )

                continue

            # ------------------------------------------------
            # TRIGGERED -> BUILD TRADE
            # ------------------------------------------------

            trade = TradeResult(
                signal=signal,

                direction=(
                    signal.signal_type
                ),

                setup_time=(
                    signal.setup_timestamp
                ),

                entry_time=current_time,

                entry_price=(
                    outcome.planned_entry
                ),

                fill_price=float(
                    outcome.fill_price
                ),

                stop_loss=float(
                    signal.stop_loss
                ),

                take_profit=float(
                    signal.take_profit
                ),

                exit_price=0.0,

                initial_risk=float(
                    outcome.initial_risk
                ),

                result_status="OPEN",

                r_multiple=0.0,

                bars_held=0,

                session=str(
                    getattr(
                        signal,
                        "session",
                        "UNKNOWN",
                    )
                ),
            )

            open_trades.append(
                (
                    trade,
                    i,
                )
            )

            total_triggered += 1

        pending_signals = still_pending

        # ====================================================
        # D. DISCOVER NEW SIGNALS
        # ====================================================

        # Give structural detectors a small warm-up period.
        if i < 4:
            continue

        try:

            # ------------------------------------------------
            # CRITICAL:
            #
            # Only the structure breaks born NOW are sent to
            # signal generation.
            #
            # This prevents historical setups from being
            # regenerated and consuming zones repeatedly.
            # ------------------------------------------------

            current_structure_breaks = (
                _current_structure_breaks(
                    all_structure_breaks,
                    current_time,
                )
            )

            if not current_structure_breaks:
                continue

            zones = current_visible_zones

            liquidity_levels = (
                _visible_liquidity_levels(
                    all_liquidity_levels,
                    current_time,
                )
            )

            if not liquidity_levels:
                continue

            # ------------------------------------------------
            # OPTIMIZED SIGNAL ENGINE
            # ------------------------------------------------

            signals = (
                generate_flip_zone_signals(
                    liquidity_levels=(
                        liquidity_levels
                    ),

                    structure_breaks=(
                        current_structure_breaks
                    ),

                    zones=zones,

                    max_bars_after_sweep=10,

                    max_bars_from_break=5,

                    max_bars_to_retest=(
                        max_bars_to_retest
                    ),

                    reward_multiple=(
                        reward_multiple
                    ),

                    stop_buffer=(
                        stop_buffer
                    ),

                    entry_mode=(
                        entry_mode
                    ),

                    precomputed_context=(
                        signal_context
                    ),

                    upto_index=i,

                    run_retest_simulation=False,

                    consumed_zones=(
                        signal_context.get(
                            "consumed_zones"
                        )
                    ),
                )
            )

            if signals is None:
                signals = []

            # ------------------------------------------------
            # REGISTER ONLY FRESH SIGNALS
            # ------------------------------------------------

            for signal in signals:

                break_idx = getattr(
                    signal,
                    "bar_index",
                    None,
                )

                setup_idx = getattr(
                    signal,
                    "setup_bar_index",
                    None,
                )

                if (
                    break_idx is None
                    or setup_idx is None
                ):
                    continue

                try:

                    break_idx = int(
                        break_idx
                    )

                    setup_idx = int(
                        setup_idx
                    )

                except (
                    TypeError,
                    ValueError,
                ):
                    continue

                # Strict chronological safety.
                if setup_idx > i:
                    continue

                if break_idx > i:
                    continue

                # Structure break must happen after sweep.
                if break_idx <= setup_idx:
                    continue

                # Signal must be pending.
                if (
                    signal.status
                    != SignalStatus.PENDING_RETEST
                ):
                    continue

                # Signal must be born on this exact bar.
                if break_idx != i:
                    continue

                # ------------------------------------------------
                # DUPLICATE PROTECTION
                # ------------------------------------------------

                signal_key = (
                    _make_setup_key(
                        signal,
                        break_idx,
                    )
                )

                if (
                    signal_key
                    in registered_setup_keys
                ):
                    continue

                registered_setup_keys.add(
                    signal_key
                )

                if mtf_filter_fn is not None:
                    if not mtf_filter_fn(signal, current_time):
                        total_mtf_rejected += 1
                        continue

                pending_signals.append(
                    (
                        signal,
                        break_idx,
                    )
                )

                total_signals += 1

        except Exception as exc:

            errors.append(
                f"{i}: "
                f"{type(exc).__name__}: "
                f"{exc}"
            )

    # ========================================================
    # 7. FINALIZE OPEN TRADES
    # ========================================================

    for (
        trade,
        _entry_idx,
    ) in open_trades:

        trade.result_status = "OPEN"

        completed_trades.append(
            trade
        )

    # ========================================================
    # 8. PERFORMANCE METRICS
    # ========================================================

    closed_trades = [
        trade
        for trade in completed_trades
        if trade.result_status
        in {
            "WIN",
            "LOSS",
        }
    ]

    wins = [
        trade
        for trade in closed_trades
        if trade.result_status == "WIN"
    ]

    if closed_trades:

        win_rate = (
            len(wins)
            / len(closed_trades)
        ) * 100.0

        expectancy = (
            sum(
                trade.r_multiple
                for trade in closed_trades
            )
            / len(closed_trades)
        )

    else:

        win_rate = 0.0
        expectancy = 0.0

    # ========================================================
    # 9. FINAL RESULT
    # ========================================================

    return BacktestResult(
        trades=completed_trades,

        total_signals_generated=(
            total_signals
        ),

        total_trades_triggered=(
            total_triggered
        ),

        total_invalidated=(
            total_invalidated
        ),

        total_expired=(
            total_expired
        ),

        total_mtf_rejected=(
            total_mtf_rejected
        ),

        win_rate=win_rate,

        expectancy=expectancy,

        errors=errors,
    )