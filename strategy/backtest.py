"""
strategy/backtest.py

Deterministic Event-Driven Backtest Engine
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


@dataclass
class TradeResult:
    signal: TradeSignal
    direction: SignalType
    setup_time: Optional[pd.Timestamp]
    entry_time: Optional[pd.Timestamp] = None
    exit_time: Optional[pd.Timestamp] = None
    entry_price: float = 0.0
    fill_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    exit_price: float = 0.0
    initial_risk: float = 0.0
    result_status: str = "OPEN"
    r_multiple: float = 0.0
    bars_held: int = 0
    session: str = "UNKNOWN"


@dataclass
class BacktestResult:
    trades: List[TradeResult] = field(default_factory=list)
    total_signals_generated: int = 0
    total_trades_triggered: int = 0
    total_invalidated: int = 0
    total_expired: int = 0
    total_mtf_rejected: int = 0
    win_rate: float = 0.0
    expectancy: float = 0.0
    errors: List[str] = field(default_factory=list)


def _calculate_r_multiple(
    direction: SignalType,
    fill_price: float,
    exit_price: float,
    initial_risk: float,
) -> float:

    if initial_risk <= EPSILON:
        return 0.0

    if direction == SignalType.LONG:
        return (exit_price - fill_price) / initial_risk

    if direction == SignalType.SHORT:
        return (fill_price - exit_price) / initial_risk

    return 0.0


def _market_fill_is_valid(
    direction: SignalType,
    fill_price: float,
    stop_loss: float,
    take_profit: float,
) -> bool:
    """A market fill is only possible if price sits between stop and target."""

    if abs(fill_price - stop_loss) <= EPSILON:
        return False

    if direction == SignalType.LONG:
        return stop_loss + EPSILON < fill_price < take_profit - EPSILON

    if direction == SignalType.SHORT:
        return take_profit + EPSILON < fill_price < stop_loss - EPSILON

    return False


def _entry_bar_exit(
    direction: SignalType,
    stop_loss: float,
    take_profit: float,
    bar_high: float,
    bar_low: float,
):
    """
    Exit check for the bar a market order was filled on.
    Stop is checked first (conservative, same as the main loop).
    Returns (status, exit_price) or None.
    """

    if direction == SignalType.LONG:
        hit_sl = bar_low <= stop_loss + EPSILON
        hit_tp = bar_high >= take_profit - EPSILON
    elif direction == SignalType.SHORT:
        hit_sl = bar_high >= stop_loss - EPSILON
        hit_tp = bar_low <= take_profit + EPSILON
    else:
        return None

    if hit_sl:
        return "LOSS", stop_loss

    if hit_tp:
        return "WIN", take_profit

    return None


def _make_setup_key(
    signal: TradeSignal,
    setup_idx: int,
) -> tuple:

    return (
        int(setup_idx),
        getattr(signal, "setup_timestamp", None),
        signal.signal_type,
        getattr(signal, "zone_id", None),
        getattr(signal, "zone_stable_key", None),
        float(signal.entry_price),
        float(signal.stop_loss),
        float(signal.take_profit),
    )


def _validate_dataframe(
    df: pd.DataFrame,
) -> pd.DataFrame:

    if not isinstance(df, pd.DataFrame):
        raise ValueError("Backtest input must be a pandas DataFrame.")

    result = df.copy()

    result.columns = [
        str(column).strip().lower()
        for column in result.columns
    ]

    required = {"open", "high", "low", "close"}

    missing = required - set(result.columns)

    if missing:
        raise ValueError(
            "Backtest dataframe missing required "
            f"columns: {sorted(missing)}"
        )

    if not isinstance(result.index, pd.DatetimeIndex):
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

    result = result.sort_index(kind="stable")

    for column in ("open", "high", "low", "close"):
        result[column] = pd.to_numeric(result[column], errors="coerce")

    if result[["open", "high", "low", "close"]].isna().any().any():
        raise ValueError(
            "Backtest dataframe contains invalid OHLC values."
        )

    invalid_high = (
        result["high"]
        < result[["open", "close", "low"]].max(axis=1)
    )

    invalid_low = (
        result["low"]
        > result[["open", "close", "high"]].min(axis=1)
    )

    if invalid_high.any() or invalid_low.any():
        raise ValueError(
            "Backtest dataframe contains invalid OHLC candle ranges."
        )

    return result


def _visible_structure_breaks(all_breaks, current_time):

    visible = []

    for structure_break in all_breaks:

        broken_at = getattr(structure_break, "broken_at", None)

        if broken_at is None:
            continue

        try:
            broken_at = pd.Timestamp(broken_at)
        except (TypeError, ValueError):
            continue

        if broken_at <= current_time:
            visible.append(structure_break)

    return visible


def _current_structure_breaks(all_breaks, current_time):

    current = []

    for structure_break in all_breaks:

        broken_at = getattr(structure_break, "broken_at", None)

        if broken_at is None:
            continue

        try:
            broken_at = pd.Timestamp(broken_at)
        except (TypeError, ValueError):
            continue

        if broken_at == current_time:
            current.append(structure_break)

    return current


def _visible_zones(
    all_zones: List[Zone],
    current_time: pd.Timestamp,
) -> List[Zone]:

    visible_zones = []

    for zone in all_zones:

        formed_at = getattr(zone, "formed_from_break_at", None)

        if formed_at is None:
            continue

        try:
            formed_at = pd.Timestamp(formed_at)
        except (TypeError, ValueError):
            continue

        if formed_at > current_time:
            continue

        mitigated = getattr(zone, "mitigated", False)
        is_mitigated = getattr(zone, "is_mitigated", False)

        mitigated = bool(mitigated or is_mitigated)

        mitigated_at = getattr(zone, "mitigated_at", None)

        if mitigated_at is not None:
            try:
                mitigated_at = pd.Timestamp(mitigated_at)
            except (TypeError, ValueError):
                mitigated_at = None

        mitigation_reached = (
            mitigated
            and mitigated_at is not None
            and mitigated_at <= current_time
        )

        visible_zones.append(
            replace(
                zone,
                mitigated=mitigation_reached,
                mitigated_at=(
                    mitigated_at if mitigation_reached else None
                ),
            )
        )

    return visible_zones


def _visible_liquidity_levels(
    all_levels: List[LiquidityLevel],
    current_time: pd.Timestamp,
) -> List[LiquidityLevel]:

    visible_levels = []

    for level in all_levels:

        formed_at = getattr(level, "formed_at", None)

        if formed_at is None:
            continue

        try:
            formed_at = pd.Timestamp(formed_at)
        except (TypeError, ValueError):
            continue

        if formed_at > current_time:
            continue

        status = getattr(level, "status", None)
        swept_at = getattr(level, "swept_at", None)

        if swept_at is not None:
            try:
                swept_at = pd.Timestamp(swept_at)
            except (TypeError, ValueError):
                swept_at = None

        sweep_reached = (
            status == LiquidityStatus.SWEPT
            and swept_at is not None
            and swept_at <= current_time
        )

        visible_levels.append(
            LiquidityLevel(
                liquidity_type=level.liquidity_type,
                price=float(level.price),
                formed_at=formed_at,
                swing=level.swing,
                status=(
                    LiquidityStatus.SWEPT
                    if sweep_reached
                    else LiquidityStatus.ACTIVE
                ),
                sweep_bar_index=(
                    level.sweep_bar_index if sweep_reached else None
                ),
                swept_at=(swept_at if sweep_reached else None),
            )
        )

    return visible_levels


def _replay_zone_mitigation(
    df: pd.DataFrame,
    zones: List[Zone],
) -> None:

    for (
        timestamp,
        open_price,
        high,
        low,
        close,
    ) in df[["open", "high", "low", "close"]].itertuples(
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

        update_mitigation(zones, candle, timestamp)


def _close_trade(
    trade: TradeResult,
    exit_time: pd.Timestamp,
    exit_price: float,
    result_status: str,
    round_trip_cost: float = 0.0,
) -> None:

    trade.result_status = result_status
    trade.exit_price = float(exit_price)
    trade.exit_time = exit_time

    gross_r = _calculate_r_multiple(
        trade.direction,
        trade.fill_price,
        trade.exit_price,
        trade.initial_risk,
    )

    if trade.initial_risk > EPSILON:
        cost_r = round_trip_cost / trade.initial_risk
    else:
        cost_r = 0.0

    trade.r_multiple = gross_r - cost_r


def run_backtest(
    df: pd.DataFrame,
    max_bars_to_retest: int = 20,
    reward_multiple: float = 2.0,
    stop_buffer: float = 0.0,
    entry_mode: str = "midpoint",
    mtf_filter_fn: Optional[
        Callable[[TradeSignal, pd.Timestamp], bool]
    ] = None,
    spread: float = 0.0,
    slippage: float = 0.0,
    commission: float = 0.0,
    strict_entry_fill: bool = False,
    fill_mode: str = "touch",
) -> BacktestResult:
    """
    fill_mode:
        "touch" (default) -- original behaviour (optionally combined
            with strict_entry_fill).
        "market_on_close" -- the signal triggers on zone-edge touch at
            bar t, and the market order fills at the OPEN of bar t+1.
            Mirrors what the live bot does (one fresh tick after a
            closed candle triggers the signal).
    """

    if df is None:
        return BacktestResult()

    df = _validate_dataframe(df)

    if df.empty or len(df) < 5:
        return BacktestResult()

    if max_bars_to_retest <= 0:
        raise ValueError("max_bars_to_retest must be > 0")

    if reward_multiple <= 0:
        raise ValueError("reward_multiple must be > 0")

    if stop_buffer < 0:
        raise ValueError("stop_buffer must be >= 0")

    if entry_mode not in {"midpoint", "extreme"}:
        raise ValueError(
            "entry_mode must be 'midpoint' or 'extreme'"
        )

    if spread < 0 or slippage < 0 or commission < 0:
        raise ValueError(
            "spread, slippage and commission must be >= 0"
        )

    if fill_mode not in {"touch", "market_on_close"}:
        raise ValueError(
            "fill_mode must be 'touch' or 'market_on_close'"
        )

    if fill_mode == "market_on_close" and strict_entry_fill:
        raise ValueError(
            "strict_entry_fill cannot be combined with market_on_close"
        )

    round_trip_cost = spread + (2.0 * slippage) + commission

    all_swings = find_swings(df)

    (
        all_structure_breaks,
        _,
        _,
    ) = analyze_structure(df, all_swings)

    all_zones = find_zones(df, all_structure_breaks)

    all_liquidity_levels = detect_liquidity_levels(df, all_swings)

    all_liquidity_levels = update_liquidity_sweeps(
        df,
        all_liquidity_levels,
    )

    signal_context = _prepare_full_context(df)

    _replay_zone_mitigation(df, all_zones)

    completed_trades: List[TradeResult] = []
    pending_signals: List[Tuple[TradeSignal, int]] = []
    open_trades: List[Tuple[TradeResult, int]] = []
    registered_setup_keys: Set[tuple] = set()
    moc_queue: List[Tuple[TradeSignal, float]] = []

    total_signals = 0
    total_triggered = 0
    total_invalidated = 0
    total_expired = 0
    total_mtf_rejected = 0

    errors: List[str] = []

    for i in range(len(df)):

        current_time = pd.Timestamp(df.index[i])

        row = df.iloc[i]

        current_open = float(row["open"])
        current_high = float(row["high"])
        current_low = float(row["low"])

        current_visible_zones = _visible_zones(
            all_zones,
            current_time,
        )

        # A2. FILL QUEUED MARKET-ON-CLOSE ENTRIES AT THIS BAR'S OPEN

        if moc_queue:

            queued, moc_queue = moc_queue, []

            for signal, planned_entry in queued:

                fill_price = current_open
                stop = float(signal.stop_loss)
                target = float(signal.take_profit)

                if not _market_fill_is_valid(
                    signal.signal_type, fill_price, stop, target
                ):
                    total_invalidated += 1
                    continue

                signal.entry_price = float(fill_price)
                signal.recalculate_risk_reward()

                trade = TradeResult(
                    signal=signal,
                    direction=signal.signal_type,
                    setup_time=signal.setup_timestamp,
                    entry_time=current_time,
                    entry_price=planned_entry,
                    fill_price=float(fill_price),
                    stop_loss=stop,
                    take_profit=target,
                    exit_price=0.0,
                    initial_risk=abs(fill_price - stop),
                    result_status="OPEN",
                    r_multiple=0.0,
                    bars_held=0,
                    session=str(
                        getattr(signal, "session", "UNKNOWN")
                    ),
                )

                total_triggered += 1

                exit_info = _entry_bar_exit(
                    trade.direction,
                    stop,
                    target,
                    current_high,
                    current_low,
                )

                if exit_info is not None:
                    status, exit_price = exit_info
                    _close_trade(
                        trade=trade,
                        exit_time=current_time,
                        exit_price=exit_price,
                        result_status=status,
                        round_trip_cost=round_trip_cost,
                    )
                    completed_trades.append(trade)
                else:
                    open_trades.append((trade, i))

        # B. MANAGE OPEN TRADES

        still_open: List[Tuple[TradeResult, int]] = []

        for trade, entry_idx in open_trades:

            if i <= entry_idx:
                still_open.append((trade, entry_idx))
                continue

            trade.bars_held += 1

            if trade.direction == SignalType.LONG:

                hit_sl = current_low <= trade.stop_loss + EPSILON
                hit_tp = current_high >= trade.take_profit - EPSILON

            elif trade.direction == SignalType.SHORT:

                hit_sl = current_high >= trade.stop_loss - EPSILON
                hit_tp = current_low <= trade.take_profit + EPSILON

            else:

                errors.append(
                    f"{i}: unsupported trade direction "
                    f"{trade.direction}"
                )
                still_open.append((trade, entry_idx))
                continue

            if hit_sl:

                _close_trade(
                    trade=trade,
                    exit_time=current_time,
                    exit_price=trade.stop_loss,
                    result_status="LOSS",
                    round_trip_cost=round_trip_cost,
                )

                completed_trades.append(trade)

            elif hit_tp:

                _close_trade(
                    trade=trade,
                    exit_time=current_time,
                    exit_price=trade.take_profit,
                    result_status="WIN",
                    round_trip_cost=round_trip_cost,
                )

                completed_trades.append(trade)

            else:

                still_open.append((trade, entry_idx))

        open_trades = still_open

        # C. MANAGE PENDING SIGNALS

        still_pending: List[Tuple[TradeSignal, int]] = []

        for signal, setup_idx in pending_signals:

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
                strict_entry_fill=strict_entry_fill,
            )

            if outcome.outcome == PendingOutcomeType.EXPIRED:
                total_expired += 1
                continue

            if outcome.outcome == PendingOutcomeType.INVALIDATED:
                total_invalidated += 1
                continue

            if outcome.outcome == PendingOutcomeType.STILL_PENDING:
                still_pending.append((signal, setup_idx))
                continue

            if fill_mode == "market_on_close":
                moc_queue.append(
                    (signal, float(outcome.planned_entry))
                )
                continue

            trade = TradeResult(
                signal=signal,
                direction=signal.signal_type,
                setup_time=signal.setup_timestamp,
                entry_time=current_time,
                entry_price=outcome.planned_entry,
                fill_price=float(outcome.fill_price),
                stop_loss=float(signal.stop_loss),
                take_profit=float(signal.take_profit),
                exit_price=0.0,
                initial_risk=float(outcome.initial_risk),
                result_status="OPEN",
                r_multiple=0.0,
                bars_held=0,
                session=str(
                    getattr(signal, "session", "UNKNOWN")
                ),
            )

            open_trades.append((trade, i))
            total_triggered += 1

        pending_signals = still_pending

        # D. DISCOVER NEW SIGNALS

        if i < 4:
            continue

        try:

            current_structure_breaks = _current_structure_breaks(
                all_structure_breaks,
                current_time,
            )

            if not current_structure_breaks:
                continue

            zones = current_visible_zones

            liquidity_levels = _visible_liquidity_levels(
                all_liquidity_levels,
                current_time,
            )

            if not liquidity_levels:
                continue

            signals = generate_flip_zone_signals(
                liquidity_levels=liquidity_levels,
                structure_breaks=current_structure_breaks,
                zones=zones,
                max_bars_after_sweep=10,
                max_bars_from_break=5,
                max_bars_to_retest=max_bars_to_retest,
                reward_multiple=reward_multiple,
                stop_buffer=stop_buffer,
                entry_mode=entry_mode,
                precomputed_context=signal_context,
                upto_index=i,
                run_retest_simulation=False,
                consumed_zones=signal_context.get("consumed_zones"),
            )

            if signals is None:
                signals = []

            for signal in signals:

                break_idx = getattr(signal, "bar_index", None)
                setup_idx = getattr(
                    signal, "setup_bar_index", None
                )

                if break_idx is None or setup_idx is None:
                    continue

                try:
                    break_idx = int(break_idx)
                    setup_idx = int(setup_idx)
                except (TypeError, ValueError):
                    continue

                if setup_idx > i:
                    continue

                if break_idx > i:
                    continue

                if break_idx <= setup_idx:
                    continue

                if signal.status != SignalStatus.PENDING_RETEST:
                    continue

                if break_idx != i:
                    continue

                signal_key = _make_setup_key(signal, break_idx)

                if signal_key in registered_setup_keys:
                    continue

                registered_setup_keys.add(signal_key)

                if mtf_filter_fn is not None:
                    if not mtf_filter_fn(signal, current_time):
                        total_mtf_rejected += 1
                        continue

                pending_signals.append((signal, break_idx))
                total_signals += 1

        except Exception as exc:

            errors.append(
                f"{i}: {type(exc).__name__}: {exc}"
            )

    # FINALIZE OPEN TRADES

    for trade, _entry_idx in open_trades:
        trade.result_status = "OPEN"
        completed_trades.append(trade)

    # PERFORMANCE METRICS

    closed_trades = [
        trade
        for trade in completed_trades
        if trade.result_status in {"WIN", "LOSS"}
    ]

    wins = [
        trade
        for trade in closed_trades
        if trade.result_status == "WIN"
    ]

    if closed_trades:

        win_rate = (len(wins) / len(closed_trades)) * 100.0

        expectancy = (
            sum(trade.r_multiple for trade in closed_trades)
            / len(closed_trades)
        )

    else:
        win_rate = 0.0
        expectancy = 0.0

    return BacktestResult(
        trades=completed_trades,
        total_signals_generated=total_signals,
        total_trades_triggered=total_triggered,
        total_invalidated=total_invalidated,
        total_expired=total_expired,
        total_mtf_rejected=total_mtf_rejected,
        win_rate=win_rate,
        expectancy=expectancy,
        errors=errors,
    )
