
"""
phase_03_paper/strategy/adapter.py

Phase 3 Strategy Adapter
========================

Thin integration bridge between the Phase 3 paper-trading runtime and
the existing Phase 1 SMC strategy.

IMPORTANT:
This module does NOT reimplement strategy logic.

The existing strategy/ package remains responsible for:
- swings
- structure
- zones
- liquidity
- signal discovery
- pending-retest state transitions

The adapter is responsible only for:
- receiving one candle at a time
- maintaining causal candle history
- running the existing strategy pipeline
- maintaining pending signal state across candles
- returning newly TRIGGERED signals to Phase 3
- reporting adapter-level errors

Architecture:

    MarketDataEngine
           |
           v
    StrategyAdapter
           |
           v
    Existing strategy/
           |
           v
    PENDING_RETEST
           |
           v
    retest_engine.py
           |
           v
    TRIGGERED SIGNAL
           |
           v
    PaperTradeEngine


DESIGN NOTES
------------

1. The adapter currently reuses several private helpers from
   strategy.backtest:

       _prepare_full_context
       _visible_zones
       _visible_liquidity_levels
       _current_structure_breaks
       _make_setup_key

   This is intentional for the first correctness-focused implementation.

   A future strategy refactor may extract these helpers into a shared
   public strategy context module. That refactor should NOT be performed
   silently as part of Phase 3.

2. The structural pipeline is currently recomputed from the complete
   adapter history on every candle.

   This is intentionally the simple correctness-first implementation.

   Do NOT optimize this prematurely.

3. Pending signals, consumed zones, and registered setup keys persist
   across candles because Phase 3 is a long-running process.

4. The adapter does NOT:
   - execute orders
   - simulate fills
   - manage open positions
   - calculate account risk
   - make session decisions
   - contain dashboard/UI logic
   - generate notifications
   - perform journal/persistence work

5. The caller is responsible for checking whether trading is permitted
   before asking the adapter to process a candle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Set, Tuple

import pandas as pd

from phase_03_paper.market.engine import Candle

from strategy.swings import find_swings
from strategy.structure import analyze_structure
from strategy.zones import find_zones
from strategy.liquidity import (
    detect_liquidity_levels,
    update_liquidity_sweeps,
)
from strategy.signals import (
    generate_flip_zone_signals,
    TradeSignal,
    SignalStatus,
)
from strategy.retest_engine import (
    advance_pending_signal,
    PendingOutcomeType,
)

from strategy.backtest import (
    _prepare_full_context,
    _visible_zones,
    _visible_liquidity_levels,
    _current_structure_breaks,
    _make_setup_key,
)


@dataclass(frozen=True)
class AdapterSignalEvent:
    """
    Phase 3 event emitted when an existing strategy signal
    becomes TRIGGERED on the current candle.

    This is an integration object only.

    PaperTradeEngine is responsible for execution behavior.
    """

    signal: TradeSignal
    setup_bar_index: int
    trigger_bar_index: int
    fill_price: float
    initial_risk: float


class StrategyAdapter:
    """
    Thin bridge between Phase 3 runtime candles and the existing
    Phase 1 SMC strategy.
    """

    def __init__(
        self,
        max_bars_to_retest: int = 20,
        reward_multiple: float = 2.0,
        stop_buffer: float = 0.0,
        entry_mode: str = "midpoint",
        max_bars_after_sweep: int = 10,
        max_bars_from_break: int = 5,
        warmup_bars: int = 4,
    ) -> None:

        if max_bars_to_retest < 0:
            raise ValueError("max_bars_to_retest must be >= 0")

        if reward_multiple <= 0:
            raise ValueError("reward_multiple must be > 0")

        if max_bars_after_sweep < 0:
            raise ValueError("max_bars_after_sweep must be >= 0")

        if max_bars_from_break < 0:
            raise ValueError("max_bars_from_break must be >= 0")

        if warmup_bars < 0:
            raise ValueError("warmup_bars must be >= 0")

        if entry_mode not in {"midpoint", "extreme"}:
            raise ValueError(
                "entry_mode must be 'midpoint' or 'extreme'"
            )

        self._max_bars_to_retest = max_bars_to_retest
        self._reward_multiple = reward_multiple
        self._stop_buffer = stop_buffer
        self._entry_mode = entry_mode
        self._max_bars_after_sweep = max_bars_after_sweep
        self._max_bars_from_break = max_bars_from_break
        self._warmup_bars = warmup_bars

        # --------------------------------------------------------
        # Causal candle history
        # --------------------------------------------------------

        self._rows: List[dict] = []

        # --------------------------------------------------------
        # Cached signal-context state
        # --------------------------------------------------------
        #
        # _prepare_full_context() performs preparation work that
        # does not need to be repeated for rows already processed.
        #
        # The context is prepared once and then extended
        # chronologically as new candles arrive.
        #
        # IMPORTANT:
        # This cache does NOT change the structural strategy
        # pipeline. Swings, structure, zones, and liquidity are
        # still recalculated from the complete causal history.
        # --------------------------------------------------------

        self._prepared_context = None

        # --------------------------------------------------------
        # Persistent cross-candle strategy state
        # --------------------------------------------------------

        self._pending_signals: List[
            Tuple[TradeSignal, int]
        ] = []

        self._registered_setup_keys: Set[tuple] = set()

        self._consumed_zones: Set = set()

        # --------------------------------------------------------
        # Adapter diagnostics
        # --------------------------------------------------------

        self._errors: List[str] = []

    # ============================================================
    # PUBLIC API
    # ============================================================

    def on_candle(
        self,
        candle: Candle,
    ) -> List[AdapterSignalEvent]:
        """
        Process exactly one new candle.

        Returns:
            A list of signals that became TRIGGERED on this candle.

        Normally this list is empty.

        Expired and invalidated pending signals are removed from
        adapter state. Detailed audit/journal handling belongs to
        the appropriate Phase 3 reporting layer.
        """

        self._append_candle(candle)

        current_index = len(self._rows) - 1

        # --------------------------------------------------------
        # Warmup
        # --------------------------------------------------------

        if current_index < self._warmup_bars:
            return []

        # --------------------------------------------------------
        # Build causal DataFrame
        # --------------------------------------------------------

        df = self._build_dataframe()

        current_time = pd.Timestamp(
            df.index[current_index]
        )

        current_open = float(
            df.iloc[current_index]["open"]
        )

        current_high = float(
            df.iloc[current_index]["high"]
        )

        current_low = float(
            df.iloc[current_index]["low"]
        )

        # --------------------------------------------------------
        # Existing strategy pipeline
        # --------------------------------------------------------

        try:
            (
                all_structure_breaks,
                all_zones,
                all_liquidity_levels,
                signal_context,
            ) = self._build_strategy_context(df)

        except Exception as exc:
            self._record_error(
                current_index,
                "strategy pipeline",
                exc,
            )
            return []

        current_visible_zones = _visible_zones(
            all_zones,
            current_time,
        )

        triggered_events: List[AdapterSignalEvent] = []

        # --------------------------------------------------------
        # 1. Advance existing pending signals
        # --------------------------------------------------------

        self._advance_pending_signals(
            current_index=current_index,
            current_time=current_time,
            current_open=current_open,
            current_high=current_high,
            current_low=current_low,
            visible_zones=current_visible_zones,
            triggered_events=triggered_events,
        )

        # --------------------------------------------------------
        # 2. Discover new signals born on this exact candle
        # --------------------------------------------------------

        self._discover_new_signals(
            current_index=current_index,
            current_time=current_time,
            structure_breaks=all_structure_breaks,
            visible_zones=current_visible_zones,
            liquidity_levels=all_liquidity_levels,
            signal_context=signal_context,
            triggered_events=triggered_events,
        )

        return triggered_events

    # ============================================================
    # CANDLE HISTORY
    # ============================================================

    def _append_candle(self, candle: Candle) -> None:
        """Append one validated candle to causal history."""

        self._rows.append(
            {
                "timestamp": pd.Timestamp(candle.timestamp),
                "open": float(candle.open),
                "high": float(candle.high),
                "low": float(candle.low),
                "close": float(candle.close),
            }
        )

    def _build_dataframe(self) -> pd.DataFrame:
        """Build the causal OHLC DataFrame used by the strategy."""

        df = pd.DataFrame(self._rows)

        df = df.set_index("timestamp")

        df.index = pd.DatetimeIndex(df.index)

        return df

    # ============================================================
    # STRATEGY CONTEXT
    # ============================================================

    def _build_strategy_context(
        self,
        df: pd.DataFrame,
    ):
        """
        Run the existing Phase 1 structural pipeline.

        No strategy logic is implemented here.
        """

        all_swings = find_swings(df)

        all_structure_breaks, _, _ = analyze_structure(
            df,
            all_swings,
        )

        all_zones = find_zones(
            df,
            all_structure_breaks,
        )

        all_liquidity_levels = detect_liquidity_levels(
            df,
            all_swings,
        )

        all_liquidity_levels = update_liquidity_sweeps(
            df,
            all_liquidity_levels,
        )

        # --------------------------------------------------------
        # Prepared signal context
        # --------------------------------------------------------
        #
        # First candle:
        #     Prepare the complete signal context.
        #
        # Later candles:
        #     Extend the existing context with exactly one new
        #     causal candle.
        #
        # Persistent state already stored inside the context,
        # including bar_index_cache and consumed_zones, remains
        # attached to the same context object.
        # --------------------------------------------------------

        if self._prepared_context is None:
            signal_context = _prepare_full_context(df)
        else:
            signal_context = self._extend_prepared_context(
                self._prepared_context,
                self._rows[-1],
            )

        self._prepared_context = signal_context

        return (
            all_structure_breaks,
            all_zones,
            all_liquidity_levels,
            signal_context,
        )

    def _extend_prepared_context(
        self,
        context: dict,
        new_row: dict,
    ) -> dict:
        """
        Extend the prepared signal context by one causal candle.
        This avoids repeating full-context preparation for rows
        that have already been processed. Strategy logic is
        intentionally unchanged.
        """

        working = context["working"]
        timestamp_column = context["timestamp_column"]

        # --------------------------------------------------------
        # Normalize timestamp consistently with the existing
        # signal preparation pipeline.
        # --------------------------------------------------------
        timestamp = pd.Timestamp(new_row["timestamp"])

        # --------------------------------------------------------
        # Append exactly one new row.
        # --------------------------------------------------------
        new_index = len(working)

        working.loc[new_index] = {
            timestamp_column: timestamp,
            "open": float(new_row["open"]),
            "high": float(new_row["high"]),
            "low": float(new_row["low"]),
            "close": float(new_row["close"]),
        }

        # --------------------------------------------------------
        # Extend timestamp lookup without replacing existing
        # entries for duplicate timestamps.
        # --------------------------------------------------------
        context["timestamp_lookup"].setdefault(
            timestamp,
            new_index,
        )

        # --------------------------------------------------------
        # Refresh the arrays consumed by the signal engine.
        # --------------------------------------------------------
        context["highs"] = working["high"].to_numpy()
        context["lows"] = working["low"].to_numpy()
        context["opens"] = working["open"].to_numpy()
        context["timestamps"] = working[timestamp_column].to_numpy()

        return context

    # ============================================================
    # PENDING SIGNAL STATE MACHINE
    # ============================================================

    def _advance_pending_signals(
        self,
        *,
        current_index: int,
        current_time: pd.Timestamp,
        current_open: float,
        current_high: float,
        current_low: float,
        visible_zones,
        triggered_events: List[AdapterSignalEvent],
    ) -> None:
        """
        Advance every pending signal through the shared
        retest state machine.
        """

        still_pending: List[
            Tuple[TradeSignal, int]
        ] = []

        for signal, setup_index in self._pending_signals:

            outcome = advance_pending_signal(
                signal=signal,
                setup_idx=setup_index,
                current_index=current_index,
                current_time=current_time,
                current_open=current_open,
                current_high=current_high,
                current_low=current_low,
                visible_zones=visible_zones,
                max_bars_to_retest=self._max_bars_to_retest,
            )

            if outcome.outcome in {
                PendingOutcomeType.EXPIRED,
                PendingOutcomeType.INVALIDATED,
            }:
                continue

            if outcome.outcome == PendingOutcomeType.STILL_PENDING:
                still_pending.append(
                    (signal, setup_index)
                )
                continue

            if outcome.outcome == PendingOutcomeType.TRIGGERED:

                if (
                    outcome.fill_price is None
                    or outcome.initial_risk is None
                ):
                    self._errors.append(
                        f"{current_index}: triggered signal "
                        f"returned incomplete execution data"
                    )
                    continue

                triggered_events.append(
                    AdapterSignalEvent(
                        signal=signal,
                        setup_bar_index=setup_index,
                        trigger_bar_index=current_index,
                        fill_price=float(
                            outcome.fill_price
                        ),
                        initial_risk=float(
                            outcome.initial_risk
                        ),
                    )
                )

        self._pending_signals = still_pending

    # ============================================================
    # NEW SIGNAL DISCOVERY
    # ============================================================

    def _discover_new_signals(
        self,
        *,
        current_index: int,
        current_time: pd.Timestamp,
        structure_breaks,
        visible_zones,
        liquidity_levels,
        signal_context,
        triggered_events: List[AdapterSignalEvent],
    ) -> None:
        """
        Discover only signals whose structure break was born
        on the current candle.
        """

        try:
            current_structure_breaks = (
                _current_structure_breaks(
                    structure_breaks,
                    current_time,
                )
            )

            if not current_structure_breaks:
                return

            visible_liquidity_levels = (
                _visible_liquidity_levels(
                    liquidity_levels,
                    current_time,
                )
            )

            if not visible_liquidity_levels:
                return

            signals = generate_flip_zone_signals(
                liquidity_levels=visible_liquidity_levels,
                structure_breaks=current_structure_breaks,
                zones=visible_zones,
                max_bars_after_sweep=self._max_bars_after_sweep,
                max_bars_from_break=self._max_bars_from_break,
                max_bars_to_retest=self._max_bars_to_retest,
                reward_multiple=self._reward_multiple,
                stop_buffer=self._stop_buffer,
                entry_mode=self._entry_mode,
                precomputed_context=signal_context,
                upto_index=current_index,
                run_retest_simulation=False,
                consumed_zones=self._consumed_zones,
            )

            if not signals:
                return

            for signal in signals:
                self._register_signal_if_valid(
                    signal=signal,
                    current_index=current_index,
                )

        except Exception as exc:
            self._record_error(
                current_index,
                "signal discovery",
                exc,
            )

    def _register_signal_if_valid(
        self,
        *,
        signal: TradeSignal,
        current_index: int,
    ) -> None:
        """Validate and register a newly discovered pending signal."""

        break_index = getattr(
            signal,
            "bar_index",
            None,
        )

        setup_index = getattr(
            signal,
            "setup_bar_index",
            None,
        )

        if break_index is None or setup_index is None:
            return

        try:
            break_index = int(break_index)
            setup_index = int(setup_index)
        except (TypeError, ValueError):
            return

        # Signal cannot reference the future.
        if setup_index > current_index:
            return

        if break_index > current_index:
            return

        # Structure break must occur after setup.
        if break_index <= setup_index:
            return

        # Only pending signals enter the adapter's pending state.
        if signal.status != SignalStatus.PENDING_RETEST:
            return

        # New signal must be born on this exact candle.
        if break_index != current_index:
            return

        signal_key = _make_setup_key(
            signal,
            break_index,
        )

        if signal_key in self._registered_setup_keys:
            return

        self._registered_setup_keys.add(
            signal_key
        )

        self._pending_signals.append(
            (signal, break_index)
        )

    # ============================================================
    # DIAGNOSTICS
    # ============================================================

    def _record_error(
        self,
        current_index: int,
        stage: str,
        exc: Exception,
    ) -> None:
        """Record a structured adapter error."""

        self._errors.append(
            f"{current_index}: {stage} error: "
            f"{type(exc).__name__}: {exc}"
        )

    @property
    def errors(self) -> List[str]:
        """Return a copy of adapter errors."""

        return list(self._errors)

    @property
    def pending_count(self) -> int:
        """Number of currently pending strategy signals."""

        return len(self._pending_signals)

    @property
    def candle_count(self) -> int:
        """Number of candles processed by the adapter."""

        return len(self._rows)

