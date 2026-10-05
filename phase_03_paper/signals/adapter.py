"""
phase_03_paper/signals/adapter.py

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

DESIGN NOTES
------------

1. The adapter reuses several private helpers from strategy.backtest:
       _prepare_full_context
       _visible_zones
       _visible_liquidity_levels
       _current_structure_breaks
       _make_setup_key

2. Swing detection, structure, zones, and liquidity are incremental.

3. Pending signals, consumed zones, registered setup keys, and
   liquidity levels persist across candles.

4. The prepared signal context is persistent.

5. The adapter does NOT:
   - execute orders
   - simulate fills
   - manage open positions
   - calculate account risk
   - make session decisions
   - contain dashboard/UI logic
   - generate notifications
   - perform journal/persistence work

ENTRY FILL MODE
---------------

The adapter accepts strict_entry_fill (default False), forwarded
verbatim to strategy/retest_engine.advance_pending_signal.

    False -- original rule: triggers on zone-edge touch.
    True  -- limit-order realism: triggers only when price trades
             through the signal's entry price.
"""


from __future__ import annotations

from dataclasses import dataclass
from typing import List, Set, Tuple

import pandas as pd

from phase_03_paper.market.engine import Candle

from strategy.swings import SwingState
from strategy.structure import analyze_structure, StructureState
from strategy.zones import Zone, find_zones, compute_zone_for_break
from strategy.liquidity import (
    LiquidityLevel,
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

from strategy.mtf_structure import build_mtf_dataset_with_structure
from strategy.confluence import build_mtf_signal_filter
from dashboard.mtf_context import MTFConfig


@dataclass(frozen=True)
class AdapterSignalEvent:
    """
    Phase 3 event emitted when an existing strategy signal
    becomes TRIGGERED on the current candle.
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
        mtf_enabled: bool = True,
        mtf_macro_tf: str = "1h",
        mtf_internal_tf: str = "15min",
        mtf_soft_internal_conflict: bool = False,
        mtf_min_bars: int = 200,
        strict_entry_fill: bool = False,
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
        self._strict_entry_fill = bool(strict_entry_fill)

        self._mtf_enabled = mtf_enabled
        self._mtf_macro_tf = mtf_macro_tf
        self._mtf_internal_tf = mtf_internal_tf
        self._mtf_soft_internal_conflict = mtf_soft_internal_conflict
        self._mtf_min_bars = mtf_min_bars
        self._mtf_filter_fn = None
        self._mtf_df = None

        self._rows: List[dict] = []

        self._prepared_context = None

        self._pending_signals: List[
            Tuple[TradeSignal, int]
        ] = []

        self._registered_setup_keys: Set[tuple] = set()

        self._consumed_zones: Set = set()

        self._liquidity_levels: List[LiquidityLevel] = []

        self._processed_swing_keys: Set[tuple] = set()

        self._swing_state = SwingState()

        from strategy.swings import find_swings as _mtf_swings_fn
        self._mtf_swings_fn = _mtf_swings_fn
        self._confirmed_swings: List = []
        self._swing_rows_processed = 0

        self._structure_state = StructureState()
        self._structure_rows_processed = 0
        self._structure_swings_fed = 0

        self._zones: List[Zone] = []
        self._processed_break_keys: Set[tuple] = set()

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

        Returns a list of signals that became TRIGGERED on this
        candle. Normally empty.
        """

        self._append_candle(candle)

        current_index = len(self._rows) - 1

        if current_index < self._warmup_bars:
            return []

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

        self._mtf_df = df
        self._mtf_filter_fn = None

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

        self._advance_pending_signals(
            current_index=current_index,
            current_time=current_time,
            current_open=current_open,
            current_high=current_high,
            current_low=current_low,
            visible_zones=current_visible_zones,
            triggered_events=triggered_events,
        )

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
    # STABLE KEYS
    # ============================================================

    @staticmethod
    def _make_swing_key(swing) -> tuple:
        return (
            swing.swing_type,
            swing.price,
            swing.formed_at,
            swing.confirmed_at,
        )

    @staticmethod
    def _make_break_key(brk) -> tuple:
        return (
            brk.break_type,
            brk.direction,
            brk.broken_at,
            brk.break_price,
        )

    # ============================================================
    # INCREMENTAL SWING STATE
    # ============================================================

    def _advance_swing_state(self, df: pd.DataFrame) -> List:

        total_rows = len(df)

        for i in range(self._swing_rows_processed, total_rows):
            ts = df.index[i]
            high = float(df.iloc[i]["high"])
            low = float(df.iloc[i]["low"])
            close = float(df.iloc[i]["close"])

            confirmed = self._swing_state.step(
                ts=ts,
                high=high,
                low=low,
                close=close,
            )

            if confirmed is not None:
                self._confirmed_swings.append(confirmed)

        self._swing_rows_processed = total_rows

        return list(self._confirmed_swings)

    # ============================================================
    # INCREMENTAL STRUCTURE STATE
    # ============================================================

    def _advance_structure_state(self, df: pd.DataFrame):

        total_rows = len(df)

        for i in range(self._structure_rows_processed, total_rows):
            ts = df.index[i]
            close = float(df.iloc[i]["close"])

            while (
                self._structure_swings_fed < len(self._confirmed_swings)
                and self._confirmed_swings[
                    self._structure_swings_fed
                ].confirmed_at <= ts
            ):
                self._structure_state.add_swings(
                    [self._confirmed_swings[self._structure_swings_fed]]
                )
                self._structure_swings_fed += 1

            self._structure_state.step(ts=ts, close=close)

        self._structure_rows_processed = total_rows

        return (
            self._structure_state.breaks,
            self._structure_state.classified_swings,
            self._structure_state.current_trend,
        )

    # ============================================================
    # INCREMENTAL ZONE STATE
    # ============================================================

    def _advance_zone_state(
        self,
        df: pd.DataFrame,
        all_structure_breaks,
    ) -> List:

        for brk in all_structure_breaks:
            key = self._make_break_key(brk)

            if key in self._processed_break_keys:
                continue

            break_idx = df.index.get_loc(brk.broken_at)

            zone = compute_zone_for_break(df, brk, break_idx)

            self._processed_break_keys.add(key)

            if zone is not None:
                self._zones.append(zone)

        return self._zones

    # ============================================================
    # STRATEGY CONTEXT
    # ============================================================

    def _build_strategy_context(
        self,
        df: pd.DataFrame,
    ):

        all_swings = self._advance_swing_state(df)

        all_structure_breaks, _, _ = self._advance_structure_state(df)

        all_zones = self._advance_zone_state(
            df,
            all_structure_breaks,
        )

        new_swings = []

        for swing in all_swings:
            swing_key = self._make_swing_key(swing)

            if swing_key not in self._processed_swing_keys:
                new_swings.append(swing)

        if new_swings:
            new_levels = detect_liquidity_levels(
                df,
                new_swings,
            )

            self._liquidity_levels.extend(new_levels)

            for swing in new_swings:
                self._processed_swing_keys.add(
                    self._make_swing_key(swing)
                )

        self._liquidity_levels = update_liquidity_sweeps(
            df,
            self._liquidity_levels,
        )

        all_liquidity_levels = self._liquidity_levels

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

        working = context["working"]
        timestamp_column = context["timestamp_column"]

        timestamp = pd.Timestamp(new_row["timestamp"])

        new_index = len(working)

        working.loc[new_index] = {
            timestamp_column: timestamp,
            "open": float(new_row["open"]),
            "high": float(new_row["high"]),
            "low": float(new_row["low"]),
            "close": float(new_row["close"]),
        }

        context["timestamp_lookup"].setdefault(
            timestamp,
            new_index,
        )

        context["highs"] = working["high"].to_numpy()
        context["lows"] = working["low"].to_numpy()
        context["opens"] = working["open"].to_numpy()
        context["timestamps"] = working[
            timestamp_column
        ].to_numpy()

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
                strict_entry_fill=self._strict_entry_fill,
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
                        fill_price=float(outcome.fill_price),
                        initial_risk=float(outcome.initial_risk),
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
                    current_time=current_time,
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
        current_time: pd.Timestamp,
    ) -> None:

        break_index = getattr(signal, "bar_index", None)
        setup_index = getattr(signal, "setup_bar_index", None)

        if break_index is None or setup_index is None:
            return

        try:
            break_index = int(break_index)
            setup_index = int(setup_index)
        except (TypeError, ValueError):
            return

        if setup_index > current_index:
            return

        if break_index > current_index:
            return

        if break_index <= setup_index:
            return

        if signal.status != SignalStatus.PENDING_RETEST:
            return

        if break_index != current_index:
            return

        signal_key = _make_setup_key(signal, break_index)

        if signal_key in self._registered_setup_keys:
            return

        if self._mtf_enabled:
            mtf_filter = self._get_mtf_filter(current_index)
            if not mtf_filter(signal, current_time):
                return

        self._registered_setup_keys.add(signal_key)

        self._pending_signals.append(
            (signal, break_index)
        )

    # ============================================================
    # MTF FILTER
    # ============================================================

    def _get_mtf_filter(self, current_index: int):
        """Build (once per tick, on demand) the MTF confluence filter."""

        if self._mtf_filter_fn is not None:
            return self._mtf_filter_fn

        df = self._mtf_df

        if df is None or len(df) < self._mtf_min_bars:
            self._mtf_filter_fn = lambda signal, ts: False
            return self._mtf_filter_fn

        try:
            df_enriched = build_mtf_dataset_with_structure(
                df,
                macro_swings_fn=self._mtf_swings_fn,
                internal_swings_fn=self._mtf_swings_fn,
                config=MTFConfig(
                    macro_tf=self._mtf_macro_tf,
                    internal_tf=self._mtf_internal_tf,
                ),
            )
            self._mtf_filter_fn = build_mtf_signal_filter(
                df_enriched,
                soft_internal_conflict=self._mtf_soft_internal_conflict,
            )
        except Exception as exc:
            self._record_error(current_index, "mtf context", exc)
            self._mtf_filter_fn = lambda signal, ts: False

        return self._mtf_filter_fn

    # ============================================================
    # DIAGNOSTICS
    # ============================================================

    def _record_error(
        self,
        current_index: int,
        stage: str,
        exc: Exception,
    ) -> None:

        self._errors.append(
            f"{current_index}: {stage} error: "
            f"{type(exc).__name__}: {exc}"
        )

    @property
    def errors(self) -> List[str]:
        return list(self._errors)

    @property
    def pending_count(self) -> int:
        return len(self._pending_signals)

    @property
    def candle_count(self) -> int:
        return len(self._rows)
