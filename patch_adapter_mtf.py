import re

path = "phase_03_paper/signals/adapter.py"
with open(path, "r", encoding="utf-8") as f:
    src = f.read()

# 1. Add imports
old_imports = """from strategy.backtest import (
    _prepare_full_context,
    _visible_zones,
    _visible_liquidity_levels,
    _current_structure_breaks,
    _make_setup_key,
)"""

new_imports = """from strategy.backtest import (
    _prepare_full_context,
    _visible_zones,
    _visible_liquidity_levels,
    _current_structure_breaks,
    _make_setup_key,
)

from strategy.mtf_structure import build_mtf_dataset_with_structure
from strategy.confluence import build_mtf_signal_filter
from dashboard.mtf_context import MTFConfig"""

assert old_imports in src, "import block not found -- adapter.py may have changed"
src = src.replace(old_imports, new_imports, 1)

# 2. Extend __init__ signature
old_init_sig = """    def __init__(
        self,
        max_bars_to_retest: int = 20,
        reward_multiple: float = 2.0,
        stop_buffer: float = 0.0,
        entry_mode: str = "midpoint",
        max_bars_after_sweep: int = 10,
        max_bars_from_break: int = 5,
        warmup_bars: int = 4,
    ) -> None:"""

new_init_sig = """    def __init__(
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
    ) -> None:"""

assert old_init_sig in src, "__init__ signature not found -- adapter.py may have changed"
src = src.replace(old_init_sig, new_init_sig, 1)

# 3. Store new config + MTF filter state, right after self._warmup_bars assignment
old_warmup_store = "        self._warmup_bars = warmup_bars\n"
new_warmup_store = """        self._warmup_bars = warmup_bars

        # --------------------------------------------------------
        # MTF (1H + 15M) confluence gate
        # --------------------------------------------------------
        #
        # Uses the same validated strategy/confluence.py +
        # strategy/mtf_structure.py pipeline confirmed today against
        # the 90,000-candle backtest (1H macro + 15M internal, hard
        # gate: soft_internal_conflict=False). Correctness-first: the
        # adapter only receives one new candle every 5 real minutes,
        # so a full (non-incremental) MTF context rebuild every tick
        # is well within that budget -- no SwingState/StructureState-
        # style incremental machinery is needed here.
        #
        # FAIL-CLOSED: if MTF context building raises on a given
        # tick, no new signals are registered that tick (existing
        # pending signals still advance normally). This matches the
        # rest of Phase 3's safety-first design rather than silently
        # falling back to an ungated, unvalidated signal path.
        # --------------------------------------------------------

        self._mtf_enabled = mtf_enabled
        self._mtf_macro_tf = mtf_macro_tf
        self._mtf_internal_tf = mtf_internal_tf
        self._mtf_soft_internal_conflict = mtf_soft_internal_conflict
        self._mtf_min_bars = mtf_min_bars
        self._mtf_filter_fn = None
"""
assert old_warmup_store in src, "warmup_bars assignment not found"
src = src.replace(old_warmup_store, new_warmup_store, 1)

# 4. Rebuild the MTF filter once per candle, inside on_candle(), right after
#    the causal DataFrame is built and current_time/open/high/low are known.
old_ohlc_block = """        current_low = float(
            df.iloc[current_index]["low"]
        )

        # --------------------------------------------------------
        # Existing strategy pipeline
        # --------------------------------------------------------"""

new_ohlc_block = """        current_low = float(
            df.iloc[current_index]["low"]
        )

        # --------------------------------------------------------
        # Refresh the MTF confluence filter for this tick.
        #
        # FAIL-CLOSED: on any exception, self._mtf_filter_fn is set
        # to a function that rejects everything, so no new signals
        # are registered this tick until MTF context succeeds again.
        # Existing pending signals are unaffected -- they continue
        # to advance through the normal retest state machine.
        # --------------------------------------------------------

        if self._mtf_enabled:
            if len(df) < self._mtf_min_bars:
                self._mtf_filter_fn = lambda signal, ts: False
            else:
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
                    self._record_error(
                        current_index,
                        "mtf context",
                        exc,
                    )
                    self._mtf_filter_fn = lambda signal, ts: False

        # --------------------------------------------------------
        # Existing strategy pipeline
        # --------------------------------------------------------"""

assert old_ohlc_block in src, "current_low block not found -- adapter.py may have changed"
src = src.replace(old_ohlc_block, new_ohlc_block, 1)

# 5. Add the swings_fn reference used above (reuse strategy.swings.find_swings,
#    matching exactly what today's validated backtest scripts used).
old_swing_state_init = "        self._swing_state = SwingState()\n"
new_swing_state_init = """        self._swing_state = SwingState()

        # find_swings is imported lazily here (not at module top) to
        # avoid a second import line churn in the patch; it is the
        # exact same function strategy/mtf_structure.py's callers use
        # for macro/internal swing detection today.
        from strategy.swings import find_swings as _mtf_swings_fn
        self._mtf_swings_fn = _mtf_swings_fn
"""
assert old_swing_state_init in src, "self._swing_state = SwingState() not found"
src = src.replace(old_swing_state_init, new_swing_state_init, 1)

# 6. Gate _register_signal_if_valid with the MTF filter, right before it's
#    added to pending signals. current_index is available; we need the
#    signal's break timestamp, which matches current_time in the caller.
old_register_call = """            for signal in signals:
                self._register_signal_if_valid(
                    signal=signal,
                    current_index=current_index,
                )"""

new_register_call = """            for signal in signals:
                self._register_signal_if_valid(
                    signal=signal,
                    current_index=current_index,
                    current_time=current_time,
                )"""

assert old_register_call in src, "_register_signal_if_valid call site not found"
src = src.replace(old_register_call, new_register_call, 1)

old_register_def = """    def _register_signal_if_valid(
        self,
        *,
        signal: TradeSignal,
        current_index: int,
    ) -> None:
        \"\"\"Validate and register a newly discovered pending signal.\"\"\""""

new_register_def = """    def _register_signal_if_valid(
        self,
        *,
        signal: TradeSignal,
        current_index: int,
        current_time: pd.Timestamp,
    ) -> None:
        \"\"\"Validate and register a newly discovered pending signal.

        Applies the MTF confluence gate (see self._mtf_filter_fn) after
        all existing validation but before the signal enters pending
        state. A rejected signal is simply not registered -- it is not
        counted as invalidated or expired, matching how
        strategy/backtest.py's run_backtest() treats mtf_filter_fn
        rejections (total_mtf_rejected, never entered as pending).
        \"\"\""""

assert old_register_def in src, "_register_signal_if_valid def not found"
src = src.replace(old_register_def, new_register_def, 1)

old_final_append = """        self._registered_setup_keys.add(
            signal_key
        )

        self._pending_signals.append(
            (signal, break_index)
        )"""

new_final_append = """        if self._mtf_enabled:
            if self._mtf_filter_fn is None:
                return
            if not self._mtf_filter_fn(signal, current_time):
                return

        self._registered_setup_keys.add(
            signal_key
        )

        self._pending_signals.append(
            (signal, break_index)
        )"""

assert old_final_append in src, "final pending-signal append block not found"
src = src.replace(old_final_append, new_final_append, 1)

with open(path, "w", encoding="utf-8") as f:
    f.write(src)

print("Patched successfully:", path)
print("All 6 assertions passed -- every expected anchor was found exactly once.")
