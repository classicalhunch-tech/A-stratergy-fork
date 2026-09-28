path = "phase_03_paper/signals/adapter.py"
with open(path, "r", encoding="utf-8") as f:
    src = f.read()

# 0. init: add self._mtf_df
old = "        self._mtf_min_bars = mtf_min_bars\n        self._mtf_filter_fn = None\n"
new = "        self._mtf_min_bars = mtf_min_bars\n        self._mtf_filter_fn = None\n        self._mtf_df = None\n"
assert src.count(old) == 1, "init anchor"
src = src.replace(old, new, 1)

# A. replace the eager per-candle rebuild inside on_candle
start_marker = "        if self._mtf_enabled:\n            if len(df) < self._mtf_min_bars:"
end_marker = "                    self._mtf_filter_fn = lambda signal, ts: False\n"
s = src.index(start_marker)
e = src.index(end_marker, src.index("except Exception as exc:", s)) + len(end_marker)
lazy = (
    "        # Lazy MTF: the filter is built only if a signal needs it this\n"
    "        # tick (see _get_mtf_filter). Same causal DataFrame, same result\n"
    "        # as eager building, without rebuilding on every candle.\n"
    "        self._mtf_df = df\n"
    "        self._mtf_filter_fn = None\n"
)
src = src[:s] + lazy + src[e:]

# B. gate call in _register_signal_if_valid
old = ("        if self._mtf_enabled:\n"
       "            if self._mtf_filter_fn is None:\n"
       "                return\n"
       "            if not self._mtf_filter_fn(signal, current_time):\n"
       "                return\n")
new = ("        if self._mtf_enabled:\n"
       "            mtf_filter = self._get_mtf_filter(current_index)\n"
       "            if not mtf_filter(signal, current_time):\n"
       "                return\n")
assert src.count(old) == 1, "gate anchor"
src = src.replace(old, new, 1)

# C. add _get_mtf_filter before _record_error
anchor = "    def _record_error(\n"
assert src.count(anchor) == 1, "record_error anchor"
method = '''    def _get_mtf_filter(self, current_index: int):
        """Build (once per tick, on demand) the MTF confluence filter.

        FAIL-CLOSED: returns a reject-everything predicate when there is
        too little history or MTF context building raises.
        """
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

'''
src = src.replace(anchor, method + anchor, 1)

with open(path, "w", encoding="utf-8") as f:
    f.write(src)
print("Lazy MTF patch applied; all anchors matched.")
