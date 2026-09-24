path = "phase_03_paper/signals/adapter.py"

with open(path, "r", encoding="utf-8") as f:
    content = f.read()

OLD_INIT = """        self._rows: List[dict] = []

        # --------------------------------------------------------
        # Persistent cross-candle strategy state
        # --------------------------------------------------------"""

NEW_INIT = """        self._rows: List[dict] = []

        # --------------------------------------------------------
        # Cached signal-context state (avoids re-copying/sorting/
        # rebuilding lookups for the full history on every candle)
        # --------------------------------------------------------

        self._prepared_context = None

        # --------------------------------------------------------
        # Persistent cross-candle strategy state
        # --------------------------------------------------------"""

OLD_CONTEXT = """        all_liquidity_levels = update_liquidity_sweeps(
            df,
            all_liquidity_levels,
        )

        signal_context = _prepare_full_context(df)

        return (
            all_structure_breaks,
            all_zones,
            all_liquidity_levels,
            signal_context,
        )

    # ============================================================
    # PENDING SIGNAL STATE MACHINE
    # ============================================================"""

NEW_CONTEXT = """        all_liquidity_levels = update_liquidity_sweeps(
            df,
            all_liquidity_levels,
        )

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
        \"\"\"
        Cheaply extend an existing prepared signal context with
        exactly one new causal row, instead of re-copying,
        re-sorting, and rebuilding the timestamp lookup for the
        entire history the way _prepare_full_context does.

        No strategy logic is changed here -- this only removes
        redundant data-preparation work for rows already prepared
        on a previous candle.
        \"\"\"

        working = context["working"]
        timestamp_column = context["timestamp_column"]

        new_index = len(working)

        working.loc[new_index] = {
            timestamp_column: new_row["timestamp"],
            "open": new_row["open"],
            "high": new_row["high"],
            "low": new_row["low"],
            "close": new_row["close"],
        }

        context["timestamp_lookup"].setdefault(
            new_row["timestamp"],
            new_index,
        )

        context["highs"] = working["high"].to_numpy()
        context["lows"] = working["low"].to_numpy()
        context["opens"] = working["open"].to_numpy()
        context["timestamps"] = working[timestamp_column].to_numpy()

        return context

    # ============================================================
    # PENDING SIGNAL STATE MACHINE
    # ============================================================"""

assert OLD_INIT in content, "INIT anchor not found -- aborting, no changes made"
assert OLD_CONTEXT in content, "CONTEXT anchor not found -- aborting, no changes made"

content = content.replace(OLD_INIT, NEW_INIT)
content = content.replace(OLD_CONTEXT, NEW_CONTEXT)

with open(path, "w", encoding="utf-8") as f:
    f.write(content)

print("Patch applied successfully.")