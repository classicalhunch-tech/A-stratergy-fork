from pathlib import Path

# ============================================================
# TARGET FILE SETUP
# ============================================================
path = Path("phase_03_paper/signals/adapter.py")
if not path.exists():
    raise FileNotFoundError(f"Target file not found: {path}")

with path.open("r", encoding="utf-8") as f:
    content = f.read()

# ============================================================
# PATCH 1 — INITIALIZATION STATE
# ============================================================

OLD_INIT = """        self._rows: List[dict] = []

        # --------------------------------------------------------
        # Persistent cross-candle strategy state
        # --------------------------------------------------------"""

NEW_INIT = """        self._rows: List[dict] = []

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
        # --------------------------------------------------------"""

if content.count(OLD_INIT) != 1:
    raise RuntimeError(
        "INIT anchor was not found exactly once. "
        "No changes were made."
    )


# ============================================================
# PATCH 2 — PREPARED SIGNAL CONTEXT & EXTENSION METHOD
# ============================================================

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
        \"\"\"
        Extend the prepared signal context by one causal candle.
        This avoids repeating full-context preparation for rows
        that have already been processed. Strategy logic is
        intentionally unchanged.
        \"\"\"

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
    # ============================================================"""

if content.count(OLD_CONTEXT) != 1:
    raise RuntimeError(
        "CONTEXT anchor was not found exactly once. "
        "No changes were made."
    )


# ============================================================
# APPLY EXACTLY TWO REPLACEMENTS
# ============================================================

content = content.replace(OLD_INIT, NEW_INIT, 1)
content = content.replace(OLD_CONTEXT, NEW_CONTEXT, 1)


# ============================================================
# WRITE CHANGES TO FILE
# ============================================================

with path.open("w", encoding="utf-8") as f:
    f.write(content)


# ============================================================
# RESULT OUTPUT
# ============================================================

print("=" * 70)
print("ADAPTER OPTIMIZATION PATCH")
print("=" * 70)
print("PASS: patch applied successfully.")
print(f"File modified: {path}")
print()
print("Modified Components:")
print("  - Added persistent `_prepared_context` tracking")
print("  - Built full context once on the initial candle")
print("  - Extended context causally for subsequent candles")
print("  - Maintained incremental timestamp lookup & NumPy updates")
print("  - Preserved existing `bar_index_cache` and `consumed_zones`")
print()
print("Intentionally Unchanged:")
print("  - Swing detection, market structure, and zones")
print("  - Liquidity detection & strategy core signaling rules")
print()
print("Next steps:")
print("  1. Run compile check: python -m py_compile phase_03_paper/signals/adapter.py")
print("  2. Run smoke tests:   python -m unittest phase_03_paper.tests.test_strategy_adapter_smoke -v")
print("=" * 70)