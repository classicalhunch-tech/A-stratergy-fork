"""
phase_02_optimization/warmup_probe.py

Purpose
-------
Determine reasonable warmup-size candidates for Phase 2 walk-forward
analysis by measuring how the strategy's historical structural state
changes as more candles are supplied.

IMPORTANT
---------
This is a DIAGNOSTIC PROBE.

It is:
    - NOT the walk-forward engine
    - NOT an optimizer
    - NOT a performance test
    - NOT a final warmup selector

It does not change strategy parameters and does not select the
"best" warmup.

The output is used to identify reasonable candidates that are then
validated with the canonical walk-forward engine.

Run from the project root:

    python -m phase_02_optimization.warmup_probe

Dataset:
    your_data_file.csv

IMPORTANT INPUT NOTE
--------------------
The canonical run_backtest() engine may normalize dataframe column names
before passing data into the strategy components.

This probe intentionally does NOT guess the private
strategy.backtest._validate_dataframe() signature.

Column check is EXACT-CASE, not case-insensitive: a case-insensitive
check that passed but left the DataFrame's actual column casing
unchanged would give false confidence — it wouldn't guarantee
find_swings()/analyze_structure()/etc. can actually index the columns
they expect. If the dataset's columns aren't already exactly:

    open
    high
    low
    close

this probe raises immediately with the real column names shown, rather
than silently proceeding on a check that doesn't match what downstream
code will actually do.

What this probe measures
------------------------
For progressively larger historical slices it measures:

    - swings
    - structure breaks
    - classified swings
    - zones
    - liquidity levels
    - current trend
    - computation time

These are structural diagnostics.

A larger number does NOT mean "better".

The objective is to see whether structural state begins to show
diminishing changes as more historical context is supplied.

A candidate warmup must still be validated with walk_forward.py.
"""


from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pandas as pd

from strategy.swings import find_swings
from strategy.structure import analyze_structure
from strategy.zones import find_zones
from strategy.liquidity import (
    detect_liquidity_levels,
    update_liquidity_sweeps,
)


# ============================================================================
# Configuration
# ============================================================================

DATA_FILE = Path("your_data_file.csv")

# Historical-context candidates to inspect.
#
# These are candidates, NOT recommendations.
WARMUP_CANDIDATES = (
    50,
    100,
    200,
    300,
    500,
    750,
    1000,
)

MAX_WARMUP_REQUIRED = max(WARMUP_CANDIDATES)

REQUIRED_OHLC_COLUMNS = {
    "open",
    "high",
    "low",
    "close",
}


# ============================================================================
# Data loading
# ============================================================================

def load_dataset(path: Path) -> pd.DataFrame:
    """
    Load and perform basic validation of the historical dataset.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path.resolve()}"
        )

    df = pd.read_csv(
        path,
        index_col=0,
        parse_dates=True,
    )

    if df.empty:
        raise ValueError(
            "Dataset is empty."
        )

    if not isinstance(
        df.index,
        pd.DatetimeIndex,
    ):
        raise ValueError(
            "Dataset index must be a pandas DatetimeIndex."
        )

    if df.index.has_duplicates:
        raise ValueError(
            "Dataset contains duplicate timestamps."
        )

    if not df.index.is_monotonic_increasing:
        df = df.sort_index()

    # ------------------------------------------------------------------------
    # Required OHLC columns — EXACT case, not case-insensitive.
    # ------------------------------------------------------------------------
    #
    # We deliberately do NOT silently rename columns here.
    #
    # A case-insensitive check (e.g. matching "Open" as satisfying "open")
    # would pass even though slice_df still has "Open" — and
    # find_swings()/analyze_structure()/etc. would then fail (or silently
    # misbehave) trying to index the exact lowercase names they expect.
    # The canonical backtest engine may have its own normalization logic;
    # guessing it here could make this diagnostic disagree with the real
    # backtest pipeline, so we fail loudly instead of guessing.
    #
    missing = REQUIRED_OHLC_COLUMNS - set(df.columns)

    if missing:
        raise ValueError(
            "Dataset does not expose the exact required OHLC columns "
            f"(case-sensitive). Missing: {sorted(missing)}. "
            f"Actual columns found: {list(df.columns)}. "
            "Confirm the actual column names and the normalization "
            "performed by strategy.backtest._validate_dataframe() before "
            "renaming — do not guess it here."
        )

    return df


# ============================================================================
# Structural-state probe
# ============================================================================

def probe_history(
    df: pd.DataFrame,
    n: int,
) -> dict[str, Any]:
    """
    Run the actual structural components over the first n candles.

    This is deliberately diagnostic.

    No strategy parameters are changed.
    No trades are evaluated.
    No optimization is performed.
    """

    if n <= 0:
        raise ValueError(
            "Probe size n must be positive."
        )

    if n > len(df):
        raise ValueError(
            f"Probe size {n} exceeds dataset length {len(df)}."
        )

    start_time = time.perf_counter()

    slice_df = df.iloc[:n].copy()

    # ------------------------------------------------------------------------
    # Swing detection
    # ------------------------------------------------------------------------

    swings = find_swings(
        slice_df
    )

    # ------------------------------------------------------------------------
    # Market structure
    # ------------------------------------------------------------------------

    (
        breaks,
        classified_swings,
        trend,
    ) = analyze_structure(
        slice_df,
        swings,
    )

    # ------------------------------------------------------------------------
    # Supply / demand zones
    # ------------------------------------------------------------------------

    zones = find_zones(
        slice_df,
        breaks,
    )

    # ------------------------------------------------------------------------
    # Liquidity
    # ------------------------------------------------------------------------

    liquidity_levels = detect_liquidity_levels(
        slice_df,
        swings,
    )

    liquidity_levels = update_liquidity_sweeps(
        slice_df,
        liquidity_levels,
    )

    elapsed = (
        time.perf_counter()
        - start_time
    )

    return {
        "n": n,
        "start": slice_df.index[0],
        "end": slice_df.index[-1],
        "swings": len(swings),
        "breaks": len(breaks),
        "classified_swings": len(
            classified_swings
        ),
        "zones": len(zones),
        "liquidity": len(
            liquidity_levels
        ),
        "trend": str(trend),
        "elapsed_sec": elapsed,
    }


# ============================================================================
# Output
# ============================================================================

def print_results(
    results: list[dict[str, Any]],
) -> None:
    """
    Print the warmup diagnostic table.
    """

    print()
    print("=" * 112)
    print("PHASE 2 WARMUP PROBE")
    print("=" * 112)

    print()
    print(
        "This probe measures structural state as historical context increases."
    )

    print(
        "It does NOT determine the final warmup automatically."
    )

    print()
    print(
        f"{'Warmup':>8} "
        f"{'Swings':>8} "
        f"{'Breaks':>8} "
        f"{'Class.':>8} "
        f"{'Zones':>8} "
        f"{'Liquidity':>11} "
        f"{'Time(s)':>9} "
        f"{'Trend':>18}"
    )

    print("-" * 112)

    for result in results:
        print(
            f"{result['n']:>8} "
            f"{result['swings']:>8} "
            f"{result['breaks']:>8} "
            f"{result['classified_swings']:>8} "
            f"{result['zones']:>8} "
            f"{result['liquidity']:>11} "
            f"{result['elapsed_sec']:>9.3f} "
            f"{result['trend']:>18}"
        )

    print("-" * 112)

    # ------------------------------------------------------------------------
    # Structural observations
    # ------------------------------------------------------------------------

    print()
    print("STRUCTURAL OBSERVATIONS")
    print("-" * 112)

    for previous, current in zip(
        results,
        results[1:],
    ):
        trend_note = (
            f" | TREND CHANGED: {previous['trend']} -> {current['trend']}"
            if previous["trend"] != current["trend"]
            else ""
        )

        print(
            f"{previous['n']:>4} -> "
            f"{current['n']:<4} candles | "
            f"swings {previous['swings']:>4} -> "
            f"{current['swings']:<4} | "
            f"breaks {previous['breaks']:>4} -> "
            f"{current['breaks']:<4} | "
            f"zones {previous['zones']:>4} -> "
            f"{current['zones']:<4} | "
            f"liquidity {previous['liquidity']:>4} -> "
            f"{current['liquidity']:<4}"
            f"{trend_note}"
        )

    # ------------------------------------------------------------------------
    # Interpretation
    # ------------------------------------------------------------------------

    print()
    print("HOW TO INTERPRET THIS")
    print("-" * 112)

    print(
        "1. Do not choose a warmup simply because it has the most "
        "swings, breaks, zones, or liquidity."
    )

    print(
        "2. Look for candidate ranges where adding substantially more "
        "historical context produces progressively smaller structural "
        "changes."
    )

    print(
        "3. A TREND CHANGED note between two consecutive candidates means "
        "the structural trend read is still unstable at that warmup size "
        "— treat that as evidence the smaller of the two is too little "
        "context, not as evidence the larger one is correct either."
    )

    print(
        "4. Treat convergence as evidence for a candidate warmup, not "
        "proof that the candidate is optimal."
    )

    print(
        "5. The final candidate must be tested with the canonical "
        "walk-forward engine."
    )

    print(
        "6. Do not use trading performance from this probe to select "
        "the warmup. That would turn warmup selection into an optimization."
    )

    print()
    print("IMPORTANT")
    print("-" * 112)

    print(
        "This probe analyzes the FIRST n candles of the dataset."
    )

    print(
        "That means it is a structural-context diagnostic, not yet a "
        "full fixed-evaluation warmup validation."
    )

    print(
        "The walk-forward validation stage is still required before "
        "declaring a final warmup_size."
    )

    total_elapsed = sum(r["elapsed_sec"] for r in results)
    print()
    print(
        f"Total probe time across all {len(results)} candidates: "
        f"{total_elapsed:.3f}s — use this as a rough per-window cost "
        "reference before choosing run_walk_forward()'s window count."
    )

    print()
    print("NEXT STEP")
    print("-" * 112)

    print(
        "Run the probe, inspect the structural progression, choose "
        "reasonable candidate(s), then validate them with walk_forward.py."
    )

    print()


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    """
    Run the warmup diagnostic.
    """

    print()
    print("=" * 112)
    print("LOADING DATASET")
    print("=" * 112)

    df = load_dataset(
        DATA_FILE
    )

    print(
        f"Dataset : {DATA_FILE}"
    )

    print(
        f"Candles : {len(df):,}"
    )

    print(
        f"Start   : {df.index[0]}"
    )

    print(
        f"End     : {df.index[-1]}"
    )

    print(
        f"Columns : {list(df.columns)}"
    )

    if len(df) < MAX_WARMUP_REQUIRED:
        raise ValueError(
            f"Dataset contains only {len(df)} candles, "
            f"but the largest configured warmup candidate "
            f"requires {MAX_WARMUP_REQUIRED} candles."
        )

    print()
    print("Warmup candidates:")
    print(
        "  "
        + ", ".join(
            str(n)
            for n in WARMUP_CANDIDATES
        )
    )

    print()
    print(
        "Starting structural diagnostics..."
    )

    results: list[
        dict[str, Any]
    ] = []

    for n in WARMUP_CANDIDATES:

        print(
            f"\nRunning n={n:,} ..."
        )

        try:
            result = probe_history(
                df,
                n,
            )

        except Exception as exc:
            raise RuntimeError(
                f"Warmup probe failed at n={n} "
                f"with {type(exc).__name__}: {exc}"
            ) from exc

        results.append(
            result
        )

    print_results(
        results
    )


if __name__ == "__main__":
    main()