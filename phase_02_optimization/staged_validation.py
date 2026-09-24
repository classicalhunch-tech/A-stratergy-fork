"""
phase_02_optimization/staged_validation.py

Purpose:
    Validate StrategyAdapter scaling across increasing historical
    candle counts.

This is a BENCHMARKING / VALIDATION script.

It does NOT:
    - modify strategy logic
    - optimize strategy parameters
    - change adapter behavior
    - reuse adapter state between stages

Current validation stages:
    500
    1,000
    2,000
    3,000
    4,000
    5,000 (full dataset)

This is the final stage in the staged validation sequence.

Run from the project root:

    python -m phase_02_optimization.staged_validation
"""

from pathlib import Path
import time

import pandas as pd

from phase_03_paper.market.engine import Candle
from phase_03_paper.signals.adapter import StrategyAdapter


def run_stage(csv_path: str, label: str):
    """
    Run one independent StrategyAdapter scaling stage.

    A fresh StrategyAdapter is created for every stage so that each
    dataset is measured from a clean state.
    """
    path_obj = Path(csv_path)
    if not path_obj.exists():
        raise FileNotFoundError(
            f"Validation dataset not found: {csv_path}. "
            "Please ensure the CSV file exists in the expected directory."
        )

    df = pd.read_csv(path_obj)

    # Normalize timestamp values before constructing Candle objects.
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # Normalize column names for consistent access.
    df.columns = [c.strip().lower() for c in df.columns]

    # IMPORTANT:
    # Every stage gets a completely fresh adapter.
    adapter = StrategyAdapter()

    start = time.perf_counter()

    processed = 0
    triggered_total = 0

    # Use itertuples() for higher throughput across large candle sets
    for row in df.itertuples(index=False):
        candle = Candle(
            timestamp=row.timestamp,
            open=row.open,
            high=row.high,
            low=row.low,
            close=row.close,
        )

        events = adapter.on_candle(candle)

        triggered_total += len(events)
        processed += 1

    elapsed = time.perf_counter() - start

    avg_per_candle = (
        elapsed / processed
        if processed
        else 0.0
    )

    print("=" * 70)
    print(f"STAGE: {label}")
    print("=" * 70)
    print(f"Candles processed   : {processed}")
    print(f"Elapsed time        : {elapsed:.2f}s")
    print(f"Avg per candle      : {avg_per_candle:.5f}s")
    print(f"Triggered events    : {triggered_total}")
    print(f"Pending signals     : {adapter.pending_count}")
    print(f"Adapter errors      : {len(adapter.errors)}")
    print()

    return elapsed, processed, triggered_total, adapter.pending_count, len(
        adapter.errors
    )


# ---------------------------------------------------------------------------
# Validation stages
# ---------------------------------------------------------------------------
#
# Sequence: 500 -> 1,000 -> 2,000 -> 3,000 -> 4,000 -> 5,000.
#
# All stages reviewed and enabled. This is the final measurement
# in the staged validation sequence.
#
stages = [
    ("your_data_file_500.csv", "500 candles"),
    ("your_data_file_1000.csv", "1,000 candles"),
    ("your_data_file_2000.csv", "2,000 candles"),
    ("your_data_file_3000.csv", "3,000 candles"),
    ("your_data_file_4000.csv", "4,000 candles"),
    ("your_data_file.csv", "5,000 candles (FULL)"),
]


# ---------------------------------------------------------------------------
# Run validation
# ---------------------------------------------------------------------------

results = []

for path, label in stages:
    (
        elapsed,
        processed,
        triggered,
        pending,
        errors,
    ) = run_stage(path, label)

    results.append(
        (
            label,
            processed,
            elapsed,
            triggered,
            pending,
            errors,
        )
    )


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print("=" * 70)
print("SCALING VALIDATION SUMMARY")
print("=" * 70)

for (
    label,
    processed,
    elapsed,
    triggered,
    pending,
    errors,
) in results:
    per_candle = (
        elapsed / processed
        if processed
        else 0.0
    )

    print(
        f"{label:24s} "
        f"candles={processed:5d}  "
        f"total={elapsed:7.2f}s  "
        f"avg/candle={per_candle:.5f}s  "
        f"triggered={triggered:4d}  "
        f"pending={pending:3d}  "
        f"errors={errors:3d}"
    )