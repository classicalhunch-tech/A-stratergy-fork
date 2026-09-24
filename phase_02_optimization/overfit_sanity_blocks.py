"""
phase_02_optimization/overfit_sanity_blocks.py

PBO sanity check using a different even CSCV block count than the
primary PBO run.

Primary PBO:
    CSCV_BLOCKS = 10

Recommended sanity check:
    CSCV_BLOCKS = 8

Usage:
    python -m phase_02_optimization.overfit_sanity_blocks 8

Examples:
    python -m phase_02_optimization.overfit_sanity_blocks 8
    python -m phase_02_optimization.overfit_sanity_blocks 6

Safety:
    - Does NOT modify strategy logic.
    - Does NOT modify the canonical backtest.
    - Does NOT modify trial_registry.py.
    - Does NOT modify overfit_detection.py on disk.
    - Does NOT overwrite the original 10-block PBO cache.
    - Uses a separate cache for each block count.
"""

import sys
from pathlib import Path

import phase_02_optimization.overfit_detection as od


PRIMARY_BLOCKS = 10


def parse_block_count() -> int:
    """Read and validate the CSCV block count from the command line."""

    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: "
            "python -m phase_02_optimization.overfit_sanity_blocks "
            "<EVEN_BLOCK_COUNT>"
        )

    try:
        n_blocks = int(sys.argv[1])
    except ValueError:
        raise SystemExit(
            "CSCV_BLOCKS must be an integer."
        )

    if n_blocks < 2:
        raise SystemExit(
            "CSCV_BLOCKS must be an even integer >= 2."
        )

    if n_blocks % 2 != 0:
        raise SystemExit(
            "CSCV_BLOCKS must be an even integer >= 2."
        )

    if n_blocks == PRIMARY_BLOCKS:
        raise SystemExit(
            "Use a block count different from the primary "
            f"value of {PRIMARY_BLOCKS}."
        )

    return n_blocks


def main() -> None:
    """Run an isolated PBO sanity check."""

    n_blocks = parse_block_count()

    # Save the module's original values so this process remains isolated.
    original_blocks = od.CSCV_BLOCKS
    original_cache = od.PBO_CACHE_JSON

    # Use the requested block count only for this process.
    od.CSCV_BLOCKS = n_blocks

    # IMPORTANT:
    # Never touch the original 10-block cache.
    od.PBO_CACHE_JSON = Path(
        f"phase_02_optimization/"
        f"pbo_backtest_cache_blocks{n_blocks}.json"
    )

    try:
        print()
        print("=" * 100)
        print(
            f"PBO SANITY CHECK -- CSCV_BLOCKS = {n_blocks}"
        )
        print("=" * 100)

        print(
            f"Primary PBO block count : {PRIMARY_BLOCKS}"
        )
        print(
            f"Sanity-check block count: {n_blocks}"
        )
        print(
            f"Separate cache         : {od.PBO_CACHE_JSON}"
        )
        print(
            "Original 10-block cache: NOT modified"
        )

        print()
        print("Loading dataset for PBO...")

        df = od.load_dataset(od.DATA_FILE)

        print(
            f"Dataset candles loaded : {len(df)}"
        )

        ts_index = od.build_timestamp_index(df)

        print()
        print("=" * 100)
        print("BUILDING SANITY-CHECK PERFORMANCE MATRIX")
        print("=" * 100)

        labels, matrix = od.build_block_performance_matrix(
            df,
            ts_index,
        )

        print()
        print("=" * 100)
        print("RUNNING SANITY-CHECK PBO")
        print("=" * 100)

        od.run_pbo(labels, matrix)

        print()
        print("=" * 100)
        print("PBO SANITY CHECK COMPLETE")
        print("=" * 100)

        print(
            f"Sanity-check cache retained at: {od.PBO_CACHE_JSON}"
        )
        print(
            "Original 10-block cache was NOT modified."
        )
        print()

    finally:
        # Restore the imported module's values before exiting.
        od.CSCV_BLOCKS = original_blocks
        od.PBO_CACHE_JSON = original_cache


if __name__ == "__main__":
    main()