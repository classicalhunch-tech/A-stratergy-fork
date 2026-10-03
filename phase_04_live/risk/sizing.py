"""
main.py

End-to-end multi-timeframe SMC pipeline runner.

Pipeline
--------
5M OHLC
    v
1H Macro + 15M Internal Structure (MTF context)
    v
run_backtest() [tested, validated engine]
    - Swings
    - Market Structure
    - Liquidity (Active -> Swept)
    - Supply/Demand Zones
    - Flip Zone signal generation
    - MTF Confluence Guard (per-signal filter, before execution)
    - Retest -> Trigger -> WIN/LOSS/OPEN lifecycle
    v
Final trade results

Architecture
------------
main.py is the orchestration layer only.

The strategy engines remain responsible for:
    - Swing detection
    - Market structure
    - Liquidity
    - Zones
    - Signal generation
    - Retest / trigger / exit lifecycle
    - MTF structure
    - MTF confluence

The gap this version closes: previously, main.py generated raw 5M
candidates and reported which ones WOULD pass MTF confluence, but
never executed a single trade -- run_backtest() (the tested engine
with 79+ passing tests, restart-equivalence, walk-forward validation)
had no MTF awareness at all. Now run_backtest() accepts an optional
mtf_filter_fn hook: when None (the default), behavior is unchanged
from every existing test. When provided, only MTF-approved signals
proceed to entry/retest/execution. No execution logic is duplicated
here -- the confluence gate is checked, then the same tested engine
runs exactly as it always has.
"""

import argparse
import sys
import time
import traceback
from pathlib import Path

import pandas as pd

from strategy.backtest import run_backtest
from strategy.mtf_structure import build_mtf_dataset_with_structure
from strategy.confluence import build_mtf_signal_filter
from strategy.swings import find_swings
from dashboard.mtf_context import MTFConfig


# =====================================================================
# Constants
# =====================================================================

REQUIRED_OHLC_COLUMNS = {"open", "high", "low", "close"}

# Multi-timeframe settings (one place to change them)
MTF_MACRO_TF = "1h"
MTF_INTERNAL_TF = "15min"

# Errors we treat as EXPECTED pipeline conditions (bad input, bad data).
# These get a clean one-line message because the user can act on them.
EXPECTED_ERRORS = (FileNotFoundError, ValueError)


# =====================================================================
# Step 1: Load and clean raw 5M OHLC data
# =====================================================================

def load_5m_data(csv_path: str) -> pd.DataFrame:
    """
    Load, clean, and validate the raw 5M OHLC CSV.

    Expected format:
        timestamp,open,high,low,close[,volume]

    The timestamp is stored as the DatetimeIndex used by the rest
    of the strategy architecture.
    """

    path = Path(csv_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Data file not found: {path}\n"
            f"Pass the correct path with --data <file.csv>."
        )

    df = pd.read_csv(path, parse_dates=True, index_col=0)

    # Normalize column names
    df.columns = [str(column).strip().lower() for column in df.columns]

    # Validate timestamp index
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")

    invalid_timestamp_count = int(df.index.isna().sum())
    if invalid_timestamp_count:
        print(f"   WARNING: removing {invalid_timestamp_count} invalid timestamp row(s).")
        df = df[df.index.notna()]

    df = df.sort_index()

    duplicate_count = int(df.index.duplicated().sum())
    if duplicate_count:
        print(f"   WARNING: {duplicate_count} duplicate timestamp(s) found -- keeping the first occurrence.")
        df = df[~df.index.duplicated(keep="first")]

    # Defensive copy: everything above is boolean-mask filtering, which
    # can leave df as a view. Make "this is my own frame now" explicit.
    df = df.copy()

    missing = REQUIRED_OHLC_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"5M dataset is missing required columns: {sorted(missing)}")

    for column in sorted(REQUIRED_OHLC_COLUMNS):
        df[column] = pd.to_numeric(df[column], errors="coerce")

    invalid_ohlc_mask = df[list(REQUIRED_OHLC_COLUMNS)].isna().any(axis=1)
    invalid_ohlc_count = int(invalid_ohlc_mask.sum())
    if invalid_ohlc_count:
        print(f"   WARNING: removing {invalid_ohlc_count} row(s) with invalid OHLC values.")
        df = df[~invalid_ohlc_mask]

    if df.empty:
        raise ValueError("5M dataset is empty after loading and cleaning.")

    if not df.index.is_monotonic_increasing:
        raise ValueError("5M dataset index is not chronologically sorted.")

    print(f"   Loaded {len(df)} rows of 5M data.")
    print(f"   Data range: {df.index[0]} -> {df.index[-1]}")

    return df


# =====================================================================
# Step 2: Build MTF structural context
# =====================================================================

def build_mtf_context(df_5m: pd.DataFrame) -> pd.DataFrame:
    """
    Build the lookahead-safe MTF dataset.

    Higher-timeframe structure:
        1H = macro context
        15M = internal context

    The actual MTF logic remains inside strategy.mtf_structure.
    """

    df_enriched = build_mtf_dataset_with_structure(
        df_5m,
        macro_swings_fn=find_swings,
        internal_swings_fn=find_swings,
        config=MTFConfig(
            macro_tf=MTF_MACRO_TF,
            internal_tf=MTF_INTERNAL_TF,
        ),
    )

    print("   MTF enrichment complete.")

    mtf_columns = [
        column for column in df_enriched.columns
        if column.startswith("macro_") or column.startswith("internal_")
    ]

    print(f"   MTF columns ({len(mtf_columns)}):")
    for column in mtf_columns:
        print(f"      - {column}")

    return df_enriched


# =====================================================================
# Step 3: Run the tested backtest engine, gated by MTF confluence
# =====================================================================

def run_mtf_gated_backtest(
    df_5m: pd.DataFrame,
    df_enriched: pd.DataFrame,
    allow_neutral_internal: bool = True,
):
    """
    Run the SAME tested run_backtest() engine used by Phase 1/2/3,
    with an MTF confluence filter gating which signals are allowed
    to proceed to entry/retest/execution.

    No execution logic is duplicated here or in confluence.py --
    the filter is a predicate checked per-signal at the exact point
    a fresh signal would otherwise be registered as pending. The
    trade lifecycle itself (retest, trigger, SL/TP, WIN/LOSS/OPEN)
    is untouched, tested code.
    """

    mtf_filter_fn = build_mtf_signal_filter(
        df_enriched,
        allow_neutral_internal=allow_neutral_internal,
    )

    result = run_backtest(
        df_5m,
        mtf_filter_fn=mtf_filter_fn,
    )

    print("--------------------------------------------------")
    print("MTF-GATED BACKTEST COMPLETE")
    print("--------------------------------------------------")
    print(f"Signals generated (MTF-approved): {result.total_signals_generated}")
    print(f"Rejected by MTF confluence:       {result.total_mtf_rejected}")
    print(f"Triggered:                        {result.total_trades_triggered}")
    print(f"Invalidated:                      {result.total_invalidated}")
    print(f"Expired:                          {result.total_expired}")
    print(f"Win rate:                         {result.win_rate:.2f}%")
    print(f"Expectancy:                       {result.expectancy:.4f}R")

    if result.errors:
        print(f"\nWARNING: {len(result.errors)} error(s) during replay:")
        for err in result.errors[:10]:
            print(f"   {err}")

    return result


# =====================================================================
# Main entry point
# =====================================================================

def main():

    parser = argparse.ArgumentParser(
        description="Run the multi-timeframe SMC pipeline (MTF-gated backtest)."
    )

    parser.add_argument(
        "--data",
        default="your_data_file.csv",
        help="Path to the 5M OHLC CSV (default: your_data_file.csv)",
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Optional path for saving closed trades as CSV.",
    )

    parser.add_argument(
        "--no-mtf-neutral",
        action="store_true",
        help="Reject signals when internal (15M) trend is neutral, instead of allowing them.",
    )

    args = parser.parse_args()

    start_time = time.perf_counter()

    print("==================================================")
    print("RUNNING TRUE MTF SMC PIPELINE (MTF-gated backtest)")
    print("==================================================")

    try:

        print("\n1. Loading raw 5M OHLC dataset...")
        df_5m = load_5m_data(args.data)

        print("\n2. Building 1H macro and 15M internal structure...")
        df_enriched = build_mtf_context(df_5m)

        print("\n3. Running MTF-gated backtest (tested engine + confluence guard)...")
        result = run_mtf_gated_backtest(
            df_5m,
            df_enriched,
            allow_neutral_internal=not args.no_mtf_neutral,
        )

        if args.output:
            closed = [t for t in result.trades if t.result_status in {"WIN", "LOSS"}]
            rows = [
                {
                    "setup_time": t.setup_time,
                    "entry_time": t.entry_time,
                    "exit_time": t.exit_time,
                    "direction": getattr(t.direction, "value", t.direction),
                    "entry_price": t.fill_price,
                    "exit_price": t.exit_price,
                    "stop_loss": t.stop_loss,
                    "take_profit": t.take_profit,
                    "result_status": t.result_status,
                    "r_multiple": t.r_multiple,
                    "bars_held": t.bars_held,
                    "session": t.session,
                }
                for t in closed
            ]
            pd.DataFrame(rows).to_csv(args.output, index=False)
            print(f"\nSaved {len(rows)} closed trade(s) to {args.output}")

    except EXPECTED_ERRORS as exc:
        print(f"\nPIPELINE ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    except Exception:
        print(
            "\nUNEXPECTED ERROR -- this looks like a bug, not a data problem:",
            file=sys.stderr,
        )
        traceback.print_exc()
        sys.exit(1)

    elapsed = time.perf_counter() - start_time
    print(f"\nTotal runtime: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
