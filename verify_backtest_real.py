"""
verify_backtest_real.py

Phase 1/2 backtest integration test — REAL DATA VERSION.

Same purpose as verify_backtest.py, but runs the full pipeline
against the actual ~5,000-candle dataset (your_data_file.csv)
instead of synthetic data.

The main thing this script exists to measure is EXECUTION TIME,
to confirm whether the precompute + visibility-snapshot fix
resolved the earlier O(n^2) freeze seen in verify_monte_carlo.py.
"""

import sys
import time
import traceback

import pandas as pd

from strategy.backtest import run_backtest


DATA_FILE = "your_data_file.csv"


def load_real_data(path: str) -> pd.DataFrame:
    """
    Load the real OHLC dataset and shape it exactly like
    generate_sample_data() does: DatetimeIndex, lowercase
    open/high/low/close columns, sorted ascending, no gaps in dtype.
    """

    df = pd.read_csv(path)

    if "timestamp" not in df.columns:
        raise ValueError(
            f"Expected a 'timestamp' column in {path}, "
            f"found columns: {list(df.columns)}"
        )

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp")
    df = df.sort_index()

    required_cols = {"open", "high", "low", "close"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[["open", "high", "low", "close"]]

    # Sanity checks before we hand this to the backtest engine
    if df.index.duplicated().any():
        dup_count = df.index.duplicated().sum()
        print(f"WARNING: {dup_count} duplicate timestamps found in data.")

    if df.isna().any().any():
        na_count = df.isna().sum().sum()
        print(f"WARNING: {na_count} NaN values found in OHLC data.")

    if not df.index.is_monotonic_increasing:
        raise ValueError("Timestamp index is not sorted ascending after sort_index().")

    return df


def fmt_metric(value, suffix: str = "") -> str:
    if value is None:
        return "N/A (no trades)"
    try:
        if isinstance(value, float) and pd.isna(value):
            return "N/A (no trades)"
    except TypeError:
        pass
    try:
        return f"{value:.2f}{suffix}"
    except (ValueError, TypeError):
        return f"{value!r}"


def main() -> None:
    print("=" * 60)
    print("BACKTEST INTEGRATION TEST — REAL DATA (your_data_file.csv)")
    print("=" * 60)

    print(f"\n[1/3] Loading {DATA_FILE}...")

    try:
        df = load_real_data(DATA_FILE)
    except Exception:
        print("\n❌ FAILED TO LOAD DATA")
        print("-" * 60)
        traceback.print_exc()
        print("-" * 60)
        sys.exit(1)

    print(f"Loaded candles : {len(df)}")
    print(f"Start          : {df.index[0]}")
    print(f"End            : {df.index[-1]}")

    print("\n[2/3] Running backtest engine on real data...")
    print("(this may take a while on the first run — timing it now)")

    start_time = time.perf_counter()

    try:
        result = run_backtest(
            df=df,
            max_bars_to_retest=20,
            reward_multiple=2.0,
            stop_buffer=0.1,
            entry_mode="midpoint",
        )
    except Exception:
        elapsed = time.perf_counter() - start_time
        print(f"\n❌ run_backtest() RAISED AN EXCEPTION after {elapsed:.2f}s")
        print("-" * 60)
        traceback.print_exc()
        print("-" * 60)
        return

    elapsed = time.perf_counter() - start_time

    print(f"\n[3/3] Backtest completed in {elapsed:.3f}s "
          f"({len(df) / max(elapsed, 1e-9):.1f} candles/sec).")

    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)

    print(f"Total Signals Generated : {result.total_signals_generated}")
    print(f"Total Trades Triggered  : {result.total_trades_triggered}")
    print(f"Total Invalidated       : {result.total_invalidated}")
    print(f"Total Expired           : {result.total_expired}")
    print(f"Win Rate                : {fmt_metric(result.win_rate, '%')}")
    print(f"Expectancy (Avg R)      : {fmt_metric(result.expectancy, 'R')}")

    trades = result.trades if result.trades is not None else []
    print(f"Total Trades Completed  : {len(trades)}")

    status_counts = {}
    for t in trades:
        status = getattr(t, "result_status", None)
        status_key = str(status)
        status_counts[status_key] = status_counts.get(status_key, 0) + 1

    print("\nTrade Status Breakdown:")
    for status_key, count in sorted(status_counts.items()):
        print(f"  {status_key}: {count}")

    errors = result.errors if result.errors is not None else []

    print("\n" + "=" * 60)

    if errors:
        print(f"ERRORS ENCOUNTERED: {len(errors)}")
        for error in errors[:5]:
            print(f"  - {error}")
        print("\n❌ REAL-DATA BACKTEST FAILED.")
    else:
        print("✅ BACKTEST EXECUTED CLEANLY ON REAL DATA.")
        print("✅ ZERO ERRORS REPORTED.")
        print(
            "\nNext step:"
            "\nIf this timing looks reasonable (roughly linear vs. the"
            "\n500-bar run, not exponential), proceed to rerun"
            "\nverify_monte_carlo.py on this same dataset."
        )

    print("=" * 60)


if __name__ == "__main__":
    main()