"""
profile_backtest.py

Diagnostic script — NOT part of the strategy engine.

Purpose:
    1. Profile run_backtest() on the real dataset to identify
       the functions consuming most of the execution time.
    2. Inspect completed trades and their R-multiples.
    3. Investigate why win_rate and expectancy were reported as 0.00.
    4. Confirm the backtest is executing cleanly without modifying
       any strategy logic.

This script only observes the existing backtest engine.
"""

import cProfile
import io
import math
import pstats
import time

import pandas as pd

from strategy.backtest import run_backtest


DATA_FILE = "your_data_file.csv"

MAX_BARS_TO_RETEST = 20
REWARD_MULTIPLE = 2.0
STOP_BUFFER = 0.1
ENTRY_MODE = "midpoint"


def load_real_data(path: str) -> pd.DataFrame:
    """Load and validate the real OHLC dataset."""

    df = pd.read_csv(path)

    required_columns = {"timestamp", "open", "high", "low", "close"}
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing)}"
        )

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df = (
        df.set_index("timestamp")
        .sort_index()
    )

    df = df[["open", "high", "low", "close"]]

    if df.index.duplicated().any():
        duplicate_count = int(df.index.duplicated().sum())
        print(
            f"WARNING: dataset contains {duplicate_count} "
            "duplicate timestamps."
        )

    if df.isna().any().any():
        missing_values = int(df.isna().sum().sum())
        print(
            f"WARNING: dataset contains {missing_values} "
            "missing OHLC values."
        )

    if not df.index.is_monotonic_increasing:
        raise ValueError(
            "Dataset index is not monotonically increasing."
        )

    return df


def print_backtest_summary(result) -> None:
    """Print the main BacktestResult metrics if available.

    Uses the field names confirmed from the real run's output and the
    original project spec (section 18): total_signals_generated,
    total_trades_triggered, total_invalidated, total_expired, win_rate,
    expectancy, trades, errors. Prints "<attribute not found>" instead
    of silently skipping a field, so a schema mismatch is visible
    rather than hidden.
    """

    print("=" * 70)
    print("BACKTEST SUMMARY")
    print("=" * 70)

    summary_fields = (
        "total_signals_generated",
        "total_trades_triggered",
        "total_invalidated",
        "total_expired",
        "win_rate",
        "expectancy",
    )

    _MISSING = object()

    for field in summary_fields:
        value = getattr(result, field, _MISSING)
        if value is _MISSING:
            print(f"{field:24}: <attribute not found>")
        else:
            print(f"{field:24}: {value}")

    trades = getattr(result, "trades", None)

    if trades is not None:
        print(f"{'completed_trades':24}: {len(trades)}")

    errors = getattr(result, "errors", None)

    if errors is not None:
        print(f"{'errors':24}: {len(errors)}")


def print_trade_objects(result) -> None:
    """Print completed trade objects without assuming their schema."""

    trades = getattr(result, "trades", None)

    print("=" * 70)
    print("COMPLETED TRADE OBJECTS")
    print("=" * 70)

    if not trades:
        print("No completed trades to inspect.")
        return

    for i, trade in enumerate(trades, start=1):
        print(f"\nTrade {i}")

        if hasattr(trade, "__dict__"):
            attrs = vars(trade)

            if attrs:
                for key, value in attrs.items():
                    print(f"  {key}: {value}")
            else:
                print("  <trade object has an empty __dict__>")
        else:
            print(f"  {trade!r}")


def extract_r_values(result) -> list:
    """
    Attempt to extract R-multiples from completed trade objects.

    This deliberately checks several possible attribute names because
    the exact Trade class schema has not been assumed.
    """

    trades = getattr(result, "trades", None)

    if not trades:
        return []

    possible_names = (
        "r_multiple",
        "r",
        "pnl_r",
        "result_r",
    )

    r_values = []

    for trade in trades:
        found = False

        for attr_name in possible_names:
            if hasattr(trade, attr_name):
                value = getattr(trade, attr_name)
                r_values.append(value)
                found = True
                break

        if not found:
            r_values.append(None)

    return r_values


def print_r_analysis(result) -> None:
    """Analyze extracted R-multiple values."""

    print("=" * 70)
    print("R-MULTIPLE ANALYSIS")
    print("=" * 70)

    r_values = extract_r_values(result)

    if not r_values:
        print("No R-multiple values could be extracted.")
        return

    print(f"Raw extracted values: {r_values}")

    numeric_values = []

    for value in r_values:
        if isinstance(value, bool):
            continue

        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue

        if math.isfinite(numeric):
            numeric_values.append(numeric)

    if not numeric_values:
        print("No valid numeric R-multiples were found.")
        return

    total_r = sum(numeric_values)
    average_r = total_r / len(numeric_values)
    wins = sum(1 for r in numeric_values if r > 0)
    losses = sum(1 for r in numeric_values if r < 0)
    breakeven = sum(1 for r in numeric_values if r == 0)

    print(f"Numeric R-values : {numeric_values}")
    print(f"Trade count      : {len(numeric_values)}")
    print(f"Sum R            : {total_r:.4f}")
    print(f"Average R        : {average_r:.4f}")
    print(f"Wins             : {wins}")
    print(f"Losses           : {losses}")
    print(f"Breakeven        : {breakeven}")

    if wins + losses > 0:
        observed_win_rate = wins / (wins + losses) * 100
        print(f"Win rate by sign : {observed_win_rate:.2f}%")

    if any(r == 0 for r in numeric_values):
        print(
            "\nNOTE: Zero-valued R-multiples were found. "
            "This may be relevant to the 0.00 expectancy result."
        )


def print_profile(profiler: cProfile.Profile) -> None:
    """Print cumulative-time and self-time profiler reports."""

    print("=" * 70)
    print("TOP 25 FUNCTIONS BY CUMULATIVE TIME")
    print("=" * 70)

    stream = io.StringIO()

    stats = pstats.Stats(
        profiler,
        stream=stream,
    )

    stats.sort_stats("cumulative")
    stats.print_stats(25)

    print(stream.getvalue())

    print("=" * 70)
    print("TOP 15 FUNCTIONS BY SELF (TOTAL) TIME")
    print("=" * 70)

    stream2 = io.StringIO()

    stats2 = pstats.Stats(
        profiler,
        stream=stream2,
    )

    stats2.sort_stats("tottime")
    stats2.print_stats(15)

    print(stream2.getvalue())


def main() -> None:
    print("=" * 70)
    print("BACKTEST PROFILER")
    print("=" * 70)
    print(f"Dataset: {DATA_FILE}")
    print("Purpose: identify performance bottlenecks and inspect trade R-values.")
    print()

    # ---------------------------------------------------------------
    # Load data
    # ---------------------------------------------------------------

    try:
        df = load_real_data(DATA_FILE)
    except Exception as exc:
        print(f"ERROR loading dataset: {exc}")
        return

    print(f"Candles loaded : {len(df)}")

    if not df.empty:
        print(f"Start          : {df.index[0]}")
        print(f"End            : {df.index[-1]}")

    print()

    # ---------------------------------------------------------------
    # Profile run_backtest()
    # ---------------------------------------------------------------

    print("Starting profiled backtest...")
    print("cProfile adds overhead, so this may take longer than 271 seconds.")
    print()

    profiler = cProfile.Profile()

    wall_start = time.perf_counter()

    try:
        # runcall() handles enable()/disable() internally — do not call
        # profiler.enable() manually anywhere else in this file, or
        # cProfile raises "Another profiling tool is already active".
        result = profiler.runcall(
            run_backtest,
            df=df,
            max_bars_to_retest=MAX_BARS_TO_RETEST,
            reward_multiple=REWARD_MULTIPLE,
            stop_buffer=STOP_BUFFER,
            entry_mode=ENTRY_MODE,
        )
    except Exception as exc:
        elapsed = time.perf_counter() - wall_start

        print()
        print("=" * 70)
        print("BACKTEST FAILED")
        print("=" * 70)
        print(f"Elapsed before failure: {elapsed:.3f} seconds")
        print(f"Error: {exc}")
        raise

    elapsed = time.perf_counter() - wall_start

    print()
    print("=" * 70)
    print("PROFILED BACKTEST FINISHED")
    print("=" * 70)
    print(f"Wall-clock time: {elapsed:.3f} seconds")
    print(f"Candles        : {len(df)}")

    if len(df) > 0 and elapsed > 0:
        print(f"Candles/sec    : {len(df) / elapsed:.2f}")

    print()

    # ---------------------------------------------------------------
    # Part 1: performance profile
    # ---------------------------------------------------------------

    print_profile(profiler)

    # ---------------------------------------------------------------
    # Part 2: backtest metrics
    # ---------------------------------------------------------------

    print_backtest_summary(result)

    print()

    # ---------------------------------------------------------------
    # Part 3: inspect individual trades
    # ---------------------------------------------------------------

    print_trade_objects(result)

    print()

    # ---------------------------------------------------------------
    # Part 4: analyze R-multiples
    # ---------------------------------------------------------------

    print_r_analysis(result)

    print()
    print("=" * 70)
    print("PROFILING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()