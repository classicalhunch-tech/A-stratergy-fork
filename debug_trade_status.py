"""
debug_trade_status.py

Focused diagnostic for the real-data backtest.

Purpose:
- Run the current strategy/backtest engine on real data.
- Inspect BacktestResult counters.
- Compare triggered trades with result.trades.
- Break down WIN / LOSS / OPEN statuses.
- Inspect the first 10 TradeResult objects.
- Measure execution time.
- Detect obvious result-accounting inconsistencies.

This script is diagnostic only.
It does NOT modify strategy logic.
"""

from __future__ import annotations

import time
import traceback
from pathlib import Path

import pandas as pd

from strategy.backtest import run_backtest


# ============================================================
# CONFIGURATION
# ============================================================

DATA_FILE = "your_data_file.csv"

MAX_BARS_TO_RETEST = 20
REWARD_MULTIPLE = 2.0
STOP_BUFFER = 0.001
ENTRY_MODE = "midpoint"

SAMPLE_TRADE_COUNT = 10


# ============================================================
# DATA LOADER
# ============================================================


def load_real_data(path: str) -> pd.DataFrame:
    """
    Load and normalize the real OHLC dataset.

    Expected columns:
        timestamp
        open
        high
        low
        close
    """

    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found at: {file_path.resolve()}"
        )

    df = pd.read_csv(file_path)

    # Normalize column names.
    df.columns = [
        str(column).strip().lower()
        for column in df.columns
    ]

    required_columns = {
        "timestamp",
        "open",
        "high",
        "low",
        "close",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns in CSV: "
            f"{sorted(missing)}"
        )

    # Parse timestamps.
    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
    )

    invalid_timestamps = int(
        df["timestamp"].isna().sum()
    )

    if invalid_timestamps:
        print(
            f"WARNING: {invalid_timestamps} rows have "
            f"invalid timestamps and will be dropped."
        )

        df = df.dropna(
            subset=["timestamp"]
        )

    # Build the exact format expected by run_backtest().
    df = (
        df.set_index("timestamp")
        .sort_index()
        [["open", "high", "low", "close"]]
    )

    if df.empty:
        raise ValueError(
            "Dataset is empty after timestamp parsing."
        )

    # Duplicate timestamp diagnostic.
    duplicate_count = int(
        df.index.duplicated().sum()
    )

    if duplicate_count:
        print(
            f"WARNING: {duplicate_count} duplicate "
            f"timestamps found."
        )

    # OHLC NaN diagnostic.
    nan_count = int(
        df.isna().sum().sum()
    )

    if nan_count:
        print(
            f"WARNING: {nan_count} NaN OHLC values found."
        )

    # Chronological check.
    if not df.index.is_monotonic_increasing:
        raise ValueError(
            "Timestamp index is not monotonically increasing."
        )

    return df


# ============================================================
# STATUS NORMALIZATION
# ============================================================


def normalize_status(status) -> str:
    """
    Convert enum-like or string statuses into a simple
    comparable string.
    """

    if hasattr(status, "value"):
        return str(status.value).upper()

    return str(status).upper()


# ============================================================
# MAIN DIAGNOSTIC
# ============================================================


def main() -> None:

    print("=" * 70)
    print("BACKTEST TRADE-STATUS DEBUG DIAGNOSTIC")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. LOAD DATA
    # --------------------------------------------------------

    print(
        f"\n[1/4] Loading dataset: {DATA_FILE}"
    )

    try:
        df = load_real_data(DATA_FILE)

    except Exception:
        print(
            "\n[FATAL] Failed to load or parse dataset."
        )
        traceback.print_exc()
        return

    print(
        f"Candles loaded : {len(df):,}"
    )

    print(
        f"Start          : {df.index[0]}"
    )

    print(
        f"End            : {df.index[-1]}"
    )

    # --------------------------------------------------------
    # 2. RUN BACKTEST
    # --------------------------------------------------------

    print(
        "\n[2/4] Running backtest engine..."
    )

    print(
        f"Parameters:"
        f"\n  max_bars_to_retest = {MAX_BARS_TO_RETEST}"
        f"\n  reward_multiple    = {REWARD_MULTIPLE}"
        f"\n  stop_buffer        = {STOP_BUFFER}"
        f"\n  entry_mode         = {ENTRY_MODE}"
    )

    start_time = time.perf_counter()

    try:

        result = run_backtest(
            df=df,
            max_bars_to_retest=MAX_BARS_TO_RETEST,
            reward_multiple=REWARD_MULTIPLE,
            stop_buffer=STOP_BUFFER,
            entry_mode=ENTRY_MODE,
        )

    except Exception:

        elapsed = (
            time.perf_counter()
            - start_time
        )

        print(
            f"\n[FATAL] run_backtest() crashed "
            f"after {elapsed:.3f}s"
        )

        traceback.print_exc()
        return

    elapsed = (
        time.perf_counter()
        - start_time
    )

    candles_per_second = (
        len(df)
        / max(elapsed, 1e-9)
    )

    print(
        f"\nBacktest completed successfully."
    )

    print(
        f"Runtime        : {elapsed:.3f}s"
    )

    print(
        f"Candles/sec    : {candles_per_second:,.1f}"
    )

    # --------------------------------------------------------
    # 3. INSPECT RESULT
    # --------------------------------------------------------

    print(
        "\n[3/4] Inspecting BacktestResult..."
    )

    trades = result.trades or []

    print("\n" + "=" * 70)
    print("BACKTEST COUNTERS")
    print("=" * 70)

    print(
        f"Signals Generated : "
        f"{result.total_signals_generated:,}"
    )

    print(
        f"Trades Triggered  : "
        f"{result.total_trades_triggered:,}"
    )

    print(
        f"Invalidated       : "
        f"{result.total_invalidated:,}"
    )

    print(
        f"Expired           : "
        f"{result.total_expired:,}"
    )

    print(
        f"Trades Returned   : "
        f"{len(trades):,}"
    )

    print(
        f"Win Rate          : "
        f"{result.win_rate}"
    )

    print(
        f"Expectancy        : "
        f"{result.expectancy}"
    )

    # --------------------------------------------------------
    # STATUS BREAKDOWN
    # --------------------------------------------------------

    status_counts: dict[str, int] = {}

    for trade in trades:

        status = normalize_status(
            getattr(
                trade,
                "result_status",
                None,
            )
        )

        status_counts[status] = (
            status_counts.get(status, 0) + 1
        )

    print("\n" + "=" * 70)
    print("TRADE STATUS BREAKDOWN")
    print("=" * 70)

    if not status_counts:

        print(
            "No trades returned in result.trades."
        )

    else:

        for status, count in sorted(
            status_counts.items()
        ):
            print(
                f"  {status:<20} : {count:,}"
            )

    # --------------------------------------------------------
    # 4. SAMPLE TRADES
    # --------------------------------------------------------

    print(
        "\n[4/4] Inspecting sample trades..."
    )

    print("\n" + "=" * 70)
    print(
        f"SAMPLE TRADES (FIRST {SAMPLE_TRADE_COUNT})"
    )
    print("=" * 70)

    if not trades:

        print(
            "No trades available to inspect."
        )

    else:

        for number, trade in enumerate(
            trades[:SAMPLE_TRADE_COUNT],
            start=1,
        ):

            print(
                f"\nTrade #{number}"
            )

            print(
                f"  Direction     : "
                f"{getattr(trade, 'direction', None)}"
            )

            print(
                f"  Setup time    : "
                f"{getattr(trade, 'setup_time', None)}"
            )

            print(
                f"  Entry time    : "
                f"{getattr(trade, 'entry_time', None)}"
            )

            print(
                f"  Entry price   : "
                f"{getattr(trade, 'entry_price', None)}"
            )

            print(
                f"  Fill price    : "
                f"{getattr(trade, 'fill_price', None)}"
            )

            print(
                f"  Stop loss     : "
                f"{getattr(trade, 'stop_loss', None)}"
            )

            print(
                f"  Take profit   : "
                f"{getattr(trade, 'take_profit', None)}"
            )

            print(
                f"  Exit price    : "
                f"{getattr(trade, 'exit_price', None)}"
            )

            print(
                f"  Result status : "
                f"{normalize_status(getattr(trade, 'result_status', None))}"
            )

            print(
                f"  R multiple    : "
                f"{getattr(trade, 'r_multiple', None)}"
            )

            print(
                f"  Bars held     : "
                f"{getattr(trade, 'bars_held', None)}"
            )

            print(
                f"  Session       : "
                f"{getattr(trade, 'session', None)}"
            )

    # --------------------------------------------------------
    # CONSISTENCY CHECK
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("RESULT CONSISTENCY CHECK")
    print("=" * 70)

    triggered = result.total_trades_triggered
    returned = len(trades)

    print(
        f"Triggered counter : {triggered:,}"
    )

    print(
        f"Trades returned   : {returned:,}"
    )

    if triggered == returned:

        print(
            "\n  [PASS] Every triggered trade "
            "has a returned TradeResult."
        )

    else:

        difference = triggered - returned

        print(
            f"\n  [NOTICE] Triggered and returned "
            f"trade counts differ by {difference:,}."
        )

        print(
            "  Investigate only if this difference "
            "is unexpected for the current backtest design."
        )

    # --------------------------------------------------------
    # STATUS SANITY CHECK
    # --------------------------------------------------------

    expected_statuses = {
        "WIN",
        "LOSS",
        "OPEN",
    }

    unexpected_statuses = (
        set(status_counts)
        - expected_statuses
    )

    print("\n" + "=" * 70)
    print("STATUS SANITY CHECK")
    print("=" * 70)

    if unexpected_statuses:

        print(
            "  [NOTICE] Unexpected trade statuses found:"
        )

        for status in sorted(
            unexpected_statuses
        ):
            print(
                f"    - {status}"
            )

    else:

        print(
            "  [PASS] All returned trades use "
            "expected result statuses."
        )

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)

    print(
        f"Runtime        : {elapsed:.3f}s"
    )

    print(
        f"Candles        : {len(df):,}"
    )

    print(
        f"Signals        : "
        f"{result.total_signals_generated:,}"
    )

    print(
        f"Triggered      : "
        f"{result.total_trades_triggered:,}"
    )

    print(
        f"Returned trades: "
        f"{len(trades):,}"
    )

    print(
        "\nNo strategy code was modified by this diagnostic."
    )


if __name__ == "__main__":
    main()