"""
verify_backtest.py

Phase 1 backtest smoke/integration test.

Purpose:
    Verify that the complete backtest pipeline can execute cleanly
    from synthetic OHLC data through run_backtest().

This is NOT a profitability test.
It is an execution/integration test.

A random synthetic dataset may legitimately produce zero signals.
The important first checks are:
    - no syntax errors
    - no runtime errors
    - backtest completes
    - result object is returned correctly
"""

import time
import traceback

import numpy as np
import pandas as pd

from strategy.backtest import run_backtest


def generate_sample_data(n_bars: int = 500) -> pd.DataFrame:
    """Generate deterministic synthetic OHLC data for smoke testing."""

    np.random.seed(42)

    timestamps = pd.date_range(
        start="2026-01-01",
        periods=n_bars,
        freq="1h",
    )

    close_prices = 100 + np.cumsum(
        np.random.randn(n_bars) * 0.5
    )

    high_prices = (
        close_prices
        + np.random.rand(n_bars) * 0.4
    )

    low_prices = (
        close_prices
        - np.random.rand(n_bars) * 0.4
    )

    open_prices = (
        close_prices
        + (np.random.rand(n_bars) - 0.5) * 0.2
    )

    df = pd.DataFrame(
        {
            "open": open_prices,
            "high": high_prices,
            "low": low_prices,
            "close": close_prices,
        },
        index=timestamps,
    )

    return df


def fmt_metric(value, suffix: str = "") -> str:
    """
    Safely format a metric that may be None/NaN when zero trades
    were triggered (e.g. win_rate, expectancy with no completed trades).
    Avoids crashing the smoke test on a formatting error, which would
    hide the real signal (pipeline ran cleanly, just no trades).
    """
    if value is None:
        return "N/A (no trades)"
    try:
        if isinstance(value, float) and np.isnan(value):
            return "N/A (no trades)"
    except TypeError:
        pass
    try:
        return f"{value:.2f}{suffix}"
    except (ValueError, TypeError):
        return f"{value!r}"


def main() -> None:
    """Run the complete backtest smoke test."""

    print("=" * 60)
    print("BACKTEST SMOKE / INTEGRATION TEST")
    print("=" * 60)

    print("\n[1/3] Generating synthetic OHLC data...")

    df = generate_sample_data(500)

    print(f"Generated candles : {len(df)}")
    print(f"Start             : {df.index[0]}")
    print(f"End               : {df.index[-1]}")

    print("\n[2/3] Running backtest engine...")

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
        print(
            "\nThis is a runtime error inside run_backtest() itself "
            "(see traceback above for the exact file/function/line)."
        )
        return

    elapsed = time.perf_counter() - start_time

    print(f"\n[3/3] Backtest completed in {elapsed:.3f}s.")

    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)

    print(
        f"Total Signals Generated : "
        f"{result.total_signals_generated}"
    )

    print(
        f"Total Trades Triggered  : "
        f"{result.total_trades_triggered}"
    )

    print(
        f"Total Invalidated       : "
        f"{result.total_invalidated}"
    )

    print(
        f"Total Expired           : "
        f"{result.total_expired}"
    )

    print(
        f"Win Rate                : "
        f"{fmt_metric(result.win_rate, '%')}"
    )

    print(
        f"Expectancy (Avg R)      : "
        f"{fmt_metric(result.expectancy, 'R')}"
    )

    trades = result.trades if result.trades is not None else []
    print(
        f"Total Trades Completed  : "
        f"{len(trades)}"
    )

    errors = result.errors if result.errors is not None else []

    print("\n" + "=" * 60)

    if errors:
        print(
            f"ERRORS ENCOUNTERED: "
            f"{len(errors)}"
        )

        for error in errors[:5]:
            print(f"  - {error}")

        print("\n❌ BACKTEST SMOKE TEST FAILED.")

    else:
        print("✅ BACKTEST EXECUTED CLEANLY.")
        print("✅ ZERO ERRORS REPORTED.")

        if result.total_signals_generated == 0:
            print(
                "\nNOTE:"
                "\nThe random synthetic dataset produced zero signals."
                "\nThis does NOT automatically mean the strategy is broken."
                "\nThe dedicated strategy tests are responsible for"
                "\nverifying specific SMC signal behavior."
            )

        print(
            "\nNext step:"
            "\nRun the real 5,000-candle dataset and measure"
            "\nbacktest execution time before running Monte Carlo."
        )

    print("=" * 60)


if __name__ == "__main__":
    main()