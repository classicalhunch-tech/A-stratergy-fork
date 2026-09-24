"""
generate_synthetic_data.py

Generates a synthetic M5 EURUSD-like OHLC dataset with valid
candle relationships, purely for exercising the BACKTEST ENGINE's
mechanics on a lower scalping timeframe (5-minute bars).

IMPORTANT: this is a random walk, not real market data. Good/bad
results here test the engine's code correctness only -- not
whether Machet Mechanics has a real edge.
"""

import sys

import numpy as np
import pandas as pd

DEFAULT_FILENAME = "your_data_file.csv"
DEFAULT_ROWS = 5000  # More rows for M5 to cover a solid time window
DEFAULT_SEED = 42


def generate_eurusd_m5(
    filename: str = DEFAULT_FILENAME,
    rows: int = DEFAULT_ROWS,
    seed: int = DEFAULT_SEED,
    start_price: float = 1.0850,
):
    print(f"Generating {rows:,} rows of synthetic M5 EURUSD scalping data (seed={seed})...")

    index = pd.date_range(
        start="2026-01-01",
        periods=rows,
        freq="5min",
    )

    rng = np.random.default_rng(seed)

    returns = rng.normal(
        loc=0.0000005,
        scale=0.00015,
        size=rows,
    )

    close_prices = start_price * np.exp(np.cumsum(returns))

    open_prices = np.empty(rows)
    open_prices[0] = start_price
    open_prices[1:] = close_prices[:-1]

    upper_wicks = rng.uniform(0.00002, 0.0002, size=rows)
    lower_wicks = rng.uniform(0.00002, 0.0002, size=rows)

    high_prices = np.maximum(open_prices, close_prices) + upper_wicks
    low_prices = np.minimum(open_prices, close_prices) - lower_wicks

    df = pd.DataFrame(
        {
            "open": open_prices,
            "high": high_prices,
            "low": low_prices,
            "close": close_prices,
        },
        index=index,
    )
    df.index.name = "timestamp"

    assert (df["high"] >= df["open"]).all(), "Invalid high/open relationship"
    assert (df["high"] >= df["close"]).all(), "Invalid high/close relationship"
    assert (df["low"] <= df["open"]).all(), "Invalid low/open relationship"
    assert (df["low"] <= df["close"]).all(), "Invalid low/close relationship"

    if (df["low"] <= 0).any():
        print("ERROR: random walk drifted to non-positive prices.")
        sys.exit(1)

    df.to_csv(filename)

    print(f"Successfully saved M5 dataset to: {filename}")
    print(f"Rows: {len(df):,}")
    print(f"Timeframe: M5 (Scalping)")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"Columns: {list(df.columns)}")
    print("\nReminder: synthetic data -- engine mechanics test only, "
          "not strategy validation.")


if __name__ == "__main__":
    generate_eurusd_m5()