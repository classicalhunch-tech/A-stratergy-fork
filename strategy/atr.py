"""
strategy/atr.py

Causal ATR. known_atr[i] uses only bars up to i-1, so it is fully known at
the open of bar i (where market-on-close fills happen).
"""

import pandas as pd


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    parts = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    )
    return parts.max(axis=1)


def atr_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
    if period <= 0:
        raise ValueError("period must be > 0")
    return true_range(df).rolling(period, min_periods=period).mean()


def known_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """ATR as known at the OPEN of each bar (excludes the bar itself)."""
    return atr_series(df, period).shift(1)
