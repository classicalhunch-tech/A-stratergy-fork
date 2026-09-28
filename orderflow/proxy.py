"""
orderflow/proxy.py

Pure, causal OHLCV-derived order-flow proxy calculations.

IMPORTANT -- HONESTY REQUIREMENT
---------------------------------
Nothing in this file computes true bid/ask delta. The current
datasets contain only open/high/low/close/volume. Every value
produced here is an ESTIMATE derived from candle shape and volume,
and must be treated (and named, and reported) as a proxy only.

CAUSALITY REQUIREMENT
----------------------
Every function here operates on whatever slice of the dataframe the
caller passes in. This module has no notion of "now" or "future" --
it is the caller's job (see orderflow/attach.py) to only ever pass
in bars at or before the decision point. Nothing here reaches
outside the slice it is given.
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd


def candle_pressure(row: pd.Series) -> Tuple[float, float]:
    """
    Estimate the buy/sell volume split for a single OHLCV candle.

    This is the standard OHLCV order-flow proxy: volume traded near
    the candle's high is treated as more likely buyer-driven, and
    volume traded near the low as more likely seller-driven. It is
    NOT a substitute for genuine bid/ask trade classification.

    Returns (est_buy_volume, est_sell_volume), which always sum to
    row['volume'].
    """
    high = float(row["high"])
    low = float(row["low"])
    close = float(row["close"])
    volume = float(row["volume"])

    candle_range = high - low

    if candle_range <= 0 or volume <= 0:
        # Doji / zero-range or zero-volume candle: split evenly --
        # nothing in the candle shape supports a directional estimate.
        return volume / 2.0, volume / 2.0

    buy_fraction = (close - low) / candle_range
    buy_fraction = min(max(buy_fraction, 0.0), 1.0)

    est_buy = volume * buy_fraction
    est_sell = volume - est_buy

    return est_buy, est_sell


def window_pressure(window_df: pd.DataFrame) -> Tuple[float, float]:
    """
    Sum candle_pressure() over every bar in window_df.

    window_df must already be restricted to the causal window by the
    caller -- this function does not know about "now" or "future".
    """
    if window_df.empty:
        return 0.0, 0.0

    highs = window_df["high"].astype(float).to_numpy()
    lows = window_df["low"].astype(float).to_numpy()
    closes = window_df["close"].astype(float).to_numpy()
    volumes = window_df["volume"].astype(float).to_numpy()

    ranges = highs - lows

    buy_fraction = np.where(
        ranges > 0,
        np.clip((closes - lows) / np.where(ranges > 0, ranges, 1.0), 0.0, 1.0),
        0.5,
    )

    est_buy = volumes * buy_fraction
    est_sell = volumes - est_buy

    return float(est_buy.sum()), float(est_sell.sum())


def relative_volume(
    window_df: pd.DataFrame,
    lookback_df: pd.DataFrame,
) -> Optional[float]:
    """
    window_df's average volume divided by lookback_df's average
    volume. lookback_df must be a trailing slice ending at or before
    the trigger bar -- the caller is responsible for causality here.

    Returns None if lookback_df has no usable volume.
    """
    if lookback_df is None or lookback_df.empty:
        return None

    lookback_avg = lookback_df["volume"].astype(float).mean()

    if not np.isfinite(lookback_avg) or lookback_avg <= 0:
        return None

    window_avg = window_df["volume"].astype(float).mean()

    return float(window_avg / lookback_avg)


def volume_at_price_band(
    window_df: pd.DataFrame,
    price_bottom: float,
    price_top: float,
) -> Optional[float]:
    """
    Fraction (0-1) of window_df's total volume that occurred on bars
    whose [low, high] range overlapped [price_bottom, price_top].

    "Overlapped" is used rather than "closed inside", since a candle
    can wick through a zone without closing there -- consistent with
    strategy/zones.py's own mitigation rule (wick touch, not close).
    """
    if window_df.empty or price_top <= price_bottom:
        return None

    total_volume = window_df["volume"].astype(float).sum()

    if total_volume <= 0:
        return None

    overlap_mask = (
        (window_df["low"].astype(float) <= price_top)
        & (window_df["high"].astype(float) >= price_bottom)
    )

    zone_volume = window_df.loc[overlap_mask, "volume"].astype(float).sum()

    return float(zone_volume / total_volume)