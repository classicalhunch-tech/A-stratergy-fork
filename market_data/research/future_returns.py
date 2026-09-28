"""
market_data/research/future_returns.py

Computes forward returns -- the OUTCOME LABEL used in research, not a
feature. This module intentionally looks into the future.

CRITICAL DISTINCTION FROM features.py
--------------------------------------
features.py (imbalance_at, add_rolling_imbalance) is causal: it must never
see a trade after the point being evaluated, because those values are meant
to eventually be usable as real-time decision inputs.

This module is the opposite on purpose: it answers "what happened to price
AFTER this point in time?" -- that is the thing we are trying to predict,
not something we are allowed to feed back in as an input. Do not import
anything from this module into attach.py, a live feature pipeline, or
anything that touches strategy signal generation. It exists only to score
research features against outcomes, exactly like `r_multiple` and
`result_status` already do for the OHLCV proxy in orderflow/.
"""

from __future__ import annotations

from typing import List

import pandas as pd


def add_forward_returns(
    df: pd.DataFrame,
    horizons_seconds: List[int],
    price_col: str = "price",
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """
    For each row, find the trade price at or after (timestamp + horizon)
    and compute the simple return from the row's own price to that future
    price.

    Adds, for each horizon h in horizons_seconds:
        future_price_{h}s
        future_return_{h}s   (NaN if no trade exists that far into the
                               future within this dataset -- typically only
                               near the very end of the loaded range)

    Parameters
    ----------
    df : DataFrame with at least [timestamp_col, price_col], any row order
    horizons_seconds : list of horizons, e.g. [10, 30, 60, 300]

    Returns a new DataFrame sorted by timestamp_col (does not mutate input).
    """
    required = {timestamp_col, price_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    work = df.sort_values(timestamp_col, kind="mergesort").reset_index(drop=True)
    price_lookup = work[[timestamp_col, price_col]].rename(
        columns={timestamp_col: "_lookup_ts", price_col: "_lookup_price"}
    )

    for h in horizons_seconds:
        target = pd.DataFrame({"timestamp": work[timestamp_col] + h * 1000})
        merged = pd.merge_asof(
            target,
            price_lookup,
            left_on="timestamp",
            right_on="_lookup_ts",
            direction="forward",
        )

        future_price_col = f"future_price_{h}s"
        future_return_col = f"future_return_{h}s"

        work[future_price_col] = merged["_lookup_price"].to_numpy()
        work[future_return_col] = (
            work[future_price_col] - work[price_col]
        ) / work[price_col]

    return work