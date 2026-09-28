"""
market_data/research/oos_validation.py

Reusable chronological out-of-sample split -- the same idea as the
check_oos_split.py script already used for the gold OHLCV proxy, but
generalized so it can be reused for new research features.

This module uses a strict chronological split:
- first portion = development/train data
- later portion = out-of-sample/test data

No shuffling or random sampling is used.
"""

from __future__ import annotations

import pandas as pd

from .imbalance_response import imbalance_deciles_vs_returns


def split_chronological(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    train_frac: float = 0.7,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Sort by timestamp_col and split into (train, test) by chronological
    position.

    The first train_frac of rows becomes the development portion.
    The remaining rows become the out-of-sample portion.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (train, test)
    """
    if timestamp_col not in df.columns:
        raise ValueError(
            f"Missing timestamp column: {timestamp_col}"
        )

    if not 0.0 < train_frac < 1.0:
        raise ValueError(
            "train_frac must be strictly between 0 and 1."
        )

    work = (
        df.sort_values(timestamp_col, kind="mergesort")
        .reset_index(drop=True)
    )

    if len(work) < 2:
        raise ValueError(
            "At least two rows are required for an OOS split."
        )

    split_idx = int(len(work) * train_frac)

    if split_idx <= 0 or split_idx >= len(work):
        raise ValueError(
            "train_frac does not leave data in both portions."
        )

    train = work.iloc[:split_idx].copy()
    test = work.iloc[split_idx:].copy()

    return train, test


def compare_deciles_oos(
    research_df: pd.DataFrame,
    imbalance_col: str,
    return_col: str,
    n_buckets: int = 10,
    train_frac: float = 0.7,
    timestamp_col: str = "timestamp",
) -> dict[str, pd.DataFrame]:
    """
    Run the same imbalance-decile-vs-return analysis independently on
    the development and out-of-sample portions.

    This allows a first-pass visual/manual check of whether the observed
    relationship is reasonably consistent across time.

    This function does not:
    - optimize thresholds
    - select a best result
    - calculate trading P&L
    - perform significance testing
    - claim causation

    Returns
    -------
    dict[str, pd.DataFrame]
        {
            "train": development summary,
            "test": out-of-sample summary,
        }
    """
    train, test = split_chronological(
        research_df,
        timestamp_col=timestamp_col,
        train_frac=train_frac,
    )

    train_summary = imbalance_deciles_vs_returns(
        train,
        imbalance_col,
        return_col,
        n_buckets=n_buckets,
    )

    test_summary = imbalance_deciles_vs_returns(
        test,
        imbalance_col,
        return_col,
        n_buckets=n_buckets,
    )

    return {
        "train": train_summary,
        "test": test_summary,
    }