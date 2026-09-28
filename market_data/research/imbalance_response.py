"""
market_data/research/imbalance_response.py

Step 7 in the plan: "feature research". Joins the causal imbalance feature
(features.py) against forward-return outcome labels (future_returns.py) to
see, descriptively, whether imbalance carries any relationship to what
price does next.

This module does NOT select a threshold, does NOT filter anything, and
does NOT claim causation. It produces a descriptive bucket table --
analogous to the pressure_bucket agree/conflict tables already built for
the OHLCV proxy on gold. Threshold selection is explicitly deferred to a
later, separate step (see threshold_analysis.py's docstring for why).
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from ..features import add_rolling_imbalance
from .future_returns import add_forward_returns


def build_research_table(
    trades_df: pd.DataFrame,
    time_window: str = "60s",
    trade_window: int = 500,
    return_horizons_seconds: List[int] = [10, 30, 60, 300],
    min_trades: int = 1,
) -> pd.DataFrame:
    """
    Build one combined research DataFrame: causal imbalance features +
    forward-return labels, aligned by trade row.

    min_trades : passed through to add_rolling_imbalance -- windows with
        fewer trades than this report NaN imbalance instead of a
        potentially meaningless +-1 from a tiny sample. Check the
        n_trades_* columns' distribution on your data before picking a
        value; on bursty data, 60s windows can have a median trade count
        far below the overall average rate.
    """
    enriched = add_rolling_imbalance(
        trades_df, time_window=time_window, trade_window=trade_window, min_trades=min_trades
    )
    enriched = add_forward_returns(enriched, horizons_seconds=return_horizons_seconds)
    return enriched


def imbalance_deciles_vs_returns(
    research_df: pd.DataFrame,
    imbalance_col: str,
    return_col: str,
    n_buckets: int = 10,
) -> pd.DataFrame:
    """
    Descriptive table: bucket rows into n_buckets quantile groups of
    imbalance_col, report mean/median/count of return_col per bucket.

    This is descriptive only -- it does not pick a threshold. Use it to
    see the shape of the relationship (monotonic? flat? U-shaped?) before
    even considering threshold_analysis.py.
    """
    work = research_df[[imbalance_col, return_col]].dropna()
    if work.empty:
        raise ValueError(f"No non-null rows for {imbalance_col} / {return_col}")

    work = work.copy()
    try:
        work["bucket"] = pd.qcut(work[imbalance_col], q=n_buckets, duplicates="drop")
    except ValueError as e:
        raise ValueError(
            f"Could not form {n_buckets} quantile buckets for {imbalance_col} "
            f"(possibly too many repeated values, e.g. many exact +-1.0 rows): {e}"
        )

    summary = work.groupby("bucket", observed=True).agg(
        n=(return_col, "count"),
        mean_return=(return_col, "mean"),
        median_return=(return_col, "median"),
        std_return=(return_col, "std"),
    )
    return summary


def correlation_summary(
    research_df: pd.DataFrame,
    imbalance_cols: List[str],
    return_cols: List[str],
) -> pd.DataFrame:
    """
    Simple Pearson correlation matrix between each imbalance feature and
    each forward-return horizon. A quick screening tool, not a conclusion --
    correlation on its own says nothing about robustness or OOS stability.
    """
    rows = []
    for icol in imbalance_cols:
        for rcol in return_cols:
            paired = research_df[[icol, rcol]].dropna()
            if len(paired) < 30:
                corr = np.nan
            else:
                corr = paired[icol].corr(paired[rcol])
            rows.append({"imbalance_feature": icol, "return_horizon": rcol, "n": len(paired), "pearson_r": corr})
    return pd.DataFrame(rows)