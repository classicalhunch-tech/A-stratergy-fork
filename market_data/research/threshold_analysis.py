"""
market_data/research/threshold_analysis.py

IMPORTANT -- READ BEFORE USING
-------------------------------
Per project rule ("Do NOT optimize thresholds immediately", Section 14/25 of
the project spec): this module does NOT search for the threshold that
maximizes any backtest metric. That search-and-pick-the-best pattern is a
direct overfitting risk -- it will always find *something* that looks good
on the sample it was tuned on.

What this module does instead: given a threshold you specify (not one it
found for you), it reports how the outcome distribution splits above vs
below that threshold, so you can inspect a candidate honestly, one at a
time, with full visibility into sample sizes on each side.

If you want to see the full shape of the relationship without picking any
threshold yet, use imbalance_response.imbalance_deciles_vs_returns instead
-- that's the right tool for exploration. This module is for checking one
specific, deliberately chosen candidate threshold.
"""

from __future__ import annotations

import pandas as pd


def evaluate_threshold(
    research_df: pd.DataFrame,
    imbalance_col: str,
    return_col: str,
    threshold: float,
) -> pd.DataFrame:
    """
    Split rows into imbalance_col > threshold vs <= threshold, and report
    return_col statistics on each side. One threshold, chosen by the
    caller -- this function does not search for one.
    """
    work = research_df[[imbalance_col, return_col]].dropna()
    if work.empty:
        raise ValueError(f"No non-null rows for {imbalance_col} / {return_col}")

    work = work.copy()
    work["side"] = pd.Series(
        ["above" if v > threshold else "at_or_below" for v in work[imbalance_col]],
        index=work.index,
    )

    summary = work.groupby("side").agg(
        n=(return_col, "count"),
        mean_return=(return_col, "mean"),
        median_return=(return_col, "median"),
        std_return=(return_col, "std"),
    )
    summary["threshold_used"] = threshold
    return summary


def evaluate_threshold_grid_for_inspection(
    research_df: pd.DataFrame,
    imbalance_col: str,
    return_col: str,
    candidate_thresholds: list,
) -> pd.DataFrame:
    """
    Reports the same split for several candidate thresholds side by side,
    purely for you to *read* and reason about -- e.g. "does the mean return
    on the 'above' side increase steadily as the threshold rises, or does
    it jump around unpredictably (a sign of overfitting risk / small
    samples at the extremes)?"

    This function deliberately does NOT return "the best" row, does NOT
    sort by performance, and does NOT compute a backtest P&L. Selecting a
    winner from this table defeats its purpose -- eyeball the whole table,
    check sample sizes, and remember this is Step 5 of the project's 7-step
    process (build features -> record -> analyze -> baseline -> simple
    deterministic rule -> validate OOS -> *then* consider thresholds).
    """
    rows = []
    for t in candidate_thresholds:
        result = evaluate_threshold(research_df, imbalance_col, return_col, t)
        for side, row in result.iterrows():
            rows.append(
                {
                    "threshold": t,
                    "side": side,
                    "n": row["n"],
                    "mean_return": row["mean_return"],
                    "median_return": row["median_return"],
                    "std_return": row["std_return"],
                }
            )
    return pd.DataFrame(rows)