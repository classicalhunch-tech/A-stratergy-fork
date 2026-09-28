"""
market_data/research/regime_breakdown.py

The decisive test that a single 2-day window couldn't provide: does the
imbalance -> forward-return relationship hold up across days with
different trends, or does it only appear on days that happened to be
trending?

For each day (from download_daily.py's manifest), independently computes:
  - Pearson correlation: imbalance vs future_return_60s
  - decile spread: mean_return in the top decile minus the bottom decile
    (a simple, robust-to-shape summary of "how much does the relationship
    move return, if at all")

Then prints one row per day next to that day's own net move and regime
label (UP/DOWN/FLAT), so the pattern is visible at a glance.

NOTE ON WINDOW BOUNDARIES: each day is processed independently, so a
window near the very start of a day only sees that day's trades (not
carried over from the previous day), and forward returns near the end of
a day may be NaN for longer horizons if no later trade exists within that
day's file. This is a known, intentional simplification -- it slightly
under-populates the first/last few minutes of each day rather than
silently reaching across day boundaries.

Usage:
    python -m market_data.research.regime_breakdown \
        --manifest market_data/data/daily/BTCUSDT_regime_manifest.csv \
        --time-window 60s --trade-window 500 --min-trades 30
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from ..features import add_rolling_imbalance
from .future_returns import add_forward_returns


def _decile_spread(df: pd.DataFrame, imbalance_col: str, return_col: str, n_buckets: int = 10) -> float:
    work = df[[imbalance_col, return_col]].dropna()
    if len(work) < n_buckets * 10:  # need enough rows per bucket to mean anything
        return np.nan
    try:
        work = work.copy()
        work["bucket"] = pd.qcut(work[imbalance_col], q=n_buckets, duplicates="drop")
    except ValueError:
        return np.nan
    means = work.groupby("bucket", observed=True)[return_col].mean()
    if len(means) < 2:
        return np.nan
    return float(means.iloc[-1] - means.iloc[0])


def analyze_one_day(file_path: str, time_window: str, trade_window: int, min_trades: int) -> dict:
    df = pd.read_csv(file_path)
    if df.empty or len(df) < trade_window:
        return {"n_trades": len(df), "corr_t": np.nan, "corr_n": np.nan, "spread_t": np.nan, "spread_n": np.nan}

    enriched = add_rolling_imbalance(
        df, time_window=time_window, trade_window=trade_window, min_trades=min_trades
    )
    enriched = add_forward_returns(enriched, horizons_seconds=[60])

    t_col = f"imbalance_t{time_window}"
    n_col = f"imbalance_n{trade_window}"
    r_col = "future_return_60s"

    def _corr(col):
        paired = enriched[[col, r_col]].dropna()
        return float(paired[col].corr(paired[r_col])) if len(paired) >= 30 else np.nan

    return {
        "n_trades": len(df),
        "corr_t": _corr(t_col),
        "corr_n": _corr(n_col),
        "spread_t": _decile_spread(enriched, t_col, r_col),
        "spread_n": _decile_spread(enriched, n_col, r_col),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--time-window", default="60s")
    parser.add_argument("--trade-window", type=int, default=500)
    parser.add_argument("--min-trades", type=int, default=30)
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    print(f"Analyzing {len(manifest)} days from {args.manifest} ...\n")

    rows = []
    for _, row in manifest.iterrows():
        print(f"  {row['day']} ({row['regime']}, {row['pct_move']:+.2f}%) ...")
        result = analyze_one_day(row["file"], args.time_window, args.trade_window, args.min_trades)
        rows.append(
            {
                "day": row["day"],
                "regime": row["regime"],
                "pct_move": row["pct_move"],
                **result,
            }
        )

    summary = pd.DataFrame(rows)

    print("\n" + "=" * 100)
    print("PER-DAY SUMMARY")
    print("=" * 100)
    print(summary.to_string(index=False))

    print("\n" + "=" * 100)
    print("BY REGIME (mean correlation/spread across days in each regime)")
    print("=" * 100)
    by_regime = summary.groupby("regime").agg(
        n_days=("day", "count"),
        mean_corr_t=("corr_t", "mean"),
        mean_corr_n=("corr_n", "mean"),
        mean_spread_t=("spread_t", "mean"),
        mean_spread_n=("spread_n", "mean"),
    )
    print(by_regime.to_string())

    print("\nInterpretation guide:")
    print("- If corr_t / corr_n are consistently positive across UP, DOWN, and FLAT days")
    print("  alike, that's real evidence of a general relationship, not a trend artifact.")
    print("- If they're strongly positive on UP days but near zero or negative on DOWN days,")
    print("  the original correlation was very likely a trend confound, not a tradeable signal.")

    out_path = args.manifest.replace("_regime_manifest.csv", "_regime_breakdown.csv")
    summary.to_csv(out_path, index=False)
    print(f"\nWrote per-day summary -> {out_path}")


if __name__ == "__main__":
    main()