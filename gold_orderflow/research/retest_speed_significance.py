"""
gold_orderflow/research/retest_speed_significance.py

Retest speed (bars from liquidity sweep to actual retest trigger),
derived directly from trades_90000.csv's own setup_time/entry_time
columns -- no new data collection needed. Fixed buckets decided
before looking at any outcome:
    FAST   <= 5 bars  (<=25 min)
    NORMAL 6-20 bars  (30-100 min)
    SLOW   > 20 bars  (>100 min)
"""

import argparse

import numpy as np
import pandas as pd

BAR_MINUTES = 5
FAST_MAX_BARS = 5
NORMAL_MAX_BARS = 20


def load_trades(csv_path: str, outcome_col: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["setup_time"] = pd.to_datetime(df["setup_time"], utc=True, errors="coerce")
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True, errors="coerce")
    df = df.dropna(subset=["setup_time", "entry_time"])

    minutes = (df["entry_time"] - df["setup_time"]).dt.total_seconds() / 60.0
    df["retest_bars"] = minutes / BAR_MINUTES

    df["retest_speed"] = np.select(
        [df["retest_bars"] <= FAST_MAX_BARS,
         df["retest_bars"] <= NORMAL_MAX_BARS],
        ["FAST", "NORMAL"],
        default="SLOW",
    )
    return df


def permutation_test(valid, outcome_col, group_a, group_b, n_perm, seed=42):
    outcome = pd.to_numeric(valid[outcome_col], errors="coerce")
    mask = valid["retest_speed"].isin([group_a, group_b]) & outcome.notna()
    labels = valid.loc[mask, "retest_speed"].to_numpy()
    values = outcome[mask].to_numpy()

    def mean_diff(lab, val):
        return val[lab == group_a].mean() - val[lab == group_b].mean()

    def win_rate_diff(lab, val):
        return (val[lab == group_a] > 0).mean() - (val[lab == group_b] > 0).mean()

    obs_mean_diff = mean_diff(labels, values)
    obs_wr_diff = win_rate_diff(labels, values)

    rng = np.random.default_rng(seed)
    mean_diffs = np.empty(n_perm)
    wr_diffs = np.empty(n_perm)
    for i in range(n_perm):
        shuffled = rng.permutation(labels)
        mean_diffs[i] = mean_diff(shuffled, values)
        wr_diffs[i] = win_rate_diff(shuffled, values)

    p_mean = (np.abs(mean_diffs) >= np.abs(obs_mean_diff)).mean()
    p_wr = (np.abs(wr_diffs) >= np.abs(obs_wr_diff)).mean()
    n_a = (labels == group_a).sum()
    n_b = (labels == group_b).sum()
    print(f"\n{group_a} (n={n_a}) vs {group_b} (n={n_b})")
    print(f"  observed mean-R diff:  {obs_mean_diff:+.4f}   permutation p = {p_mean:.4f}")
    print(f"  observed win-rate diff: {obs_wr_diff:+.4f}   permutation p = {p_wr:.4f}")


def main(trades_csv, outcome_col, n_perm):
    trades = load_trades(trades_csv, outcome_col)
    print(f"trades: {len(trades)}")
    print(trades["retest_speed"].value_counts())

    print(f"\nOUTCOME ({outcome_col}) BY RETEST SPEED")
    outcome_numeric = pd.to_numeric(trades[outcome_col], errors="coerce")
    trades = trades.copy()
    trades["_outcome_numeric"] = outcome_numeric
    summary = trades.groupby("retest_speed")["_outcome_numeric"].agg(
        n="count", mean="mean", median="median", sum="sum",
        win_rate=lambda s: (s > 0).mean(),
    )
    print(summary.to_string())

    print(f"\nrunning {n_perm} permutations per comparison...")
    permutation_test(trades, outcome_col, "FAST", "NORMAL", n_perm)
    permutation_test(trades, outcome_col, "FAST", "SLOW", n_perm)
    permutation_test(trades, outcome_col, "NORMAL", "SLOW", n_perm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trades-csv", default="trades_90000.csv")
    parser.add_argument("--outcome-col", default="r")
    parser.add_argument("--n-perm", type=int, default=10000)
    args = parser.parse_args()
    main(args.trades_csv, args.outcome_col, args.n_perm)
