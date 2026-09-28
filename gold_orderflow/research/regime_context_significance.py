"""
gold_orderflow/research/regime_context_significance.py

Attaches the causal market-regime label (RANGING/TRANSITIONAL/TRENDING,
from efficiency_ratio.py) to each of the 464 strategy trades using a
strictly backward merge_asof (nearest prior completed M5 bar's regime
at or before entry_time -- never a future bar).

Then runs the same permutation-test significance check used for the
quote_pressure context-filter question, on regime instead.
"""

import argparse

import numpy as np
import pandas as pd


def load_trades(csv_path: str, time_col: str, outcome_col: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["trade_dt"] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    df = df.dropna(subset=["trade_dt"]).sort_values("trade_dt").reset_index(drop=True)
    return df


def load_regimes(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp", "regime"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df[["timestamp", "regime", "efficiency_ratio"]]


def attach_regime(trades: pd.DataFrame, regimes: pd.DataFrame,
                   max_staleness_minutes: float) -> pd.DataFrame:
    merged = pd.merge_asof(
        trades, regimes,
        left_on="trade_dt", right_on="timestamp",
        direction="backward",
    )
    staleness = merged["trade_dt"] - merged["timestamp"]
    too_stale = staleness > pd.Timedelta(minutes=max_staleness_minutes)
    n_stale = too_stale.sum()
    if n_stale > 0:
        print(f"NOTE: {n_stale} trades had no regime bar within "
              f"{max_staleness_minutes} min before entry; excluded.")
        merged.loc[too_stale, ["regime", "efficiency_ratio"]] = np.nan
    return merged


def permutation_test(valid: pd.DataFrame, outcome_col: str, group_a: str,
                      group_b: str, n_perm: int, seed: int = 42):
    outcome = pd.to_numeric(valid[outcome_col], errors="coerce")
    mask = valid["regime"].isin([group_a, group_b]) & outcome.notna()
    labels = valid.loc[mask, "regime"].to_numpy()
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


def main(trades_csv, regimes_csv, time_col, outcome_col, max_staleness_minutes, n_perm):
    trades = load_trades(trades_csv, time_col, outcome_col)
    regimes = load_regimes(regimes_csv)
    merged = attach_regime(trades, regimes, max_staleness_minutes)

    valid = merged.dropna(subset=["regime"])
    print(f"trades with usable regime context: {len(valid)} / {len(merged)}")
    print(valid["regime"].value_counts())

    print(f"\nOUTCOME ({outcome_col}) BY REGIME")
    outcome_numeric = pd.to_numeric(valid[outcome_col], errors="coerce")
    valid = valid.copy()
    valid["_outcome_numeric"] = outcome_numeric
    summary = valid.groupby("regime")["_outcome_numeric"].agg(
        n="count", mean="mean", median="median", sum="sum",
        win_rate=lambda s: (s > 0).mean(),
    )
    print(summary.to_string())

    print(f"\nrunning {n_perm} permutations per comparison...")
    permutation_test(valid, outcome_col, "TRENDING", "RANGING", n_perm)
    permutation_test(valid, outcome_col, "TRENDING", "TRANSITIONAL", n_perm)
    permutation_test(valid, outcome_col, "RANGING", "TRANSITIONAL", n_perm)

    print(
        "\nREMINDER: research observation only. OOS validation and "
        "stability testing still required before any live use."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trades-csv", default="trades_90000.csv")
    parser.add_argument("--regimes-csv", default="gold_orderflow/data/xauusd_efficiency_ratio.csv")
    parser.add_argument("--time-col", default="entry_time")
    parser.add_argument("--outcome-col", default="r")
    parser.add_argument("--max-staleness-minutes", type=float, default=10.0)
    parser.add_argument("--n-perm", type=int, default=10000)
    args = parser.parse_args()
    main(args.trades_csv, args.regimes_csv, args.time_col, args.outcome_col,
         args.max_staleness_minutes, args.n_perm)
