"""
gold_orderflow/research/context_filter_significance.py

Permutation-test significance check on the AGREE/DISAGREE/NEUTRAL
result from context_filter_backtest.py. Answers: could a mean-R or
win-rate gap this large between groups have arisen by chance, given
these group sizes, if quote_pressure's agreement label carried no
real information?

Reuses load_trades/load_ticks/normalize_direction/attach_context from
context_filter_backtest.py unchanged -- only the significance test
itself is new.
"""

import argparse

import numpy as np
import pandas as pd

from gold_orderflow.research.context_filter_backtest import (
    load_trades, load_ticks, normalize_direction, attach_context,
)


def label_agreement(merged: pd.DataFrame, direction_col_norm: str) -> pd.DataFrame:
    valid = merged.dropna(subset=["quote_pressure", direction_col_norm]).copy()
    valid["agreement"] = np.select(
        [
            valid["quote_pressure"] * valid[direction_col_norm] > 0,
            valid["quote_pressure"] * valid[direction_col_norm] < 0,
        ],
        ["AGREE", "DISAGREE"],
        default="NEUTRAL",
    )
    return valid


def permutation_test(valid: pd.DataFrame, outcome_col: str, group_a: str,
                      group_b: str, n_perm: int, seed: int = 42):
    outcome = pd.to_numeric(valid[outcome_col], errors="coerce")
    mask = valid["agreement"].isin([group_a, group_b]) & outcome.notna()
    labels = valid.loc[mask, "agreement"].to_numpy()
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


def main(trades_csv, ticks_csv, time_col, direction_col, outcome_col,
         max_staleness_minutes, n_perm):
    from datetime import timedelta

    trades = load_trades(trades_csv, time_col, direction_col, outcome_col)
    ticks = load_ticks(ticks_csv)
    trades["_direction_norm"] = normalize_direction(trades[direction_col])
    merged = attach_context(trades, ticks, max_staleness=timedelta(minutes=max_staleness_minutes))
    valid = label_agreement(merged, "_direction_norm")

    print(f"trades with usable context: {len(valid)}")
    print(valid["agreement"].value_counts())
    print(f"\nrunning {n_perm} permutations per comparison...")

    permutation_test(valid, outcome_col, "AGREE", "NEUTRAL", n_perm)
    permutation_test(valid, outcome_col, "DISAGREE", "NEUTRAL", n_perm)
    permutation_test(valid, outcome_col, "AGREE", "DISAGREE", n_perm)

    print(
        "\nREMINDER: p-values here describe whether THIS sample's group "
        "gap is distinguishable from random labeling noise. They do not "
        "by themselves validate a live filter -- OOS testing and "
        "stability across periods/regimes are still required."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trades-csv", required=True)
    parser.add_argument("--ticks-csv", required=True)
    parser.add_argument("--time-col", default="entry_time")
    parser.add_argument("--direction-col", default="direction")
    parser.add_argument("--outcome-col", required=True)
    parser.add_argument("--max-staleness-minutes", type=float, default=60.0)
    parser.add_argument("--n-perm", type=int, default=10000)
    args = parser.parse_args()
    main(args.trades_csv, args.ticks_csv, args.time_col, args.direction_col,
         args.outcome_col, args.max_staleness_minutes, args.n_perm)
