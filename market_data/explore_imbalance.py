"""
market_data/explore_imbalance.py

Quick descriptive look at the rolling imbalance feature on real trade data.
This is exploration only -- Step 7 in the plan ("feature research") --
not a filter, not a strategy signal, nothing causal-sensitive beyond what
features.py already guarantees.

Usage:
    python -m market_data.explore_imbalance --file orderflow/market_data/data/BTCUSDT_trades_normalized.csv
    (or wherever your normalized CSV actually landed, e.g. market_data/data/...)
"""

from __future__ import annotations

import argparse

import pandas as pd

from .features import add_rolling_imbalance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="Path to normalized trades CSV")
    parser.add_argument("--time-window", default="60s", help="e.g. 60s, 5min")
    parser.add_argument("--trade-window", type=int, default=500)
    args = parser.parse_args()

    print(f"Loading {args.file} ...")
    df = pd.read_csv(args.file)
    print(f"  {len(df):,} trades loaded")

    print(f"Computing imbalance (time_window={args.time_window}, trade_window={args.trade_window}) ...")
    enriched = add_rolling_imbalance(
        df,
        time_window=args.time_window,
        trade_window=args.trade_window,
    )

    t_col = f"imbalance_t{args.time_window}"
    n_col = f"imbalance_n{args.trade_window}"

    print(f"\n--- {t_col} ---")
    print(enriched[t_col].describe())

    print(f"\n--- {n_col} ---")
    print(enriched[n_col].describe())

    # Quick sanity: does the time-window and trade-window imbalance agree
    # in sign most of the time? (They measure related but not identical
    # things, so this is a plausibility check, not a requirement.)
    both_valid = enriched[[t_col, n_col]].dropna()
    if not both_valid.empty:
        same_sign = (both_valid[t_col] > 0) == (both_valid[n_col] > 0)
        print(f"\nSign agreement between {t_col} and {n_col}: {same_sign.mean() * 100:.1f}% of rows")

    out_path = args.file.replace(".csv", "_with_imbalance.csv")
    enriched.to_csv(out_path, index=False)
    print(f"\nWrote enriched file -> {out_path}")


if __name__ == "__main__":
    main()