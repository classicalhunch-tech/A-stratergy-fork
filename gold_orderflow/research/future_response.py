"""
gold_orderflow/research/future_response.py

Research-only. Tests whether XAUUSD quote/microstructure features
(quote_pressure, spread, spread_change) computed at time T carry any
relationship with mid-price movement after T.

This is explicitly QUOTE-side research, not order-flow research. No
buyer/seller aggressor claim is made anywhere in this module.

Causality rule enforced throughout:
    feature[t]  is computed only from data at or before t
    target[t]   is computed only from data strictly after t
Feature and target are never allowed to share information.

This module does NOT decide whether anything is tradeable. It only
measures whether a relationship exists, at what horizons, and whether
it survives basic out-of-sample and decile checks.
"""

import argparse

import numpy as np
import pandas as pd


def load_features(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["dt"])
    df = df.sort_values("dt").reset_index(drop=True)
    return df


def add_forward_returns(df: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """
    horizons are given in NUMBER OF TICKS ahead (not time), since this
    is an irregularly-spaced tick stream. Each forward return uses only
    mid prices strictly after the current row -- no lookahead into the
    feature side.
    """
    out = df.copy()
    for h in horizons:
        out[f"fwd_ret_{h}"] = out["mid"].shift(-h) - out["mid"]
    return out


def correlation_report(df: pd.DataFrame, feature_cols: list[str],
                        horizons: list[int]) -> pd.DataFrame:
    rows = []
    for feat in feature_cols:
        for h in horizons:
            target_col = f"fwd_ret_{h}"
            valid = df[[feat, target_col]].dropna()
            if len(valid) < 100:
                continue
            corr = np.corrcoef(valid[feat], valid[target_col])[0, 1]
            rows.append({
                "feature": feat,
                "horizon_ticks": h,
                "n": len(valid),
                "corr": corr,
            })
    return pd.DataFrame(rows)


def decile_report(df: pd.DataFrame, feature: str, horizon: int) -> pd.DataFrame:
    target_col = f"fwd_ret_{horizon}"
    valid = df[[feature, target_col]].dropna().copy()
    if len(valid) < 1000:
        return pd.DataFrame()

    valid["decile"] = pd.qcut(valid[feature], 10, labels=False, duplicates="drop")
    summary = valid.groupby("decile").agg(
        n=(target_col, "size"),
        feature_mean=(feature, "mean"),
        fwd_ret_mean=(target_col, "mean"),
        fwd_ret_std=(target_col, "std"),
    )
    return summary


def chronological_oos_split(df: pd.DataFrame, train_frac: float = 0.7):
    cutoff = int(len(df) * train_frac)
    return df.iloc[:cutoff].copy(), df.iloc[cutoff:].copy()


def main(csv_path: str, out_prefix: str):
    df = load_features(csv_path)

    horizons = [1, 5, 20, 50, 100, 500]
    df = add_forward_returns(df, horizons)

    feature_cols = ["quote_pressure", "spread", "spread_change"]

    print("=" * 70)
    print("FULL-SAMPLE CORRELATION: feature[t] vs fwd_ret[t+h]")
    print("=" * 70)
    full_corr = correlation_report(df, feature_cols, horizons)
    print(full_corr.to_string(index=False))
    full_corr.to_csv(f"{out_prefix}_full_corr.csv", index=False)

    train, test = chronological_oos_split(df, train_frac=0.7)

    print("\n" + "=" * 70)
    print("IN-SAMPLE (first 70%, chronological) CORRELATION")
    print("=" * 70)
    train_corr = correlation_report(train, feature_cols, horizons)
    print(train_corr.to_string(index=False))

    print("\n" + "=" * 70)
    print("OUT-OF-SAMPLE (last 30%, chronological) CORRELATION")
    print("=" * 70)
    test_corr = correlation_report(test, feature_cols, horizons)
    print(test_corr.to_string(index=False))

    combined = train_corr.merge(
        test_corr, on=["feature", "horizon_ticks"],
        suffixes=("_in_sample", "_oos"),
    )
    combined.to_csv(f"{out_prefix}_oos_corr.csv", index=False)
    print("\n" + "=" * 70)
    print("IN-SAMPLE vs OOS (same sign + similar magnitude = more credible)")
    print("=" * 70)
    print(combined.to_string(index=False))

    print("\n" + "=" * 70)
    print("DECILE REPORT: quote_pressure vs fwd_ret_20 (full sample)")
    print("=" * 70)
    deciles = decile_report(df, "quote_pressure", horizon=20)
    print(deciles.to_string())
    if not deciles.empty:
        deciles.to_csv(f"{out_prefix}_decile_quote_pressure_h20.csv")

    print("\n" + "=" * 70)
    print("DECILE REPORT: spread vs fwd_ret_20 (full sample)")
    print("=" * 70)
    deciles_spread = decile_report(df, "spread", horizon=20)
    print(deciles_spread.to_string())
    if not deciles_spread.empty:
        deciles_spread.to_csv(f"{out_prefix}_decile_spread_h20.csv")

    print(f"\nwrote result files with prefix: {out_prefix}_*.csv")
    print(
        "\nREMINDER: correlation alone is not a trading edge. This is "
        "step 3 of the roadmap (future-response research) only. Regime "
        "analysis, significance testing, cost/stress testing, and full "
        "OOS validation still follow before any integration decision."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True,
                         help="path to a quote_diagnostics.py output CSV")
    parser.add_argument("--out-prefix", default="gold_orderflow/data/future_response")
    args = parser.parse_args()
    main(args.csv, args.out_prefix)