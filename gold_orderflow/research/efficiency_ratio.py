"""
gold_orderflow/research/efficiency_ratio.py

Kaufman's Efficiency Ratio (ER) as a trending/ranging regime measure,
computed causally over M5 OHLC closes.

ER[t] = |close[t] - close[t-N]| /
        sum(|close[i] - close[i-1]| for i in t-N+1..t)

Fixed regime buckets (defined BEFORE looking at trade outcomes):
    ER < 0.2          -> RANGING
    0.2 <= ER <= 0.5  -> TRANSITIONAL
    ER > 0.5          -> TRENDING

Research-only:
    This file does not generate trades, alter the validated strategy,
    or use future information.
"""

import argparse

import numpy as np
import pandas as pd


N = 20
RANGING_MAX = 0.2
TRENDING_MIN = 0.5


def compute_efficiency_ratio(
    df: pd.DataFrame,
    n: int = N,
) -> pd.DataFrame:
    """Compute causal Efficiency Ratio and fixed regime labels."""

    if n <= 0:
        raise ValueError("n must be greater than 0")

    required = {"close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required column(s): {sorted(missing)}"
        )

    out = df.copy()

    out["close"] = pd.to_numeric(out["close"], errors="coerce")

    close = out["close"]

    net_change = (close - close.shift(n)).abs()

    path_length = close.diff().abs().rolling(window=n).sum()

    er = net_change.div(path_length)
    er = er.replace([np.inf, -np.inf], np.nan)

    out["efficiency_ratio"] = er

    out["regime"] = np.select(
        [
            out["efficiency_ratio"] < RANGING_MAX,
            out["efficiency_ratio"] > TRENDING_MIN,
        ],
        [
            "RANGING",
            "TRENDING",
        ],
        default="TRANSITIONAL",
    )

    out.loc[
        out["efficiency_ratio"].isna(),
        "regime",
    ] = None

    return out


def validate_input(
    df: pd.DataFrame,
    time_col: str,
) -> pd.DataFrame:
    """Validate and normalize the OHLC input before calculation."""

    if time_col not in df.columns:
        raise ValueError(
            f"Time column '{time_col}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    out = df.copy()

    timestamps = pd.to_datetime(
        out[time_col],
        utc=True,
        errors="coerce",
    )

    bad_timestamps = timestamps.isna().sum()

    if bad_timestamps:
        print(
            f"WARNING: {bad_timestamps} rows have unparseable "
            f"timestamps in '{time_col}' and will be dropped."
        )

    out[time_col] = timestamps
    out = out.dropna(subset=[time_col])

    out["close"] = pd.to_numeric(
        out["close"],
        errors="coerce",
    )

    bad_close = out["close"].isna().sum()

    if bad_close:
        print(
            f"WARNING: {bad_close} rows have invalid/missing "
            f"close values and will be dropped."
        )

    out = out.dropna(subset=["close"])

    out = out.sort_values(time_col).reset_index(drop=True)

    duplicate_count = out[time_col].duplicated().sum()

    if duplicate_count:
        print(
            f"WARNING: {duplicate_count} duplicate timestamps "
            f"found; keeping the first occurrence."
        )
        out = out.drop_duplicates(
            subset=[time_col],
            keep="first",
        ).reset_index(drop=True)

    if not out[time_col].is_monotonic_increasing:
        raise ValueError(
            "Timestamp column is not strictly chronological "
            "after sorting."
        )

    return out


def main(
    ohlc_csv: str,
    out_csv: str,
    time_col: str,
) -> None:
    df = pd.read_csv(ohlc_csv)

    rows_loaded = len(df)

    df = validate_input(
        df,
        time_col=time_col,
    )

    rows_after_validation = len(df)

    result = compute_efficiency_ratio(df)

    valid_er = result["efficiency_ratio"].notna()

    n_valid = valid_er.sum()

    print()
    print("Efficiency Ratio research")
    print("-------------------------")
    print(f"rows loaded: {rows_loaded}")
    print(f"rows after validation: {rows_after_validation}")
    print(f"N: {N}")
    print(f"RANGING: ER < {RANGING_MAX}")
    print(
        f"TRANSITIONAL: {RANGING_MAX} <= ER <= {TRENDING_MIN}"
    )
    print(f"TRENDING: ER > {TRENDING_MIN}")
    print(f"valid ER rows: {n_valid}")
    print()
    print("Regime counts:")
    print(result["regime"].value_counts(dropna=False))

    if n_valid:
        print()
        print("ER summary:")
        print(result.loc[valid_er, "efficiency_ratio"].describe())

    result.to_csv(
        out_csv,
        index=False,
    )

    print()
    print(f"wrote {len(result)} rows -> {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Compute causal Kaufman Efficiency Ratio "
            "and fixed market-regime buckets."
        )
    )

    parser.add_argument(
        "--ohlc-csv",
        required=True,
        help="Input OHLC CSV containing timestamp and close.",
    )

    parser.add_argument(
        "--out-csv",
        default=(
            "gold_orderflow/data/"
            "xauusd_efficiency_ratio.csv"
        ),
    )

    parser.add_argument(
        "--time-col",
        default="timestamp",
        help="Timestamp column in the OHLC CSV.",
    )

    args = parser.parse_args()

    main(
        ohlc_csv=args.ohlc_csv,
        out_csv=args.out_csv,
        time_col=args.time_col,
    )
