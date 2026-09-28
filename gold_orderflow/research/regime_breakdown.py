"""
gold_orderflow/research/regime_breakdown.py

Research-only. Tests whether the preliminary quote_pressure vs
forward-return relationship survives across different market regimes.

Regimes:
  1. Price trend: UP / DOWN / FLAT
  2. Volatility: HIGH / LOW
  3. Quote activity: HIGH / LOW

Strict causality:
- Trend uses only a trailing price window.
- Volatility uses only trailing mid-price changes.
- Activity uses only trailing inter-arrival times.
- Regime thresholds use ONLY prior observations.
- No future information is used to label a row.

quote_pressure is {-1, 0, +1}, so it is grouped explicitly rather
than using qcut/deciles.

This module is descriptive research only.
It does not establish significance, profitability, or integration readiness.
"""

import argparse

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Fixed research parameters
# ---------------------------------------------------------------------

TREND_WINDOW_TICKS = 2000
TREND_THRESHOLD = 0.15

VOL_WINDOW_TICKS = 2000
ACTIVITY_WINDOW_TICKS = 2000

# Historical baseline used to classify current volatility/activity.
# IMPORTANT: shifted by one row, so the current observation can never
# influence its own regime threshold.
REGIME_BASELINE_WINDOW_TICKS = 20000

HORIZONS = [1, 5, 20, 50]

MIN_GROUP_ROWS = 200
MIN_CORR_ROWS = 500

# Minimum rows required after causal regime labeling before continuing.
# Below this, warm-up windows have likely consumed most/all of the data.
MIN_ROWS_AFTER_LABELING = 5000


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------

def load_features(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["dt"])

    df = (
        df.sort_values("dt")
        .drop_duplicates(subset=["dt"], keep="first")
        .reset_index(drop=True)
    )

    required = [
        "dt",
        "mid",
        "mid_change",
        "inter_arrival_ms",
        "quote_pressure",
    ]

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {', '.join(missing)}"
        )

    return df


# ---------------------------------------------------------------------
# Forward returns
# ---------------------------------------------------------------------

def add_forward_returns(
    df: pd.DataFrame,
    horizons: list[int],
) -> pd.DataFrame:
    out = df.copy()

    for h in horizons:
        # Strictly future mid-price.
        out[f"fwd_ret_{h}"] = out["mid"].shift(-h) - out["mid"]

    return out


# ---------------------------------------------------------------------
# Causal regime labels
# ---------------------------------------------------------------------

def add_regime_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # ================================================================
    # 1. TREND REGIME
    # ================================================================

    trailing_move = (
        out["mid"]
        - out["mid"].shift(TREND_WINDOW_TICKS)
    )

    # Start as missing, NOT FLAT.
    # The first TREND_WINDOW_TICKS rows do not have enough history.
    trend = pd.Series(
        pd.NA,
        index=out.index,
        dtype="object",
    )

    valid_trend = trailing_move.notna()

    trend.loc[
        valid_trend & (trailing_move > TREND_THRESHOLD)
    ] = "UP"

    trend.loc[
        valid_trend & (trailing_move < -TREND_THRESHOLD)
    ] = "DOWN"

    trend.loc[
        valid_trend
        & (trailing_move >= -TREND_THRESHOLD)
        & (trailing_move <= TREND_THRESHOLD)
    ] = "FLAT"

    out["trend_regime"] = trend

    # ================================================================
    # 2. VOLATILITY REGIME
    # ================================================================

    trailing_vol = (
        out["mid_change"]
        .rolling(
            VOL_WINDOW_TICKS,
            min_periods=VOL_WINDOW_TICKS,
        )
        .std()
    )

    # The baseline is based ONLY on prior observations.
    # Current trailing_vol is shifted out before calculating
    # the historical median.
    vol_threshold = (
        trailing_vol
        .shift(1)
        .rolling(
            REGIME_BASELINE_WINDOW_TICKS,
            min_periods=REGIME_BASELINE_WINDOW_TICKS,
        )
        .median()
    )

    out["vol_regime"] = pd.Series(
        pd.NA,
        index=out.index,
        dtype="object",
    )

    valid_vol = (
        trailing_vol.notna()
        & vol_threshold.notna()
    )

    out.loc[
        valid_vol & (trailing_vol >= vol_threshold),
        "vol_regime",
    ] = "HIGH"

    out.loc[
        valid_vol & (trailing_vol < vol_threshold),
        "vol_regime",
    ] = "LOW"

    # ================================================================
    # 3. QUOTE ACTIVITY REGIME
    # ================================================================

    trailing_activity = (
        out["inter_arrival_ms"]
        .rolling(
            ACTIVITY_WINDOW_TICKS,
            min_periods=ACTIVITY_WINDOW_TICKS,
        )
        .mean()
    )

    # Lower inter-arrival time = more active quote stream.
    #
    # Again, shift by one so the current observation cannot affect
    # its own historical threshold.
    activity_threshold = (
        trailing_activity
        .shift(1)
        .rolling(
            REGIME_BASELINE_WINDOW_TICKS,
            min_periods=REGIME_BASELINE_WINDOW_TICKS,
        )
        .median()
    )

    out["activity_regime"] = pd.Series(
        pd.NA,
        index=out.index,
        dtype="object",
    )

    valid_activity = (
        trailing_activity.notna()
        & activity_threshold.notna()
    )

    out.loc[
        valid_activity
        & (trailing_activity <= activity_threshold),
        "activity_regime",
    ] = "HIGH"

    out.loc[
        valid_activity
        & (trailing_activity > activity_threshold),
        "activity_regime",
    ] = "LOW"

    # Only keep rows for which ALL regime dimensions have
    # sufficient historical information.
    out = out.dropna(
        subset=[
            "trend_regime",
            "vol_regime",
            "activity_regime",
        ]
    ).reset_index(drop=True)

    return out


# ---------------------------------------------------------------------
# Quote-pressure group analysis
# ---------------------------------------------------------------------

def quote_pressure_group_report(
    df: pd.DataFrame,
    horizon: int,
    regime_col: str,
) -> pd.DataFrame:

    target_col = f"fwd_ret_{horizon}"
    rows = []

    for regime_val, regime_df in df.groupby(regime_col):

        for qp_val, qp_df in regime_df.groupby(
            "quote_pressure",
            dropna=True,
        ):

            valid = qp_df[target_col].dropna()

            if len(valid) < MIN_GROUP_ROWS:
                continue

            rows.append({
                "regime_dim": regime_col,
                "regime_value": regime_val,
                "quote_pressure": qp_val,
                "horizon_ticks": horizon,
                "n": len(valid),
                "fwd_ret_mean": valid.mean(),
                "fwd_ret_std": valid.std(),
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Correlation by regime
# ---------------------------------------------------------------------

def correlation_by_regime(
    df: pd.DataFrame,
    horizon: int,
    regime_col: str,
) -> pd.DataFrame:

    target_col = f"fwd_ret_{horizon}"
    rows = []

    for regime_val, regime_df in df.groupby(regime_col):

        valid = regime_df[
            ["quote_pressure", target_col]
        ].dropna()

        if len(valid) < MIN_CORR_ROWS:
            continue

        corr = np.corrcoef(
            valid["quote_pressure"],
            valid[target_col],
        )[0, 1]

        rows.append({
            "regime_dim": regime_col,
            "regime_value": regime_val,
            "horizon_ticks": horizon,
            "n": len(valid),
            "corr": corr,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Main research run
# ---------------------------------------------------------------------

def main(
    csv_path: str,
    out_prefix: str,
):

    df = load_features(csv_path)
    n_loaded = len(df)

    # Regime labels are created BEFORE forward returns.
    # This makes the causal separation explicit.
    df = add_regime_labels(df)

    print(
        f"rows loaded: {n_loaded}  |  "
        f"rows available after causal regime labeling: {len(df)}  |  "
        f"consumed by warm-up: {n_loaded - len(df)}"
    )

    if len(df) < MIN_ROWS_AFTER_LABELING:
        raise RuntimeError(
            f"Only {len(df)} rows remain after regime labeling "
            f"(minimum required: {MIN_ROWS_AFTER_LABELING}). "
            f"The dataset is too short relative to "
            f"VOL_WINDOW_TICKS + REGIME_BASELINE_WINDOW_TICKS warm-up "
            f"requirements. Use a longer CSV or reduce those constants "
            f"(before inspecting results, not after)."
        )

    df = add_forward_returns(
        df,
        HORIZONS,
    )

    print("\nREGIME COUNTS")

    print(
        "\ntrend_regime:\n",
        df["trend_regime"].value_counts(),
    )

    print(
        "\nvol_regime:\n",
        df["vol_regime"].value_counts(),
    )

    print(
        "\nactivity_regime:\n",
        df["activity_regime"].value_counts(),
    )

    regime_cols = [
        "trend_regime",
        "vol_regime",
        "activity_regime",
    ]

    all_corr = []
    all_groups = []

    for regime_col in regime_cols:

        print("\n" + "=" * 70)
        print(
            f"CORRELATION BY {regime_col.upper()}"
        )
        print("=" * 70)

        for h in HORIZONS:

            corr_df = correlation_by_regime(
                df,
                h,
                regime_col,
            )

            if corr_df.empty:
                print(
                    f"\nh={h}: insufficient data"
                )
            else:
                print(
                    corr_df.to_string(
                        index=False
                    )
                )

            all_corr.append(corr_df)

        print("\n" + "-" * 70)
        print(
            f"QUOTE_PRESSURE GROUP MEANS BY "
            f"{regime_col.upper()} "
            f"(horizon=20)"
        )
        print("-" * 70)

        group_df = quote_pressure_group_report(
            df,
            20,
            regime_col,
        )

        if group_df.empty:
            print("insufficient data")
        else:
            print(
                group_df.to_string(
                    index=False
                )
            )

        all_groups.append(group_df)

    # Remove empty frames before concatenation.
    all_corr = [
        x for x in all_corr
        if not x.empty
    ]

    all_groups = [
        x for x in all_groups
        if not x.empty
    ]

    if not all_corr:
        raise RuntimeError(
            "No correlation results were produced."
        )

    corr_out = pd.concat(
        all_corr,
        ignore_index=True,
    )

    if all_groups:
        group_out = pd.concat(
            all_groups,
            ignore_index=True,
        )
    else:
        group_out = pd.DataFrame()

    corr_out.to_csv(
        f"{out_prefix}_regime_corr.csv",
        index=False,
    )

    group_out.to_csv(
        f"{out_prefix}_regime_groups.csv",
        index=False,
    )

    # ================================================================
    # SIGN CONSISTENCY
    # ================================================================

    print("\n" + "=" * 70)
    print(
        "SIGN CONSISTENCY CHECK"
    )
    print(
        "(Does correlation keep the same sign across "
        "all available regime values?)"
    )
    print("=" * 70)

    for regime_col in regime_cols:

        sub = corr_out[
            corr_out["regime_dim"] == regime_col
        ]

        for h in HORIZONS:

            h_sub = sub[
                sub["horizon_ticks"] == h
            ]

            if h_sub.empty:
                continue

            values = h_sub["corr"]

            all_positive = (
                values > 0
            ).all()

            all_negative = (
                values < 0
            ).all()

            consistent = (
                all_positive
                or all_negative
            )

            value_dict = dict(
                zip(
                    h_sub["regime_value"],
                    h_sub["corr"].round(5),
                )
            )

            print(
                f"{regime_col:>16} | "
                f"h={h:>3} | "
                f"consistent_sign={consistent} | "
                f"values={value_dict}"
            )

    print(
        f"\nwrote result files with prefix: "
        f"{out_prefix}_*.csv"
    )

    print(
        "\nREMINDER: this is regime analysis only. "
        "Sign consistency does not establish significance, "
        "economic value, robustness after costs, or a trading edge."
    )


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--csv",
        required=True,
        help="path to quote_diagnostics.py output CSV",
    )

    parser.add_argument(
        "--out-prefix",
        default="gold_orderflow/data/regime_breakdown",
    )

    args = parser.parse_args()

    main(
        args.csv,
        args.out_prefix,
    )