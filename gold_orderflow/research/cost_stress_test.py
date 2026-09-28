"""
gold_orderflow/research/cost_stress_test.py

Research-only execution-friction stress test for the XAUUSD
quote_pressure vs forward-price relationship.

Purpose
-------
Test whether the preliminary quote_pressure relationship survives:

1. BID/ASK SPREAD
   A position must cross the spread to enter and exit.

2. EXECUTION LAG
   The signal is observed at tick t, but execution occurs after
   a configurable number of ticks.

3. SLIPPAGE
   Additional adverse price movement is applied to BOTH entry
   and exit legs.

This module does NOT define the production trading strategy.
It simply asks whether the research feature has enough raw
directional movement to survive realistic execution friction.

Execution convention
---------------------
LONG:
    signal = +1
    entry = ask + slippage
    exit  = bid - slippage

SHORT:
    signal = -1
    entry = bid - slippage
    exit  = ask + slippage

Because only bid, ask and mid are available at each tick, actual
future execution is approximated using the observed future quote.

Costs are explicitly decomposed as:

    spread_cost
    + slippage_cost
    = total_cost

The analysis is intentionally conservative.

IMPORTANT
---------
lag=0 is an instantaneous-execution BASELINE, not a claim that
zero-latency live execution is achievable.

This module:
    - does not size positions
    - does not use risk gates
    - does not use the existing strategy
    - does not compound trades
    - does not model portfolio effects
    - does not authorize strategy integration

Research sequence:

    feed truth
        ->
    causal features
        ->
    future response
        ->
    regime robustness
        ->
    significance challenge
        ->
    costs / execution stress       <-- THIS MODULE
        ->
    stronger OOS
        ->
    paper validation
        ->
    possible confluence integration
"""

import argparse

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Fixed research parameters
# ---------------------------------------------------------------------

HORIZONS = [1, 5, 20, 50]

LAG_TICKS = [0, 1, 5, 20]

SLIPPAGE_POINTS = [0.00, 0.05, 0.10]

MIN_TRADES = 500


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------

def load_features(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(
        csv_path,
        parse_dates=["dt"],
    )

    df = (
        df.sort_values("dt")
        .drop_duplicates(
            subset=["dt"],
            keep="first",
        )
        .reset_index(drop=True)
    )

    required = [
        "dt",
        "bid",
        "ask",
        "mid",
        "spread",
        "quote_pressure",
    ]

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {', '.join(missing)}"
        )

    numeric_cols = [
        "bid",
        "ask",
        "mid",
        "spread",
        "quote_pressure",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    df = df.dropna(
        subset=numeric_cols
    ).reset_index(drop=True)

    return df


# ---------------------------------------------------------------------
# Simulate execution
# ---------------------------------------------------------------------

def simulate_net_edge(
    df: pd.DataFrame,
    horizon: int,
    lag: int,
    slippage: float,
) -> pd.DataFrame:
    """
    Simulate one independent feature-level trade for every nonzero
    quote_pressure observation.

    Signal at t:
        +1 -> LONG
        -1 -> SHORT

    Entry:
        t + lag

    Exit:
        t + lag + horizon

    Returns one row per simulated trade.
    """

    n = len(df)

    signal = df["quote_pressure"].to_numpy(
        dtype=float
    )

    mid = df["mid"].to_numpy(
        dtype=float
    )

    spread = df["spread"].to_numpy(
        dtype=float
    )

    entry_idx = (
        np.arange(n)
        + lag
    )

    exit_idx = (
        entry_idx
        + horizon
    )

    valid = (
        (signal != 0)
        & (entry_idx < n)
        & (exit_idx < n)
    )

    idx = np.where(valid)[0]

    if len(idx) == 0:
        return pd.DataFrame()

    sig = signal[idx]

    e_idx = entry_idx[idx]
    x_idx = exit_idx[idx]

    entry_mid = mid[e_idx]
    exit_mid = mid[x_idx]

    entry_spread = spread[e_idx]
    exit_spread = spread[x_idx]

    is_long = sig > 0

    # ---------------------------------------------------------------
    # Entry execution
    # ---------------------------------------------------------------

    long_entry = (
        entry_mid
        + entry_spread / 2.0
        + slippage
    )

    short_entry = (
        entry_mid
        - entry_spread / 2.0
        - slippage
    )

    entry_fill = np.where(
        is_long,
        long_entry,
        short_entry,
    )

    # ---------------------------------------------------------------
    # Exit execution
    # ---------------------------------------------------------------

    long_exit = (
        exit_mid
        - exit_spread / 2.0
        - slippage
    )

    short_exit = (
        exit_mid
        + exit_spread / 2.0
        + slippage
    )

    exit_fill = np.where(
        is_long,
        long_exit,
        short_exit,
    )

    # ---------------------------------------------------------------
    # Raw directional movement
    # ---------------------------------------------------------------

    raw_edge = np.where(
        is_long,
        exit_mid - entry_mid,
        entry_mid - exit_mid,
    )

    # ---------------------------------------------------------------
    # Realized simulated edge after execution friction
    # ---------------------------------------------------------------

    net_edge = np.where(
        is_long,
        exit_fill - entry_fill,
        entry_fill - exit_fill,
    )

    # ---------------------------------------------------------------
    # Explicit cost decomposition
    #
    # Each leg pays half of its observed spread.
    # Therefore:
    #
    # total spread cost =
    #     entry_spread/2 + exit_spread/2
    #
    # Slippage is applied to both legs.
    # ---------------------------------------------------------------

    spread_cost = (
        entry_spread / 2.0
        + exit_spread / 2.0
    )

    slippage_cost = (
        2.0 * slippage
    )

    total_cost = (
        spread_cost
        + slippage_cost
    )

    # Numerical consistency check:
    # raw_edge - total_cost should equal net_edge.
    cost_check_error = (
        net_edge
        - (raw_edge - total_cost)
    )

    return pd.DataFrame(
        {
            "signal": sig,
            "direction": np.where(
                is_long,
                "LONG",
                "SHORT",
            ),
            "raw_edge": raw_edge,
            "spread_cost": spread_cost,
            "slippage_cost": np.full(
                len(idx),
                slippage_cost,
            ),
            "total_cost": total_cost,
            "net_edge": net_edge,
            "cost_check_error": cost_check_error,
        }
    )


# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------

def summarize_condition(
    trades: pd.DataFrame,
    horizon: int,
    lag: int,
    slippage: float,
    direction: str,
) -> dict:

    if direction == "ALL":
        subset = trades
    else:
        subset = trades[
            trades["direction"] == direction
        ]

    n = len(subset)

    if n < MIN_TRADES:
        return {
            "horizon_ticks": horizon,
            "lag_ticks": lag,
            "slippage_points": slippage,
            "direction": direction,
            "n_trades": n,
            "mean_raw_edge": np.nan,
            "median_raw_edge": np.nan,
            "mean_spread_cost": np.nan,
            "mean_slippage_cost": np.nan,
            "mean_total_cost": np.nan,
            "mean_net_edge": np.nan,
            "median_net_edge": np.nan,
            "net_win_rate": np.nan,
            "positive_net_edge_fraction": np.nan,
            "mean_cost_to_raw_ratio": np.nan,
            "net_edge_positive": False,
        }

    mean_raw = subset[
        "raw_edge"
    ].mean()

    median_raw = subset[
        "raw_edge"
    ].median()

    mean_spread = subset[
        "spread_cost"
    ].mean()

    mean_slippage = subset[
        "slippage_cost"
    ].mean()

    mean_total_cost = subset[
        "total_cost"
    ].mean()

    mean_net = subset[
        "net_edge"
    ].mean()

    median_net = subset[
        "net_edge"
    ].median()

    win_rate = (
        subset["net_edge"] > 0
    ).mean()

    positive_fraction = win_rate

    if mean_raw > 0:
        cost_to_raw = (
            mean_total_cost
            / mean_raw
        )
    else:
        cost_to_raw = np.nan

    return {
        "horizon_ticks": horizon,
        "lag_ticks": lag,
        "slippage_points": slippage,
        "direction": direction,
        "n_trades": n,
        "mean_raw_edge": mean_raw,
        "median_raw_edge": median_raw,
        "mean_spread_cost": mean_spread,
        "mean_slippage_cost": mean_slippage,
        "mean_total_cost": mean_total_cost,
        "mean_net_edge": mean_net,
        "median_net_edge": median_net,
        "net_win_rate": win_rate,
        "positive_net_edge_fraction": positive_fraction,
        "mean_cost_to_raw_ratio": cost_to_raw,
        "net_edge_positive": mean_net > 0,
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main(
    csv_path: str,
    out_csv: str,
):

    df = load_features(
        csv_path
    )

    print(
        f"rows loaded: {len(df)}"
    )

    nonzero = (
        df["quote_pressure"] != 0
    ).sum()

    print(
        f"nonzero quote_pressure rows: "
        f"{nonzero}"
    )

    print(
        f"horizons: {HORIZONS}"
    )

    print(
        f"execution lags: {LAG_TICKS}"
    )

    print(
        f"slippage points: "
        f"{SLIPPAGE_POINTS}"
    )

    results = []

    for horizon in HORIZONS:

        for lag in LAG_TICKS:

            for slippage in SLIPPAGE_POINTS:

                trades = simulate_net_edge(
                    df,
                    horizon,
                    lag,
                    slippage,
                )

                if trades.empty:
                    continue

                for direction in [
                    "ALL",
                    "LONG",
                    "SHORT",
                ]:

                    result = summarize_condition(
                        trades,
                        horizon,
                        lag,
                        slippage,
                        direction,
                    )

                    results.append(
                        result
                    )

    out = pd.DataFrame(
        results
    )

    pd.set_option(
        "display.width",
        240,
    )

    # ---------------------------------------------------------------
    # Full stress grid
    # ---------------------------------------------------------------

    print(
        "\n"
        + "=" * 110
    )

    print(
        "COST / EXECUTION / SLIPPAGE STRESS GRID"
    )

    print(
        "=" * 110
    )

    print(
        out.to_string(
            index=False
        )
    )

    out.to_csv(
        out_csv,
        index=False,
    )

    print(
        f"\nwrote {out_csv}"
    )

    # ---------------------------------------------------------------
    # Survival summary
    # ---------------------------------------------------------------

    print(
        "\n"
        + "=" * 110
    )

    print(
        "SURVIVAL SUMMARY — ALL DIRECTIONS"
    )

    print(
        "=" * 110
    )

    all_results = out[
        out["direction"] == "ALL"
    ]

    for horizon in HORIZONS:

        h_sub = all_results[
            all_results["horizon_ticks"]
            == horizon
        ]

        if h_sub.empty:
            continue

        survived = int(
            h_sub[
                "net_edge_positive"
            ].sum()
        )

        total = len(
            h_sub
        )

        print(
            f"horizon={horizon:>3} ticks | "
            f"positive mean net edge: "
            f"{survived}/{total}"
        )

    # ---------------------------------------------------------------
    # Execution-lag survival
    # ---------------------------------------------------------------

    print(
        "\n"
        + "=" * 110
    )

    print(
        "SURVIVAL BY EXECUTION LAG — ALL DIRECTIONS"
    )

    print(
        "=" * 110
    )

    for lag in LAG_TICKS:

        lag_sub = all_results[
            all_results["lag_ticks"]
            == lag
        ]

        survived = int(
            lag_sub[
                "net_edge_positive"
            ].sum()
        )

        total = len(
            lag_sub
        )

        print(
            f"lag={lag:>2} ticks | "
            f"positive mean net edge: "
            f"{survived}/{total}"
        )

    # ---------------------------------------------------------------
    # Internal accounting check
    # ---------------------------------------------------------------

    print(
        "\n"
        + "=" * 110
    )

    print(
        "EXECUTION COST ACCOUNTING CHECK"
    )

    print(
        "=" * 110
    )

    # Re-run one baseline condition for a deterministic check.
    baseline = simulate_net_edge(
        df,
        horizon=20,
        lag=0,
        slippage=0.0,
    )

    if baseline.empty:
        print(
            "baseline check unavailable"
        )
    else:
        max_abs_error = float(
            baseline[
                "cost_check_error"
            ].abs().max()
        )

        print(
            f"maximum absolute accounting error: "
            f"{max_abs_error:.12f}"
        )

        if max_abs_error < 1e-10:
            print(
                "PASS: raw_edge - total_cost == net_edge"
            )
        else:
            print(
                "WARNING: cost accounting mismatch detected"
            )

    # ---------------------------------------------------------------
    # Interpretation reminder
    # ---------------------------------------------------------------

    print(
        "\n"
        + "=" * 110
    )

    print(
        "RESEARCH INTERPRETATION"
    )

    print(
        "=" * 110
    )

    print(
        "This is NOT a production trading strategy."
    )

    print(
        "It asks whether the raw quote_pressure relationship "
        "survives spread, execution lag and slippage."
    )

    print(
        "\nImportant:"
    )

    print(
        "- lag=0 is an instantaneous-execution baseline, "
        "not a realistic live-latency assumption."
    )

    print(
        "- Positive mean net edge does not establish profitability."
    )

    print(
        "- No position sizing or risk management is modeled."
    )

    print(
        "- Every nonzero quote_pressure observation is treated "
        "as a hypothetical signal."
    )

    print(
        "- No existing strategy filter is applied."
    )

    print(
        "- No portfolio interaction is modeled."
    )

    print(
        "- Costs are expressed in the dataset's price-point units."
    )

    print(
        "\nNEXT RESEARCH STEP:"
    )

    print(
        "Stronger chronological out-of-sample validation, "
        "using only assumptions that survive this stress test."
    )


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--csv",
        required=True,
        help=(
            "path to quote_diagnostics.py "
            "output CSV"
        ),
    )

    parser.add_argument(
        "--out-csv",
        default=(
            "gold_orderflow/data/"
            "cost_stress_test.csv"
        ),
    )

    args = parser.parse_args()

    main(
        args.csv,
        args.out_csv,
    )