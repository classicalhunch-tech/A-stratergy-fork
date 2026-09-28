"""
gold_orderflow/research/context_filter_backtest.py

Research-only. Tests quote_pressure as a CONTEXT/CONFLUENCE FILTER on
top of the existing validated strategy's own trades -- NOT as a
standalone signal.

Critical distinction from cost_stress_test.py:
    cost_stress_test.py asked "can quote_pressure trade profitably on
    its own, paying its own round-trip spread?" -- answer was a
    decisive NO (0/48 conditions survived).

    THIS module asks a different question: "at the moment the existing
    strategy already fires a signal (and will pay its round-trip
    spread regardless), does quote_pressure's direction at that
    instant correlate with which of those setups become winners vs
    losers?"

    This adds ZERO incremental transaction cost, because no new trade
    is created. quote_pressure is used purely as a CONFIRM/REJECT gate
    or a post-hoc explanatory variable on trades the strategy already
    takes -- exactly the CONFLUENCE FILTERS role in the architecture:

        existing strategy -> valid entry setup -> confluence filters
        (order flow / MTF / session / etc.) -> confirm/reject ->
        risk gates -> execution

Causality rule:
    For each existing-strategy trade entry at time T, only tick data
    at or before T is used to determine quote_pressure context. No
    tick after T is ever read for that trade. This is enforced via
    pandas.merge_asof(direction="backward").

This module does NOT modify the existing strategy, does NOT execute
trades, and does NOT establish that quote_pressure should gate live
entries. It only measures whether a correlation exists worth acting on
later, after further validation (larger sample, OOS split, etc.).
"""

import argparse
from datetime import timedelta

import numpy as np
import pandas as pd


def load_ticks(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "dt" not in df.columns:
        raise ValueError("tick CSV missing required column: 'dt'")

    # Explicit UTC parsing -- pd.read_csv(parse_dates=[...]) can silently
    # leave a column as plain strings (dtype=object) if it can't
    # uniformly infer the format, which then breaks any downstream
    # comparison against a real Timestamp with a confusing TypeError.
    df["dt"] = pd.to_datetime(df["dt"], utc=True, errors="coerce")
    bad = df["dt"].isna().sum()
    if bad > 0:
        print(f"WARNING: {bad} rows in ticks CSV had unparseable "
              f"timestamps in 'dt' and will be dropped.")
    df = df.dropna(subset=["dt"])

    df = df.sort_values("dt").drop_duplicates(subset=["dt"]).reset_index(drop=True)

    required = ["dt", "quote_pressure", "spread", "mid"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"tick CSV missing required columns: {missing}")

    return df


def load_trades(csv_path: str, time_col: str, direction_col: str,
                 outcome_col: str | None) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if time_col not in df.columns:
        raise ValueError(
            f"--time-col '{time_col}' not found in trades CSV. "
            f"Available columns: {list(df.columns)}"
        )
    if direction_col not in df.columns:
        raise ValueError(
            f"--direction-col '{direction_col}' not found in trades CSV. "
            f"Available columns: {list(df.columns)}"
        )
    if outcome_col is not None and outcome_col not in df.columns:
        raise ValueError(
            f"--outcome-col '{outcome_col}' not found in trades CSV. "
            f"Available columns: {list(df.columns)}"
        )

    df["trade_dt"] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    bad = df["trade_dt"].isna().sum()
    if bad > 0:
        print(f"WARNING: {bad} rows had unparseable timestamps in "
              f"'{time_col}' and will be dropped.")
    df = df.dropna(subset=["trade_dt"]).sort_values("trade_dt").reset_index(drop=True)

    return df


def normalize_direction(series: pd.Series) -> pd.Series:
    """
    Normalize a direction column into +1 (long) / -1 (short).
    Accepts common representations: LONG/SHORT, BUY/SELL, 1/-1, 1/0.
    """
    s = series.astype(str).str.strip().str.upper()
    mapped = pd.Series(np.nan, index=series.index, dtype=float)
    mapped[s.isin(["LONG", "BUY", "1", "L"])] = 1.0
    mapped[s.isin(["SHORT", "SELL", "-1", "S"])] = -1.0

    # fall back to raw numeric parsing if strings didn't match
    still_na = mapped.isna()
    if still_na.any():
        numeric = pd.to_numeric(series[still_na], errors="coerce")
        mapped.loc[still_na] = np.sign(numeric)

    return mapped


def check_time_overlap(trades: pd.DataFrame, ticks: pd.DataFrame) -> None:
    t_start, t_end = trades["trade_dt"].min(), trades["trade_dt"].max()
    k_start, k_end = ticks["dt"].min(), ticks["dt"].max()

    print(f"trades span:    {t_start} -> {t_end}")
    print(f"tick data span: {k_start} -> {k_end}")

    overlap_start = max(t_start, k_start)
    overlap_end = min(t_end, k_end)

    if overlap_start > overlap_end:
        raise RuntimeError(
            "NO TIME OVERLAP between trades and tick data. These two "
            "datasets do not cover the same period, so no valid join "
            "is possible. Re-run the existing strategy over the tick "
            "window above, or capture tick data over the trades' "
            "window, before re-running this module."
        )

    n_trades_in_window = (
        (trades["trade_dt"] >= overlap_start)
        & (trades["trade_dt"] <= overlap_end)
    ).sum()

    print(f"overlapping window: {overlap_start} -> {overlap_end}")
    print(f"trades falling inside overlap window: "
          f"{n_trades_in_window} / {len(trades)}")

    if n_trades_in_window == 0:
        raise RuntimeError(
            "Time ranges technically overlap but zero trades fall "
            "inside the overlap window. Check timezone handling in "
            "both files -- a naive/UTC mismatch is a common cause."
        )
    if n_trades_in_window < 20:
        print(
            f"WARNING: only {n_trades_in_window} trades fall inside "
            f"the overlap window. Any comparison below will be based "
            f"on a very small sample and should not be trusted."
        )


def attach_context(trades: pd.DataFrame, ticks: pd.DataFrame,
                    max_staleness: timedelta) -> pd.DataFrame:
    """
    For each trade, attach the most recent tick's quote_pressure/spread
    at or before the trade's entry time (strictly causal, backward-only
    merge). If the nearest prior tick is older than max_staleness, the
    context is treated as unavailable (NaN) rather than silently reused.
    """
    trades_sorted = trades.sort_values("trade_dt").reset_index(drop=True)
    ticks_sorted = ticks.sort_values("dt").reset_index(drop=True)

    merged = pd.merge_asof(
        trades_sorted,
        ticks_sorted[["dt", "quote_pressure", "spread", "mid"]],
        left_on="trade_dt",
        right_on="dt",
        direction="backward",
    )

    staleness = merged["trade_dt"] - merged["dt"]
    too_stale = staleness > max_staleness
    n_stale = too_stale.sum()
    if n_stale > 0:
        print(f"NOTE: {n_stale} / {len(merged)} trades had no tick "
              f"within {max_staleness} before entry and are excluded "
              f"from context analysis (context = NaN).")
        merged.loc[too_stale, ["quote_pressure", "spread", "mid"]] = np.nan

    return merged


def analyze(merged: pd.DataFrame, direction_col_norm: str,
            outcome_col: str | None) -> None:
    valid = merged.dropna(subset=["quote_pressure", direction_col_norm]).copy()
    n_valid = len(valid)
    print(f"\ntrades with usable context: {n_valid} / {len(merged)}")

    if n_valid < 20:
        print("Too few trades with valid context to draw any conclusion.")
        return

    # agreement: quote_pressure sign matches trade direction sign
    valid["agreement"] = np.select(
        [
            valid["quote_pressure"] * valid[direction_col_norm] > 0,
            valid["quote_pressure"] * valid[direction_col_norm] < 0,
        ],
        ["AGREE", "DISAGREE"],
        default="NEUTRAL",
    )

    print("\nAGREEMENT GROUP COUNTS")
    print(valid["agreement"].value_counts())

    if outcome_col is None:
        print(
            "\nNo --outcome-col provided, so win rate / R comparison "
            "cannot be computed. Provide the column holding each "
            "trade's result (e.g. R-multiple, or win/loss) to enable "
            "that comparison."
        )
        return

    print(f"\nOUTCOME ({outcome_col}) BY AGREEMENT GROUP")
    outcome_numeric = pd.to_numeric(valid[outcome_col], errors="coerce")
    valid["_outcome_numeric"] = outcome_numeric

    summary = valid.groupby("agreement")["_outcome_numeric"].agg(
        n="count", mean="mean", median="median", sum="sum",
        win_rate=lambda s: (s > 0).mean(),
    )
    print(summary.to_string())

    print(
        "\nREMINDER: this is a research observation on the existing "
        "strategy's own trade sample, not a validated filter. Sample "
        "size, OOS testing, and stability over more data are all "
        "still required before considering this as a live confluence "
        "gate."
    )


def main(trades_csv: str, ticks_csv: str, time_col: str, direction_col: str,
         outcome_col: str | None, max_staleness_minutes: float):
    trades = load_trades(trades_csv, time_col, direction_col, outcome_col)
    ticks = load_ticks(ticks_csv)

    print(f"trades loaded: {len(trades)}")
    print(f"ticks loaded: {len(ticks)}")

    check_time_overlap(trades, ticks)

    trades["_direction_norm"] = normalize_direction(trades[direction_col])
    unparsed = trades["_direction_norm"].isna().sum()
    if unparsed > 0:
        print(f"WARNING: {unparsed} rows had an unrecognized direction "
              f"value in '{direction_col}' and will be excluded.")

    merged = attach_context(
        trades, ticks, max_staleness=timedelta(minutes=max_staleness_minutes)
    )

    analyze(merged, "_direction_norm", outcome_col)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trades-csv", required=True,
                         help="existing strategy's trade log CSV")
    parser.add_argument("--ticks-csv", required=True,
                         help="quote_diagnostics.py output CSV")
    parser.add_argument("--time-col", default="entry_time",
                         help="column in trades CSV holding entry timestamp")
    parser.add_argument("--direction-col", default="direction",
                         help="column in trades CSV holding LONG/SHORT (or similar)")
    parser.add_argument("--outcome-col", default=None,
                         help="column in trades CSV holding trade result "
                              "(R-multiple or win/loss numeric)")
    parser.add_argument("--max-staleness-minutes", type=float, default=5.0,
                         help="max allowed gap between a trade's entry and "
                              "the nearest prior tick before context is "
                              "treated as unavailable")
    args = parser.parse_args()
    main(args.trades_csv, args.ticks_csv, args.time_col, args.direction_col,
         args.outcome_col, args.max_staleness_minutes)