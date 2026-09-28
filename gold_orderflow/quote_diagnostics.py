"""
gold_orderflow/quote_diagnostics.py

Research-diagnostic only. Determines whether the available XAUUSD BID/ASK
quote stream contains observable market information such as spread behavior,
mid-price movement, quote update frequency/velocity, and defensible
quote-pressure proxies.

This module does NOT compute or imply genuine trade-side/order-flow
information. There is no usable 'last' price or 'volume' in this feed
(confirmed separately). Everything here is quote/microstructure only and
must be labeled as such downstream.

This module does NOT test relationships with future returns. That belongs
in a separate future-response research module so causality/lookahead
boundaries remain explicit.
"""

import argparse
from datetime import datetime, timedelta, timezone

import MetaTrader5 as mt5
import numpy as np
import pandas as pd


def fetch_ticks(symbol: str, minutes_back: float) -> pd.DataFrame:
    """Fetch historical XAUUSD quote ticks from MT5."""
    if not mt5.initialize():
        raise RuntimeError(
            f"mt5.initialize() failed: {mt5.last_error()}"
        )

    try:
        if not mt5.symbol_select(symbol, True):
            raise RuntimeError(
                f"mt5.symbol_select('{symbol}', True) failed: "
                f"{mt5.last_error()}"
            )

        utc_to = datetime.now(timezone.utc)
        utc_from = utc_to - timedelta(minutes=minutes_back)

        raw = mt5.copy_ticks_range(
            symbol,
            utc_from,
            utc_to,
            mt5.COPY_TICKS_ALL,
        )

        if raw is None or len(raw) == 0:
            raise RuntimeError(
                f"No ticks returned for {symbol} in window "
                f"{utc_from.isoformat()} -> {utc_to.isoformat()}"
            )

        df = pd.DataFrame(raw)

    finally:
        mt5.shutdown()

    df["dt"] = pd.to_datetime(
        df["time_msc"],
        unit="ms",
        utc=True,
    )

    df = (
        df.sort_values("dt")
        .reset_index(drop=True)
    )

    # Keep only rows carrying valid positive BID/ASK quotes.
    df = df[
        (df["bid"] > 0) &
        (df["ask"] > 0)
    ].reset_index(drop=True)

    if df.empty:
        raise RuntimeError(
            f"No valid BID/ASK quote ticks remained for {symbol} "
            "after filtering."
        )

    return df


def compute_quote_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute strictly causal quote/microstructure features.

    Each row uses only:
    - the current row's BID/ASK
    - the immediately preceding row

    No future information is used.
    """
    out = df.copy()

    # Basic quote state.
    out["mid"] = (out["bid"] + out["ask"]) / 2.0
    out["spread"] = out["ask"] - out["bid"]

    # Previous quote state.
    out["bid_prev"] = out["bid"].shift(1)
    out["ask_prev"] = out["ask"].shift(1)
    out["mid_prev"] = out["mid"].shift(1)
    out["dt_prev"] = out["dt"].shift(1)

    # Tick-to-tick changes.
    out["bid_change"] = out["bid"] - out["bid_prev"]
    out["ask_change"] = out["ask"] - out["ask_prev"]
    out["mid_change"] = out["mid"] - out["mid_prev"]
    out["spread_change"] = (
        out["spread"] - out["spread"].shift(1)
    )

    # Time between quote updates.
    out["inter_arrival_ms"] = (
        out["dt"] - out["dt_prev"]
    ).dt.total_seconds() * 1000.0

    # Defensible quote-pressure proxy.
    #
    # +1:
    #   one quote side moves upward while the other stays flat.
    #
    # -1:
    #   one quote side moves downward while the other stays flat.
    #
    #  0:
    #   both sides move, neither moves, or the movement is mixed.
    #
    # This is explicitly a QUOTE-side proxy.
    # It is NOT buyer/seller aggressor classification.
    bid_up = out["bid_change"] > 0
    bid_dn = out["bid_change"] < 0
    ask_up = out["ask_change"] > 0
    ask_dn = out["ask_change"] < 0

    bid_flat = out["bid_change"] == 0
    ask_flat = out["ask_change"] == 0

    pressure = np.zeros(len(out), dtype=float)

    pressure[
        (bid_up & ask_flat) |
        (ask_up & bid_flat)
    ] = 1.0

    pressure[
        (bid_dn & ask_flat) |
        (ask_dn & bid_flat)
    ] = -1.0

    out["quote_pressure"] = pressure

    # The first row has no previous quote and therefore cannot have
    # a causal tick-to-tick change.
    out = out.dropna(
        subset=["bid_prev"]
    ).reset_index(drop=True)

    return out


def summarize(df: pd.DataFrame) -> None:
    """Print descriptive quote/microstructure diagnostics only."""
    n = len(df)

    if n == 0:
        print("No rows available after feature construction.")
        return

    span = df["dt"].iloc[-1] - df["dt"].iloc[0]

    print(f"n_ticks (after causal shift): {n}")
    print(
        f"time span: {df['dt'].iloc[0]} -> "
        f"{df['dt'].iloc[-1]} ({span})"
    )
    print("-" * 70)

    print("SPREAD")
    print(
        df["spread"].describe(
            percentiles=[0.05, 0.25, 0.50, 0.75, 0.95]
        )
    )
    print()

    print("MID PRICE")
    print(df["mid"].describe())
    print()

    print("SPREAD CHANGE (tick-to-tick)")
    print(
        df["spread_change"].describe(
            percentiles=[0.05, 0.50, 0.95]
        )
    )
    print()

    print("MID CHANGE (tick-to-tick, in points)")
    print(
        df["mid_change"].describe(
            percentiles=[0.05, 0.50, 0.95]
        )
    )

    nonzero_mid_moves = int(
        (df["mid_change"] != 0).sum()
    )

    print(
        f"nonzero mid moves: {nonzero_mid_moves} / {n} "
        f"({100 * nonzero_mid_moves / n:.1f}%)"
    )
    print()

    print("QUOTE UPDATE FREQUENCY (inter-arrival, ms)")
    print(
        df["inter_arrival_ms"].describe(
            percentiles=[0.05, 0.50, 0.95]
        )
    )
    print()

    print("QUOTE PRESSURE PROXY")
    counts = (
        df["quote_pressure"]
        .value_counts()
        .sort_index()
    )

    print(counts)
    print(
        f"mean quote_pressure: "
        f"{df['quote_pressure'].mean():.5f}"
    )
    print()

    print("DATA AVAILABILITY")
    print(
        f"bid nonzero: "
        f"{int((df['bid'] > 0).sum())} / {n}"
    )
    print(
        f"ask nonzero: "
        f"{int((df['ask'] > 0).sum())} / {n}"
    )

    print()
    print(
        "No future-response correlation is calculated here. "
        "Future-response testing belongs to the next research stage."
    )


def main(
    symbol: str,
    minutes_back: float,
    out_csv: str | None,
) -> None:
    raw = fetch_ticks(symbol, minutes_back)
    features = compute_quote_features(raw)

    summarize(features)

    if out_csv:
        features.to_csv(
            out_csv,
            index=False,
        )
        print(
            f"\nwrote {len(features)} rows -> {out_csv}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--symbol",
        default="XAUUSD",
    )

    parser.add_argument(
        "--minutes-back",
        type=float,
        default=1440.0,
        help="How far back to pull ticks (default: 24 hours).",
    )

    parser.add_argument(
        "--out-csv",
        default=None,
        help="Optional path to write the feature table.",
    )

    args = parser.parse_args()

    main(
        args.symbol,
        args.minutes_back,
        args.out_csv,
    )