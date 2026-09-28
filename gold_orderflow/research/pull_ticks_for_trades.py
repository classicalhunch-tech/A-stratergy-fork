"""
gold_orderflow/research/pull_ticks_for_trades.py

Pulls a short window of XAUUSD quote ticks before each trade entry in
trades_90000.csv, computes the same causal quote features as
quote_diagnostics.py (reused, not reimplemented), and writes one combined
feature CSV in the same schema quote_diagnostics.py already produces.

This does NOT compute quote_pressure at entry itself -- that causal
"most recent tick at or before entry" match is context_filter_backtest.py's
job via merge_asof(direction="backward"). This script's only job is to
make sure tick coverage exists around each trade entry.
"""

import argparse
from datetime import timedelta, timezone

import MetaTrader5 as mt5
import pandas as pd

from gold_orderflow.quote_diagnostics import compute_quote_features


def fetch_ticks_range(symbol: str, utc_from, utc_to) -> pd.DataFrame | None:
    """Historical variant of fetch_ticks -- explicit start/end, no
    mt5.initialize()/shutdown() (caller manages the session)."""
    raw = mt5.copy_ticks_range(symbol, utc_from, utc_to, mt5.COPY_TICKS_ALL)

    if raw is None or len(raw) == 0:
        return None

    df = pd.DataFrame(raw)
    df["dt"] = pd.to_datetime(df["time_msc"], unit="ms", utc=True)
    df = df.sort_values("dt").reset_index(drop=True)
    df = df[(df["bid"] > 0) & (df["ask"] > 0)].reset_index(drop=True)

    if df.empty:
        return None

    return df


def main(trades_csv: str, out_csv: str, lookback_minutes: float) -> None:
    if not mt5.initialize():
        raise RuntimeError(f"mt5.initialize() failed: {mt5.last_error()}")

    symbol = "XAUUSD"
    if not mt5.symbol_select(symbol, True):
        mt5.shutdown()
        raise RuntimeError(f"symbol_select failed: {mt5.last_error()}")

    trades = pd.read_csv(trades_csv, parse_dates=["entry_time"])
    lookback = timedelta(minutes=lookback_minutes)

    chunks = []
    missing = []

    try:
        for i, row in trades.iterrows():
            entry = row["entry_time"]
            if entry.tzinfo is None:
                entry = entry.tz_localize(timezone.utc)

            utc_from = entry - lookback
            utc_to = entry

            raw = fetch_ticks_range(symbol, utc_from, utc_to)
            if raw is None:
                missing.append(entry)
                continue

            features = compute_quote_features(raw)
            if features.empty:
                missing.append(entry)
                continue

            chunks.append(features)

            if (i + 1) % 50 == 0:
                print(f"processed {i + 1}/{len(trades)} trades...")
    finally:
        mt5.shutdown()

    print(f"\ntrades with tick coverage: {len(chunks)} / {len(trades)}")
    print(f"trades missing coverage:  {len(missing)}")
    if missing:
        print("missing entry_times (first 10):", missing[:10])

    if not chunks:
        print("No tick data collected -- nothing written.")
        return

    combined = pd.concat(chunks, ignore_index=True)
    before = len(combined)
    combined = combined.drop_duplicates(subset=["dt"]).sort_values("dt").reset_index(drop=True)
    print(f"combined rows: {before} -> {len(combined)} after de-dup")

    combined.to_csv(out_csv, index=False)
    print(f"wrote {len(combined)} rows -> {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trades-csv", default="trades_90000.csv")
    parser.add_argument("--out-csv", default="gold_orderflow/data/xauusd_quote_features_all_trades.csv")
    parser.add_argument("--lookback-minutes", type=float, default=60.0)
    args = parser.parse_args()

    main(args.trades_csv, args.out_csv, args.lookback_minutes)
