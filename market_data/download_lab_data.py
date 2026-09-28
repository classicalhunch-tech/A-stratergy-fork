"""
CLI: download and normalize historical Binance spot TRADE events for the
microstructure lab.

Usage:
    python -m orderflow.market_data.download_lab_data \
        --symbol BTCUSDT --start 2026-09-20 --end 2026-09-21 \
        --out orderflow/market_data/data/BTCUSDT_trades_normalized.csv

Notes:
- Dates are UTC calendar days, inclusive on both ends. Binance publishes
  each day's file the following day, so very recent days may 404 -- the
  downloader logs and skips those rather than failing.
- This produces TRADE events only. For BOOK events, run
  binance_book_live.py separately, going forward, since a confirmed free
  historical spot book archive was not found -- see that module's
  docstring for why.
- Nothing here touches strategy/ or the XAUUSD orderflow/ proxy layer.
  This is a self-contained lab.
"""

from __future__ import annotations

import argparse
from datetime import date

from .binance_trades import download_agg_trades, normalize_trades, trades_to_dataframe


def main() -> None:
    parser = argparse.ArgumentParser(description="Download + normalize Binance historical trades.")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--start", required=True, help="YYYY-MM-DD (UTC, inclusive)")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD (UTC, inclusive)")
    parser.add_argument("--out", default="orderflow/market_data/data/{symbol}_trades_normalized.csv")
    args = parser.parse_args()

    start_day = date.fromisoformat(args.start)
    end_day = date.fromisoformat(args.end)
    out_path = args.out.format(symbol=args.symbol)

    print(f"Downloading {args.symbol} aggTrades from {start_day} to {end_day} ...")
    raw = download_agg_trades(args.symbol, start_day, end_day)

    if raw.empty:
        print("No data downloaded -- check symbol/date range (files publish next-day).")
        return

    print(f"Normalizing {len(raw)} raw trades into TradeEvent schema ...")
    events = normalize_trades(raw)
    df = trades_to_dataframe(events)

    import os
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    df.to_csv(out_path, index=False)

    buy = (df["side"] == "BUY").sum()
    sell = (df["side"] == "SELL").sum()
    print(f"\nWrote {len(df)} normalized TRADE events -> {out_path}")
    print(f"  BUY (aggressor bought):  {buy}  ({buy / len(df) * 100:.1f}%)")
    print(f"  SELL (aggressor sold):   {sell}  ({sell / len(df) * 100:.1f}%)")
    print(f"  Time range: {df['timestamp'].min()} -> {df['timestamp'].max()}  (epoch ms, UTC)")


if __name__ == "__main__":
    main()