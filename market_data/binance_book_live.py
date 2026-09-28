"""
Polls Binance's public top-of-book endpoint and appends normalized BOOK events
to a CSV, going forward in real time.

WHY LIVE POLLING INSTEAD OF A HISTORICAL DOWNLOAD:
Binance's free public archive (data.binance.vision) is well-documented and
confirmed for klines, trades, and aggTrades. A historical top-of-book/bookTicker
archive for SPOT markets was not confirmed as reliably available during
research for this module. Rather than guess at an unverified URL and silently
produce wrong or empty data, this module builds book history honestly: by
polling the live endpoint and logging it yourself, starting now.

If you need book history for a period before you started polling, that gap
should be treated as genuinely missing data -- not backfilled with guesses.

Endpoint used (no API key required):
    GET https://api.binance.com/api/v3/ticker/bookTicker?symbol=BTCUSDT
    -> {"symbol": "BTCUSDT", "bidPrice": "...", "bidQty": "...",
        "askPrice": "...", "askQty": "..."}

Usage:
    python -m orderflow.market_data.binance_book_live --symbol BTCUSDT \
        --interval 1.0 --out orderflow/market_data/data/BTCUSDT_book_live.csv

Run it in a separate terminal/process for as long as you want book history
to accumulate. It appends, so it's safe to stop and restart.
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from datetime import datetime, timezone

import requests

from .models import BookEvent

URL = "https://api.binance.com/api/v3/ticker/bookTicker"

FIELDS = ["timestamp", "bid", "ask", "bid_size", "ask_size", "event_type"]


def poll_once(symbol: str, session: requests.Session) -> BookEvent:
    resp = session.get(URL, params={"symbol": symbol}, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    ts = int(datetime.now(timezone.utc).timestamp() * 1000)
    return BookEvent(
        timestamp=ts,
        bid=float(data["bidPrice"]),
        ask=float(data["askPrice"]),
        bid_size=float(data["bidQty"]),
        ask_size=float(data["askQty"]),
    )


def run(symbol: str, interval_seconds: float, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    file_exists = os.path.isfile(out_path)

    session = requests.Session()
    print(f"Polling {symbol} bookTicker every {interval_seconds}s -> {out_path}")
    print("Press Ctrl+C to stop.")

    with open(out_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if not file_exists:
            writer.writeheader()

        try:
            while True:
                event = poll_once(symbol, session)
                writer.writerow(event.to_dict())
                f.flush()
                time.sleep(interval_seconds)
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Poll Binance top-of-book and log normalized BOOK events.")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--interval", type=float, default=1.0, help="Seconds between polls.")
    parser.add_argument("--out", default="orderflow/market_data/data/BTCUSDT_book_live.csv")
    args = parser.parse_args()
    run(args.symbol, args.interval, args.out)