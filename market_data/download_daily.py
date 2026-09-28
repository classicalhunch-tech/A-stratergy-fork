"""
market_data/download_daily.py

Downloads a longer history (e.g. a month) one day at a time, saving each
day to its own normalized CSV rather than one giant combined file. This
avoids holding tens of millions of trades in memory at once, and lets
regime_breakdown.py analyze each day independently.

Also writes a manifest CSV with each day's net price move, so days can be
labeled UP / DOWN / FLAT for regime comparison.

Usage:
    python -m market_data.download_daily --symbol BTCUSDT \
        --start 2026-08-27 --end 2026-09-25 \
        --out-dir market_data/data/daily
"""

from __future__ import annotations

import argparse
import os
from datetime import date, timedelta

import pandas as pd
import requests

from .binance_trades import _daterange, _download_one_day, normalize_trades_df

FLAT_THRESHOLD_PCT = 1.0  # +-1% net move => FLAT, else UP/DOWN


def classify_regime(pct_move: float) -> str:
    if pct_move > FLAT_THRESHOLD_PCT:
        return "UP"
    if pct_move < -FLAT_THRESHOLD_PCT:
        return "DOWN"
    return "FLAT"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--start", required=True, help="YYYY-MM-DD (UTC, inclusive)")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD (UTC, inclusive)")
    parser.add_argument("--out-dir", default="market_data/data/daily")
    args = parser.parse_args()

    symbol = args.symbol.upper().strip()
    start_day = date.fromisoformat(args.start)
    end_day = date.fromisoformat(args.end)
    os.makedirs(args.out_dir, exist_ok=True)

    session = requests.Session()
    manifest_rows = []

    for day in _daterange(start_day, end_day):
        print(f"\n=== {day} ===")
        try:
            raw = _download_one_day(symbol, day, session)
        except Exception as e:
            print(f"  [error] {day}: {e} -- skipping this day")
            continue

        if raw.empty:
            print(f"  [skip] {day}: no data")
            continue

        normalized = normalize_trades_df(raw)

        out_path = os.path.join(args.out_dir, f"{symbol}_{day.isoformat()}.csv")
        normalized.to_csv(out_path, index=False)

        start_price = float(normalized["price"].iloc[0])
        end_price = float(normalized["price"].iloc[-1])
        pct_move = (end_price / start_price - 1.0) * 100.0
        regime = classify_regime(pct_move)

        print(f"  wrote {len(normalized):,} trades -> {out_path}")
        print(f"  net move: {pct_move:+.2f}%  ({start_price} -> {end_price})  regime={regime}")

        manifest_rows.append(
            {
                "day": day.isoformat(),
                "symbol": symbol,
                "n_trades": len(normalized),
                "start_price": start_price,
                "end_price": end_price,
                "pct_move": pct_move,
                "regime": regime,
                "file": out_path,
            }
        )

    session.close()

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = os.path.join(args.out_dir, f"{symbol}_regime_manifest.csv")
    manifest.to_csv(manifest_path, index=False)

    print("\n" + "=" * 60)
    print(f"Wrote manifest -> {manifest_path}")
    if not manifest.empty:
        print(manifest[["day", "pct_move", "regime", "n_trades"]].to_string(index=False))
        print(f"\nRegime counts: {manifest['regime'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()