"""
gold_orderflow/check_tick_history.py

Diagnostic-only. Tests MT5 copy_ticks_range() across several time windows
to determine whether historical XAUUSD tick data is actually retrievable,
and if so, what fields it actually contains.
"""

import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone

import MetaTrader5 as mt5

FLAG_BITS = {2: "BID", 4: "ASK", 8: "LAST", 16: "VOLUME", 32: "BUY", 64: "SELL"}


def decode_flags(flags: int):
    return [name for bit, name in FLAG_BITS.items() if flags & bit]


def check_window(symbol: str, minutes_back: float, label: str):
    utc_to = datetime.now(timezone.utc)
    utc_from = utc_to - timedelta(minutes=minutes_back)

    ticks = mt5.copy_ticks_range(symbol, utc_from, utc_to, mt5.COPY_TICKS_ALL)
    n = 0 if ticks is None else len(ticks)

    print(f"\n=== {label} ({utc_from.isoformat()} -> {utc_to.isoformat()}) ===")
    print(f"requested minutes back: {minutes_back}")
    print(f"n_ticks: {n}")

    if n == 0:
        print("last_error:", mt5.last_error())
        return

    first, last = ticks[0], ticks[-1]
    print(f"first tick time: {datetime.fromtimestamp(first['time'], tz=timezone.utc)}")
    print(f"last  tick time: {datetime.fromtimestamp(last['time'], tz=timezone.utc)}")

    nonzero_last = int((ticks["last"] != 0).sum())
    nonzero_volume = int((ticks["volume"] != 0).sum())

    flag_counter = Counter(int(f) for f in ticks["flags"])

    print(f"nonzero 'last' count:   {nonzero_last} / {n}")
    print(f"nonzero 'volume' count: {nonzero_volume} / {n}")
    print("flag distribution:")
    for flag_val, count in flag_counter.most_common():
        flag_str = ", ".join(decode_flags(flag_val)) or "(none)"
        print(f"    {flag_val:>4}  {flag_str:<20}  x{count}")


def main(symbol: str):
    if not mt5.initialize():
        print("mt5.initialize() failed:", mt5.last_error())
        return

    mt5.symbol_select(symbol, True)

    windows = [
        (1, "last 1 minute"),
        (5, "last 5 minutes"),
        (60, "last 1 hour"),
        (60 * 6, "last 6 hours"),
        (60 * 24, "last 24 hours"),
        (60 * 24 * 5, "last 5 days"),
    ]

    for minutes_back, label in windows:
        check_window(symbol, minutes_back, label)

    mt5.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="XAUUSD")
    args = parser.parse_args()
    main(args.symbol)