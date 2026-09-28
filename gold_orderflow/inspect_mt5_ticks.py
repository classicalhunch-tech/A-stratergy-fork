"""
gold_orderflow/inspect_mt5_ticks.py

DIAGNOSTIC ONLY -- run this before building anything. It answers the
honest question: "what does my broker's MT5 feed actually give me at the
tick level, and is any of it genuine trade-aggressor information, or is
it only bid/ask quote updates?"

This must be run on Windows with a running MT5 terminal and the
`MetaTrader5` python package installed (`pip install MetaTrader5`) --
it cannot be tested in a sandboxed environment without that terminal, so
treat this as a first draft to run and report back on, not a verified tool.

WHAT IT CHECKS
--------------
1. Does copy_ticks_range return data at all for XAUUSD, and how dense
   is it (ticks per second)?
2. Are 'last' and 'volume' populated (would indicate real trade prints)
   or are they always zero (would mean this is a quote-only feed with no
   trade prints at all)?
3. What flag values actually appear, and how often? Specifically:
   - TICK_FLAG_BUY / TICK_FLAG_SELL: do these appear at all? If never,
     your feed doesn't expose any aggressor-like information.
   - If they DO appear, do they look like real trade classification or
     like simple tick-direction (does TICK_FLAG_BUY only ever coincide
     with price ticking up, and TICK_FLAG_SELL with price ticking down)?
     If it's 1:1 with tick direction, it's very likely just a synthetic
     up/down flag, not real executed-trade-side information -- this
     matters because tick-direction "aggressor" proxies have well-known
     biases and are NOT the same as Binance's is_buyer_maker.

Usage (on Windows, in your venv, with MT5 terminal running and logged in):
    python -m gold_orderflow.inspect_mt5_ticks --symbol XAUUSD --hours 2
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

import pandas as pd


# From MT5 docs -- TICK_FLAG bit values
TICK_FLAG_BID = 2
TICK_FLAG_ASK = 4
TICK_FLAG_LAST = 8
TICK_FLAG_VOLUME = 16
TICK_FLAG_BUY = 32
TICK_FLAG_SELL = 64


def _has_flag(flags: int, flag: int) -> bool:
    return bool(flags & flag)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--hours", type=float, default=2.0, help="How many recent hours of ticks to inspect")
    args = parser.parse_args()

    if mt5 is None:
        print("ERROR: MetaTrader5 package not installed. Run: pip install MetaTrader5")
        print("This must be run on Windows with a running, logged-in MT5 terminal.")
        return

    if not mt5.initialize():
        print(f"ERROR: mt5.initialize() failed: {mt5.last_error()}")
        return

    try:
        end = datetime.now(timezone.utc)
        start = end - timedelta(hours=args.hours)

        print(f"Requesting {args.symbol} ticks from {start} to {end} (COPY_TICKS_ALL) ...")
        ticks = mt5.copy_ticks_range(args.symbol, start, end, mt5.COPY_TICKS_ALL)

        if ticks is None or len(ticks) == 0:
            print(f"No ticks returned. mt5.last_error(): {mt5.last_error()}")
            print("Check: is the symbol name exactly right for your broker (e.g. XAUUSDc, XAUUSD.m, etc)?")
            print("Check: mt5.symbols_get() to see exact available symbol names.")
            return

        df = pd.DataFrame(ticks)
        n = len(df)
        duration_s = (df["time_msc"].iloc[-1] - df["time_msc"].iloc[0]) / 1000.0

        print(f"\n{n:,} ticks over {duration_s:.0f} seconds ({n/duration_s:.2f} ticks/sec average)")
        print(f"Columns: {list(df.columns)}")

        # --- Check 1: is 'last'/'volume' ever populated? ---
        print("\n--- last / volume ---")
        print(f"  'last' non-zero: {(df['last'] != 0).sum():,} / {n:,} ({(df['last'] != 0).mean()*100:.1f}%)")
        print(f"  'volume' non-zero: {(df['volume'] != 0).sum():,} / {n:,} ({(df['volume'] != 0).mean()*100:.1f}%)")
        if (df["last"] == 0).all():
            print("  -> 'last' is ALWAYS zero: this feed appears to be quote-only (bid/ask), no trade prints.")
        if (df["volume"] == 0).all():
            print("  -> 'volume' is ALWAYS zero: no real trade-size information available.")

        # --- Check 2: what flags actually appear? ---
        print("\n--- flag frequency ---")
        for name, flag in [
            ("TICK_FLAG_BID", TICK_FLAG_BID),
            ("TICK_FLAG_ASK", TICK_FLAG_ASK),
            ("TICK_FLAG_LAST", TICK_FLAG_LAST),
            ("TICK_FLAG_VOLUME", TICK_FLAG_VOLUME),
            ("TICK_FLAG_BUY", TICK_FLAG_BUY),
            ("TICK_FLAG_SELL", TICK_FLAG_SELL),
        ]:
            count = df["flags"].apply(lambda f: _has_flag(int(f), flag)).sum()
            print(f"  {name}: {count:,} / {n:,} ({count/n*100:.1f}%)")

        has_buy_sell_flags = (
            df["flags"].apply(lambda f: _has_flag(int(f), TICK_FLAG_BUY) or _has_flag(int(f), TICK_FLAG_SELL)).any()
        )

        if not has_buy_sell_flags:
            print("\n  -> TICK_FLAG_BUY/SELL never appear in this feed.")
            print("     Your broker's feed does not expose any aggressor-like classification.")
            print("     Real bid/ask spread and mid-price features are still usable, but NOT a")
            print("     genuine buy/sell trade classification like Binance's is_buyer_maker.")
        else:
            # --- Check 3: does BUY/SELL just mirror tick direction? ---
            print("\n--- Checking if TICK_FLAG_BUY/SELL just mirrors price tick direction ---")
            df["mid"] = (df["bid"] + df["ask"]) / 2.0
            df["price_change"] = df["mid"].diff()
            df["is_buy_flag"] = df["flags"].apply(lambda f: _has_flag(int(f), TICK_FLAG_BUY))
            df["is_sell_flag"] = df["flags"].apply(lambda f: _has_flag(int(f), TICK_FLAG_SELL))

            buy_rows = df[df["is_buy_flag"]]
            sell_rows = df[df["is_sell_flag"]]

            if len(buy_rows) > 0:
                buy_up_pct = (buy_rows["price_change"] > 0).mean() * 100
                print(f"  Of {len(buy_rows):,} TICK_FLAG_BUY ticks, {buy_up_pct:.1f}% coincide with price ticking up")
            if len(sell_rows) > 0:
                sell_down_pct = (sell_rows["price_change"] < 0).mean() * 100
                print(f"  Of {len(sell_rows):,} TICK_FLAG_SELL ticks, {sell_down_pct:.1f}% coincide with price ticking down")

            print("\n  If both percentages above are close to 100%, TICK_FLAG_BUY/SELL is very")
            print("  likely just a tick-direction indicator (price up = 'buy', down = 'sell'),")
            print("  not genuine trade-aggressor classification -- a materially different (and")
            print("  weaker) signal than what the Binance lab used. If the percentages are far")
            print("  from 100%, it may carry real independent information -- worth investigating")
            print("  further with your broker's documentation.")

        out_path = f"{args.symbol}_tick_sample.csv"
        df.to_csv(out_path, index=False)
        print(f"\nWrote raw sample -> {out_path} for further inspection.")

    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()