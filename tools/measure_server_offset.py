"""
tools/measure_server_offset.py

Run ONCE on a Windows PC with MetaTrader 5 open, logged in, and the gold
market OPEN. It compares MT5's latest tick time with the real UTC clock
to show the broker's server-time offset.
"""

import time

import MetaTrader5 as mt5

SYMBOL = "XAUUSD"

if not mt5.initialize():
    raise SystemExit(f"initialize() failed: {mt5.last_error()}")

try:
    if not mt5.symbol_select(SYMBOL, True):
        raise SystemExit(f"symbol_select() failed: {mt5.last_error()}")

    tick = mt5.symbol_info_tick(SYMBOL)
    if tick is None:
        raise SystemExit(f"symbol_info_tick() returned None: {mt5.last_error()}")

    now_utc = time.time()
    diff_hours = (tick.time - now_utc) / 3600.0
    rounded = round(diff_hours * 2) / 2

    print(f"Raw difference:       {diff_hours:+.3f} hours")
    print(f"Server offset (est.): UTC{rounded:+.1f}")

    if abs(diff_hours - rounded) > 0.05:
        print("WARNING: the tick looks old (market closed?). Run again while the market is open.")
finally:
    mt5.shutdown()
