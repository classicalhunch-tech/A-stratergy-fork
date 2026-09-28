import MetaTrader5 as mt5
from datetime import datetime, timedelta
import sys

if not mt5.initialize():
    print("initialize() failed:", mt5.last_error())
    sys.exit(1)

symbol = "XAUUSD"
mt5.symbol_select(symbol, True)

# walk back from now in ~30-day steps, checking a 2-hour window each time
now = datetime(2026, 9, 20)
for i in range(0, 18):  # back to roughly 2025-04
    dt = now - timedelta(days=30 * i)
    ticks = mt5.copy_ticks_range(symbol, dt, dt + timedelta(hours=2), mt5.COPY_TICKS_ALL)
    n = 0 if ticks is None else len(ticks)
    print(f"{dt.date()} -> {n} ticks   {mt5.last_error()}")

mt5.shutdown()