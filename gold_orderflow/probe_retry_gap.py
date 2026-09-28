import MetaTrader5 as mt5
from datetime import datetime, timedelta
import time

if not mt5.initialize():
    print("initialize() failed:", mt5.last_error())
    raise SystemExit(1)

symbol = "XAUUSD"
mt5.symbol_select(symbol, True)

gap_points = [datetime(2026,1,23), datetime(2025,12,24), datetime(2025,11,24), datetime(2025,10,25)]

for dt in gap_points:
    for attempt in range(3):
        ticks = mt5.copy_ticks_range(symbol, dt, dt + timedelta(hours=2), mt5.COPY_TICKS_ALL)
        n = 0 if ticks is None else len(ticks)
        print(f"{dt.date()} attempt {attempt+1} -> {n} ticks   {mt5.last_error()}")
        if n > 0:
            break
        time.sleep(2)

mt5.shutdown()
