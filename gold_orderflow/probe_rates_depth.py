import MetaTrader5 as mt5
from datetime import datetime, timedelta

if not mt5.initialize():
    print("initialize() failed:", mt5.last_error())
    raise SystemExit(1)

symbol = "XAUUSD"
mt5.symbol_select(symbol, True)

# Walk back in 6-month steps, checking if M5 rates exist for a 1-week window
probe_points = [datetime(2026,6,1) - timedelta(days=182*i) for i in range(0, 10)]  # back to ~2021

for dt in probe_points:
    rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M5, dt, dt + timedelta(days=7))
    n = 0 if rates is None else len(rates)
    print(f"{dt.date()} -> {n} M5 candles   {mt5.last_error()}")

mt5.shutdown()
