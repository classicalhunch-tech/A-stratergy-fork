import MetaTrader5 as mt5
import pandas as pd

mt5.initialize(path=r"C:\Program Files\MetaTrader 5\terminal64.exe")
mt5.symbol_select("XAUUSD", True)

for n in [15000, 20000, 30000, 50000]:
    rates = mt5.copy_rates_from_pos("XAUUSD", mt5.TIMEFRAME_M5, 0, n)
    if rates is None:
        print(n, ": FAILED -", mt5.last_error())
    else:
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        earliest = df["time"].iloc[0]
        latest = df["time"].iloc[-1]
        print(n, ": got", len(df), "candles, earliest=", earliest, "latest=", latest)