import MetaTrader5 as mt5
import pandas as pd

mt5.initialize(path=r"C:\Program Files\MetaTrader 5\terminal64.exe")
mt5.symbol_select("XAUUSD", True)

rates = mt5.copy_rates_from_pos(
    "XAUUSD",
    mt5.TIMEFRAME_M5,
    1,
    90000,
)

if rates is None:
    print("FAILED:", mt5.last_error())
else:
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")

    # Match project convention:
    # timestamp, open, high, low, close, volume
    df = df.rename(columns={
        "time": "timestamp",
        "tick_volume": "volume",
    })

    df = df[[
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]]

    print("Row count:", len(df))
    print("First timestamp:", df["timestamp"].iloc[0])
    print("Last timestamp:", df["timestamp"].iloc[-1])
    print("Duplicate timestamps:", df["timestamp"].duplicated().sum())
    print(
        "Chronological:",
        df["timestamp"].is_monotonic_increasing
    )

    df.to_csv(
        "real_gold_data_mt5_90000.csv",
        index=False,
    )

    print("Saved to real_gold_data_mt5_90000.csv")

mt5.shutdown()