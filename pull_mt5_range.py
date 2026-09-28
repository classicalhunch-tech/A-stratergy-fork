import argparse
from datetime import datetime, timezone

import MetaTrader5 as mt5
import pandas as pd


def main(symbol: str, timeframe_str: str, start_str: str, end_str: str, out_csv: str):
    timeframe_map = {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "H1": mt5.TIMEFRAME_H1,
    }
    if timeframe_str not in timeframe_map:
        raise ValueError(f"Unsupported timeframe '{timeframe_str}'. "
                          f"Choose from: {list(timeframe_map)}")
    timeframe = timeframe_map[timeframe_str]

    date_from = datetime.strptime(start_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    date_to = datetime.strptime(end_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    if not mt5.initialize(path=r"C:\Program Files\MetaTrader 5\terminal64.exe"):
        print("FAILED to initialize:", mt5.last_error())
        return
    mt5.symbol_select(symbol, True)

    rates = mt5.copy_rates_range(symbol, timeframe, date_from, date_to)

    if rates is None or len(rates) == 0:
        print("FAILED or empty:", mt5.last_error())
        mt5.shutdown()
        return

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)

    df = df.rename(columns={"time": "timestamp", "tick_volume": "volume"})
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]

    print("Row count:", len(df))
    print("First timestamp:", df["timestamp"].iloc[0])
    print("Last timestamp:", df["timestamp"].iloc[-1])
    print("Duplicate timestamps:", df["timestamp"].duplicated().sum())
    print("Chronological:", df["timestamp"].is_monotonic_increasing)

    df.to_csv(out_csv, index=False)
    print(f"Saved to {out_csv}")

    mt5.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--timeframe", default="M5")
    parser.add_argument("--start", required=True, help="YYYY-MM-DD (UTC)")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD (UTC)")
    parser.add_argument("--out-csv", default="real_gold_data_mt5_range.csv")
    args = parser.parse_args()
    main(args.symbol, args.timeframe, args.start, args.end, args.out_csv)