import time
import pandas as pd

from phase_03_paper.market.engine import Candle
from phase_03_paper.signals.adapter import StrategyAdapter as NewAdapter


def time_adapter(adapter_cls, df, n):
    df_n = df.head(n)
    adapter = adapter_cls()

    start = time.perf_counter()
    for _, row in df_n.iterrows():
        candle = Candle(
            timestamp=row["timestamp"],
            open=row["open"],
            high=row["high"],
            low=row["low"],
            close=row["close"],
        )
        adapter.on_candle(candle)
    elapsed = time.perf_counter() - start

    return elapsed


df = pd.read_csv("real_gold_data_mt5_10000.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
df.columns = [c.strip().lower() for c in df.columns]

print("Timing NEW (production) adapter scaling...")
for n in [500, 1000, 2000, 3000, 4000, 5000]:
    print(f"Timing NEW adapter at {n} candles...", flush=True)
    elapsed = time_adapter(NewAdapter, df, n)
    print(f"  {n} candles: {elapsed:.2f}s", flush=True)