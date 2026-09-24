import time
import pandas as pd
from phase_03_paper.signals.adapter import StrategyAdapter
from phase_03_paper.market.engine import Candle

df = pd.read_csv("your_data_file.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
df.columns = [c.strip().lower() for c in df.columns]

adapter = StrategyAdapter()

start = time.perf_counter()

for i, row in df.iterrows():
    candle = Candle(
        timestamp=row["timestamp"],
        open=row["open"],
        high=row["high"],
        low=row["low"],
        close=row["close"],
    )

    t0 = time.perf_counter()
    adapter.on_candle(candle)
    t1 = time.perf_counter()

    if i % 25 == 0:
        elapsed_total = t1 - start
        this_candle = t1 - t0
        print(f"i={i:4d}  this_candle={this_candle:.4f}s  total_elapsed={elapsed_total:.2f}s", flush=True)

    if i >= 500:
        break

print("DONE")