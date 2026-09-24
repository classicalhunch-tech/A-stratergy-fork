import time
import pandas as pd
from collections import defaultdict

import phase_03_paper.signals.adapter as adapter_module
from phase_03_paper.signals.adapter import StrategyAdapter
from phase_03_paper.market.engine import Candle

timings = defaultdict(float)
calls = defaultdict(int)

def wrap(name, fn):
    def wrapped(*args, **kwargs):
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        t1 = time.perf_counter()
        timings[name] += (t1 - t0)
        calls[name] += 1
        return result
    return wrapped

for fn_name in [
    "analyze_structure",
    "find_zones",
    "detect_liquidity_levels",
    "update_liquidity_sweeps",
    "_prepare_full_context",
]:
    original = getattr(adapter_module, fn_name)
    setattr(adapter_module, fn_name, wrap(fn_name, original))

for method_name in [
    "_build_dataframe",
    "_build_strategy_context",
    "_advance_pending_signals",
    "_discover_new_signals",
    "_advance_swing_state",
    "_extend_prepared_context",
]:
    original = getattr(StrategyAdapter, method_name)
    setattr(StrategyAdapter, method_name, wrap(method_name, original))

df = pd.read_csv("your_data_file.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
df.columns = [c.strip().lower() for c in df.columns]

adapter = StrategyAdapter()

checkpoints = {100, 300, 500, 750, 1000, 1500, 2000}

for i, row in df.iterrows():
    candle = Candle(
        timestamp=row["timestamp"],
        open=row["open"],
        high=row["high"],
        low=row["low"],
        close=row["close"],
    )

    adapter.on_candle(candle)

    if i in checkpoints:
        print("=" * 70)
        print(f"CUMULATIVE TIMING THROUGH CANDLE {i}")
        print("=" * 70)
        for name in sorted(timings, key=lambda n: -timings[n]):
            print(f"  {name:28s} total={timings[name]:.4f}s  calls={calls[name]}")
        print()

    if i >= 2000:
        break

print("DONE")