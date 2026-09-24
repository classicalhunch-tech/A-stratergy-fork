import time
import pandas as pd
from strategy.swings import find_swings
from strategy.structure import analyze_structure
from strategy.zones import find_zones
from strategy.liquidity import detect_liquidity_levels, update_liquidity_sweeps

df_full = pd.read_csv("your_data_file.csv")
df_full["timestamp"] = pd.to_datetime(df_full["timestamp"])
df_full = df_full.set_index("timestamp")
df_full.columns = [c.lower() for c in df_full.columns]

for n in [50, 100, 150, 200, 250, 300]:
    df = df_full.iloc[:n]

    t0 = time.perf_counter()
    swings = find_swings(df)
    t1 = time.perf_counter()

    breaks, _, _ = analyze_structure(df, swings)
    t2 = time.perf_counter()

    zones = find_zones(df, breaks)
    t3 = time.perf_counter()

    levels = detect_liquidity_levels(df, swings)
    t4 = time.perf_counter()

    levels = update_liquidity_sweeps(df, levels)
    t5 = time.perf_counter()

    print(f"n={n:4d}  swings={t1-t0:.4f}s  structure={t2-t1:.4f}s  "
          f"zones={t3-t2:.4f}s  liquidity_detect={t4-t3:.4f}s  "
          f"liquidity_sweep={t5-t4:.4f}s")