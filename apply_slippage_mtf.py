import pandas as pd
import numpy as np

df = pd.read_csv("real_gold_data_mt5_90000.csv", parse_dates=["timestamp"]).set_index("timestamp")

high, low, close = df["high"], df["low"], df["close"]
prev_close = close.shift(1)
tr_range = pd.concat([
    (high - low),
    (high - prev_close).abs(),
    (low - prev_close).abs(),
], axis=1).max(axis=1)
atr14 = tr_range.rolling(14).mean()
atr_p75 = atr14.quantile(0.75)

tr = pd.read_csv("trades_90000_mtf_final.csv", parse_dates=["setup_time", "entry_time", "exit_time"])
tr_closed = tr[tr["status"].isin(["WIN", "LOSS"])].copy()

def atr_at_entry(entry_time):
    return atr14.asof(entry_time)

def classify_vol(row):
    a = atr_at_entry(row["entry_time"])
    if a is None or pd.isna(a):
        return "UNKNOWN"
    return "HIGH" if a > atr_p75 else "NORMAL"

def apply_slippage(row):
    if row["status"] != "LOSS":
        return row["r"]
    if row["vol_regime"] == "HIGH":
        return -1.15
    return -1.05

tr_closed["vol_regime"] = tr_closed.apply(classify_vol, axis=1)
tr_closed["r_slipped"] = tr_closed.apply(apply_slippage, axis=1)

print(f"=== MTF-GATED BEFORE slippage ===")
print(f"closed={len(tr_closed)}  win_rate={100*(tr_closed['status']=='WIN').mean():.2f}%")
print(f"total_R={tr_closed['r'].sum():.2f}  avg_R={tr_closed['r'].mean():.3f}")

print(f"\n=== MTF-GATED AFTER slippage ===")
print(f"total_R={tr_closed['r_slipped'].sum():.2f}  avg_R={tr_closed['r_slipped'].mean():.3f}")

r_before = tr_closed["r"].mean()
r_after = tr_closed["r_slipped"].mean()
print(f"\nExpectancy impact: {r_before:.3f}R -> {r_after:.3f}R  ({r_after-r_before:+.3f}R per trade)")

tr_closed.to_csv("trades_90000_mtf_final_with_slippage.csv", index=False)
print("saved trades_90000_mtf_final_with_slippage.csv")

print(f"\n{'='*50}")
print("FINAL COMPARISON (both with 0.05/0.15R slippage on losses):")
print(f"  Baseline (no MTF):  avg_R = 0.904  (from earlier run)")
print(f"  MTF-gated (1H+15M): avg_R = {r_after:.3f}")
print(f"  Difference: {r_after - 0.904:+.3f}R per trade")
