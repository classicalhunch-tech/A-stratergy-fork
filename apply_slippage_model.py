import pandas as pd
import numpy as np

df = pd.read_csv("real_gold_data_mt5_90000.csv", parse_dates=["timestamp"]).set_index("timestamp")

# ATR(14) for volatility-scaled slippage
high, low, close = df["high"], df["low"], df["close"]
prev_close = close.shift(1)
tr_range = pd.concat([
    (high - low),
    (high - prev_close).abs(),
    (low - prev_close).abs(),
], axis=1).max(axis=1)
atr14 = tr_range.rolling(14).mean()
atr_p75 = atr14.quantile(0.75)
print(f"ATR(14) 75th percentile threshold: {atr_p75:.4f}")

tr = pd.read_csv("trades_90000_baseline_verify.csv", parse_dates=["setup_time", "entry_time", "exit_time"])
tr_closed = tr[tr["status"].isin(["WIN", "LOSS"])].copy()

# Causal: use ATR at entry, not exit. The vol regime when the trade
# was taken determines how the fill would have slipped; ATR at exit
# reflects conditions after the stop was already being hit.
def atr_at_entry(entry_time):
    return atr14.asof(entry_time)

def classify_vol(row):
    a = atr_at_entry(row["entry_time"])
    if a is None or pd.isna(a):
        return "UNKNOWN"
    return "HIGH" if a > atr_p75 else "NORMAL"

def apply_slippage(row):
    if row["status"] != "LOSS":
        return row["r"]  # wins unaffected
    if row["vol_regime"] == "HIGH":
        return -1.15
    if row["vol_regime"] == "NORMAL":
        return -1.05
    return -1.05  # UNKNOWN -> conservative base

tr_closed["vol_regime"] = tr_closed.apply(classify_vol, axis=1)
tr_closed["r_slipped"] = tr_closed.apply(apply_slippage, axis=1)

wins = tr_closed[tr_closed["status"] == "WIN"]
losses = tr_closed[tr_closed["status"] == "LOSS"]

print(f"\n=== BEFORE slippage (original) ===")
print(f"closed={len(tr_closed)}  win_rate={100*len(wins)/len(tr_closed):.2f}%")
print(f"total_R={tr_closed['r'].sum():.2f}  avg_R={tr_closed['r'].mean():.3f}")

print(f"\n=== AFTER slippage (0.05R base / 0.15R when ATR>p75 at ENTRY) ===")
print(f"total_R={tr_closed['r_slipped'].sum():.2f}  avg_R={tr_closed['r_slipped'].mean():.3f}")

high_vol_losses = ((losses["vol_regime"] == "HIGH")).sum()
normal_vol_losses = ((losses["vol_regime"] == "NORMAL")).sum()
unknown_vol_losses = ((losses["vol_regime"] == "UNKNOWN")).sum()

print(f"\nlosses by entry-time vol regime:")
print(f"  HIGH (0.15R slippage):   {high_vol_losses}")
print(f"  NORMAL (0.05R slippage): {normal_vol_losses}")
print(f"  UNKNOWN (0.05R fallback): {unknown_vol_losses}")

r_before = tr_closed["r"].mean()
r_after = tr_closed["r_slipped"].mean()
print(f"\nExpectancy impact: {r_before:.3f}R -> {r_after:.3f}R  ({r_after-r_before:+.3f}R per trade)")

tr_closed.to_csv("trades_90000_with_slippage.csv", index=False)
print("\nsaved trades_90000_with_slippage.csv")
