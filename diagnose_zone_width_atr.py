import pandas as pd
import numpy as np
from strategy.swings import find_swings
from strategy.structure import analyze_structure
from strategy.zones import find_zones

df = pd.read_csv("real_gold_data_mt5_90000.csv", parse_dates=["timestamp"]).set_index("timestamp")

print("Computing swings + structure + zones (one-shot batch)...")
swings = find_swings(df)
breaks, _, _ = analyze_structure(df, swings)
zones = find_zones(df, breaks)
print(f"  total zones: {len(zones)}")

high, low, close = df["high"], df["low"], df["close"]
prev_close = close.shift(1)
tr = pd.concat([
    (high - low),
    (high - prev_close).abs(),
    (low - prev_close).abs(),
], axis=1).max(axis=1)
atr14 = tr.rolling(14).mean()

zone_rows = []
for z in zones:
    width = z.price_top - z.price_bottom
    atr_at_formation = atr14.asof(z.formed_from_break_at)
    ratio = width / atr_at_formation if atr_at_formation and atr_at_formation > 0 else None
    zone_rows.append({
        "formed_from_break_at": z.formed_from_break_at,
        "price_top": z.price_top,
        "price_bottom": z.price_bottom,
        "width": width,
        "atr_at_formation": atr_at_formation,
        "width_atr_ratio": ratio,
    })

zones_df = pd.DataFrame(zone_rows).dropna(subset=["width_atr_ratio"])
print(f"\nwidth/ATR ratio distribution:")
print(zones_df["width_atr_ratio"].describe().round(3).to_string())

def zone_bucket(ratio):
    if ratio < 0.5:
        return "NARROW"
    if ratio > 2.0:
        return "WIDE"
    return "NORMAL"

zones_df["bucket"] = zones_df["width_atr_ratio"].apply(zone_bucket)

tr_trades = pd.read_csv("trades_90000_baseline_verify.csv", parse_dates=["setup_time"])
tr_trades = tr_trades[tr_trades["status"].isin(["WIN", "LOSS"])].copy()

zones_sorted = zones_df.sort_values("formed_from_break_at").reset_index(drop=True)
break_times = zones_sorted["formed_from_break_at"].values
buckets_arr = zones_sorted["bucket"].values
ratios_arr = zones_sorted["width_atr_ratio"].values

def nearest_zone_bucket(setup_time, max_hours=6):
    ts64 = np.datetime64(setup_time)
    mask = break_times <= ts64
    if not mask.any():
        return "UNMATCHED", None
    pos = np.where(mask)[0][-1]
    gap_hours = (ts64 - break_times[pos]) / np.timedelta64(1, "h")
    if gap_hours > max_hours:
        return "UNMATCHED", None
    return buckets_arr[pos], ratios_arr[pos]

matched = tr_trades["setup_time"].apply(nearest_zone_bucket)
tr_trades["zone_bucket"] = [m[0] for m in matched]
tr_trades["width_atr_ratio"] = [m[1] for m in matched]

print(f"\n=== Trade quality by zone width/ATR bucket ===")
summary = tr_trades.groupby("zone_bucket").agg(
    trades=("r", "count"),
    win_rate=("status", lambda s: round(100*(s == "WIN").mean(), 2)),
    total_R=("r", lambda s: round(s.sum(), 2)),
    avg_R=("r", lambda s: round(s.mean(), 3)),
)
order = ["NARROW", "NORMAL", "WIDE", "UNMATCHED"]
summary = summary.reindex([o for o in order if o in summary.index])
print(summary.to_string())
print("\nsample-size warning: any bucket with <30 trades is noise")
print("\nNOTE: zone-to-trade matching is approximate (nearest prior zone")
print("break within 6h), not an exact zone_id join.")
