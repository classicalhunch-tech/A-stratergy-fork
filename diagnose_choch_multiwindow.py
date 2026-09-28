import pandas as pd
from strategy.swings import find_swings
from strategy.structure import analyze_structure

df = pd.read_csv("real_gold_data_mt5_90000.csv", parse_dates=["timestamp"]).set_index("timestamp")

print("Computing swings + structure on full 90k (one-shot, non-causal batch)...")
swings = find_swings(df)
breaks, classified_swings, final_trend = analyze_structure(df, swings)
print(f"  total breaks: {len(breaks)}")

break_rows = [{"broken_at": b.broken_at, "break_type": b.break_type.value} for b in breaks]
breaks_df = pd.DataFrame(break_rows).sort_values("broken_at").reset_index(drop=True)
break_times = breaks_df["broken_at"].values
break_types = breaks_df["break_type"].values

def choch_ratio_before(ts, window):
    import numpy as np
    ts64 = np.datetime64(ts)
    lo = ts64 - np.timedelta64(window)
    mask = (break_times < ts64) & (break_times >= lo)
    sub = break_types[mask]
    if len(sub) == 0:
        return None, 0
    choch_count = (sub == "choch").sum()
    return choch_count / len(sub), len(sub)

def regime_label(ratio):
    if ratio is None:
        return "UNKNOWN"
    if ratio >= 0.5:
        return "CHOPPY"
    if ratio <= 0.15:
        return "TRENDING"
    return "MIXED"

WINDOWS = {
    "12h": pd.Timedelta(hours=12),
    "24h": pd.Timedelta(hours=24),
    "48h": pd.Timedelta(hours=48),
}

def tag_trades(path, label):
    import os
    if not os.path.exists(path):
        print(f"\n[skip] {path} not found")
        return
    tr = pd.read_csv(path, parse_dates=["setup_time"])
    tr = tr[tr["status"].isin(["WIN", "LOSS"])].copy()

    print(f"\n{'='*70}")
    print(f"{label} ({path}) -- total closed trades: {len(tr)}")
    print(f"{'='*70}")

    for wname, wdelta in WINDOWS.items():
        ratios, counts = [], []
        for ts in tr["setup_time"]:
            r, c = choch_ratio_before(ts, wdelta)
            ratios.append(r)
            counts.append(c)
        tr[f"regime_{wname}"] = [regime_label(r) for r in ratios]

        print(f"\n--- window={wname} ---")
        summary = tr.groupby(f"regime_{wname}").agg(
            trades=("r", "count"),
            win_rate=("status", lambda s: round(100*(s == "WIN").mean(), 2)),
            total_R=("r", lambda s: round(s.sum(), 2)),
            avg_R=("r", lambda s: round(s.mean(), 3)),
        )
        order = ["TRENDING", "MIXED", "CHOPPY", "UNKNOWN"]
        summary = summary.reindex([o for o in order if o in summary.index])
        print(summary.to_string())

tag_trades("trades_90000_baseline_verify.csv", "BASELINE (no MTF)")
tag_trades("trades_90000_mtf_gated.csv", "4H+15M MTF-gated")
