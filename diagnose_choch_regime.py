import pandas as pd
from strategy.swings import find_swings
from strategy.structure import analyze_structure, BreakType

df = pd.read_csv("real_gold_data_mt5_90000.csv", parse_dates=["timestamp"]).set_index("timestamp")

print("Computing swings + structure on full 90k (one-shot, non-causal batch -- fast)...")
swings = find_swings(df)
breaks, classified_swings, final_trend = analyze_structure(df, swings)
print(f"  total breaks: {len(breaks)}")

break_rows = []
for b in breaks:
    break_rows.append({"broken_at": b.broken_at, "break_type": b.break_type.value})
breaks_df = pd.DataFrame(break_rows).sort_values("broken_at").reset_index(drop=True)

WINDOW = pd.Timedelta(hours=24)

def choch_ratio_before(ts):
    window_breaks = breaks_df[(breaks_df["broken_at"] < ts) & (breaks_df["broken_at"] >= ts - WINDOW)]
    if len(window_breaks) == 0:
        return None, 0
    choch_count = (window_breaks["break_type"] == "choch").sum()
    return choch_count / len(window_breaks), len(window_breaks)

def regime_label(ratio):
    if ratio is None:
        return "UNKNOWN"
    if ratio >= 0.5:
        return "CHOPPY"
    if ratio <= 0.15:
        return "TRENDING"
    return "MIXED"

def tag_trades(path, label):
    import os
    if not os.path.exists(path):
        print(f"\n[skip] {path} not found")
        return
    tr = pd.read_csv(path, parse_dates=["setup_time"])
    tr = tr[tr["status"].isin(["WIN", "LOSS"])].copy()

    ratios = []
    counts = []
    for ts in tr["setup_time"]:
        r, c = choch_ratio_before(ts)
        ratios.append(r)
        counts.append(c)
    tr["choch_ratio_24h"] = ratios
    tr["breaks_in_window"] = counts
    tr["regime"] = tr["choch_ratio_24h"].apply(regime_label)

    print(f"\n=== {label} ({path}) ===")
    print(f"total closed trades: {len(tr)}")
    summary = tr.groupby("regime").agg(
        trades=("r", "count"),
        win_rate=("status", lambda s: round(100*(s == "WIN").mean(), 2)),
        total_R=("r", lambda s: round(s.sum(), 2)),
        avg_R=("r", lambda s: round(s.mean(), 3)),
    )
    order = ["TRENDING", "MIXED", "CHOPPY", "UNKNOWN"]
    summary = summary.reindex([o for o in order if o in summary.index])
    print(summary.to_string())
    print("sample-size warning: any bucket with <30 trades is noise")

tag_trades("trades_90000_baseline_verify.csv", "BASELINE (no MTF)")
tag_trades("trades_90000_mtf_gated.csv", "4H+15M MTF-gated")
tag_trades("trades_90000_mtf_final.csv", "1H+15M MTF-gated FINAL")
