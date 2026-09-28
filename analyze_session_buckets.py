import os
import pandas as pd

DEAD = {23, 0, 1, 2}
LOW = {6, 7, 12, 13}
PEAK = {15, 16, 17, 18, 19, 20}

def bucket(hour):
    if hour in DEAD: return "DEAD"
    if hour in LOW: return "LOW"
    if hour in PEAK: return "PEAK"
    return "NORMAL"

R_COL_CANDIDATES = ["r", "realized_r", "r_multiple", "pnl_r", "R"]

def analyze(path, label):
    if not os.path.exists(path):
        print(f"[skip] {path} not found"); return

    df = pd.read_csv(path, parse_dates=["entry_time"])

    # --- sanity checks ---
    print(f"\n=== {label} ===")
    print("columns:", df.columns.tolist())
    print("statuses:", df["status"].value_counts().to_dict())
    print("entry_time dtype:", df["entry_time"].dtype)

    r_col = next((c for c in R_COL_CANDIDATES if c in df.columns), None)
    if r_col is None:
        print("!! No R column found; aborting"); return
    print("using R column:", r_col)

    df = df[df["status"].isin(["WIN", "LOSS"])].copy()
    df["hour"] = df["entry_time"].dt.hour
    df["bucket"] = df["hour"].apply(bucket)

    print("first entry_time:", df["entry_time"].iloc[0] if len(df) else "n/a")
    print("total closed trades:", len(df))

    summary = df.groupby("bucket").agg(
        trades=("hour", "count"),
        win_rate=("status", lambda s: round(100*(s == "WIN").mean(), 2)),
        total_R=(r_col, lambda s: round(s.sum(), 2)),
        avg_R=(r_col, lambda s: round(s.mean(), 3)),
    ).reindex(["DEAD", "LOW", "NORMAL", "PEAK"])

    print(summary.to_string())
    print()
    print("sample-size warning: any bucket with <30 trades is noise")

analyze("trades_90000_baseline_verify.csv", "BASELINE (no MTF)")
analyze("trades_90000_mtf_gated.csv", "4H+15M MTF-gated")
analyze("trades_90000_mtf_final.csv", "1H+15M MTF-gated FINAL")
