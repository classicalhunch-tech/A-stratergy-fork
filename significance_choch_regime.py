import pandas as pd
import numpy as np
from strategy.swings import find_swings
from strategy.structure import analyze_structure

np.random.seed(42)

df = pd.read_csv("real_gold_data_mt5_90000.csv", parse_dates=["timestamp"]).set_index("timestamp")

print("Computing swings + structure (one-shot batch)...")
swings = find_swings(df)
breaks, _, _ = analyze_structure(df, swings)
break_rows = [{"broken_at": b.broken_at, "break_type": b.break_type.value} for b in breaks]
breaks_df = pd.DataFrame(break_rows).sort_values("broken_at").reset_index(drop=True)
break_times = breaks_df["broken_at"].values
break_types = breaks_df["break_type"].values

WINDOW = np.timedelta64(24, "h")

def choch_ratio_before(ts):
    ts64 = np.datetime64(ts)
    mask = (break_times < ts64) & (break_times >= ts64 - WINDOW)
    sub = break_types[mask]
    if len(sub) == 0:
        return None
    return (sub == "choch").sum() / len(sub)

def regime_label(ratio):
    if ratio is None:
        return "UNKNOWN"
    if ratio >= 0.5:
        return "CHOPPY"
    if ratio <= 0.15:
        return "TRENDING"
    return "MIXED"

tr = pd.read_csv("trades_90000_baseline_verify.csv", parse_dates=["setup_time"])
tr = tr[tr["status"].isin(["WIN", "LOSS"])].copy()
tr["regime"] = tr["setup_time"].apply(lambda ts: regime_label(choch_ratio_before(ts)))

mixed = tr[tr["regime"] == "MIXED"]
choppy = tr[tr["regime"] == "CHOPPY"]

print(f"\nMIXED:  n={len(mixed)}  win_rate={100*(mixed['status']=='WIN').mean():.2f}%  avg_R={mixed['r'].mean():.3f}")
print(f"CHOPPY: n={len(choppy)}  win_rate={100*(choppy['status']=='WIN').mean():.2f}%  avg_R={choppy['r'].mean():.3f}")

obs_wr_diff = 100*(choppy['status']=='WIN').mean() - 100*(mixed['status']=='WIN').mean()
obs_r_diff = choppy['r'].mean() - mixed['r'].mean()

pool_r = np.concatenate([mixed['r'].values, choppy['r'].values])
pool_win = np.concatenate([(mixed['status']=='WIN').values, (choppy['status']=='WIN').values])
n_mixed, n_choppy = len(mixed), len(choppy)
n_iter = 20000

wr_null = np.zeros(n_iter)
r_null = np.zeros(n_iter)
idx = np.arange(len(pool_r))

for i in range(n_iter):
    np.random.shuffle(idx)
    choppy_idx = idx[:n_choppy]
    mixed_idx = idx[n_choppy:n_choppy+n_mixed]
    wr_null[i] = 100*pool_win[choppy_idx].mean() - 100*pool_win[mixed_idx].mean()
    r_null[i] = pool_r[choppy_idx].mean() - pool_r[mixed_idx].mean()

wr_p = (np.abs(wr_null) >= np.abs(obs_wr_diff)).mean()
r_p = (np.abs(r_null) >= np.abs(obs_r_diff)).mean()

print(f"\nObserved win_rate diff (CHOPPY - MIXED): {obs_wr_diff:+.2f} pts   permutation p = {wr_p:.4f}")
print(f"Observed mean_R diff (CHOPPY - MIXED):   {obs_r_diff:+.4f}        permutation p = {r_p:.4f}")
print("\n(p < 0.05 is conventionally 'significant' -- but with quantized +2/-1 R outcomes,")
print(" win_rate and avg_R are not independent here, so treat these as one signal, not two.)")
