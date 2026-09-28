import pandas as pd
import numpy as np

np.random.seed(42)

df = pd.read_csv("real_gold_data_mt5_90000.csv", parse_dates=["timestamp"]).set_index("timestamp")

LOOKBACK_BARS = 2000
N_BINS = 100

tr = pd.read_csv("trades_90000_baseline_verify.csv", parse_dates=["setup_time", "entry_time"])
tr = tr[tr["status"].isin(["WIN", "LOSS"])].copy()

highs = df["high"].to_numpy()
lows = df["low"].to_numpy()
vols = df["volume"].to_numpy()
ts_arr = df.index

def price_at(ts):
    idx = ts_arr.searchsorted(ts)
    if idx >= len(df):
        idx = len(df) - 1
    return df["close"].iloc[idx]

def volume_node_label(entry_time, entry_price, lookback_bars=LOOKBACK_BARS, n_bins=N_BINS):
    pos = ts_arr.searchsorted(entry_time)
    if pos <= 0:
        return "UNMATCHED", None
    start = max(0, pos - lookback_bars)
    window_high = highs[start:pos]
    window_low = lows[start:pos]
    window_vol = vols[start:pos]

    if len(window_high) < 100:
        return "UNMATCHED", None

    lo = window_low.min()
    hi = window_high.max()
    if hi <= lo:
        return "UNMATCHED", None

    bin_width = (hi - lo) / n_bins
    bin_volume = np.zeros(n_bins)
    for h, l, v in zip(window_high, window_low, window_vol):
        b_lo = int((l - lo) / bin_width)
        b_hi = int((h - lo) / bin_width)
        b_lo = max(0, min(b_lo, n_bins - 1))
        b_hi = max(0, min(b_hi, n_bins - 1))
        span = b_hi - b_lo + 1
        bin_volume[b_lo:b_hi + 1] += v / span

    entry_bin = int((entry_price - lo) / bin_width)
    entry_bin = max(0, min(entry_bin, n_bins - 1))

    entry_bin_vol = bin_volume[entry_bin]
    p25 = np.percentile(bin_volume, 25)
    p75 = np.percentile(bin_volume, 75)

    if entry_bin_vol <= p25:
        return "LVN", entry_bin_vol
    if entry_bin_vol >= p75:
        return "HVN", entry_bin_vol
    return "MID", entry_bin_vol

print(f"Labeling {len(tr)} trades (lookback={LOOKBACK_BARS}, bins={N_BINS})...")
labels, node_vols = [], []
for _, row in tr.iterrows():
    entry_time = row["entry_time"]
    entry_price = price_at(entry_time)
    label, v = volume_node_label(entry_time, entry_price)
    labels.append(label)
    node_vols.append(v)

tr["volume_node"] = labels
tr["node_volume"] = node_vols
tr.to_csv("trades_90000_volume_node_labeled.csv", index=False)
print("saved trades_90000_volume_node_labeled.csv")

def permutation_test(group_a, group_b, label_a, label_b, n_iter=20000):
    wr_a = 100 * (group_a["status"] == "WIN").mean()
    wr_b = 100 * (group_b["status"] == "WIN").mean()
    r_a = group_a["r"].mean()
    r_b = group_b["r"].mean()

    obs_wr_diff = wr_a - wr_b
    obs_r_diff = r_a - r_b

    pool_r = np.concatenate([group_a["r"].values, group_b["r"].values])
    pool_win = np.concatenate([(group_a["status"]=="WIN").values, (group_b["status"]=="WIN").values])
    n_a, n_b = len(group_a), len(group_b)

    wr_null = np.zeros(n_iter)
    r_null = np.zeros(n_iter)
    idx = np.arange(len(pool_r))

    for i in range(n_iter):
        np.random.shuffle(idx)
        a_idx = idx[:n_a]
        b_idx = idx[n_a:n_a+n_b]
        wr_null[i] = 100*pool_win[a_idx].mean() - 100*pool_win[b_idx].mean()
        r_null[i] = pool_r[a_idx].mean() - pool_r[b_idx].mean()

    wr_p = (np.abs(wr_null) >= np.abs(obs_wr_diff)).mean()
    r_p = (np.abs(r_null) >= np.abs(obs_r_diff)).mean()

    print(f"\n=== {label_a} (n={n_a}) vs {label_b} (n={n_b}) ===")
    print(f"{label_a}: win_rate={wr_a:.2f}%  avg_R={r_a:.3f}")
    print(f"{label_b}: win_rate={wr_b:.2f}%  avg_R={r_b:.3f}")
    print(f"win_rate diff: {obs_wr_diff:+.2f} pts   permutation p = {wr_p:.4f}")
    print(f"mean_R diff:   {obs_r_diff:+.4f}        permutation p = {r_p:.4f}")
    verdict = "PASS" if (wr_p < 0.05 or r_p < 0.05) else "FAIL"
    print(f"VERDICT: {verdict}")
    return verdict

lvn = tr[tr["volume_node"] == "LVN"]
mid = tr[tr["volume_node"] == "MID"]
hvn = tr[tr["volume_node"] == "HVN"]

v1 = permutation_test(lvn, mid, "LVN", "MID")
v2 = permutation_test(lvn, hvn, "LVN", "HVN")

print(f"\n{'='*50}")
print(f"STEP 1 SUMMARY: LVN vs MID = {v1}, LVN vs HVN = {v2}")
if v1 == "PASS" and v2 == "PASS":
    print("-> Proceed to Step 2 (split-half test)")
elif v1 == "PASS" or v2 == "PASS":
    print("-> Mixed result, use judgment before Step 2")
else:
    print("-> Does not clear Step 1. Do not proceed to Step 2/3.")
