import pandas as pd
import numpy as np

df = pd.read_csv("real_gold_data_mt5_90000.csv", parse_dates=["timestamp"]).set_index("timestamp")

LOOKBACK_BARS = 2000   # ~7 trading days of 5M bars, causal window before each trade
N_BINS = 100           # price bins across the lookback window's range

tr = pd.read_csv("trades_90000_baseline_verify.csv", parse_dates=["setup_time", "entry_time"])
tr = tr[tr["status"].isin(["WIN", "LOSS"])].copy()

# Need entry_price -- not in the saved CSV, so reconstruct from setup context isn't
# reliable; instead use the midpoint of high/low at entry_time bar as a proxy for
# where price was, since exact entry_price wasn't exported. Close is the safest
# single available field tied to entry_time.
df_lookup = df[["close"]].copy()

def price_at(ts):
    idx = df.index.searchsorted(ts)
    if idx >= len(df):
        idx = len(df) - 1
    return df["close"].iloc[idx]

highs = df["high"].to_numpy()
lows = df["low"].to_numpy()
vols = df["volume"].to_numpy()
idx_lookup = {ts: i for i, ts in enumerate(df.index)}

def volume_node_label(entry_time, entry_price):
    ts_arr = df.index
    pos = ts_arr.searchsorted(entry_time)
    if pos <= 0:
        return "UNMATCHED", None
    start = max(0, pos - LOOKBACK_BARS)
    window_high = highs[start:pos]
    window_low = lows[start:pos]
    window_vol = vols[start:pos]

    if len(window_high) < 100:
        return "UNMATCHED", None

    lo = window_low.min()
    hi = window_high.max()
    if hi <= lo:
        return "UNMATCHED", None

    bin_edges = np.linspace(lo, hi, N_BINS + 1)
    # distribute each candle's volume across the bins its high-low range spans,
    # weighted equally across those bins (simple, causal, no lookahead -- only
    # uses candles strictly before this trade's entry bar)
    bin_volume = np.zeros(N_BINS)
    bin_width = (hi - lo) / N_BINS
    for h, l, v in zip(window_high, window_low, window_vol):
        b_lo = int((l - lo) / bin_width)
        b_hi = int((h - lo) / bin_width)
        b_lo = max(0, min(b_lo, N_BINS - 1))
        b_hi = max(0, min(b_hi, N_BINS - 1))
        span = b_hi - b_lo + 1
        bin_volume[b_lo:b_hi + 1] += v / span

    entry_bin = int((entry_price - lo) / bin_width)
    entry_bin = max(0, min(entry_bin, N_BINS - 1))

    entry_bin_vol = bin_volume[entry_bin]
    p25 = np.percentile(bin_volume, 25)
    p75 = np.percentile(bin_volume, 75)

    if entry_bin_vol <= p25:
        return "LVN", entry_bin_vol
    if entry_bin_vol >= p75:
        return "HVN", entry_bin_vol
    return "MID", entry_bin_vol

print(f"Processing {len(tr)} trades (lookback={LOOKBACK_BARS} bars, {N_BINS} bins each)...")

labels = []
vols_out = []
for _, row in tr.iterrows():
    entry_time = row["entry_time"]
    entry_price = price_at(entry_time)
    label, v = volume_node_label(entry_time, entry_price)
    labels.append(label)
    vols_out.append(v)

tr["volume_node"] = labels
tr["node_volume"] = vols_out

print("\n=== Trade quality by volume-node classification ===")
summary = tr.groupby("volume_node").agg(
    trades=("r", "count"),
    win_rate=("status", lambda s: round(100*(s == "WIN").mean(), 2)),
    total_R=("r", lambda s: round(s.sum(), 2)),
    avg_R=("r", lambda s: round(s.mean(), 3)),
)
order = ["LVN", "MID", "HVN", "UNMATCHED"]
summary = summary.reindex([o for o in order if o in summary.index])
print(summary.to_string())
print("\nsample-size warning: any bucket with <30 trades is noise")
print("\nNOTE: entry_price is approximated as the close price at entry_time")
print("(exact entry_price was not exported in the trade CSV). Volume is")
print("distributed evenly across each candle's high-low bin span -- a")
print("simplification, not tick-level footprint data.")
