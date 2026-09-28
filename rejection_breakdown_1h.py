import time
import pandas as pd
from strategy.backtest import run_backtest
from strategy.mtf_structure import build_mtf_dataset_with_structure
from strategy.swings import find_swings
from strategy.confluence import _normalize_state
from dashboard.mtf_context import MTFConfig

t0 = time.time()
print("Loading data...")
df = pd.read_csv("real_gold_data_mt5_90000.csv", parse_dates=["timestamp"]).set_index("timestamp")
print(f"  loaded {len(df)} rows in {time.time()-t0:.1f}s")

t1 = time.time()
print("Building MTF context (1H macro + 15M internal)...")
config_1h = MTFConfig(macro_tf="1h", internal_tf="15min")
df_enriched = build_mtf_dataset_with_structure(
    df, macro_swings_fn=find_swings, internal_swings_fn=find_swings, config=config_1h
)
print(f"  MTF context built in {time.time()-t1:.1f}s")

# causal sanity check before trusting anything downstream
macro_close = df_enriched["macro_close_time"]
eval_time = df_enriched.index + pd.Timedelta("5min")
violations = (macro_close > eval_time).sum()
print(f"  causal check: {violations} lookahead violations (should be 0)")

t2 = time.time()
print("Building trend lookup table...")
trend_lookup = {
    ts: (_normalize_state(m), _normalize_state(i))
    for ts, m, i in zip(df_enriched.index, df_enriched["macro_trend"], df_enriched["internal_trend"])
}
sorted_idx = df_enriched.index.sort_values()
print(f"  lookup built in {time.time()-t2:.1f}s")

def lookup(ts):
    if ts in trend_lookup:
        return trend_lookup[ts]
    pos = sorted_idx.searchsorted(ts, side="right") - 1
    if pos < 0:
        return None, None
    return trend_lookup.get(sorted_idx[pos], (None, None))

log = []

def logging_filter(signal, current_time):
    base_signal = _normalize_state(getattr(signal, "signal_type", None))
    macro, internal = lookup(current_time)
    if base_signal not in ("LONG", "SHORT"):
        return False
    opp = "BEARISH" if base_signal == "LONG" else "BULLISH"
    macro_cat = "MACRO_CONFLICT" if macro == opp else ("MACRO_NEUTRAL" if macro is None else "MACRO_AGREE")
    internal_cat = "INTERNAL_CONFLICT" if internal == opp else ("INTERNAL_NEUTRAL" if internal is None else "INTERNAL_AGREE")
    approved = (macro_cat == "MACRO_AGREE") and (internal_cat in ("INTERNAL_AGREE", "INTERNAL_NEUTRAL"))
    log.append({"time": current_time, "direction": base_signal, "macro_cat": macro_cat, "internal_cat": internal_cat, "approved": approved})
    return approved

t3 = time.time()
print("Running backtest replay (this is the ~14min step)...")
result = run_backtest(df, mtf_filter_fn=logging_filter)
print(f"  backtest replay done in {time.time()-t3:.1f}s")

log_df = pd.DataFrame(log)
rejected = log_df[~log_df["approved"]]

print()
print("Total signals:", len(log_df), " approved:", log_df["approved"].sum(), " rejected:", len(rejected))
print()
print("--- Rejection reason breakdown (1H + 15M) ---")
print(rejected.groupby(["macro_cat","internal_cat"]).size().sort_values(ascending=False).to_string())
print()
print("--- Rejected by macro_cat alone ---")
print(rejected["macro_cat"].value_counts())

closed = [t for t in result.trades if t.result_status in ("WIN","LOSS")]
wins = [t for t in closed if t.result_status == "WIN"]
wr = 100*len(wins)/len(closed) if closed else 0
total_r = sum(t.r_multiple for t in closed)
avg_r = total_r/len(closed) if closed else 0
print()
print(f"1H+15M MTF-GATED (90k): closed={len(closed)} win_rate={wr:.2f}% total_R={total_r:.2f} avg_R={avg_r:.3f}")
print(f"TOTAL RUNTIME: {time.time()-t0:.1f}s")
