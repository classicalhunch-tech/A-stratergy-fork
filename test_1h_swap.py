import pandas as pd
from strategy.backtest import run_backtest
from strategy.mtf_structure import build_mtf_dataset_with_structure
from strategy.swings import find_swings
from strategy.confluence import build_mtf_signal_filter, _normalize_state
from dashboard.mtf_context import MTFConfig

df = pd.read_csv("real_gold_data_mt5_90000.csv", parse_dates=["timestamp"]).set_index("timestamp")
df_slice = df.iloc[:5000].copy()

def stats(trades, label):
    closed = [t for t in trades if t.result_status in ("WIN","LOSS")]
    wins = [t for t in closed if t.result_status == "WIN"]
    wr = 100*len(wins)/len(closed) if closed else 0
    total_r = sum(t.r_multiple for t in closed)
    avg_r = total_r/len(closed) if closed else 0
    print(f"{label}: closed={len(closed)} win_rate={wr:.1f}% total_R={total_r:.2f} avg_R={avg_r:.3f}")

baseline = run_backtest(df_slice)
stats(baseline.trades, "BASELINE (5k, no MTF)")

config_1h = MTFConfig(macro_tf="1h", internal_tf="15min")
df_enriched = build_mtf_dataset_with_structure(
    df_slice, macro_swings_fn=find_swings, internal_swings_fn=find_swings, config=config_1h
)
mtf_filter_fn = build_mtf_signal_filter(df_enriched)
mtf_result = run_backtest(df_slice, mtf_filter_fn=mtf_filter_fn)
stats(mtf_result.trades, "1H+15M MTF-GATED (5k)")
print(f"signals={mtf_result.total_signals_generated} rejected={mtf_result.total_mtf_rejected}")

# quick causal sanity: confirm macro_close_time never exceeds the evaluation time it's attached to
macro_close = df_enriched["macro_close_time"]
eval_time = df_enriched.index + pd.Timedelta("5min")
violations = (macro_close > eval_time).sum()
print(f"causal check: {violations} rows where macro HTF close_time is AFTER the 5M evaluation time (should be 0)")
