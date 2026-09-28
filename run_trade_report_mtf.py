import pandas as pd
from strategy.backtest import run_backtest
from strategy.mtf_structure import build_mtf_dataset_with_structure
from strategy.swings import find_swings
from strategy.confluence import build_mtf_signal_filter
from dashboard.mtf_context import MTFConfig

CSV = "real_gold_data_mt5_90000.csv"
PARAMS = dict(
    max_bars_to_retest=20,
    reward_multiple=2.0,
    stop_buffer=0.0,
    entry_mode="midpoint",
)
OUT = "trades_90000_mtf_final.csv"

df = pd.read_csv(CSV)
df.columns = [str(c).strip().lower() for c in df.columns]
time_col = next((c for c in ("timestamp", "time", "datetime", "date") if c in df.columns), df.columns[0])
df[time_col] = pd.to_datetime(df[time_col])
df = df.set_index(time_col).sort_index()
df = df[~df.index.duplicated(keep="first")]

print(f"data: {len(df)} candles, {df.index[0]} -> {df.index[-1]}")

print("Building 1H + 15M MTF context...")
config = MTFConfig(macro_tf="1h", internal_tf="15min")
df_enriched = build_mtf_dataset_with_structure(
    df, macro_swings_fn=find_swings, internal_swings_fn=find_swings, config=config
)
mtf_filter_fn = build_mtf_signal_filter(df_enriched, soft_internal_conflict=False)

print("Running MTF-corrected backtest (this is the ~40min step on this machine)...")
result = run_backtest(df, mtf_filter_fn=mtf_filter_fn, **PARAMS)
trades = result.trades

rows = [dict(
    setup_time=t.setup_time, entry_time=t.entry_time, exit_time=t.exit_time,
    direction=getattr(t.direction, "name", str(t.direction)), session=t.session,
    status=t.result_status, r=t.r_multiple, bars_held=t.bars_held, initial_risk=t.initial_risk,
) for t in trades]

tr = pd.DataFrame(rows)
tr.to_csv(OUT, index=False)

closed = tr[tr["status"] != "OPEN"].copy()
wins = closed[closed["status"] == "WIN"]
print()
print(f"total trades          : {len(tr)}")
print(f"closed / open         : {len(closed)} / {len(tr) - len(closed)}")
print(f"win rate              : {100*len(wins)/len(closed):.2f}%")
print(f"total R               : {closed['r'].sum():.2f}")
print(f"avg R                 : {closed['r'].mean():.3f}")
print(f"signals generated     : {result.total_signals_generated}")
print(f"mtf rejected          : {result.total_mtf_rejected}")
print(f"invalidated / expired : {result.total_invalidated} / {result.total_expired}")
print(f"saved                 : {OUT}")
