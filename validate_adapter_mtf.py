import pandas as pd
from strategy.backtest import run_backtest
from strategy.mtf_structure import build_mtf_dataset_with_structure
from strategy.swings import find_swings
from strategy.confluence import build_mtf_signal_filter
from dashboard.mtf_context import MTFConfig
from phase_03_paper.signals.adapter import StrategyAdapter
from phase_03_paper.market.engine import Candle

df_full = pd.read_csv("real_gold_data_mt5_90000.csv", parse_dates=["timestamp"]).set_index("timestamp")
df_slice = df_full.iloc[:3000].copy()

print(f"Testing on {len(df_slice)} candles: {df_slice.index[0]} -> {df_slice.index[-1]}")

# ---------------------------------------------------------
# PATH A: reference (today's validated backtest)
# ---------------------------------------------------------
print("\nRunning reference backtest (Path A)...")
config = MTFConfig(macro_tf="1h", internal_tf="15min")
df_enriched = build_mtf_dataset_with_structure(
    df_slice, macro_swings_fn=find_swings, internal_swings_fn=find_swings, config=config
)
mtf_filter_fn = build_mtf_signal_filter(df_enriched, soft_internal_conflict=False)
result_a = run_backtest(df_slice, mtf_filter_fn=mtf_filter_fn)

trades_a = sorted(
    [(t.setup_time, t.entry_time, getattr(t.direction, "name", str(t.direction)), round(t.r_multiple, 4))
     for t in result_a.trades],
    key=lambda x: (x[0], x[2])
)
print(f"Path A: {len(trades_a)} trades, {result_a.total_signals_generated} signals approved, {result_a.total_mtf_rejected} rejected")

# ---------------------------------------------------------
# PATH B: patched adapter, fed candle by candle
# ---------------------------------------------------------
print("\nRunning patched StrategyAdapter (Path B)...")
adapter = StrategyAdapter(
    mtf_enabled=True,
    mtf_macro_tf="1h",
    mtf_internal_tf="15min",
    mtf_soft_internal_conflict=False,
    mtf_min_bars=200,
)

triggered_b = []
for ts, row in df_slice.iterrows():
    ts_utc = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    candle = Candle(
        timestamp=ts_utc,
        open=float(row["open"]),
        high=float(row["high"]),
        low=float(row["low"]),
        close=float(row["close"]),
    )
    events = adapter.on_candle(candle)
    for ev in events:
        setup_ts = ev.signal.setup_timestamp
        entry_ts = ev.signal.entry_time
        # strip tz back off for comparison against Path A's naive timestamps
        setup_ts_naive = setup_ts.tz_localize(None) if setup_ts is not None and setup_ts.tzinfo is not None else setup_ts
        entry_ts_naive = entry_ts.tz_localize(None) if entry_ts is not None and entry_ts.tzinfo is not None else entry_ts
        triggered_b.append((
            setup_ts_naive,
            entry_ts_naive,
            getattr(ev.signal.signal_type, "name", str(ev.signal.signal_type)),
        ))

if adapter.errors:
    print(f"  ADAPTER ERRORS ({len(adapter.errors)}):")
    for e in adapter.errors[:10]:
        print("   ", e)

trades_b = sorted(triggered_b, key=lambda x: (x[0], x[2]))
print(f"Path B: {len(trades_b)} triggered signals")

# ---------------------------------------------------------
# COMPARE
# ---------------------------------------------------------
print(f"\n{'='*50}")
print(f"Path A trade count: {len(trades_a)}")
print(f"Path B trade count: {len(trades_b)}")

set_a = set((t[0], t[2]) for t in trades_a)
set_b = set((t[0], t[2]) for t in trades_b)

only_in_a = set_a - set_b
only_in_b = set_b - set_a

print(f"\nSetups in A but not B: {len(only_in_a)}")
if only_in_a:
    for x in list(only_in_a)[:10]:
        print("  ", x)

print(f"Setups in B but not A: {len(only_in_b)}")
if only_in_b:
    for x in list(only_in_b)[:10]:
        print("  ", x)

if not only_in_a and not only_in_b:
    print("\nMATCH: every setup_time+direction pair agrees between Path A and Path B.")
else:
    print("\nMISMATCH: the adapter and backtest disagree -- do NOT trust the patched adapter yet.")
