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

# Cost assumption: round-trip cost per trade in price units (gold: per ounce).
# This is a REPORTING-LAYER cost model only -- the backtest engine and MTF
# filter are unchanged.
COST_PRICE_UNITS = 0.30

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

# Cost in R: round-trip cost divided by the trade's own initial risk.
# Trades with larger R carry proportionally smaller cost.
tr["cost_R"] = COST_PRICE_UNITS / tr["initial_risk"]
tr["r_net"] = tr["r"] - tr["cost_R"]

tr.to_csv(OUT, index=False)

closed = tr[tr["status"] != "OPEN"].copy()

def summarize(g, r_col="r"):
    n = len(g)
    wins = int((g["status"] == "WIN").sum())
    return pd.Series(dict(
        trades=n,
        win_pct=round(100 * wins / n, 1) if n else 0.0,
        avg_R=round(g[r_col].mean(), 3) if n else 0.0,
        total_R=round(g[r_col].sum(), 2),
    ))

print()
print(f"total trades          : {len(tr)}")
print(f"closed / open         : {len(closed)} / {len(tr) - len(closed)}")
print(f"signals generated     : {result.total_signals_generated}")
print(f"mtf rejected          : {result.total_mtf_rejected}")
print(f"invalidated / expired : {result.total_invalidated} / {result.total_expired}")
print(f"saved                 : {OUT}")

print(f"\n=== GROSS vs NET (cost = {COST_PRICE_UNITS} price units per trade) ===")
print("Gross:")
print(summarize(closed, "r").to_string())
print("Net:")
print(summarize(closed, "r_net").to_string())

print("\n=== BY DIRECTION (net) ===")
print(closed.groupby("direction").apply(lambda g: summarize(g, "r_net")).to_string())

print("\n=== BY CHRONOLOGICAL THIRD (net) ===")
closed_net = closed.sort_values("entry_time").reset_index(drop=True)
closed_net["third"] = pd.qcut(closed_net.index, 3, labels=["early", "middle", "late"])
print(closed_net.groupby("third", observed=True).apply(lambda g: summarize(g, "r_net")).to_string())

# Gross and net drawdown
gross_by_exit = closed.sort_values("exit_time")
cum_gross = gross_by_exit["r"].cumsum()
dd_gross = (cum_gross - cum_gross.cummax()).min()

net_by_exit = closed_net.sort_values("exit_time")
cum_net = net_by_exit["r_net"].cumsum()
dd_net = (cum_net - cum_net.cummax()).min()

print(f"\nmax drawdown GROSS (R): {dd_gross:.2f}")
print(f"max drawdown NET   (R): {dd_net:.2f}")
