import pandas as pd
from strategy.signals import generate_flip_zone_signals
from strategy.swings import find_swings
from strategy.structure import analyze_structure
from strategy.zones import find_zones
from strategy.liquidity import detect_liquidity_levels, update_liquidity_sweeps
from strategy.mtf_structure import build_mtf_dataset_with_structure
from strategy.confluence import build_mtf_signal_filter

df_full = pd.read_csv("real_gold_data_mt5_90000.csv", parse_dates=["timestamp"])
df_full = df_full.set_index("timestamp")

start = pd.Timestamp("2026-03-15")
end = pd.Timestamp("2026-05-20")
df_slice = df_full[(df_full.index >= start) & (df_full.index <= end)].copy()

# Rebuild the same structural pipeline used inside run_backtest(),
# but simplified: generate ALL signals in one pass (not causal
# bar-by-bar) purely to inspect direction/trend agreement patterns.
swings = find_swings(df_slice)
breaks, _, _ = analyze_structure(df_slice, swings)
zones = find_zones(df_slice, breaks)
liquidity = detect_liquidity_levels(df_slice, swings)
liquidity = update_liquidity_sweeps(df_slice, liquidity)

signals = generate_flip_zone_signals(
    liquidity_levels=liquidity,
    structure_breaks=breaks,
    zones=zones,
    df=df_slice,
)

df_enriched = build_mtf_dataset_with_structure(
    df_slice,
    macro_swings_fn=find_swings,
    internal_swings_fn=find_swings,
)

mtf_filter_fn = build_mtf_signal_filter(df_enriched)

rows = []
for sig in signals:
    setup_time = getattr(sig, "setup_timestamp", None)
    direction = getattr(sig.signal_type, "value", sig.signal_type)

    if setup_time is None:
        continue
    setup_time = pd.Timestamp(setup_time)

    approved = mtf_filter_fn(sig, setup_time)

    macro = df_enriched["macro_trend"].asof(setup_time) if setup_time in df_enriched.index or True else None
    internal = df_enriched["internal_trend"].asof(setup_time) if setup_time in df_enriched.index or True else None

    rows.append({
        "setup_time": setup_time,
        "direction": str(direction).upper(),
        "macro_trend": macro,
        "internal_trend": internal,
        "approved": approved,
    })

result_df = pd.DataFrame(rows)

print("Total raw signals inspected:", len(result_df))
print()
print("Approval rate by direction:")
print(result_df.groupby("direction")["approved"].agg(["sum", "count", "mean"]))
print()
print("Rejected signals: direction vs macro_trend at signal time:")
rejected = result_df[~result_df["approved"]]
print(pd.crosstab(rejected["direction"], rejected["macro_trend"]))
print()
print("Rejected signals: direction vs internal_trend at signal time:")
print(pd.crosstab(rejected["direction"], rejected["internal_trend"]))