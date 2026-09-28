import pandas as pd
from strategy.signals import generate_flip_zone_signals
from strategy.swings import find_swings
from strategy.structure import analyze_structure
from strategy.zones import find_zones
from strategy.liquidity import detect_liquidity_levels, update_liquidity_sweeps
from strategy.mtf_structure import build_mtf_dataset_with_structure
from strategy.confluence import _normalize_state
from dashboard.mtf_context import MTFConfig

df = pd.read_csv("real_gold_data_mt5_90000.csv", parse_dates=["timestamp"]).set_index("timestamp")

swings = find_swings(df)
breaks, _, _ = analyze_structure(df, swings)
zones = find_zones(df, breaks)
liquidity = detect_liquidity_levels(df, swings)
liquidity = update_liquidity_sweeps(df, liquidity)

signals = generate_flip_zone_signals(
    liquidity_levels=liquidity, structure_breaks=breaks, zones=zones, df=df,
)

config_1h = MTFConfig(macro_tf="1h", internal_tf="15min")
df_enriched = build_mtf_dataset_with_structure(
    df, macro_swings_fn=find_swings, internal_swings_fn=find_swings, config=config_1h
)

rows = []
for sig in signals:
    setup_time = getattr(sig, "setup_timestamp", None)
    if setup_time is None:
        continue
    setup_time = pd.Timestamp(setup_time)
    direction = _normalize_state(getattr(sig, "signal_type", None))
    if direction not in ("LONG", "SHORT"):
        continue

    macro = _normalize_state(df_enriched["macro_trend"].asof(setup_time))
    internal = _normalize_state(df_enriched["internal_trend"].asof(setup_time))

    opp = "BEARISH" if direction == "LONG" else "BULLISH"
    macro_cat = "CONFLICT" if macro == opp else ("NEUTRAL" if macro is None else "AGREE")
    internal_cat = "CONFLICT" if internal == opp else ("NEUTRAL" if internal is None else "AGREE")
    approved = (macro_cat == "AGREE") and (internal_cat in ("AGREE", "NEUTRAL"))

    rows.append({
        "setup_time": setup_time, "direction": direction,
        "h1_trend": macro, "m15_trend": internal,
        "h1_cat": macro_cat, "m15_cat": internal_cat,
        "approved": approved,
    })

result_df = pd.DataFrame(rows)
result_df.to_csv("mtf_direction_breakdown_1h15m.csv", index=False)

print("Total candidate signals:", len(result_df))
print()
print("--- Full direction x category crosstab (direction, h1_cat, m15_cat) ---")
print(result_df.groupby(["direction","h1_cat","m15_cat"]).size().sort_values(ascending=False).to_string())
print()
print("--- Approval rate by direction ---")
print(result_df.groupby("direction")["approved"].agg(["sum","count","mean"]).to_string())
print()
print("--- Rejected: h1_trend x m15_trend raw value crosstab ---")
rejected = result_df[~result_df["approved"]]
print(pd.crosstab([rejected["direction"], rejected["h1_trend"]], rejected["m15_trend"], dropna=False).to_string())
print()
print("saved mtf_direction_breakdown_1h15m.csv")
