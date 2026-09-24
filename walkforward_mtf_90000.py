import pandas as pd
from strategy.mtf_structure import build_mtf_dataset_with_structure
from strategy.swings import find_swings
from strategy.confluence import build_mtf_signal_filter
from phase_02_optimization.walk_forward import run_walk_forward, WalkForwardConfig, print_walk_forward_report

df = pd.read_csv("real_gold_data_mt5_90000.csv", parse_dates=["timestamp"])
df = df.set_index("timestamp")

print("Building MTF context across full 90,000-candle dataset...")
df_enriched = build_mtf_dataset_with_structure(
    df,
    macro_swings_fn=find_swings,
    internal_swings_fn=find_swings,
)

mtf_filter_fn = build_mtf_signal_filter(df_enriched)

config = WalkForwardConfig(
    window_size=2000,
    step_size=1000,
    warmup_size=750,
)

print("Running MTF-gated walk-forward...")
result = run_walk_forward(df, config, mtf_filter_fn=mtf_filter_fn)
print_walk_forward_report(result)