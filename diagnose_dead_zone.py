import pandas as pd
from strategy.backtest import run_backtest
from strategy.mtf_structure import build_mtf_dataset_with_structure
from strategy.swings import find_swings
from strategy.confluence import build_mtf_signal_filter

df_full = pd.read_csv("real_gold_data_mt5_90000.csv", parse_dates=["timestamp"])
df_full = df_full.set_index("timestamp")

# The dead zone: windows 57-63 covered roughly 2026-04-02 to 2026-05-12.
# Give it generous padding on both sides for warmup context.
start = pd.Timestamp("2026-03-15")
end = pd.Timestamp("2026-05-20")

df_slice = df_full[(df_full.index >= start) & (df_full.index <= end)].copy()

print("Slice size:", len(df_slice))
print("Slice range:", df_slice.index[0], "to", df_slice.index[-1])
print()

# --- 1. Raw volatility check: candle ranges in this period vs whole dataset ---
df_slice_range = (df_slice["high"] - df_slice["low"])
df_full_range = (df_full["high"] - df_full["low"])

print("Candle range (high-low) stats:")
print("  This slice  - mean:", round(df_slice_range.mean(), 4), " median:", round(df_slice_range.median(), 4))
print("  Full dataset - mean:", round(df_full_range.mean(), 4), " median:", round(df_full_range.median(), 4))
print()

# --- 2. Ungated engine: does the base 5M engine even generate candidate signals here? ---
result_ungated = run_backtest(df_slice)
print("UNGATED (no MTF filter) on this slice:")
print("  Signals generated:", result_ungated.total_signals_generated)
print("  Triggered:", result_ungated.total_trades_triggered)
print("  Win rate:", result_ungated.win_rate, " Expectancy:", result_ungated.expectancy)
print()

# --- 3. MTF-gated engine: how many of those signals got rejected? ---
df_enriched_slice = build_mtf_dataset_with_structure(
    df_slice,
    macro_swings_fn=find_swings,
    internal_swings_fn=find_swings,
)
mtf_filter_fn = build_mtf_signal_filter(df_enriched_slice)

result_gated = run_backtest(df_slice, mtf_filter_fn=mtf_filter_fn)
print("MTF-GATED on this slice:")
print("  Signals generated (approved):", result_gated.total_signals_generated)
print("  Rejected by MTF:", result_gated.total_mtf_rejected)
print("  Triggered:", result_gated.total_trades_triggered)
print()

# --- 4. What did macro/internal trend look like during this period? ---
trend_counts_macro = df_enriched_slice["macro_trend"].value_counts(dropna=False)
trend_counts_internal = df_enriched_slice["internal_trend"].value_counts(dropna=False)
print("Macro (4H) trend distribution during this slice:")
print(trend_counts_macro)
print()
print("Internal (15M) trend distribution during this slice:")
print(trend_counts_internal)