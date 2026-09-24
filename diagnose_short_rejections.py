"""
diagnose_short_rejections.py

For every rejected SHORT signal, print the exact macro_trend and
internal_trend values the filter saw at that timestamp, plus whether
the lookup found an exact index match or fell back to as-of.
"""

import pandas as pd

from main import load_5m_data, build_mtf_context
from strategy.backtest import run_backtest
from strategy.confluence import _normalize_state

df_5m = load_5m_data("your_data_file.csv")
df_enriched = build_mtf_context(df_5m)

trend_lookup = {
    ts: (_normalize_state(macro), _normalize_state(internal))
    for ts, macro, internal in zip(
        df_enriched.index,
        df_enriched["macro_trend"],
        df_enriched["internal_trend"],
    )
}

log = []

def logging_filter(signal, current_time):
    base_signal = _normalize_state(getattr(signal, "signal_type", None))
    exact_match = current_time in trend_lookup
    macro, internal = trend_lookup.get(current_time, (None, None))
    log.append({
        "time": current_time,
        "direction": base_signal,
        "exact_index_match": exact_match,
        "macro_trend": macro,
        "internal_trend": internal,
    })
    return False  # reject everything, we only care about the log here

result = run_backtest(df_5m, mtf_filter_fn=logging_filter)

log_df = pd.DataFrame(log)
shorts = log_df[log_df["direction"] == "SHORT"]

print(f"\nTotal SHORT signals: {len(shorts)}")
print("\n--- Per-SHORT-signal trend state ---")
print(shorts.to_string(index=False))

print("\n--- exact_index_match value counts ---")
print(log_df["exact_index_match"].value_counts())