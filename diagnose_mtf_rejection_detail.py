"""
diagnose_mtf_rejection_detail.py

Wraps build_mtf_signal_filter's predicate with logging so we can see,
per rejected signal: its timestamp, direction, and the macro/internal
trend state that caused the rejection. Does not modify strategy code.
"""

import pandas as pd

from main import load_5m_data, build_mtf_context
from strategy.backtest import run_backtest
from strategy.confluence import build_mtf_signal_filter, _normalize_state

df_5m = load_5m_data("your_data_file.csv")
df_enriched = build_mtf_context(df_5m)

base_filter = build_mtf_signal_filter(df_enriched, allow_neutral_internal=True)

log = []

def logging_filter(signal, current_time):
    approved = base_filter(signal, current_time)
    base_signal = _normalize_state(getattr(signal, "signal_type", None))
    log.append({
        "time": current_time,
        "direction": base_signal,
        "approved": approved,
    })
    return approved

result = run_backtest(df_5m, mtf_filter_fn=logging_filter)

log_df = pd.DataFrame(log)
print(f"\nTotal signals checked: {len(log_df)}")
print(f"Approved: {log_df['approved'].sum()}")
print(f"Rejected: {(~log_df['approved']).sum()}")

print("\n--- Rejected signals: direction breakdown ---")
print(log_df[~log_df["approved"]]["direction"].value_counts())

print("\n--- Rejected signals: first/last timestamp ---")
rejected = log_df[~log_df["approved"]]
print(f"First rejected at: {rejected['time'].min()}")
print(f"Last rejected at:  {rejected['time'].max()}")

macro_established_at = pd.Timestamp("2026-01-06 11:55:00")
before_established = rejected[rejected["time"] < macro_established_at]
print(f"\nRejected signals BEFORE macro_trend established: {len(before_established)} / {len(rejected)}")