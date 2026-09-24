"""
diagnose_mtf_rejections.py

One-off diagnostic: why is every candidate being rejected by the
MTF confluence filter? Prints the macro/internal trend distribution
across the enriched dataset, and does NOT modify any strategy code.
"""

import pandas as pd

from main import load_5m_data, build_mtf_context

df_5m = load_5m_data("your_data_file.csv")
df_enriched = build_mtf_context(df_5m)

print("\n--- macro_trend value counts (including None/NaN) ---")
print(df_enriched["macro_trend"].value_counts(dropna=False))

print("\n--- internal_trend value counts (including None/NaN) ---")
print(df_enriched["internal_trend"].value_counts(dropna=False))

first_established = df_enriched.index[df_enriched["macro_trend"].notna()]
if len(first_established) > 0:
    print(f"\nmacro_trend first established at: {first_established[0]}")
    print(f"Dataset ends at:                  {df_enriched.index[-1]}")
else:
    print("\nmacro_trend is NEVER established in this dataset.")