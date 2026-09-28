import pandas as pd

df = pd.read_csv("trades_90000.csv", parse_dates=["entry_time"])
df["month"] = df["entry_time"].dt.to_period("M")

counts = df.groupby("month").size()
print("Trades per month:")
print(counts.to_string())
print()
print(f"Total trades: {len(df)}")
print(f"Date range: {df['entry_time'].min()} -> {df['entry_time'].max()}")
