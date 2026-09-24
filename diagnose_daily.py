"""
diagnose_daily.py

One-off diagnostic: pull raw daily GC=F data (no validation) and show
exactly which rows have High < max(Open,Close,Low) or Low > min(Open,Close,High),
so we can see whether this is a rounding issue or a real data problem.
"""

from market import market_data as md

df = md.get_candles("D", validate=False)

print(f"Total daily candles: {len(df)}")

bad_high = df[df["High"] < df[["Open", "Close", "Low"]].max(axis=1)]
bad_low = df[df["Low"] > df[["Open", "Close", "High"]].min(axis=1)]

print(f"\nBad High rows: {len(bad_high)}")
print(bad_high.head(10))

print(f"\nBad Low rows: {len(bad_low)}")
print(bad_low.head(10))

if not bad_high.empty:
    diff = df.loc[bad_high.index, ["Open", "Close", "Low"]].max(axis=1) - bad_high["High"]
    print(f"\nHigh discrepancy stats:\n{diff.describe()}")

if not bad_low.empty:
    diff = bad_low["Low"] - df.loc[bad_low.index, ["Open", "Close", "High"]].min(axis=1)
    print(f"\nLow discrepancy stats:\n{diff.describe()}")