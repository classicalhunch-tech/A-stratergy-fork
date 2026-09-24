import pandas as pd

df = pd.read_csv("real_gold_data_mt5_90000.csv", parse_dates=["timestamp"])

print("DATASET VALIDATION")
print("------------------")
print("Total candles:", len(df))
print()
print("First candle:", df["timestamp"].iloc[0])
print("Last candle:", df["timestamp"].iloc[-1])
print()
print("Duplicate timestamps:", df["timestamp"].duplicated().sum())
print("Missing OHLC values:", df[["open", "high", "low", "close"]].isna().sum().sum())

bad_ohlc = df[
    (df["high"] < df["open"]) | (df["high"] < df["close"]) | (df["high"] < df["low"]) |
    (df["low"] > df["open"]) | (df["low"] > df["close"])
]
print("Invalid OHLC rows:", len(bad_ohlc))

is_sorted = df["timestamp"].is_monotonic_increasing
print("Chronological order:", "PASS" if is_sorted else "FAIL")

diffs = df["timestamp"].diff().dropna()
normal_spacing = (diffs == pd.Timedelta(minutes=5)).sum()
print("5-minute structure:", f"{normal_spacing}/{len(diffs)} intervals at exactly 5min")
print()
print("Top gap types (non-5min intervals):")
print(diffs[diffs != pd.Timedelta(minutes=5)].value_counts().head(15))
print()

overlap_yfinance = (
    (df["timestamp"].min() <= pd.Timestamp("2026-09-16 09:25:00")) and
    (df["timestamp"].max() >= pd.Timestamp("2026-07-26 20:55:00"))
)
print("Overlaps with yfinance real dataset period (2026-07-26 to 2026-09-16):", overlap_yfinance)

print()
print("Price range - min close:", df["close"].min(), "max close:", df["close"].max())
print()
print("Dataset validation:", "PASS" if (df["timestamp"].duplicated().sum() == 0 and len(bad_ohlc) == 0 and is_sorted) else "FAIL")