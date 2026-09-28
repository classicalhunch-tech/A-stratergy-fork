import pandas as pd

df = pd.read_csv("real_gold_data_mt5_90000.csv", parse_dates=["timestamp"]).set_index("timestamp")

df["hour"] = df.index.hour
hourly_avg_volume = df.groupby("hour")["volume"].mean().round(1)

print("=== Average tick_volume by broker-time hour (0-23) ===")
print(hourly_avg_volume.to_string())

print()
print("=== Sorted by volume, highest first (top 8) ===")
print(hourly_avg_volume.sort_values(ascending=False).head(8).to_string())

print()
print("=== Sorted by volume, lowest first (top 8, likely dead/Asian hours) ===")
print(hourly_avg_volume.sort_values(ascending=True).head(8).to_string())

print()
print("First timestamp:", df.index[0])
print("Last timestamp:", df.index[-1])
print("Timezone info on index:", df.index.tz)
