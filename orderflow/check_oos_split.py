import pandas as pd

df = pd.read_csv("orderflow_research.csv", parse_dates=["entry_time"])
df = df.sort_values("entry_time").reset_index(drop=True)
df["win"] = (df["result_status"] == "WIN").astype(int)

split_idx = int(len(df) * 0.7)
first, second = df.iloc[:split_idx], df.iloc[split_idx:]

for name, part in [("FIRST 70%", first), ("LAST 30%", second)]:
    print(f"\n=== {name} ({part['entry_time'].min()} -> {part['entry_time'].max()}) ===")
    g = part.groupby("of_pressure_agrees_with_signal").agg(
        trade_count=("r_multiple", "count"),
        win_rate=("win", "mean"),
        avg_r=("r_multiple", "mean"),
        total_r=("r_multiple", "sum"),
    )
    g["win_rate"] = (g["win_rate"] * 100).round(2)
    print(g.round(3))
