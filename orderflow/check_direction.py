import pandas as pd

df = pd.read_csv("orderflow_research.csv")
df["win"] = (df["result_status"] == "WIN").astype(int)

print("=== By direction x pressure agreement ===")
g = df.groupby(["direction", "of_pressure_agrees_with_signal"]).agg(
    trade_count=("r_multiple", "count"),
    win_rate=("win", "mean"),
    avg_r=("r_multiple", "mean"),
    total_r=("r_multiple", "sum"),
)
g["win_rate"] = (g["win_rate"] * 100).round(2)
print(g.round(3))
