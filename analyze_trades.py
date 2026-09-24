import pandas as pd

t = pd.read_csv("trades_90000.csv", parse_dates=["entry_time", "exit_time"])
t["hour"] = t["entry_time"].dt.hour
t["bucket"] = (t["hour"] // 3) * 3
t["win"] = (t["status"] == "WIN").astype(int)

print("=== BY ENTRY HOUR BUCKET (data timestamps) ===")
g = t.groupby("bucket").agg(
    trades=("r", "size"),
    win_pct=("win", lambda s: round(100 * s.mean(), 1)),
    avg_R=("r", lambda s: round(s.mean(), 3)),
    total_R=("r", lambda s: round(s.sum(), 1)),
)
g.index = [f"{h:02d}-{h+3:02d}h" for h in g.index]
print(g.to_string())

print("\n=== STOP SIZE (price units) ===")
print(t["initial_risk"].describe().round(2).to_string())

print("\n=== COST SENSITIVITY (round-trip cost in price units) ===")
for c in (0.0, 0.1, 0.2, 0.3, 0.5, 1.0):
    adj = t["r"] - c / t["initial_risk"]
    print(f"cost {c:>4}: avg_R={adj.mean():6.3f}  total_R={adj.sum():8.1f}  win-after-cost%={100*(adj>0).mean():5.1f}")
