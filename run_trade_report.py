import pandas as pd
from strategy.backtest import run_backtest

# ---- EDIT THESE TO MATCH THE RUN THAT GAVE 276 ----
CSV = "real_gold_data_mt5_90000.csv"
PARAMS = dict(
    max_bars_to_retest=20,
    reward_multiple=2.0,
    stop_buffer=0.0,
    entry_mode="midpoint",
)
EXPECTED = 276
OUT = "trades_90000.csv"
# ---------------------------------------------------

df = pd.read_csv(CSV)
df.columns = [str(c).strip().lower() for c in df.columns]

time_col = next(
    (c for c in ("timestamp", "time", "datetime", "date") if c in df.columns),
    df.columns[0],
)
df[time_col] = pd.to_datetime(df[time_col])
df = df.set_index(time_col).sort_index()
dupes = int(df.index.duplicated().sum())
if dupes:
    print(f"WARNING: {dupes} duplicate timestamps - dropping duplicates")
    df = df[~df.index.duplicated(keep="first")]

print(f"data: {len(df)} candles, {df.index[0]} -> {df.index[-1]}, columns={list(df.columns)}")

result = run_backtest(df, **PARAMS)
trades = result.trades

rows = []
for t in trades:
    rows.append(
        dict(
            setup_time=t.setup_time,
            entry_time=t.entry_time,
            exit_time=t.exit_time,
            direction=getattr(t.direction, "name", str(t.direction)),
            session=t.session,
            status=t.result_status,
            r=t.r_multiple,
            bars_held=t.bars_held,
            initial_risk=t.initial_risk,
        )
    )

tr = pd.DataFrame(rows)
tr.to_csv(OUT, index=False)

closed = tr[tr["status"] != "OPEN"].copy()
print()
print(f"total trades in result : {len(tr)}   (expected {EXPECTED}: "
      f"{'MATCH' if len(tr) == EXPECTED else 'DIFFERENT'})")
print(f"closed / open          : {len(closed)} / {len(tr) - len(closed)}")
print(f"signals generated      : {result.total_signals_generated}")
print(f"invalidated / expired  : {result.total_invalidated} / {result.total_expired}")
print(f"saved                  : {OUT}")


def summarize(g):
    n = len(g)
    wins = int((g["status"] == "WIN").sum())
    return pd.Series(
        dict(
            trades=n,
            win_pct=round(100 * wins / n, 1) if n else 0.0,
            avg_R=round(g["r"].mean(), 3) if n else 0.0,
            total_R=round(g["r"].sum(), 2),
        )
    )


print("\n=== ALL CLOSED TRADES ===")
print(summarize(closed).to_string())

print("\n=== BY SESSION ===")
print(closed.groupby("session").apply(summarize).to_string())

print("\n=== BY DIRECTION ===")
print(closed.groupby("direction").apply(summarize).to_string())

closed = closed.sort_values("entry_time").reset_index(drop=True)
closed["third"] = pd.qcut(closed.index, 3, labels=["early", "middle", "late"])
print("\n=== BY CHRONOLOGICAL THIRD (equal trade counts) ===")
print(closed.groupby("third", observed=True).apply(summarize).to_string())

print("\n=== SESSION x THIRD (avg R / trades) ===")
piv = closed.pivot_table(
    index="session", columns="third", values="r",
    aggfunc=["mean", "count"], observed=True,
).round(2)
print(piv.to_string())

by_exit = closed.sort_values("exit_time")
cum = by_exit["r"].cumsum()
dd = (cum - cum.cummax()).min()
print(f"\nmax drawdown (R, closed trades in exit order): {dd:.2f}")
