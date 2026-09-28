import sys, pandas as pd
from strategy.backtest import run_backtest

DATA = "real_gold_data_mt5_90000.csv"
SPREAD = 0.33

def load():
    df = pd.read_csv(DATA)
    df.columns = [c.strip().lower() for c in df.columns]
    t = next((c for c in ("time", "timestamp", "datetime", "date") if c in df.columns), df.columns[0])
    df[t] = pd.to_datetime(df[t])
    return df.set_index(t).sort_index()

def rows(res, closed_only=True):
    out = []
    for t in res.trades:
        if closed_only and t.result_status not in ("WIN", "LOSS"):
            continue
        out.append(dict(setup_time=t.setup_time, entry_time=t.entry_time, exit_time=t.exit_time,
                        direction=str(t.direction).upper(), fill=t.fill_price, stop=t.stop_loss,
                        tp=t.take_profit, risk=t.initial_risk, r=t.r_multiple,
                        bars=t.bars_held, status=t.result_status))
    return pd.DataFrame(out)

def dump():
    df = load()
    t = rows(run_backtest(df))
    eb = df.loc[t.entry_time, ["open", "high", "low", "close"]].reset_index(drop=True).add_prefix("eb_")
    t = pd.concat([t, eb], axis=1)
    L = t.direction.str.contains("LONG")
    t["stop_touch"] = (L & (t.eb_low <= t.stop)) | (~L & (t.eb_high >= t.stop))
    t["tp_touch"] = (L & (t.eb_high >= t.tp)) | (~L & (t.eb_low <= t.tp))
    t["fill_in_bar"] = (t.fill >= t.eb_low - 1e-6) & (t.fill <= t.eb_high + 1e-6)
    t["r_worst"] = t.r.where(~t.stop_touch, -1.0)
    t["r_worst_spread"] = t.r_worst - SPREAD / t.risk
    t.to_csv("trades_audit.csv", index=False)
    print("trades:", len(t), "| reported avg R:", round(t.r.mean(), 3))
    print("entry-bar stop touched:", int(t.stop_touch.sum()), "| of which reported WIN:", int((t.stop_touch & (t.status == "WIN")).sum()))
    print("entry-bar TP touched:", int(t.tp_touch.sum()))
    print("fill outside entry bar range:", int((~t.fill_in_bar).sum()))
    print("worst-case avg R (trigger-bar stops counted):", round(t.r_worst.mean(), 3))
    print("worst-case + 1x spread:", round(t.r_worst_spread.mean(), 3))

def prefix(start, n_short, n_long):
    df = load()
    a, b = df.iloc[start:start + n_short], df.iloc[start:start + n_long]
    ra, rb = rows(run_backtest(a), False), rows(run_backtest(b), False)
    last = a.index[-1]
    key = lambda d: set(zip(d.setup_time, d.entry_time, d.direction))
    ka, kb = key(ra), key(rb[rb.entry_time <= last])
    print(f"window start={start} short={n_short} long={n_long}: entries short={len(ka)} full={len(kb)}")
    print("only in short:", sorted(ka - kb)[:5])
    print("only in full :", sorted(kb - ka)[:5])
    m = ra.merge(rb, on=["setup_time", "entry_time", "direction"], suffixes=("_a", "_b"))
    m = m[m.exit_time_b <= last]
    bad = m[(m.status_a != m.status_b) | (m.exit_time_a != m.exit_time_b)]
    print("outcome mismatches on trades closed inside short window:", len(bad))
    print("PASS" if not (ka ^ kb) and bad.empty else "FAIL: possible lookahead")

if sys.argv[1] == "dump":
    dump()
else:
    prefix(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
