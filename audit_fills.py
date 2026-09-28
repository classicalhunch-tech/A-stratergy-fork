import sys, pandas as pd
import strategy.backtest as bt
import strategy.retest_engine as rt
from strategy.retest_engine import PendingOutcome, PendingOutcomeType
from strategy.signals import SignalType, SignalStatus

EPS = 1e-8
orig = rt.advance_pending_signal
state = {"strict": False, "phantom": 0, "stop_drops": 0}

def wrapped(signal, setup_idx, current_index, current_time, current_open,
            current_high, current_low, visible_zones, max_bars_to_retest):
    planned = float(signal.entry_price)
    stop = float(signal.stop_loss)
    is_long = signal.signal_type == SignalType.LONG
    out = orig(signal, setup_idx, current_index, current_time, current_open,
               current_high, current_low, visible_zones, max_bars_to_retest)
    if out.outcome == PendingOutcomeType.INVALIDATED:
        hit = (current_low <= stop + EPS) if is_long else (current_high >= stop - EPS)
        if hit:
            state["stop_drops"] += 1
    elif out.outcome == PendingOutcomeType.TRIGGERED:
        reached = (current_low <= planned + EPS) if is_long else (current_high >= planned - EPS)
        if not reached:
            state["phantom"] += 1
            if state["strict"]:
                signal.status = SignalStatus.PENDING_RETEST
                signal.entry_price = planned
                signal.recalculate_risk_reward()
                return PendingOutcome(PendingOutcomeType.STILL_PENDING)
    return out

bt.advance_pending_signal = wrapped

df = pd.read_csv("real_gold_data_mt5_90000.csv")
df.columns = [c.strip().lower() for c in df.columns]
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.set_index("timestamp").sort_index()
start, end, mode = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
state["strict"] = (mode == "strict")
res = bt.run_backtest(df.iloc[start:end])
c = [t for t in res.trades if t.result_status in ("WIN", "LOSS")]
n = len(c); r = sum(t.r_multiple for t in c); w = sum(t.result_status == "WIN" for t in c)
d = state["stop_drops"]
print(f"mode={mode} candles={end-start}")
print(f"closed trades={n}  win rate={100*w/max(n,1):.2f}%  total R={r:.2f}  avg R={r/max(n,1):.3f}")
print(f"phantom fills (entry never traded)={state['phantom']}")
print(f"signals dropped as stop-hit invalidations={d}")
print(f"if each dropped signal were a -1R trade: n={n+d}  avg R={(r-d)/max(n+d,1):.3f}  WR={100*w/max(n+d,1):.2f}%")
