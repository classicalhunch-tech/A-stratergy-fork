import sys, pandas as pd
import strategy.backtest as bt
import strategy.retest_engine as rt
from strategy.retest_engine import PendingOutcome, PendingOutcomeType
from strategy.signals import SignalType, SignalStatus

EPS = 1e-8
RR = 2.0
orig = rt.advance_pending_signal
state = {"mode": "base", "phantom": 0, "stop_drops": 0}

def find_zone(signal, zones):
    key = getattr(signal, "zone_stable_key", None)
    if key is not None:
        for z in zones:
            if rt._stable_cache_key(z) == key:
                return z
    zid = getattr(signal, "zone_id", None)
    if zid is not None:
        for z in zones:
            if getattr(z, "zone_id", None) == zid:
                return z
    return None

def wrapped(signal, setup_idx, current_index, current_time, current_open,
            current_high, current_low, visible_zones, max_bars_to_retest):
    planned = float(signal.entry_price)
    stop = float(signal.stop_loss)
    tp0 = float(signal.take_profit)
    is_long = signal.signal_type == SignalType.LONG
    mode = state["mode"]

    if mode == "near":
        z = find_zone(signal, visible_zones)
        if z is not None:
            signal.entry_price = float(z.price_top) if is_long else float(z.price_bottom)

    out = orig(signal, setup_idx, current_index, current_time, current_open,
               current_high, current_low, visible_zones, max_bars_to_retest)

    if out.outcome == PendingOutcomeType.INVALIDATED:
        hit = (current_low <= stop + EPS) if is_long else (current_high >= stop - EPS)
        if hit:
            state["stop_drops"] += 1
        signal.entry_price = planned
    elif out.outcome == PendingOutcomeType.STILL_PENDING:
        signal.entry_price = planned
    elif out.outcome == PendingOutcomeType.TRIGGERED:
        fill = float(out.fill_price)
        reached = (current_low <= fill + EPS) if is_long else (current_high >= fill - EPS)
        if not reached:
            state["phantom"] += 1
            if mode == "strict":
                signal.status = SignalStatus.PENDING_RETEST
                signal.entry_price = planned
                signal.recalculate_risk_reward()
                return PendingOutcome(PendingOutcomeType.STILL_PENDING)
        if mode == "near":
            risk = float(out.initial_risk)
            signal.take_profit = fill + RR * risk if is_long else fill - RR * risk
    return out

bt.advance_pending_signal = wrapped

df = pd.read_csv("real_gold_data_mt5_90000.csv")
df.columns = [c.strip().lower() for c in df.columns]
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.set_index("timestamp").sort_index()
start, end, mode = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
state["mode"] = mode
res = bt.run_backtest(df.iloc[start:end])
c = [t for t in res.trades if t.result_status in ("WIN", "LOSS")]
n = len(c); r = sum(t.r_multiple for t in c); w = sum(t.result_status == "WIN" for t in c)
d = state["stop_drops"]
sp = sum(0.33 / t.initial_risk for t in c)
print(f"mode={mode} candles={end-start}")
print(f"closed trades={n}  win rate={100*w/max(n,1):.2f}%  total R={r:.2f}  avg R={r/max(n,1):.3f}")
print(f"avg R after 0.33 spread: {(r-sp)/max(n,1):.3f}")
print(f"phantom fills={state['phantom']}  stop-hit invalidations={d}")
print(f"if each stop-hit invalidation were -1R: n={n+d}  avg R={(r-d)/max(n+d,1):.3f}")
