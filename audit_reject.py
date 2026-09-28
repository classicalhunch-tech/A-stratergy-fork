import sys, pandas as pd
import strategy.backtest as bt
import strategy.retest_engine as rt
from strategy.retest_engine import PendingOutcome, PendingOutcomeType
from strategy.signals import SignalType, SignalStatus

EPS, RR, SPREAD = 1e-8, 2.0, 0.33
orig = rt.advance_pending_signal
st = {"touch": 0, "accepted": 0, "skipped": 0}

df = pd.read_csv("real_gold_data_mt5_90000.csv")
df.columns = [c.strip().lower() for c in df.columns]
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.set_index("timestamp").sort_index()

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
    is_long = signal.signal_type == SignalType.LONG
    out = orig(signal, setup_idx, current_index, current_time, current_open,
               current_high, current_low, visible_zones, max_bars_to_retest)
    if out.outcome != PendingOutcomeType.TRIGGERED:
        return out
    st["touch"] += 1
    z = find_zone(signal, visible_zones)
    close = float(df.at[current_time, "close"])
    stop = float(signal.stop_loss)
    edge = float(z.price_top) if is_long else float(z.price_bottom)
    if is_long:
        ok = close > edge and close > current_open and close > stop
    else:
        ok = close < edge and close < current_open and close < stop
    risk = abs(close - stop)
    if not ok or risk <= EPS:
        st["skipped"] += 1
        signal.status = SignalStatus.PENDING_RETEST
        signal.entry_price = planned
        signal.recalculate_risk_reward()
        return PendingOutcome(PendingOutcomeType.STILL_PENDING)
    st["accepted"] += 1
    signal.entry_price = close
    signal.take_profit = close + RR * risk if is_long else close - RR * risk
    signal.recalculate_risk_reward()
    return PendingOutcome(PendingOutcomeType.TRIGGERED, planned_entry=planned,
                          fill_price=close, initial_risk=risk)

bt.advance_pending_signal = wrapped

start, end = int(sys.argv[1]), int(sys.argv[2])
res = bt.run_backtest(df.iloc[start:end])
c = [t for t in res.trades if t.result_status in ("WIN", "LOSS")]
n = len(c)
r = sum(t.r_multiple for t in c)
w = sum(t.result_status == "WIN" for t in c)
sp = sum(SPREAD / t.initial_risk for t in c)
days = max((df.index[end - 1] - df.index[start]).days, 1)
print(f"rejection-entry candles={end-start} ({df.index[start]} -> {df.index[end-1]})")
print(f"touch bars={st['touch']}  accepted={st['accepted']}  skipped (no rejection)={st['skipped']}")
print(f"closed trades={n}  win rate={100*w/max(n,1):.2f}%  total R={r:.2f}  avg R={r/max(n,1):.3f}")
print(f"avg R after {SPREAD} spread: {(r-sp)/max(n,1):.3f}   total R after spread: {r-sp:.2f}")
print(f"trades per day: {n/days:.2f}")
