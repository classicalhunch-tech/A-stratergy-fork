import time
import pandas as pd
from strategy.backtest import run_backtest
from strategy.mtf_structure import build_mtf_dataset_with_structure
from strategy.swings import find_swings
from strategy.confluence import build_mtf_signal_filter, _mtf_confluence_approved, _normalize_state
from dashboard.mtf_context import MTFConfig

t0 = time.time()
print("Loading data...")
df = pd.read_csv("real_gold_data_mt5_90000.csv", parse_dates=["timestamp"]).set_index("timestamp")
print(f"  {len(df)} rows in {time.time()-t0:.1f}s")

print("Building 1H+15M MTF context...")
t1 = time.time()
config_1h = MTFConfig(macro_tf="1h", internal_tf="15min")
df_enriched = build_mtf_dataset_with_structure(
    df, macro_swings_fn=find_swings, internal_swings_fn=find_swings, config=config_1h
)
print(f"  built in {time.time()-t1:.1f}s")

macro_close = df_enriched["macro_close_time"]
internal_close = df_enriched["internal_close_time"]
eval_time = df_enriched.index + pd.Timedelta("5min")
macro_violations = int((macro_close > eval_time).sum())
internal_violations = int((internal_close > eval_time).sum())
print(f"  causal check: macro_violations={macro_violations} internal_violations={internal_violations} (both should be 0)")

print()
print("--- Trend distribution (full 90k, 1H macro / 15M internal) ---")
print("macro_trend (1H):")
print(df_enriched["macro_trend"].value_counts(dropna=False).to_string())
print()
print("internal_trend (15M):")
print(df_enriched["internal_trend"].value_counts(dropna=False).to_string())

print()
print("--- Mapping/unit sanity check on _mtf_confluence_approved ---")
cases = [
    ("LONG", "BULLISH", "BULLISH", True, False, True),
    ("LONG", "BULLISH", "BEARISH", True, False, False),
    ("LONG", "BULLISH", "BEARISH", True, True, True),
    ("LONG", "BEARISH", "BULLISH", True, False, False),
    ("LONG", "BEARISH", "BULLISH", True, True, False),
    ("SHORT", "BEARISH", "BEARISH", True, False, True),
    ("SHORT", "BEARISH", "BULLISH", True, False, False),
    ("SHORT", "BEARISH", "BULLISH", True, True, True),
    ("SHORT", "BULLISH", "BEARISH", True, False, False),
    ("SHORT", "BULLISH", "BEARISH", True, True, False),
]
all_pass = True
for base_signal, macro, internal, allow_neutral, soft, expected in cases:
    actual = _mtf_confluence_approved(base_signal, macro, internal, allow_neutral, soft)
    ok = actual == expected
    all_pass = all_pass and ok
    print(f"  {'PASS' if ok else 'FAIL'}: {base_signal} macro={macro} internal={internal} soft={soft} -> {actual} (expected {expected})")
print("MAPPING CHECK:", "ALL PASS -- symmetric, no mapping bug" if all_pass else "FAILED -- mapping bug present")

trend_lookup = {
    ts: (_normalize_state(m), _normalize_state(i))
    for ts, m, i in zip(df_enriched.index, df_enriched["macro_trend"], df_enriched["internal_trend"])
}
sorted_idx = df_enriched.index.sort_values()

def lookup(ts):
    if ts in trend_lookup:
        return trend_lookup[ts]
    pos = sorted_idx.searchsorted(ts, side="right") - 1
    if pos < 0:
        return None, None
    return trend_lookup.get(sorted_idx[pos], (None, None))

def run_test(label, soft_internal_conflict):
    print()
    print(f"=== {label} (soft_internal_conflict={soft_internal_conflict}) ===")
    mtf_filter = build_mtf_signal_filter(df_enriched, soft_internal_conflict=soft_internal_conflict)

    log = []
    def logging_filter(signal, current_time):
        base_signal = _normalize_state(getattr(signal, "signal_type", None))
        approved = mtf_filter(signal, current_time)
        macro, internal = lookup(current_time)
        if base_signal in ("LONG", "SHORT"):
            opp = "BEARISH" if base_signal == "LONG" else "BULLISH"
            macro_cat = "CONFLICT" if macro == opp else ("NEUTRAL" if macro is None else "AGREE")
            internal_cat = "CONFLICT" if internal == opp else ("NEUTRAL" if internal is None else "AGREE")
        else:
            macro_cat = internal_cat = None
        log.append({"time": current_time, "direction": base_signal, "macro_cat": macro_cat, "internal_cat": internal_cat, "approved": approved})
        return approved

    t = time.time()
    result = run_backtest(df, mtf_filter_fn=logging_filter)
    runtime = time.time() - t
    print(f"  backtest runtime: {runtime:.1f}s")

    log_df = pd.DataFrame(log)
    long_log = log_df[log_df["direction"] == "LONG"]
    short_log = log_df[log_df["direction"] == "SHORT"]

    print(f"  LONG candidates: {len(long_log)}  approved: {int(long_log['approved'].sum())}  approval_rate: {100*long_log['approved'].mean():.1f}%")
    print(f"  SHORT candidates: {len(short_log)}  approved: {int(short_log['approved'].sum())}  approval_rate: {100*short_log['approved'].mean():.1f}%")

    rejected = log_df[~log_df["approved"]]
    rejected_macro_conflict = int((rejected["macro_cat"] == "CONFLICT").sum())
    rejected_macro_neutral = int((rejected["macro_cat"] == "NEUTRAL").sum())
    rejected_internal_conflict_given_macro_agree = int(((rejected["macro_cat"] == "AGREE") & (rejected["internal_cat"] == "CONFLICT")).sum())

    print(f"  rejected by 1H conflict: {rejected_macro_conflict}")
    print(f"  rejected by 1H neutral (unestablished): {rejected_macro_neutral}")
    print(f"  rejected by 15M conflict (with 1H agreeing): {rejected_internal_conflict_given_macro_agree}")

    closed = [t for t in result.trades if t.result_status in ("WIN", "LOSS")]
    wins = [t for t in closed if t.result_status == "WIN"]
    losses = [t for t in closed if t.result_status == "LOSS"]
    wr = 100*len(wins)/len(closed) if closed else 0
    total_r = sum(t.r_multiple for t in closed)
    avg_r = total_r/len(closed) if closed else 0

    print(f"  signals={result.total_signals_generated} approved={result.total_signals_generated - result.total_mtf_rejected} rejected={result.total_mtf_rejected}")
    print(f"  closed={len(closed)} wins={len(wins)} losses={len(losses)} win_rate={wr:.2f}% total_R={total_r:.2f} avg_R={avg_r:.3f}")

    by_exit = sorted(closed, key=lambda tr: tr.exit_time)
    cum = 0.0
    peak = 0.0
    max_dd = 0.0
    cur_streak = 0
    max_streak = 0
    for tr in by_exit:
        cum += tr.r_multiple
        peak = max(peak, cum)
        dd = cum - peak
        max_dd = min(max_dd, dd)
        if tr.result_status == "LOSS":
            cur_streak += 1
            max_streak = max(max_streak, cur_streak)
        else:
            cur_streak = 0
    print(f"  max_drawdown_R={max_dd:.2f}  max_losing_streak={max_streak}")

    return result

result_a = run_test("TEST A: 1H hard + 15M hard", False)
result_b = run_test("TEST B: 1H hard + 15M soft", True)

print()
print(f"TOTAL SCRIPT RUNTIME: {time.time()-t0:.1f}s")
