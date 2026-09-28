import time
import pandas as pd

from strategy.backtest import run_backtest
from strategy.mtf_structure import build_mtf_dataset_with_structure
from strategy.swings import find_swings
from strategy.confluence import build_mtf_signal_filter, _normalize_state
from dashboard.mtf_context import MTFConfig


CSV = "real_gold_data_mt5_90000.csv"
OUT = "mtf_recovered_15m_conflict_trades.csv"

t0 = time.time()

print("Loading data...")
df = pd.read_csv(
    CSV,
    parse_dates=["timestamp"],
).set_index("timestamp").sort_index()

print(f"  {len(df)} rows in {time.time() - t0:.1f}s")

print("Building 1H + 15M MTF context (same as A/B test)...")

config = MTFConfig(
    macro_tf="1h",
    internal_tf="15min",
)

df_enriched = build_mtf_dataset_with_structure(
    df,
    macro_swings_fn=find_swings,
    internal_swings_fn=find_swings,
    config=config,
)

print(f"  built in {time.time() - t0:.1f}s")


# ---------------------------------------------------------------------
# CAUSALITY CHECK
# ---------------------------------------------------------------------

macro_close = df_enriched["macro_close_time"]
internal_close = df_enriched["internal_close_time"]

eval_time = df_enriched.index + pd.Timedelta("5min")

macro_violations = int((macro_close > eval_time).sum())
internal_violations = int((internal_close > eval_time).sum())

print(
    f"  causal check: "
    f"macro_violations={macro_violations} "
    f"internal_violations={internal_violations} "
    f"(both should be 0)"
)


# ---------------------------------------------------------------------
# MTF LOOKUP
# ---------------------------------------------------------------------

trend_lookup = {
    ts: (
        _normalize_state(macro),
        _normalize_state(internal),
    )
    for ts, macro, internal in zip(
        df_enriched.index,
        df_enriched["macro_trend"],
        df_enriched["internal_trend"],
    )
}

sorted_idx = df_enriched.index.sort_values()


def lookup(ts):
    if ts in trend_lookup:
        return trend_lookup[ts]

    pos = sorted_idx.searchsorted(ts, side="right") - 1

    if pos < 0:
        return None, None

    return trend_lookup.get(
        sorted_idx[pos],
        (None, None),
    )


# ---------------------------------------------------------------------
# CANDIDATE LOG
#
# IMPORTANT:
# We deliberately use soft_internal_conflict=True because this allows
# BOTH populations to pass through the same backtest:
#
#   Group A = 1H agrees + 15M agrees
#   Group B = 1H agrees + 15M conflicts
#
# 1H conflict remains a hard rejection.
# ---------------------------------------------------------------------

base_filter = build_mtf_signal_filter(
    df_enriched,
    soft_internal_conflict=True,
)

candidate_log = []


def logging_filter(signal, current_time):

    base_signal = _normalize_state(
        getattr(signal, "signal_type", None)
    )

    macro, internal = lookup(current_time)

    if base_signal == "LONG":
        wanted = "BULLISH"
    elif base_signal == "SHORT":
        wanted = "BEARISH"
    else:
        wanted = None

    if wanted is not None:
        if macro == wanted:
            macro_cat = "AGREE"
        elif macro is None:
            macro_cat = "NEUTRAL"
        else:
            macro_cat = "CONFLICT"

        if internal == wanted:
            internal_cat = "AGREE"
        elif internal is None:
            internal_cat = "NEUTRAL"
        else:
            internal_cat = "CONFLICT"
    else:
        macro_cat = None
        internal_cat = None

    approved = base_filter(signal, current_time)

    # FIX: TradeSignal exposes setup_timestamp, not setup_time.
    # TradeResult.setup_time (used below for trade_key) is built
    # directly from signal.setup_timestamp in backtest.py, so this
    # must match that attribute exactly or the key join breaks silently.
    setup_time = getattr(signal, "setup_timestamp", current_time)

    candidate_log.append(
        {
            "setup_time": pd.Timestamp(setup_time),
            "direction": base_signal,
            "macro_cat": macro_cat,
            "internal_cat": internal_cat,
            "approved": bool(approved),
        }
    )

    return approved


# ---------------------------------------------------------------------
# IMPORTANT:
# NO explicit PARAMS here.
#
# This intentionally uses the exact run_backtest defaults that were
# used by test_ab_1h15m.py (max_bars_to_retest=20, reward_multiple=2.0,
# stop_buffer=0.0, entry_mode="midpoint" -- these ARE the function
# defaults, so omitting them changes nothing).
# ---------------------------------------------------------------------

print()
print("Running diagnostic backtest...")
print("Using the exact run_backtest defaults from the A/B test.")
print("This may take ~45-50 minutes.")

t1 = time.time()

result = run_backtest(
    df,
    mtf_filter_fn=logging_filter,
)

print(f"  backtest runtime: {time.time() - t1:.1f}s")


# ---------------------------------------------------------------------
# CANDIDATE COUNTS
# ---------------------------------------------------------------------

log_df = pd.DataFrame(candidate_log)

total_candidates = len(log_df)
approved_candidates = int(log_df["approved"].sum())
rejected_candidates = total_candidates - approved_candidates

print()
print("=== CANDIDATE COUNTS ===")
print(f"total candidates : {total_candidates}")
print(f"approved         : {approved_candidates}")
print(f"rejected         : {rejected_candidates}")

if total_candidates != 740:
    print(
        f"WARNING: expected 740 candidates from the A/B test, "
        f"but observed {total_candidates}"
    )


# ---------------------------------------------------------------------
# GROUP CANDIDATES
# ---------------------------------------------------------------------

group_a_candidates = log_df[
    (log_df["macro_cat"] == "AGREE") &
    (log_df["internal_cat"] == "AGREE")
].copy()

group_b_candidates = log_df[
    (log_df["macro_cat"] == "AGREE") &
    (log_df["internal_cat"] == "CONFLICT")
].copy()

print()
print("=== MTF POPULATIONS ===")
print(
    f"GROUP A: 1H agree + 15M agree       : "
    f"{len(group_a_candidates)} candidates"
)

print(
    f"GROUP B: 1H agree + 15M conflict    : "
    f"{len(group_b_candidates)} candidates"
)

print()
print("GROUP B direction:")
print(
    group_b_candidates["direction"]
    .value_counts(dropna=False)
    .to_string()
)


# ---------------------------------------------------------------------
# STABLE TRADE KEYS
# ---------------------------------------------------------------------

def normalize_trade_direction(trade):
    return _normalize_state(
        getattr(trade.direction, "name", str(trade.direction))
    )


def trade_key(trade):
    return (
        pd.Timestamp(trade.setup_time),
        normalize_trade_direction(trade),
    )


group_a_keys = {
    (
        pd.Timestamp(row.setup_time),
        row.direction,
    )
    for _, row in group_a_candidates[
        group_a_candidates["approved"]
    ].iterrows()
}

group_b_keys = {
    (
        pd.Timestamp(row.setup_time),
        row.direction,
    )
    for _, row in group_b_candidates[
        group_b_candidates["approved"]
    ].iterrows()
}


# ---------------------------------------------------------------------
# GROUP TRADES
# ---------------------------------------------------------------------

group_a_trades = []
group_b_trades = []
unmatched_trades = []

for trade in result.trades:

    key = trade_key(trade)

    if key in group_a_keys:
        group_a_trades.append(trade)

    elif key in group_b_keys:
        group_b_trades.append(trade)

    else:
        unmatched_trades.append(trade)


if unmatched_trades:
    print(
        f"\nWARNING: {len(unmatched_trades)} returned trades "
        f"could not be matched to an MTF candidate."
    )


# ---------------------------------------------------------------------
# GROUP STATISTICS
# ---------------------------------------------------------------------

def group_stats(trades, label):

    closed = [
        t for t in trades
        if t.result_status in ("WIN", "LOSS")
    ]

    wins = [
        t for t in closed
        if t.result_status == "WIN"
    ]

    losses = [
        t for t in closed
        if t.result_status == "LOSS"
    ]

    longs = sum(
        1 for t in trades
        if normalize_trade_direction(t) == "LONG"
    )

    shorts = sum(
        1 for t in trades
        if normalize_trade_direction(t) == "SHORT"
    )

    win_rate = (
        100 * len(wins) / len(closed)
        if closed else 0
    )

    total_r = sum(
        t.r_multiple
        for t in closed
    )

    avg_r = (
        total_r / len(closed)
        if closed else 0
    )

    by_exit = sorted(
        closed,
        key=lambda t: t.exit_time,
    )

    cumulative = 0.0
    peak = 0.0
    max_dd = 0.0

    current_streak = 0
    max_streak = 0

    for trade in by_exit:

        cumulative += trade.r_multiple
        peak = max(peak, cumulative)

        max_dd = min(
            max_dd,
            cumulative - peak,
        )

        if trade.result_status == "LOSS":
            current_streak += 1
            max_streak = max(
                max_streak,
                current_streak,
            )
        else:
            current_streak = 0

    print()
    print(f"=== {label} ===")
    print(f"trades returned      : {len(trades)}")
    print(f"LONG                 : {longs}")
    print(f"SHORT                : {shorts}")
    print(f"closed               : {len(closed)}")
    print(f"wins                 : {len(wins)}")
    print(f"losses               : {len(losses)}")
    print(f"win_rate             : {win_rate:.2f}%")
    print(f"total_R              : {total_r:.2f}")
    print(f"avg_R                : {avg_r:.3f}")
    print(f"max_drawdown_R       : {max_dd:.2f}")
    print(f"max_losing_streak    : {max_streak}")

    return {
        "trades": len(trades),
        "longs": longs,
        "shorts": shorts,
        "closed": len(closed),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": win_rate,
        "total_r": total_r,
        "avg_r": avg_r,
        "max_dd": max_dd,
        "max_streak": max_streak,
    }


stats_a = group_stats(
    group_a_trades,
    "GROUP A: 1H agree + 15M agree",
)

stats_b = group_stats(
    group_b_trades,
    "GROUP B: 1H agree + 15M conflict (RECOVERED)",
)


# ---------------------------------------------------------------------
# SAVE RECOVERED TRADES
# ---------------------------------------------------------------------

rows = []

for trade in group_b_trades:

    rows.append(
        {
            "setup_time": trade.setup_time,
            "entry_time": trade.entry_time,
            "exit_time": trade.exit_time,
            "direction": normalize_trade_direction(trade),
            "status": trade.result_status,
            "r": trade.r_multiple,
            "bars_held": trade.bars_held,
            "session": trade.session,
        }
    )

pd.DataFrame(rows).to_csv(
    OUT,
    index=False,
)

print()
print(
    f"saved {len(rows)} recovered trades to {OUT}"
)


# ---------------------------------------------------------------------
# FINAL SUMMARY
# ---------------------------------------------------------------------

print()
print("=== SUMMARY ===")

print(
    "A) 1H-agree + 15M-agree: "
    f"closed={stats_a['closed']} "
    f"win_rate={stats_a['win_rate']:.2f}% "
    f"avg_R={stats_a['avg_r']:.3f} "
    f"total_R={stats_a['total_r']:.2f}"
)

print(
    "B) 1H-agree + 15M-conflict: "
    f"closed={stats_b['closed']} "
    f"win_rate={stats_b['win_rate']:.2f}% "
    f"avg_R={stats_b['avg_r']:.3f} "
    f"total_R={stats_b['total_r']:.2f}"
)

wr_diff = (
    stats_b["win_rate"]
    - stats_a["win_rate"]
)

avg_r_diff = (
    stats_b["avg_r"]
    - stats_a["avg_r"]
)

print(
    f"C) recovered-group difference: "
    f"win_rate={wr_diff:+.2f} pts "
    f"avg_R={avg_r_diff:+.3f}"
)

print()
print(
    "NOTE: This is a diagnostic experiment only. "
    "It does not change the strategy or permanently select "
    "hard/soft 15M behavior."
)

print()
print(
    f"TOTAL SCRIPT RUNTIME: "
    f"{time.time() - t0:.1f}s"
)
