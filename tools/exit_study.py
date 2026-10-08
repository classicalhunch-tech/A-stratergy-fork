"""
tools/exit_study.py

One backtest, every exit.

The engine is run ONCE with a target so far away that only the stop ends a
trade (market on close, fill-anchored, no MTF gate; the 1H/15M pattern label
is attached as a tag). For every trade the tool then replays the 5M bars
after the fill and records how far price ran in the trade's favor BEFORE the
stop was hit (the "favorable run", in R).

With that one number per trade, the result of ANY fixed-R target, a scaled
exit (part at one target, the rest at another, stop unchanged) or a
structural 15M-swing target can be computed on the same trades without
re-running the strategy.

Why this is exact: in run_backtest() a trade's fate depends only on which of
stop / target is touched first (the stop wins a same-bar tie), and the set of
trades does not depend on the target. "Target T hit before the stop" is
therefore the same event as "favorable run >= T before the stop".

What it cannot do: exits that MOVE the stop (break-even after a partial)
depend on the order of events after the partial. The favorable run does not
carry that, so those need an engine change.

Costs are applied after the fact: net = gross - cost / risk (price units).
"""

import argparse
import math
import sys
import time
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PRICE_EPS = 1e-8
R_EPS = 1e-9

FAR_TARGET_R = 1000.0
DEFAULT_COST = 0.5
MIN_TRADES = 20
FOCUS = "Z_against_both"

TARGETS = [1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 13.0]
SCALED = [(1.5, 5.0), (2.0, 5.0), (1.5, 8.0), (3.0, 8.0)]
REF_T = 2.0
SPLIT_TARGETS = [1.5, 3.0, 5.0]


# =====================================================================
# Data
# =====================================================================

def load_ohlc(path):
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.columns = [str(c).strip().lower() for c in df.columns]
    df = df[df.index.notna()].sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df[["open", "high", "low", "close"]].astype(float).dropna()


# =====================================================================
# Pure math (tested against a brute-force replay)
# =====================================================================

def favorable_run_r(is_long, fill, stop, highs, lows):
    """
    highs/lows start AT the fill bar. Returns (run_r, stopped) or None.

    run_r   : best favorable excursion, in R, reached before the stop bar.
              The stop bar itself is excluded (stop wins a same-bar tie).
    stopped : True if the stop was touched somewhere in the data.
    """

    risk = abs(fill - stop)

    if risk <= PRICE_EPS:
        return None

    if is_long:
        hit = lows <= stop + PRICE_EPS
    else:
        hit = highs >= stop - PRICE_EPS

    if bool(hit.any()):
        stop_i = int(hit.argmax())
        stopped = True
    else:
        stop_i = len(highs)
        stopped = False

    if stop_i == 0:
        return 0.0, stopped

    if is_long:
        run = float(highs[:stop_i].max()) - fill
    else:
        run = fill - float(lows[:stop_i].min())

    return max(run, 0.0) / risk, stopped


def outcome_r(run_r, stopped, target_r):
    """
    Gross R of a fixed target: +target if reached before the stop, -1 if the
    stop came first, None if neither happened before the data ended.
    """

    if run_r >= target_r - R_EPS:
        return float(target_r)

    if stopped:
        return -1.0

    return None


def scaled_outcome_r(run_r, stopped, t1, t2, frac1=0.5):
    """
    frac1 of the position exits at t1, the rest at t2, stop never moved.
    None if either half is still unresolved when the data ends.
    """

    a = outcome_r(run_r, stopped, t1)
    b = outcome_r(run_r, stopped, t2)

    if a is None or b is None:
        return None

    return frac1 * a + (1.0 - frac1) * b


def mean_se(values):
    n = len(values)

    if n == 0:
        return 0, float("nan"), float("nan")

    mean = sum(values) / n

    if n < 2:
        return n, mean, float("nan")

    var = sum((v - mean) ** 2 for v in values) / (n - 1)

    return n, mean, math.sqrt(var / n)


def z_needed(looks):
    """Rough two-sided 5% threshold after `looks` comparisons (conservative)."""

    looks = max(int(looks), 1)

    return NormalDist().inv_cdf(1.0 - 0.025 / looks)


# =====================================================================
# Formatting
# =====================================================================

def f_r(x):
    return "   n/a" if x != x else f"{x:+.3f}"


def f_se(x):
    return "  n/a" if x != x else f"{x:.3f}"


def f_t(mean, se):
    if se != se or se <= 0:
        return "  n/a"
    return f"{mean / se:+.2f}"


def quarter_label(ts):
    ts = pd.Timestamp(ts)
    return f"{ts.year}Q{(ts.month - 1) // 3 + 1}"


# =====================================================================
# Trades -> rows
# =====================================================================

def collect_rows(df, trades, skip_fn=None, advance_fn=None):
    """
    One row per triggered trade of the far-target run.

    skip_fn / advance_fn: structural target functions
    (direction, fill, stop, fill_time) -> price or None.
    """

    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)

    rows = []

    for t in trades:

        if t.initial_risk <= 0 or t.entry_time is None:
            continue

        entry_ts = pd.Timestamp(t.entry_time)

        idx = df.index.get_loc(entry_ts)

        if not isinstance(idx, (int, np.integer)):
            continue

        fill = float(t.fill_price)
        stop = float(t.stop_loss)

        direction = str(getattr(t.direction, "value", t.direction)).upper()
        is_long = direction == "LONG"

        result = favorable_run_r(is_long, fill, stop, highs[idx:], lows[idx:])

        if result is None:
            continue

        run_r, stopped = result

        risk = abs(fill - stop)

        row = {
            "label": getattr(t.signal, "pattern", "U_unknown"),
            "direction": direction,
            "entry_time": entry_ts,
            "quarter": quarter_label(entry_ts),
            "fill": fill,
            "stop": stop,
            "risk": risk,
            "run_r": run_r,
            "stopped": stopped,
            "engine_stopped": t.result_status == "LOSS",
            "skip_rr": None,
            "adv_rr": None,
        }

        for key, fn in (("skip_rr", skip_fn), ("adv_rr", advance_fn)):

            if fn is None:
                continue

            tp = fn(t.direction, fill, stop, entry_ts)

            if tp is not None:
                row[key] = abs(float(tp) - fill) / risk

        rows.append(row)

    return rows


# =====================================================================
# Report sections
# =====================================================================

def fixed_net(rows, target, cost):
    values = []
    wins = 0

    for r in rows:
        g = outcome_r(r["run_r"], r["stopped"], target)

        if g is None:
            continue

        values.append(g - cost / r["risk"])

        if g > 0:
            wins += 1

    return values, wins


def paired_gross(rows, fn_a, fn_b):
    diffs = []

    for r in rows:
        a = fn_a(r)
        b = fn_b(r)

        if a is None or b is None:
            continue

        diffs.append(a - b)

    return diffs


def section_curve(cohorts, cost):

    print()
    print("=" * 72)
    print("1. EXPECTANCY BY FIXED TARGET (same trades, stop never changes)")
    print("=" * 72)

    looks = 0

    for name, rows in cohorts:

        print()
        print(f"{name}  ({len(rows)} trades)")
        print("   target    n  win%   net R     SE      t")

        for target in TARGETS:

            values, wins = fixed_net(rows, target, cost)
            n, mean, se = mean_se(values)

            looks += 1

            if n == 0:
                print(f"   {target:5.1f}R    0")
                continue

            flag = " *" if n < MIN_TRADES else ""

            print(
                f"   {target:5.1f}R {n:4d} {100.0 * wins / n:5.1f} "
                f"{f_r(mean)}  {f_se(se)}  {f_t(mean, se)}{flag}"
            )

    print()
    print(
        f"   You are looking at {looks} cells. After correcting for that, a "
        f"t of about {z_needed(looks):.1f} is needed, not 2."
    )
    print("   (Conservative: cells share trades, so they are not independent.)")


def section_paired(cohorts):

    print()
    print("=" * 72)
    print(f"2. EACH TARGET MINUS THE CURRENT {REF_T:.0f}R (same trades, paired)")
    print("=" * 72)

    for name, rows in cohorts:

        if len(rows) < 10:
            continue

        print()
        print(f"{name}")
        print("   target    n   diff R     SE      t")

        for target in TARGETS:

            if abs(target - REF_T) < 1e-9:
                continue

            diffs = paired_gross(
                rows,
                lambda r, t=target: outcome_r(r["run_r"], r["stopped"], t),
                lambda r: outcome_r(r["run_r"], r["stopped"], REF_T),
            )

            n, mean, se = mean_se(diffs)

            if n == 0:
                continue

            print(
                f"   {target:5.1f}R {n:4d} {f_r(mean)}  {f_se(se)}  {f_t(mean, se)}"
            )


def section_scaled(cohorts, cost):

    print()
    print("=" * 72)
    print("3. SCALED EXITS: half at the first target, half at the second")
    print("   (stop never moved; compared with a plain 5R on the same trades)")
    print("=" * 72)

    for name, rows in cohorts:

        if name not in ("ALL", FOCUS) or not rows:
            continue

        print()
        print(f"{name}")
        print("   scheme            n   net R     SE   vs plain 5R   SE")

        for t1, t2 in SCALED:

            values = []

            for r in rows:
                g = scaled_outcome_r(r["run_r"], r["stopped"], t1, t2)

                if g is not None:
                    values.append(g - cost / r["risk"])

            n, mean, se = mean_se(values)

            diffs = paired_gross(
                rows,
                lambda r, a=t1, b=t2: scaled_outcome_r(r["run_r"], r["stopped"], a, b),
                lambda r: outcome_r(r["run_r"], r["stopped"], 5.0),
            )

            _, d_mean, d_se = mean_se(diffs)

            print(
                f"   {t1:g}R + {t2:g}R   {n:4d} {f_r(mean)}  {f_se(se)}   "
                f"{f_r(d_mean)}  {f_se(d_se)}"
            )


def section_structure(cohorts, cost):

    print()
    print("=" * 72)
    print("4. STRUCTURAL 15M-SWING TARGET vs FIXED TARGETS (same trades, paired)")
    print("=" * 72)

    for name, rows in cohorts:

        if name not in ("ALL", FOCUS) or not rows:
            continue

        for label, key in (("skip if < min R", "skip_rr"), ("advance to min R", "adv_rr")):

            used = [r for r in rows if r[key] is not None]

            skipped = len(rows) - len(used)

            if not used:
                print(f"\n{name} / {label}: no trades have a structural target")
                continue

            values = []
            for r in used:
                g = outcome_r(r["run_r"], r["stopped"], r[key])

                if g is not None:
                    values.append(g - cost / r["risk"])

            n, mean, se = mean_se(values)

            fixed2, _ = fixed_net(used, 2.0, cost)
            fixed5, _ = fixed_net(used, 5.0, cost)

            _, m2, _ = mean_se(fixed2)
            _, m5, _ = mean_se(fixed5)

            diffs = paired_gross(
                used,
                lambda r, k=key: outcome_r(r["run_r"], r["stopped"], r[k]),
                lambda r: outcome_r(r["run_r"], r["stopped"], 5.0),
            )

            _, d_mean, d_se = mean_se(diffs)

            planned = sorted(r[key] for r in used)
            median_rr = planned[len(planned) // 2]

            flag = " *" if n < MIN_TRADES else ""

            print()
            print(f"{name} / {label}")
            print(
                f"   trades with a target {len(used)}, rule skipped {skipped}, "
                f"median planned RR {median_rr:.2f}{flag}"
            )
            print(
                f"   structural net {f_r(mean)} (SE {f_se(se)})   "
                f"fixed 2R {f_r(m2)}   fixed 5R {f_r(m5)}   "
                f"structural minus 5R {f_r(d_mean)} (SE {f_se(d_se)})"
            )


def section_overlap(rows, cost):

    print()
    print("=" * 72)
    print("5. WHICH CARRIES THE EDGE: THE 1.5R ROOM RULE OR THE Z LABEL?")
    print("   room rule = the spec's own 'nearest 15M swing must be >= min R away'")
    print("=" * 72)

    z_rows = [r for r in rows if r["label"] == FOCUS]
    other_rows = [r for r in rows if r["label"] != FOCUS]

    for target in (2.0, 5.0):

        print()
        print(f"Fixed {target:g}R, net of cost")
        print("   group                        n    net R     SE")

        cells = {}

        for zname, zrows in (("Z", z_rows), ("not Z", other_rows)):
            for rname, keep in (("room rule passes", True), ("room rule fails", False)):

                grp = [r for r in zrows if (r["skip_rr"] is not None) == keep]
                values, _ = fixed_net(grp, target, cost)
                n, mean, se = mean_se(values)
                cells[(zname, keep)] = (n, mean, se)

                flag = " *" if n < MIN_TRADES else ""

                print(
                    f"   {zname:6s} / {rname:16s} {n:4d} {f_r(mean)}  {f_se(se)}{flag}"
                )

        def diff_line(label, a, b):
            if a[0] < 2 or b[0] < 2:
                return
            d = a[1] - b[1]
            se = math.sqrt(a[2] ** 2 + b[2] ** 2)
            print(f"   {label}: {f_r(d)} (SE {f_se(se)}, t {f_t(d, se)})")

        all_pass = [r for r in rows if r["skip_rr"] is not None]
        all_fail = [r for r in rows if r["skip_rr"] is None]

        vp, _ = fixed_net(all_pass, target, cost)
        vf, _ = fixed_net(all_fail, target, cost)

        diff_line(
            "room rule passes minus fails (all trades)",
            mean_se(vp),
            mean_se(vf),
        )
        diff_line(
            "Z minus not Z (room rule passes)",
            cells[("Z", True)],
            cells[("not Z", True)],
        )
        diff_line(
            "Z minus not Z (room rule fails) ",
            cells[("Z", False)],
            cells[("not Z", False)],
        )


def section_split(cohorts, cost):

    print()
    print("=" * 72)
    print("6. IS THE EDGE STABLE? (direction and quarter)")
    print("=" * 72)

    for name, rows in cohorts:

        if name not in ("ALL", FOCUS) or not rows:
            continue

        groups = []

        for d in ("LONG", "SHORT"):
            groups.append((f"{d}", [r for r in rows if r["direction"] == d]))

        for q in sorted({r["quarter"] for r in rows}):
            groups.append((q, [r for r in rows if r["quarter"] == q]))

        print()
        print(f"{name}")
        header = "   group      n"

        for t in SPLIT_TARGETS:
            header += f"   net@{t:g}R (SE)"

        print(header)

        for gname, grows in groups:

            if not grows:
                continue

            line = f"   {gname:8s} {len(grows):4d}"

            for t in SPLIT_TARGETS:
                values, _ = fixed_net(grows, t, cost)
                _, mean, se = mean_se(values)
                line += f"   {f_r(mean)} ({f_se(se)})"

            if len(grows) < MIN_TRADES:
                line += " *"

            print(line)


# =====================================================================
# Main
# =====================================================================

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument(
        "--rows", type=int, default=40000,
        help="Use the last N candles (default 40000, the same as the other "
             "reports). 0 = all.",
    )
    parser.add_argument(
        "--which", choices=["tail", "head"], default="tail",
        help="tail = the last N candles (default); head = the first N candles. "
             "With --rows 40000 on the 80000-row file, head is the half the "
             "other reports never use.",
    )
    parser.add_argument("--cost", type=float, default=DEFAULT_COST)
    parser.add_argument("--output", default=None, help="Optional per-trade CSV.")
    args = parser.parse_args()

    # Heavy imports live here so the pure math above can be tested alone.
    from dashboard.mtf_context import MTFConfig
    from strategy.backtest import run_backtest
    from strategy.mtf_structure import build_mtf_dataset_with_structure
    from strategy.pattern_tags import PATTERN_LABELS, build_pattern_tagger
    from strategy.structure_targets import (
        StructureTargetConfig,
        build_structure_target_fn,
    )
    from strategy.swings import find_swings

    df = load_ohlc(Path(args.data))

    if args.rows > 0:
        df = df.head(args.rows) if args.which == "head" else df.tail(args.rows)

    first_close = float(df["close"].iloc[0])
    last_close = float(df["close"].iloc[-1])

    print(f"Using {len(df)} candles: {df.index[0]} -> {df.index[-1]}")
    print(
        f"Price over the sample: {first_close:.2f} -> {last_close:.2f} "
        f"({100.0 * (last_close / first_close - 1.0):+.1f}%)"
    )

    started = time.perf_counter()

    enriched = build_mtf_dataset_with_structure(
        df,
        macro_swings_fn=find_swings,
        internal_swings_fn=find_swings,
        config=MTFConfig(macro_tf="1h", internal_tf="15min"),
    )
    tagger = build_pattern_tagger(enriched)

    def tagging_filter(signal, current_time):
        signal.pattern = tagger(signal, current_time)
        return True

    skip_fn = build_structure_target_fn(df, config=StructureTargetConfig())
    advance_fn = build_structure_target_fn(
        df, config=StructureTargetConfig(advance_to_min_rr=True)
    )

    result = run_backtest(
        df,
        reward_multiple=FAR_TARGET_R,
        mtf_filter_fn=tagging_filter,
        fill_mode="market_on_close",
        target_anchor="fill",
    )

    if result.errors:
        print(f"WARNING: {len(result.errors)} engine errors; first: {result.errors[0]}")

    rows = collect_rows(df, result.trades, skip_fn=skip_fn, advance_fn=advance_fn)

    print(
        f"Engine: candidates {result.total_signals_generated}, "
        f"triggered {result.total_trades_triggered}, "
        f"invalidated {result.total_invalidated}, expired {result.total_expired}"
    )
    keys = [(r["entry_time"], r["direction"], round(r["stop"], 4)) for r in rows]
    duplicates = len(keys) - len(set(keys))

    print(f"Trades analysed: {len(rows)}")
    print(
        f"Trades sharing entry time, direction and stop with another trade: "
        f"{duplicates} (they are counted twice in every n and SE below)"
    )

    mismatches = sum(1 for r in rows if r["stopped"] != r["engine_stopped"])
    censored = sum(1 for r in rows if not r["stopped"])

    print(
        f"Replay vs engine stop mismatches: {mismatches} (must be 0). "
        f"Trades never stopped before the data ended: {censored}."
    )

    # Reconciliation with tools/pattern_report.py (fill anchor, same --rows).
    print()
    print("Reconciliation: ALL at fixed 2R and 5R should equal tools/pattern_report.py")
    print("(target_anchor=fill) when --rows is the same:")
    for t in (2.0, 5.0):
        values, wins = fixed_net(rows, t, 0.0)
        n, mean, _ = mean_se(values)
        if n:
            print(
                f"   {t:.0f}R: {n} trades, win {100.0 * wins / n:.1f}%, gross {mean:+.3f}R"
            )

    labels = [lab for lab in PATTERN_LABELS if any(r["label"] == lab for r in rows)]

    cohorts = [("ALL", rows)] + [
        (lab, [r for r in rows if r["label"] == lab]) for lab in labels
    ]

    section_curve(cohorts, args.cost)
    section_paired(cohorts)
    section_scaled(cohorts, args.cost)
    section_structure(cohorts, args.cost)
    section_overlap(rows, args.cost)
    section_split(cohorts, args.cost)

    if args.output:
        out = pd.DataFrame(rows)
        out.to_csv(args.output, index=False)
        print(f"\nSaved {len(out)} trades to {args.output}")

    print()
    print(f"* = fewer than {MIN_TRADES} trades: treat as noise")
    print(f"Total runtime: {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
