"""
tools/z_vs_room.py

Is the edge in the Z label, or in the room rule?

Z label : the signal is against BOTH the 1H and the 15M trend
          (strategy.pattern_tags, "Z_against_both").
Room    : the nearest confirmed 15M swing beyond the fill is at least T x the
          trade's real risk away. This is the same swing, with the same
          look-ahead rule and the same max_age, that the structural exit uses
          (strategy.structure_targets). T = 1.5 is the spec's own room rule.

The engine is run ONCE with a target so far away that only the stop ends a
trade (same set-up as tools/exit_study.py: market on close, fill-anchored,
no MTF gate, the 1H/15M label attached as a tag). Every trade then gets:

    label    A / B / X / Z, from the tagger
    room_r   distance to the nearest confirmed 15M swing, in R

and its outcome at a FIXED target (default 2R and 5R). The exit is fixed on
purpose. With the structural exit, room is also the target distance, so the
room rule and the exit would be the same thing and could not be told apart.

For each room threshold T the trades fall into four cells:

    Z / room >= T     Z / room < T     not Z / room >= T     not Z / room < T

Cells hold different trades, so they are compared with a Welch (unpaired)
difference, not a paired one. A pooled regression (net R on Z and room-pass)
gives the Z effect with room held constant and the room effect with Z held
constant, using every trade at once. It is printed three ways:

    HC1                     trade-level robust SE (the original line)
    week-clustered          SE clustered by ISO week: trades opened in the same
                            days share one price path, so they are not
                            independent. This is the line to decide on.
    + stop-size control     the same, with log(stop distance) added. Room in R
                            is distance-to-swing divided by the stop, so a
                            tight stop raises room mechanically. If the room
                            effect disappears here, "room" is a stop-size
                            effect.

Identical trades (same entry time, direction and stop) are counted once.

Run on --which head and --which tail (or --which both) and compare.
Costs are applied after the fact: net = gross - cost / risk (price units).
"""

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import exit_study as es  # noqa: E402

FOCUS = es.FOCUS
FAR_TARGET_R = es.FAR_TARGET_R
DEFAULT_COST = es.DEFAULT_COST
MIN_TRADES = es.MIN_TRADES

THRESHOLDS = [1.0, 1.5, 2.0, 3.0, 5.0]
TARGETS = [2.0, 5.0]
T_CRIT = 2.0

# Any positive value works: the structural helper then returns the nearest
# confirmed swing beyond the fill whatever its distance, which is the raw room.
TINY_MIN_RR = 1e-9
ROOM_EPS = 1e-9

# Cluster-robust SEs are unreliable (often far too small) with few clusters.
# Below this many weeks the pooled line falls back to the trade-level HC1 SE
# and says so.
MIN_CLUSTERS = 15

NAN = float("nan")


# =====================================================================
# Pure math (tested without the engine)
# =====================================================================

def room_passes(room_r, threshold):
    """True if there is a swing and it is at least `threshold` R away."""

    return room_r is not None and room_r >= threshold - ROOM_EPS


def week_label(ts):
    """ISO year-week of a timestamp, used as the cluster key."""

    iso = pd.Timestamp(ts).isocalendar()

    return f"{int(iso[0])}-W{int(iso[1]):02d}"


def dedupe_rows(rows):
    """
    Drops trades that repeat another trade's entry time, direction and stop
    (to 4 decimals). Such trades are the same position counted twice, which
    shrinks every SE. Returns (kept rows, number dropped).
    """

    seen = set()
    kept = []

    for r in rows:
        key = (r["entry_time"], r["direction"], round(r["stop"], 4))

        if key in seen:
            continue

        seen.add(key)
        kept.append(r)

    return kept, len(rows) - len(kept)


def resolved_trades(rows, target, cost):
    """
    One dict per trade whose fixed-target outcome is known (the target or the
    stop was reached before the data ended). Trades still open at the end of
    the data are left out, exactly as tools/exit_study.py does.
    """

    out = []

    for r in rows:
        g = es.outcome_r(r["run_r"], r["stopped"], target)

        if g is None:
            continue

        out.append(
            {
                "net": g - cost / r["risk"],
                "win": g > 0,
                "z": r["label"] == FOCUS,
                "room": r["room_r"],
                "risk": r["risk"],
                "week": week_label(r["entry_time"]),
                # carried through so callers can split the trades afterwards
                "quarter": r.get("quarter"),
                "direction": r.get("direction"),
                "window": r.get("window"),
            }
        )

    return out


def cell_stats(trades):
    """(n, win %, mean net R, SE) for a list of resolved trades."""

    n, mean, se = es.mean_se([t["net"] for t in trades])
    wins = sum(1 for t in trades if t["win"])
    win_pct = 100.0 * wins / n if n else NAN

    return n, win_pct, mean, se


def welch_diff(a, b, min_n=2):
    """
    Difference of two independent cell means.
    a, b: (n, win %, mean, SE). Returns (diff, SE, t); NaNs if either cell has
    fewer than `min_n` trades (at least 2) or no spread. A cell of 2 to 5
    trades can have a near-zero SE by luck, which makes t meaningless, so
    callers working with small groups should raise min_n.
    """

    min_n = max(int(min_n), 2)

    if a[0] < min_n or b[0] < min_n or a[3] != a[3] or b[3] != b[3]:
        return NAN, NAN, NAN

    diff = a[2] - b[2]
    se = math.sqrt(a[3] ** 2 + b[3] ** 2)

    if se <= 0:
        return diff, se, NAN

    return diff, se, diff / se


def ols_robust(X, y, groups=None):
    """
    OLS coefficients and robust standard errors.

    groups None  : HC1 (trade-level), scaled by n / (n - k).
    groups given : cluster-robust (CR1), scaled by G/(G-1) * (n-1)/(n-k).
                   With every trade its own cluster this equals HC1.
    Falls back to HC1 when there are fewer than 2 clusters.
    """

    n, k = X.shape

    xtx_inv = np.linalg.inv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    resid = y - X @ beta

    labels = None if groups is None else sorted(set(groups))

    if labels is None or len(labels) < 2:
        meat = (X * (resid ** 2)[:, None]).T @ X
        cov = xtx_inv @ meat @ xtx_inv * (n / (n - k))
    else:
        index = {g: [] for g in labels}

        for i, g in enumerate(groups):
            index[g].append(i)

        meat = np.zeros((k, k))

        for rows in index.values():
            s = X[rows].T @ resid[rows]
            meat += np.outer(s, s)

        g_count = len(labels)
        cov = (
            xtx_inv @ meat @ xtx_inv
            * (g_count / (g_count - 1))
            * ((n - 1) / (n - k))
        )

    return beta, np.sqrt(np.diag(cov))


def pooled_effects(trades, threshold, cluster=True, control=False):
    """
    OLS of net R on [1, Z, room-pass] (plus log stop distance if `control`).

    cluster=True clusters the SE by ISO week (needs trades from
    resolved_trades); cluster=False is the trade-level HC1 line.

    Returns {"z": (coef, se), "room": (coef, se), "clusters": G or None,
    "note": text or None} or None when the model cannot be estimated (too few
    trades, or Z or room never varies). With fewer than MIN_CLUSTERS weeks the
    clustered request falls back to HC1 and "note" says so.
    """

    n = len(trades)
    k = 4 if control else 3

    if n <= k + 1:
        return None

    y = np.array([t["net"] for t in trades], dtype=float)
    z = np.array([1.0 if t["z"] else 0.0 for t in trades])
    p = np.array(
        [1.0 if room_passes(t["room"], threshold) else 0.0 for t in trades]
    )

    columns = [np.ones(n), z, p]

    if control:
        columns.append(np.log(np.array([t["risk"] for t in trades], dtype=float)))

    X = np.column_stack(columns)

    if np.linalg.matrix_rank(X) < k:
        return None

    groups = [t["week"] for t in trades] if cluster else None
    note = None

    if groups is not None and len(set(groups)) < MIN_CLUSTERS:
        note = (
            f"only {len(set(groups))} weeks (< {MIN_CLUSTERS}): "
            f"trade-level HC1 used instead"
        )
        groups = None

    beta, se = ols_robust(X, y, groups)

    return {
        "z": (float(beta[1]), float(se[1])),
        "room": (float(beta[2]), float(se[2])),
        "clusters": len(set(groups)) if groups is not None else None,
        "note": note,
    }


def _sig(t, crit):
    return t == t and t > crit


def verdict(t_z_pass, t_room_z, t_room_notz, crit=T_CRIT):
    """
    Maps the three key t-stats to the handoff's outcomes.

        t_z_pass     Z pass minus notZ pass
        t_room_z     room pass minus fail, within Z
        t_room_notz  room pass minus fail, within not Z

    Only a positive t above `crit` counts as evidence.
    """

    z = _sig(t_z_pass, crit)
    rz = _sig(t_room_z, crit)
    rn = _sig(t_room_notz, crit)

    if z and rn:
        return "BOTH matter (outcome 3)"

    if z:
        return "Z is the edge (outcome 1)"

    if rz and rn:
        return "ROOM is the edge (outcome 2)"

    if rz or rn:
        return "room leans, one side only (not conclusive)"

    return "NEITHER distinguishable: sample too small (outcome 4)"


def significant_notes(t_stats, pooled, crit=T_CRIT):
    """
    Every comparison whose |t| clears `crit`, in EITHER direction.

    verdict() only credits positive evidence, so a significantly NEGATIVE
    result (for example Z losing money) would otherwise read as "nothing
    here". t_stats: {label: t}; pooled: pooled_effects() output or None.
    """

    notes = []

    for label, t in t_stats.items():
        if t == t and abs(t) > crit:
            notes.append(f"{label} t {t:+.2f}")

    if pooled:
        for key, label in (("z", "pooled Z effect"), ("room", "pooled room effect")):
            coef, se = pooled[key]

            if se > 0 and abs(coef / se) > crit:
                notes.append(f"{label} t {coef / se:+.2f}")

    return notes


def sign_flips(pooled_a, pooled_b):
    """
    Names of the pooled effects ("Z", "room") whose sign differs between two
    windows. Empty if either window could not be estimated.
    """

    if not pooled_a or not pooled_b:
        return []

    flips = []

    for key, name in (("z", "Z"), ("room", "room")):
        if (pooled_a[key][0] > 0) != (pooled_b[key][0] > 0):
            flips.append(name)

    return flips


# =====================================================================
# Formatting
# =====================================================================

def f_n(x):
    return "   n/a" if x != x else f"{x:5.1f}"


def f_diff(label, d):
    diff, se, t = d

    if diff != diff:
        return f"   {label}: n/a (a cell has fewer than 2 trades)"

    t_txt = "n/a" if t != t else f"{t:+.2f}"

    return f"   {label}: {es.f_r(diff)} (SE {es.f_se(se)}, t {t_txt})"


def f_pooled_line(name, pooled):
    if pooled is None:
        return f"   {name}: not estimable (Z or room never varies)"

    zc, zs = pooled["z"]
    rc, rs = pooled["room"]

    clusters = pooled.get("clusters")
    extra = "" if clusters is None else f" [{clusters} weeks]"

    if pooled.get("note"):
        extra += f" ({pooled['note']})"

    return (
        f"   {name}{extra}:  Z effect (room held fixed) "
        f"{es.f_r(zc)} (SE {es.f_se(zs)}, t {es.f_t(zc, zs)})   "
        f"room effect (Z held fixed) {es.f_r(rc)} "
        f"(SE {es.f_se(rs)}, t {es.f_t(rc, rs)})"
    )


# =====================================================================
# Engine
# =====================================================================

def run_far_target(df):
    """
    The same one-run set-up as tools/exit_study.py. Returns (rows, result).
    Each row is exit_study's row plus "room_r" (R to the nearest confirmed 15M
    swing, None if there is none within max_age).
    """

    # Heavy imports live here so the pure math above can be tested alone.
    from dashboard.mtf_context import MTFConfig
    from strategy.backtest import run_backtest
    from strategy.mtf_structure import build_mtf_dataset_with_structure
    from strategy.pattern_tags import build_pattern_tagger
    from strategy.structure_targets import (
        StructureTargetConfig,
        build_structure_target_fn,
    )
    from strategy.swings import find_swings

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

    room_fn = build_structure_target_fn(
        df, config=StructureTargetConfig(min_rr=TINY_MIN_RR)
    )

    result = run_backtest(
        df,
        reward_multiple=FAR_TARGET_R,
        mtf_filter_fn=tagging_filter,
        fill_mode="market_on_close",
        target_anchor="fill",
    )

    rows = es.collect_rows(df, result.trades, skip_fn=room_fn, advance_fn=None)

    for r in rows:
        r["room_r"] = r.pop("skip_rr")
        r.pop("adv_rr", None)

    return rows, result


def slice_window(df, which, rows):
    if rows <= 0:
        return df

    return df.head(rows) if which == "head" else df.tail(rows)


# =====================================================================
# Report
# =====================================================================

def report_target(window, rows, target, thresholds, cost, crit):
    """Prints one fixed-target block. Returns {threshold: summary dict}."""

    trades = resolved_trades(rows, target, cost)
    z_all = [t for t in trades if t["z"]]
    nz_all = [t for t in trades if not t["z"]]

    print()
    print("=" * 72)
    print(f"[{window}] FIXED {target:g}R EXIT, net of cost {cost:g}")
    print("=" * 72)
    print(f"   resolved trades {len(trades)} of {len(rows)} "
          f"(the rest were still open when the data ended)")
    print("   group               n   win%   net R     SE      t")

    for name, grp in (("ALL", trades), ("Z", z_all), ("not Z", nz_all)):
        n, win, mean, se = cell_stats(grp)
        flag = " *" if n < MIN_TRADES else ""
        print(
            f"   {name:16s} {n:4d} {f_n(win)}  {es.f_r(mean)}  "
            f"{es.f_se(se)}  {es.f_t(mean, se)}{flag}"
        )

    summaries = {}

    for T in thresholds:

        cells = {}
        for zname, zgrp in (("Z", z_all), ("notZ", nz_all)):
            cells[(zname, True)] = [t for t in zgrp if room_passes(t["room"], T)]
            cells[(zname, False)] = [
                t for t in zgrp if not room_passes(t["room"], T)
            ]

        stats = {k: cell_stats(v) for k, v in cells.items()}
        no_swing = sum(1 for t in trades if t["room"] is None)

        print()
        print(f"--- room threshold T = {T:g}R "
              f"(no swing at all within max_age: {no_swing} trades, "
              f"counted as room FAIL) ---")
        print("   Cell                  n   win%   net R     SE")

        order = (
            ("Z / room pass", ("Z", True)),
            ("Z / room fail", ("Z", False)),
            ("notZ / room pass", ("notZ", True)),
            ("notZ / room fail", ("notZ", False)),
        )

        for label, key in order:
            n, win, mean, se = stats[key]
            flag = " *" if n < MIN_TRADES else ""
            print(
                f"   {label:18s} {n:4d} {f_n(win)}  {es.f_r(mean)}  "
                f"{es.f_se(se)}{flag}"
            )

        n_pass = stats[("Z", True)][0] + stats[("notZ", True)][0]
        n_fail = stats[("Z", False)][0] + stats[("notZ", False)][0]

        print(
            f"   Group sizes the holdout rules count: Z trades {len(z_all)}; "
            f"room pass {n_pass}, room fail {n_fail} "
            f"(smaller room group {min(n_pass, n_fail)})"
        )

        d_z_pass = welch_diff(stats[("Z", True)], stats[("notZ", True)])
        d_z_fail = welch_diff(stats[("Z", False)], stats[("notZ", False)])
        d_room_z = welch_diff(stats[("Z", True)], stats[("Z", False)])
        d_room_nz = welch_diff(stats[("notZ", True)], stats[("notZ", False)])

        print()
        print("   Differences (different trades in each cell, so Welch, not paired):")
        print(f_diff("Z pass minus notZ pass       ", d_z_pass))
        print(f_diff("Z fail minus notZ fail       ", d_z_fail))
        print(f_diff("room pass minus fail (Z)     ", d_room_z))
        print(f_diff("room pass minus fail (notZ)  ", d_room_nz))

        pooled_hc = pooled_effects(trades, T, cluster=False)
        pooled = pooled_effects(trades, T, cluster=True)
        pooled_ctrl = pooled_effects(trades, T, cluster=True, control=True)

        print(f_pooled_line("Pooled OLS, robust SE (HC1, trade-level)", pooled_hc))
        print(f_pooled_line("Pooled OLS, week-clustered SE", pooled))
        print(f_pooled_line("Pooled OLS + stop-size control, week-clustered", pooled_ctrl))

        v = verdict(d_z_pass[2], d_room_z[2], d_room_nz[2], crit)
        print(f"   Reading at t > {crit:g}: {v}")

        notes = significant_notes(
            {
                "Z pass minus notZ pass": d_z_pass[2],
                "Z fail minus notZ fail": d_z_fail[2],
                "room pass minus fail (Z)": d_room_z[2],
                "room pass minus fail (notZ)": d_room_nz[2],
            },
            pooled,
            crit,
        )
        if notes:
            print(f"   |t| > {crit:g} in either direction: " + "; ".join(notes))
        else:
            print(f"   No comparison reaches |t| > {crit:g} in either direction.")

        small = [
            lbl for lbl, key in order if 0 < stats[key][0] < MIN_TRADES
        ]
        if small:
            print(f"   (* cells under {MIN_TRADES} trades: {', '.join(small)})")

        summaries[T] = {
            "verdict": v,
            "t_z_pass": d_z_pass[2],
            "t_room_z": d_room_z[2],
            "t_room_notz": d_room_nz[2],
            "d_z_pass": d_z_pass[0],
            "d_room_notz": d_room_nz[0],
            "pooled": pooled,
            "pooled_hc": pooled_hc,
            "pooled_control": pooled_ctrl,
            "n_room_pass": n_pass,
        }

    return summaries


def report_stability(results, targets, thresholds):
    """Compares head and tail verdicts (results[window][target][T])."""

    print()
    print("=" * 72)
    print("HEAD vs TAIL: DOES THE READING HOLD ACROSS TIME?")
    print("=" * 72)

    def pooled_t(summary, key):
        pooled = summary["pooled"]

        if not pooled or pooled[key][1] <= 0:
            return "   n/a"

        return f"{pooled[key][0] / pooled[key][1]:+6.2f}"

    def pooled_c(summary, key):
        pooled = summary["pooled"]

        return "  n/a" if not pooled else f"{pooled[key][0]:+.2f}"

    any_flip = False

    for target in targets:
        print()
        print(f"Fixed {target:g}R  (pooled OLS, week-clustered: coefficient in R, "
              f"then t; positive = better)")
        print("   T      Z effect head          Z effect tail          "
              "room effect head       room effect tail       sign flip?")

        for T in thresholds:
            h = results["head"][target][T]
            t = results["tail"][target][T]
            flips = sign_flips(h["pooled"], t["pooled"])
            any_flip = any_flip or bool(flips)

            print(
                f"   {T:<5g}  "
                f"{pooled_c(h, 'z'):>6s} (t {pooled_t(h, 'z')})   "
                f"{pooled_c(t, 'z'):>6s} (t {pooled_t(t, 'z')})   "
                f"{pooled_c(h, 'room'):>6s} (t {pooled_t(h, 'room')})   "
                f"{pooled_c(t, 'room'):>6s} (t {pooled_t(t, 'room')})   "
                f"{'YES: ' + ', '.join(flips) if flips else 'no'}"
            )

        print("   Reading by the handoff rule (positive t > crit only):")
        for T in thresholds:
            h = results["head"][target][T]["verdict"]
            t = results["tail"][target][T]["verdict"]
            same = "same" if h == t else "DIFFERENT"
            print(f"   {T:<5g}  head: {h}  |  tail: {t}  [{same}]")

    print()
    if any_flip:
        print("   SIGN FLIP between head and tail: this is outcome 5. The")
        print("   readings above do not describe a stable effect. Do NOT pivot")
        print("   the strategy yet. Grow the sample (more symbols or more")
        print("   history) and re-measure.")
    else:
        print("   No sign flip in the pooled effects between head and tail.")
        print("   That is necessary, not sufficient: check the t-stats above.")


# =====================================================================
# Main
# =====================================================================

def parse_floats(text):
    return [float(x) for x in str(text).split(",") if x.strip()]


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument(
        "--rows", type=int, default=40000,
        help="Candles per window (default 40000). 0 = all.",
    )
    parser.add_argument(
        "--which", choices=["tail", "head", "both"], default="tail",
        help="tail = last N candles (default); head = first N candles; "
             "both = run head then tail and print a stability table. "
             "Each window is a full backtest (about 10-15 minutes).",
    )
    parser.add_argument("--cost", type=float, default=DEFAULT_COST,
                        help="Round-trip cost in price units (default 0.5).")
    parser.add_argument(
        "--targets", default=",".join(f"{t:g}" for t in TARGETS),
        help="Fixed-R exits to evaluate, comma separated (default 2,5).",
    )
    parser.add_argument(
        "--thresholds", default=",".join(f"{t:g}" for t in THRESHOLDS),
        help="Room thresholds in R, comma separated (default 1,1.5,2,3,5).",
    )
    parser.add_argument("--t-crit", type=float, default=T_CRIT,
                        help="t needed for the reading line (default 2).")
    parser.add_argument("--output", default=None, help="Optional per-trade CSV.")
    args = parser.parse_args()

    if args.which == "both" and args.rows <= 0:
        parser.error("--which both needs --rows > 0 (otherwise head and tail "
                     "are the same candles)")

    targets = parse_floats(args.targets)
    thresholds = parse_floats(args.thresholds)

    full = es.load_ohlc(Path(args.data))

    windows = ["head", "tail"] if args.which == "both" else [args.which]

    if args.which == "both" and 2 * args.rows > len(full):
        print(
            f"WARNING: head and tail of {args.rows} candles overlap in a "
            f"{len(full)}-candle file, so they are not independent halves."
        )

    started = time.perf_counter()

    results = {}
    all_rows = []

    for window in windows:

        df = slice_window(full, window, args.rows)

        print()
        print("#" * 72)
        print(f"# WINDOW: {window}")
        print("#" * 72)
        print(f"Using {len(df)} candles: {df.index[0]} -> {df.index[-1]}")
        print(
            f"Price over the sample: {float(df['close'].iloc[0]):.2f} -> "
            f"{float(df['close'].iloc[-1]):.2f}"
        )

        rows, result = run_far_target(df)

        if result.errors:
            print(f"WARNING: {len(result.errors)} engine errors; "
                  f"first: {result.errors[0]}")

        print(
            f"Engine: candidates {result.total_signals_generated}, "
            f"triggered {result.total_trades_triggered}, "
            f"invalidated {result.total_invalidated}, "
            f"expired {result.total_expired}"
        )

        rows, duplicates = dedupe_rows(rows)
        mismatches = sum(1 for r in rows if r["stopped"] != r["engine_stopped"])
        censored = sum(1 for r in rows if not r["stopped"])

        print(f"Trades analysed: {len(rows)} "
              f"(dropped {duplicates} repeat(s) of another trade's entry "
              f"time, direction and stop)")
        print(f"Replay vs engine stop mismatches: {mismatches} (must be 0). "
              f"Trades never stopped before the data ended: {censored}.")

        n_z = sum(1 for r in rows if r["label"] == FOCUS)
        n_room = sum(1 for r in rows if room_passes(r["room_r"], 1.5))
        print(f"Z trades: {n_z}. Room >= 1.5R (the spec's rule): {n_room} "
              f"trades (should be close to the structure(skip) trade count).")

        results[window] = {}

        for target in targets:
            results[window][target] = report_target(
                window, rows, target, thresholds, args.cost, args.t_crit
            )

        for r in rows:
            r["window"] = window
        all_rows.extend(rows)

    if args.which == "both":
        report_stability(results, targets, thresholds)

    looks = len(windows) * len(targets) * len(thresholds) * 4
    print()
    print(
        f"You looked at about {looks} comparisons. After correcting for that, "
        f"a t of about {es.z_needed(looks):.1f} is needed, not "
        f"{args.t_crit:g}. (Conservative: cells and thresholds share trades.)"
    )

    if args.output:
        out = pd.DataFrame(all_rows)
        out["is_z"] = out["label"] == FOCUS
        out.to_csv(args.output, index=False)
        print(f"\nSaved {len(out)} trades to {args.output}")

    print()
    print(f"* = fewer than {MIN_TRADES} trades: treat as noise")
    print(f"Total runtime: {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
