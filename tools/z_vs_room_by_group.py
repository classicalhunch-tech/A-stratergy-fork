"""
tools/z_vs_room_by_group.py

Does the Z-vs-room answer hold inside each quarter and each direction?

A drop-in refinement of the first by-group script (same flags, plus the ones
below). What changed and why:

  * --from-csv reads the per-trade CSV that tools/z_vs_room.py writes with
    --output, so no backtest is re-run (a full head+tail run takes ~6.5 min).
    The CSV holds every trade of both windows, so grouping can run over the
    whole 80k candles at once. Head and tail do not overlap in time, so a
    quarter that straddles the boundary (2026Q1) is one group, not two.
  * A Welch difference is only printed when BOTH cells hold at least
    --min-cell trades (default 8). Cells of 2 to 5 trades can have a
    near-zero SE by luck and give a huge, meaningless t.
  * The pooled regression is only fitted when a group has at least
    --min-pooled trades (default 30).
  * One compact table per split (one line per group) instead of four cell
    blocks per group. --detail brings the cell blocks back.
  * A sign test across the groups. The groups hold different trades, so
    "Z beat not-Z in 5 of 6 groups" means something that one noisy t does
    not. The p-value is exact (two-sided binomial) and still weak with few
    groups; read it that way.
  * The number of comparisons used for the multiple-testing note is counted,
    not guessed.

Run (fast, uses the saved trades):
    python tools/z_vs_room_by_group.py --from-csv z_vs_room_trades.csv

Run (re-runs the backtests, ~6.5 min):
    python tools/z_vs_room_by_group.py --data data/real_gold_data_5m.csv \\
        --rows 40000 --which both --cost 0.5
"""

import argparse
import math
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import exit_study as es  # noqa: E402
from tools import z_vs_room as zr  # noqa: E402

FOCUS = es.FOCUS
DEFAULT_COST = es.DEFAULT_COST
MIN_TRADES = es.MIN_TRADES

TARGETS = [2.0, 5.0]
THRESHOLDS = [1.5]
MIN_CELL = 8
MIN_POOLED = 30

NAN = float("nan")

CSV_REQUIRED = {
    "label", "direction", "entry_time", "quarter",
    "risk", "run_r", "stopped", "room_r",
}


# =====================================================================
# Pure functions (tested without the engine)
# =====================================================================

def rows_from_csv(path):
    """Rows in the shape tools/z_vs_room.py produces, from its --output CSV."""

    data = pd.read_csv(path)

    missing = CSV_REQUIRED - set(data.columns)

    if missing:
        raise ValueError(
            f"{path} is missing columns {sorted(missing)}; "
            f"write it with tools/z_vs_room.py --output"
        )

    rows = data.to_dict("records")

    for r in rows:
        r["entry_time"] = pd.Timestamp(r["entry_time"])
        r["stopped"] = bool(r["stopped"])
        r["room_r"] = None if r["room_r"] != r["room_r"] else float(r["room_r"])
        r.setdefault("window", "csv")

        if r["window"] != r["window"]:
            r["window"] = "csv"

    return rows


def group_trades(trades, key):
    """[(group name, [trades])], sorted by name. Missing key -> "UNKNOWN"."""

    buckets = {}

    for t in trades:
        buckets.setdefault(t.get(key) or "UNKNOWN", []).append(t)

    return sorted(buckets.items(), key=lambda kv: kv[0])


def sign_tally(values):
    """
    (positives, total, two-sided exact binomial p) over the non-NaN, non-zero
    values. Tests "positive as often as negative" across disjoint groups.
    """

    vals = [v for v in values if v == v and v != 0]
    m = len(vals)

    if m == 0:
        return 0, 0, NAN

    k = sum(1 for v in vals if v > 0)
    tail = sum(math.comb(m, i) for i in range(0, min(k, m - k) + 1))

    return k, m, min(1.0, 2.0 * tail / 2.0 ** m)


def summarize_group(trades, threshold, min_cell=MIN_CELL, min_pooled=MIN_POOLED):
    """
    Everything the compact table needs for one group of resolved trades.
    Diffs are NaN when a cell is under `min_cell`; pooled is None when the
    group is under `min_pooled` or a factor never varies.
    """

    z = [t for t in trades if t["z"]]
    nz = [t for t in trades if not t["z"]]
    passed = [t for t in trades if zr.room_passes(t["room"], threshold)]
    failed = [t for t in trades if not zr.room_passes(t["room"], threshold)]

    s_z, s_nz = zr.cell_stats(z), zr.cell_stats(nz)
    s_pass, s_fail = zr.cell_stats(passed), zr.cell_stats(failed)

    pooled = (
        zr.pooled_effects(trades, threshold) if len(trades) >= min_pooled else None
    )

    cells = {}
    for zname, grp in (("Z", z), ("notZ", nz)):
        cells[(zname, True)] = zr.cell_stats(
            [t for t in grp if zr.room_passes(t["room"], threshold)]
        )
        cells[(zname, False)] = zr.cell_stats(
            [t for t in grp if not zr.room_passes(t["room"], threshold)]
        )

    return {
        "n": len(trades),
        "n_z": len(z),
        "z_net": s_z[2],
        "nz_net": s_nz[2],
        "z_minus_nz": zr.welch_diff(s_z, s_nz, min_n=min_cell),
        "pass_minus_fail": zr.welch_diff(s_pass, s_fail, min_n=min_cell),
        "pooled": pooled,
        "cells": cells,
    }


# =====================================================================
# Formatting
# =====================================================================

def f_diff_t(d):
    """'+0.123 (t +1.20)' or '        n/a' when the cells were too small."""

    diff, _, t = d

    if diff != diff:
        return "           n/a"

    t_txt = "  n/a" if t != t else f"{t:+.2f}"

    return f"{diff:+.3f} (t {t_txt})"


def f_pooled(pooled, key):
    if not pooled:
        return "          n/a"

    coef, se = pooled[key]

    return f"{coef:+.3f} (t {es.f_t(coef, se)})"


def print_cells(summary, indent="      "):
    order = (
        ("Z / room pass", ("Z", True)),
        ("Z / room fail", ("Z", False)),
        ("notZ / room pass", ("notZ", True)),
        ("notZ / room fail", ("notZ", False)),
    )

    for label, key in order:
        n, win, mean, se = summary["cells"][key]
        win_txt = "  n/a" if win != win else f"{win:5.1f}"
        flag = " *" if n < MIN_TRADES else ""
        print(
            f"{indent}{label:18s} {n:4d} {win_txt}  "
            f"{es.f_r(mean)}  {es.f_se(se)}{flag}"
        )


# =====================================================================
# Report
# =====================================================================

def report_split(rows, key, target, cost, threshold, title, min_cell,
                 min_pooled, detail):
    """
    One compact table: one line per group of `key` (quarter / direction).
    Returns (summaries, number of t-stats shown).
    """

    trades = zr.resolved_trades(rows, target, cost)
    groups = group_trades(trades, key)

    print()
    print("=" * 100)
    print(f"{title}: SPLIT BY {key.upper()}, fixed {target:g}R, cost {cost:g}, "
          f"room threshold {threshold:g}R")
    print("=" * 100)
    print(f"   resolved trades {len(trades)} of {len(rows)}. Differences need "
          f">= {min_cell} trades in each cell; pooled needs >= {min_pooled} "
          f"trades in the group.")

    if not groups:
        print("   (no resolved trades)")
        return {}, 0

    print()
    print(f"   {'group':9s} {'n':>4s} {'nZ':>3s}  {'Z net':>7s} {'notZ net':>8s}  "
          f"{'Z minus notZ':>20s}  {'room pass-fail':>20s}  "
          f"{'pooled Z':>20s}  {'pooled room':>20s}")

    summaries = {}
    shown = 0

    for name, g_trades in groups:

        s = summarize_group(g_trades, threshold, min_cell, min_pooled)
        summaries[name] = s

        shown += sum(
            1 for d in (s["z_minus_nz"], s["pass_minus_fail"]) if d[0] == d[0]
        )
        if s["pooled"]:
            shown += 2

        flag = " *" if s["n_z"] < MIN_TRADES else ""

        print(
            f"   {str(name):9s} {s['n']:4d} {s['n_z']:3d}  "
            f"{es.f_r(s['z_net']):>7s} {es.f_r(s['nz_net']):>8s}  "
            f"{f_diff_t(s['z_minus_nz']):>20s}  "
            f"{f_diff_t(s['pass_minus_fail']):>20s}  "
            f"{f_pooled(s['pooled'], 'z'):>20s}  "
            f"{f_pooled(s['pooled'], 'room'):>20s}{flag}"
        )

        if detail:
            print_cells(s)

    print()
    print("   * = fewer than "
          f"{MIN_TRADES} Z trades in the group: treat that line as noise.")

    for label, getter in (
        ("Z minus notZ      ", lambda s: s["z_minus_nz"][0]),
        ("room pass - fail  ", lambda s: s["pass_minus_fail"][0]),
        ("pooled Z effect   ", lambda s: s["pooled"]["z"][0] if s["pooled"] else NAN),
        ("pooled room effect", lambda s: s["pooled"]["room"][0] if s["pooled"] else NAN),
    ):
        k, m, p = sign_tally([getter(s) for s in summaries.values()])

        if m == 0:
            print(f"   Sign test, {label}: no group had enough trades")
        else:
            p_txt = "n/a" if p != p else f"{p:.2f}"
            print(f"   Sign test, {label}: positive in {k} of {m} groups "
                  f"(two-sided p {p_txt})")

    return summaries, shown


# =====================================================================
# Main
# =====================================================================

def parse_floats(text):
    return [float(x) for x in str(text).split(",") if x.strip()]


def collect_rows_by_backtest(args, full):
    """The same one-run-per-window set-up as tools/z_vs_room.py."""

    windows = ["head", "tail"] if args.which == "both" else [args.which]
    rows = []

    for window in windows:

        df = zr.slice_window(full, window, args.rows)

        print(f"[{window}] {len(df)} candles: {df.index[0]} -> {df.index[-1]}")

        window_rows, result = zr.run_far_target(df)

        if result.errors:
            print(f"WARNING: {len(result.errors)} engine errors; "
                  f"first: {result.errors[0]}")

        print(f"[{window}] triggered {result.total_trades_triggered}, "
              f"analysed {len(window_rows)}")

        for r in window_rows:
            r["window"] = window

        rows.extend(window_rows)

    return rows


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--from-csv", default=None,
                        help="Per-trade CSV written by tools/z_vs_room.py "
                             "--output. Skips the backtests.")
    parser.add_argument("--data", default=None,
                        help="5M OHLC CSV (needed when --from-csv is not used).")
    parser.add_argument("--rows", type=int, default=40000,
                        help="Candles per window (default 40000). 0 = all.")
    parser.add_argument("--which", choices=["tail", "head", "both"],
                        default="both")
    parser.add_argument("--cost", type=float, default=DEFAULT_COST)
    parser.add_argument("--targets", default=",".join(f"{t:g}" for t in TARGETS),
                        help="Fixed-R exits, comma separated (default 2,5).")
    parser.add_argument("--thresholds",
                        default=",".join(f"{t:g}" for t in THRESHOLDS),
                        help="Room thresholds in R (default 1.5).")
    parser.add_argument("--min-cell", type=int, default=MIN_CELL,
                        help=f"Trades needed in each cell for a difference "
                             f"(default {MIN_CELL}).")
    parser.add_argument("--min-pooled", type=int, default=MIN_POOLED,
                        help=f"Trades needed in a group for the pooled "
                             f"regression (default {MIN_POOLED}).")
    parser.add_argument("--per-window", action="store_true",
                        help="Report head and tail separately instead of "
                             "pooling them (quarters then split at the "
                             "boundary).")
    parser.add_argument("--detail", action="store_true",
                        help="Also print the four room cells for every group.")
    args = parser.parse_args()

    if not args.from_csv and not args.data:
        parser.error("give --from-csv or --data")

    targets = parse_floats(args.targets)
    thresholds = parse_floats(args.thresholds)

    started = time.perf_counter()

    if args.from_csv:
        rows = rows_from_csv(args.from_csv)
        print(f"Read {len(rows)} trades from {args.from_csv}")
    else:
        rows = collect_rows_by_backtest(args, es.load_ohlc(Path(args.data)))

    if not rows:
        print("No trades to analyse.")
        return

    times = [r["entry_time"] for r in rows]
    print(f"Trades: {len(rows)}, entries {min(times)} -> {max(times)}")

    if args.per_window:
        sets = [
            (w, [r for r in rows if r.get("window") == w])
            for w in sorted({r.get("window") for r in rows})
        ]
    else:
        sets = [("ALL", rows)]

    looks = 0

    for set_name, set_rows in sets:
        for target in targets:
            for T in thresholds:
                for key in ("quarter", "direction"):
                    _, shown = report_split(
                        set_rows, key, target, args.cost, T,
                        f"[{set_name}]", args.min_cell, args.min_pooled,
                        args.detail,
                    )
                    looks += shown

    print()
    if looks:
        print(
            f"You looked at {looks} t-statistics. After correcting for that, "
            f"a t of about {es.z_needed(looks):.1f} is needed, not 2. "
            f"(Conservative: the splits share trades.)"
        )

    print(f"Total runtime: {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
