"""
tools/z_vs_room_by_group.py

Splits trades by quarter and by direction, then runs the same
Z-vs-room analysis within each group.

This is a thin extension of tools/z_vs_room.py. It reuses:
    exit_study.collect_rows        (row collection)
    exit_study.load_ohlc           (data loading)
    exit_study.mean_se             (mean, SE)
    exit_study.outcome_r           (fixed-target outcome)
    exit_study.f_r, f_se, f_t      (formatting)
    exit_study.quarter_label       (quarter string)
    exit_study.z_needed            (multiple-testing t threshold)
    z_vs_room.room_passes          (room threshold test)
    z_vs_room.resolved_trades      (net R after cost)
    z_vs_room.cell_stats           (n, win %, mean, SE)
    z_vs_room.welch_diff           (unpaired difference)
    z_vs_room.pooled_effects       (OLS with robust SE)
    z_vs_room.run_far_target       (one backtest, far target)

Run:
    python tools/z_vs_room_by_group.py --data data/real_gold_data_5m.csv \\
        --rows 40000 --which both --cost 0.5

Only the head/tail window(s) requested are run. Within each window,
the same resolved trades are grouped by quarter and by direction.
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import exit_study as es       # noqa: E402
from tools import z_vs_room as zr        # noqa: E402

FOCUS = es.FOCUS
DEFAULT_COST = es.DEFAULT_COST
MIN_TRADES = es.MIN_TRADES

TARGETS = [2.0, 5.0]
THRESHOLDS = [1.5]


# =====================================================================
# Group reporting
# =====================================================================

def group_trades(trades, group_key):
    """
    Returns a list of (group_name, [trades...]) sorted by group name.

    group_key must be a key present on every resolved trade dict.
    The caller adds the key (see resolved_with_group below).
    """

    buckets = {}

    for t in trades:
        g = t.get(group_key, "UNKNOWN")
        buckets.setdefault(g, []).append(t)

    return sorted(buckets.items(), key=lambda kv: kv[0])


def resolved_with_group(rows, target, cost, group_key):
    """
    Like z_vs_room.resolved_trades, but keeps the group_key value on
    each resolved trade so we can split after the fact.
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
                "group": r.get(group_key, "UNKNOWN"),
            }
        )

    return out


def print_cells_and_diffs(trades, threshold, indent="   "):
    """
    Prints the four room cells and the three key Welch differences.
    Returns the three key t-stats for the summary line.
    """

    z_all = [t for t in trades if t["z"]]
    nz_all = [t for t in trades if not t["z"]]

    cells = {
        ("Z", True):  [t for t in z_all  if zr.room_passes(t["room"], threshold)],
        ("Z", False): [t for t in z_all  if not zr.room_passes(t["room"], threshold)],
        ("notZ", True):  [t for t in nz_all if zr.room_passes(t["room"], threshold)],
        ("notZ", False): [t for t in nz_all if not zr.room_passes(t["room"], threshold)],
    }

    stats = {k: zr.cell_stats(v) for k, v in cells.items()}

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
            f"{indent}{label:18s} {n:4d} {_f_n(win)}  "
            f"{es.f_r(mean)}  {es.f_se(se)}{flag}"
        )

    d_z_pass = zr.welch_diff(stats[("Z", True)], stats[("notZ", True)])
    d_room_z = zr.welch_diff(stats[("Z", True)], stats[("Z", False)])
    d_room_nz = zr.welch_diff(stats[("notZ", True)], stats[("notZ", False)])

    print()
    print(f"{indent}Z pass minus notZ pass    : "
          f"{_f_diff(d_z_pass)}")
    print(f"{indent}room pass minus fail (Z)  : "
          f"{_f_diff(d_room_z)}")
    print(f"{indent}room pass minus fail (notZ): "
          f"{_f_diff(d_room_nz)}")

    return d_z_pass[2], d_room_z[2], d_room_nz[2]


def print_summary_line(trades, threshold, t_z_pass, t_room_z, t_room_nz):
    """
    One line per group: Z net R with SE, pooled Z effect, pooled room
    effect with t. This is what you scan across quarters/directions.
    """

    z_all = [t for t in trades if t["z"]]

    _, _, z_mean, z_se = zr.cell_stats(z_all)

    pooled = zr.pooled_effects(trades, threshold)

    if pooled is None:
        print(
            f"   Summary: Z net {es.f_r(z_mean)} (SE {es.f_se(z_se)})  "
            f"pooled not estimable"
        )
        return

    zc, zs = pooled["z"]
    rc, rs = pooled["room"]

    print(
        f"   Summary: Z net {es.f_r(z_mean)} (SE {es.f_se(z_se)})  "
        f"pooled Z {es.f_r(zc)} (t {es.f_t(zc, zs)})  "
        f"pooled room {es.f_r(rc)} (t {es.f_t(rc, rs)})"
    )


def _f_n(x):
    return "  n/a" if x != x else f"{x:5.1f}"


def _f_diff(d):
    diff, se, t = d

    if diff != diff:
        return "n/a (a cell has fewer than 2 trades)"

    t_txt = "n/a" if t != t else f"{t:+.2f}"

    return f"{es.f_r(diff)} (SE {es.f_se(se)}, t {t_txt})"


def report_by_group(rows, group_key, target, cost, thresholds, window, label):
    """
    Runs the Z-vs-room comparison within each value of group_key.

    group_key: "quarter" or "direction"
    label:     human-readable name used in headers ("QUARTER", "DIRECTION")
    """

    trades = resolved_with_group(rows, target, cost, group_key)
    groups = group_trades(trades, "group")

    print()
    print("=" * 72)
    print(f"[{window}] SPLIT BY {label}, fixed {target:g}R, cost {cost:g}")
    print("=" * 72)
    print(f"   resolved trades {len(trades)} of {len(rows)}")

    if not groups:
        print("   (no resolved trades in any group)")
        return {}

    summary = {}

    for gname, g_trades in groups:

        print()
        print(f"{gname}   ({len(g_trades)} trades)")

        z_all = [t for t in g_trades if t["z"]]
        nz_all = [t for t in g_trades if not t["z"]]

        print("   group               n   win%   net R     SE      t")

        for name, grp in (("ALL", g_trades), ("Z", z_all), ("not Z", nz_all)):
            n, win, mean, se = zr.cell_stats(grp)
            flag = " *" if n < MIN_TRADES else ""
            print(
                f"   {name:16s} {n:4d} {_f_n(win)}  "
                f"{es.f_r(mean)}  {es.f_se(se)}  "
                f"{es.f_t(mean, se)}{flag}"
            )

        for T in thresholds:

            print()
            print(f"   --- threshold T = {T:g}R ---")

            t_z_pass, t_room_z, t_room_nz = print_cells_and_diffs(
                g_trades, T, indent="   "
            )

            print_summary_line(g_trades, T, t_z_pass, t_room_z, t_room_nz)

            summary[(gname, T)] = {
                "n": len(g_trades),
                "n_z": len(z_all),
                "t_z_pass": t_z_pass,
                "t_room_z": t_room_z,
                "t_room_nz": t_room_nz,
            }

    return summary


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
        "--which", choices=["tail", "head", "both"], default="both",
        help="tail = last N candles; head = first N candles; "
             "both = run head then tail (default).",
    )
    parser.add_argument("--cost", type=float, default=DEFAULT_COST)
    parser.add_argument(
        "--targets",
        default=",".join(f"{t:g}" for t in TARGETS),
        help="Fixed-R exits to evaluate, comma separated (default 2,5).",
    )
    parser.add_argument(
        "--thresholds",
        default=",".join(f"{t:g}" for t in THRESHOLDS),
        help="Room thresholds in R, comma separated (default 1.5).",
    )
    args = parser.parse_args()

    targets = parse_floats(args.targets)
    thresholds = parse_floats(args.thresholds)

    full = es.load_ohlc(Path(args.data))

    windows = ["head", "tail"] if args.which == "both" else [args.which]

    started = time.perf_counter()

    for window in windows:

        df = (
            full.head(args.rows)
            if (window == "head" and args.rows > 0)
            else full.tail(args.rows)
            if (args.rows > 0)
            else full
        )

        print()
        print("#" * 72)
        print(f"# WINDOW: {window}")
        print("#" * 72)
        print(f"Using {len(df)} candles: {df.index[0]} -> {df.index[-1]}")
        print(
            f"Price over the sample: "
            f"{float(df['close'].iloc[0]):.2f} -> "
            f"{float(df['close'].iloc[-1]):.2f}"
        )

        rows, result = zr.run_far_target(df)

        if result.errors:
            print(
                f"WARNING: {len(result.errors)} engine errors; "
                f"first: {result.errors[0]}"
            )

        print(
            f"Engine: candidates {result.total_signals_generated}, "
            f"triggered {result.total_trades_triggered}, "
            f"invalidated {result.total_invalidated}, "
            f"expired {result.total_expired}"
        )

        for target in targets:
            report_by_group(
                rows, "quarter", target, args.cost, thresholds,
                window, "QUARTER",
            )
            report_by_group(
                rows, "direction", target, args.cost, thresholds,
                window, "DIRECTION",
            )

    # Multiple-testing correction note: the number of comparisons is
    # roughly windows x targets x 2 group keys x cells per group x
    # thresholds. Print a conservative correction so the reader knows
    # what t to look for.
    looks = len(windows) * len(targets) * 2 * 5 * len(thresholds)
    print()
    print(
        f"You looked at about {looks} comparisons. After correcting "
        f"for that, a t of about {es.z_needed(looks):.1f} is needed, "
        f"not 2."
    )

    print()
    print(f"* = fewer than {MIN_TRADES} trades: treat as noise")
    print(f"Total runtime: {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
