"""
tools/structure_target_report.py

Market-on-close, no MTF gate. Compares fixed-R targets anchored to the fill
with the structural 15M-swing target (skip variant and advance variant),
split by pattern label. Costs applied after the fact.

Two views:
  1. Per-run tables with a standard error on every row. A group is not
     distinguishable from zero unless |net| is well above 2 x SE.
  2. A paired comparison on the trades that BOTH runs took, so exit
     differences are not mixed up with which trades each run selected.
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

from dashboard.mtf_context import MTFConfig
from strategy.backtest import run_backtest
from strategy.mtf_structure import build_mtf_dataset_with_structure
from strategy.pattern_tags import PATTERN_LABELS, build_pattern_tagger
from strategy.structure_targets import (
    StructureTargetConfig,
    build_structure_target_fn,
)
from strategy.swings import find_swings

COST = 0.5
MIN_TRADES = 20


def load_ohlc(path):
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.columns = [str(c).strip().lower() for c in df.columns]
    df = df[df.index.notna()].sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df[["open", "high", "low", "close"]].astype(float).dropna()


def closed(result):
    return [
        t for t in result.trades
        if t.result_status in ("WIN", "LOSS") and t.initial_risk > 0
    ]


def net_r(trade):
    """R after cost. The backtest runs with zero cost, so r_multiple is gross."""
    return trade.r_multiple - COST / trade.initial_risk


def mean_se(values):
    n = len(values)
    if n == 0:
        return 0, float("nan"), float("nan")
    mean = sum(values) / n
    if n < 2:
        return n, mean, float("nan")
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return n, mean, math.sqrt(var / n)


def trade_key(trade):
    direction = getattr(trade.direction, "value", trade.direction)
    return (
        pd.Timestamp(trade.entry_time),
        str(direction),
        round(float(trade.stop_loss), 4),
    )


def summarize(label, trades):
    if not trades:
        print(f"   {label:16s}    0 trades")
        return

    wins = sum(1 for t in trades if t.result_status == "WIN")
    gross = sum(t.r_multiple for t in trades) / len(trades)
    _, net, se = mean_se([net_r(t) for t in trades])
    stops = sorted(t.initial_risk for t in trades)
    planned = sorted(
        abs(t.take_profit - t.fill_price) / t.initial_risk for t in trades
    )
    flag = " *" if len(trades) < MIN_TRADES else ""

    print(
        f"   {label:16s} {len(trades):4d} trades  win {100.0 * wins / len(trades):5.1f}%"
        f"  gross {gross:+.3f}R  net@{COST} {net:+.3f}R (SE {se:.3f})"
        f"  median stop {stops[len(stops) // 2]:.2f}"
        f"  median planned RR {planned[len(planned) // 2]:.2f}{flag}"
    )


def paired(name_a, trades_a, name_b, trades_b):
    """Compare two runs on the trades both of them took."""
    by_key_a = {}
    for t in trades_a:
        by_key_a.setdefault(trade_key(t), t)

    by_key_b = {}
    for t in trades_b:
        by_key_b.setdefault(trade_key(t), t)

    common = sorted(set(by_key_a) & set(by_key_b))

    print(f"   {name_a}  vs  {name_b}")
    print(
        f"      trades in A: {len(by_key_a)}, in B: {len(by_key_b)}, "
        f"in both: {len(common)}"
    )

    if len(common) < 2:
        print("      too few common trades to compare")
        return

    a_vals = [net_r(by_key_a[k]) for k in common]
    b_vals = [net_r(by_key_b[k]) for k in common]
    diffs = [a - b for a, b in zip(a_vals, b_vals)]

    _, mean_a, se_a = mean_se(a_vals)
    _, mean_b, se_b = mean_se(b_vals)
    _, mean_d, se_d = mean_se(diffs)

    flag = " *" if len(common) < MIN_TRADES else ""

    print(f"      A net {mean_a:+.3f}R (SE {se_a:.3f})   B net {mean_b:+.3f}R (SE {se_b:.3f})")
    print(f"      A minus B {mean_d:+.3f}R (SE {se_d:.3f}){flag}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--rows", type=int, default=40000)
    args = parser.parse_args()

    df = load_ohlc(Path(args.data)).tail(args.rows)
    print(f"Using {len(df)} candles: {df.index[0]} -> {df.index[-1]}")
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

    runs = [
        ("fill 2R", "fill", 2.0, None),
        ("fill 5R", "fill", 5.0, None),
        ("structure (skip if <1.5R)", "structure", 2.0, skip_fn),
        ("structure (advance to 1.5R)", "structure", 2.0, advance_fn),
    ]

    trades_by_run = {}
    baseline_invalidated = None

    for name, anchor, rm, fn in runs:
        result = run_backtest(
            df,
            reward_multiple=rm,
            mtf_filter_fn=tagging_filter,
            fill_mode="market_on_close",
            target_anchor=anchor,
            target_fn=fn,
        )
        trades = closed(result)
        trades_by_run[name] = trades

        if baseline_invalidated is None:
            baseline_invalidated = result.total_invalidated

        print()
        print(f"{name}  (market on close, no MTF gate)")
        print(
            f"   candidates {result.total_signals_generated}, "
            f"triggered {result.total_trades_triggered}, "
            f"invalidated/skipped {result.total_invalidated}, "
            f"expired {result.total_expired}"
        )

        if anchor == "structure":
            skipped = result.total_invalidated - baseline_invalidated
            print(f"   skipped by the target rule (vs fill 2R): about {skipped}")

        summarize("ALL", trades)
        for label in PATTERN_LABELS:
            group = [
                t for t in trades
                if getattr(t.signal, "pattern", "U_unknown") == label
            ]
            summarize(label, group)

        if result.errors:
            print(f"   WARNING: {len(result.errors)} errors; first: {result.errors[0]}")

    print()
    print("=" * 60)
    print("PAIRED COMPARISON (only trades both runs took)")
    print("=" * 60)

    for struct_name in (
        "structure (skip if <1.5R)",
        "structure (advance to 1.5R)",
    ):
        for fixed_name in ("fill 5R", "fill 2R"):
            print()
            paired(
                struct_name,
                trades_by_run[struct_name],
                fixed_name,
                trades_by_run[fixed_name],
            )

    print()
    print(f"* = fewer than {MIN_TRADES} trades: treat as noise")
    print("A result is not distinguishable from zero unless |net| is well above 2 x SE.")
    print(f"Total runtime: {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
