"""
tools/structure_target_report.py

Market-on-close, no MTF gate. Compares fixed-R targets anchored to the fill
with the structural 15M-swing target (skip variant and advance variant),
split by pattern label. Costs applied after the fact.
"""

import argparse
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


def summarize(label, trades):
    if not trades:
        print(f"   {label:16s}    0 trades")
        return

    wins = sum(1 for t in trades if t.result_status == "WIN")
    gross = sum(t.r_multiple for t in trades) / len(trades)
    net = sum(t.r_multiple - COST / t.initial_risk for t in trades) / len(trades)
    stops = sorted(t.initial_risk for t in trades)
    planned = sorted(abs(t.take_profit - t.fill_price) / t.initial_risk for t in trades)
    flag = " *" if len(trades) < MIN_TRADES else ""

    print(
        f"   {label:16s} {len(trades):4d} trades  win {100.0 * wins / len(trades):5.1f}%"
        f"  gross {gross:+.3f}R  net@{COST} {net:+.3f}R"
        f"  median stop {stops[len(stops) // 2]:.2f}"
        f"  median planned RR {planned[len(planned) // 2]:.2f}{flag}"
    )


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

        print()
        print(f"{name}  (market on close, no MTF gate)")
        print(
            f"   candidates {result.total_signals_generated}, "
            f"triggered {result.total_trades_triggered}, "
            f"invalidated/skipped {result.total_invalidated}, "
            f"expired {result.total_expired}"
        )
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
    print(f"* = fewer than {MIN_TRADES} trades")
    print(f"Total runtime: {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
