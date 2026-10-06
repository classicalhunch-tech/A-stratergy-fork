"""
tools/mtf_split.py

Market-on-close backtest with the MTF filter turned into a TAG instead of a gate.
Prints expectancy for MTF-approved vs MTF-rejected trades, and sweeps the
reward multiple. Costs applied after the fact (same as compare_fill_modes).

Usage:
    python tools/mtf_split.py --data data/real_gold_data_5m.csv --rows 40000
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
from strategy.confluence import build_mtf_signal_filter
from strategy.mtf_structure import build_mtf_dataset_with_structure
from strategy.swings import find_swings

COST = 0.5  # round-trip, price units
REWARD_MULTIPLES = [1.5, 2.0, 3.0, 5.0]


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
        print(f"   {label:10s}  0 trades")
        return
    wins = sum(1 for t in trades if t.result_status == "WIN")
    gross = sum(t.r_multiple for t in trades) / len(trades)
    net = sum(t.r_multiple - COST / t.initial_risk for t in trades) / len(trades)
    print(
        f"   {label:10s} {len(trades):4d} trades  win {100.0 * wins / len(trades):5.1f}%"
        f"  gross {gross:+.3f}R  net@{COST} {net:+.3f}R"
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
    real_filter = build_mtf_signal_filter(enriched)

    def tagging_filter(signal, current_time):
        signal.mtf_approved = bool(real_filter(signal, current_time))
        return True  # never block; we only tag

    for rm in REWARD_MULTIPLES:
        result = run_backtest(
            df,
            reward_multiple=rm,
            mtf_filter_fn=tagging_filter,
            fill_mode="market_on_close",
        )
        trades = closed(result)
        approved = [t for t in trades if getattr(t.signal, "mtf_approved", False)]
        rejected = [t for t in trades if not getattr(t.signal, "mtf_approved", False)]

        print()
        print(f"reward_multiple = {rm}  (market on close, MTF as tag)")
        summarize("ALL", trades)
        summarize("approved", approved)
        summarize("rejected", rejected)
        if result.errors:
            print(f"   WARNING: {len(result.errors)} errors; first: {result.errors[0]}")

    print()
    print(f"Total runtime: {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
