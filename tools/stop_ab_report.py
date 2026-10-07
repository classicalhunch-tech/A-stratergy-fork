"""
tools/stop_ab_report.py

A/B of the ATR stop buffer beyond the zone edge: 0 (today), 0.1 x ATR(14),
0.25 x ATR(14). Market on close, no MTF gate. Two target styles:
fill-anchored 5R and the structural 15M-swing target.

A stop-out is "wick-only" when the bar that hit the stop CLOSED back on the
safe side of it (the stop was tagged, then price recovered within the bar).
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from strategy.backtest import run_backtest
from strategy.signals import SignalType
from strategy.structure_targets import build_structure_target_fn

COST = 0.5
MIN_TRADES = 20
MULTS = [0.0, 0.1, 0.25]


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


def wick_only_share(df, losses):
    if not losses:
        return float("nan")
    wick = 0
    for t in losses:
        try:
            exit_close = float(df.loc[t.exit_time, "close"])
        except KeyError:
            continue
        if t.direction == SignalType.LONG and exit_close > t.stop_loss:
            wick += 1
        elif t.direction == SignalType.SHORT and exit_close < t.stop_loss:
            wick += 1
    return 100.0 * wick / len(losses)


def summarize(label, df, trades):
    if not trades:
        print(f"   {label:10s}    0 trades")
        return
    wins = sum(1 for t in trades if t.result_status == "WIN")
    losses = [t for t in trades if t.result_status == "LOSS"]
    net = sum(t.r_multiple - COST / t.initial_risk for t in trades) / len(trades)
    gross = sum(t.r_multiple for t in trades) / len(trades)
    stops = sorted(t.initial_risk for t in trades)
    flag = " *" if len(trades) < MIN_TRADES else ""
    print(
        f"   {label:10s} {len(trades):4d} trades  win {100.0 * wins / len(trades):5.1f}%"
        f"  gross {gross:+.3f}R  net@{COST} {net:+.3f}R"
        f"  median stop {stops[len(stops) // 2]:.2f}"
        f"  wick-only losses {wick_only_share(df, losses):4.1f}%{flag}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--rows", type=int, default=40000)
    args = parser.parse_args()

    df = load_ohlc(Path(args.data)).tail(args.rows)
    print(f"Using {len(df)} candles: {df.index[0]} -> {df.index[-1]}")
    started = time.perf_counter()

    structure_fn = build_structure_target_fn(df)

    styles = [
        ("fill 5R", dict(reward_multiple=5.0, target_anchor="fill")),
        ("structure", dict(reward_multiple=2.0, target_anchor="structure", target_fn=structure_fn)),
    ]

    for name, kwargs in styles:
        print()
        print(f"{name}  (market on close, no MTF gate)")
        for mult in MULTS:
            result = run_backtest(
                df, fill_mode="market_on_close", stop_atr_mult=mult, **kwargs
            )
            summarize(f"stop +{mult}ATR", df, closed(result))
            if result.errors:
                print(f"   WARNING: {len(result.errors)} errors; first: {result.errors[0]}")

    print()
    print(f"* = fewer than {MIN_TRADES} trades")
    print(f"Total runtime: {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
