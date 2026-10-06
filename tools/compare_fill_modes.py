"""
tools/compare_fill_modes.py

Compares the backtest under two fill rules and shows how robust the result is.

  default fill rule : a setup fills as soon as price touches the zone edge
  strict fill rule  : a setup fills only if price trades through the entry
                      price (limit-order realism, strict_entry_fill=True)

Both runs use zero trading costs. Costs never change which trades trigger or
where they exit, so net results for several cost levels are computed after
the fact:

    net R = gross R - round_trip_cost / initial_risk

where round_trip_cost = spread + 2 * slippage + commission, in price units.

Usage:
    python tools/compare_fill_modes.py --data data/real_gold_data_5m.csv --rows 40000
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

# Round-trip cost levels in price units.
# XAUUSD: 0.50 = 0.30 spread + 2 x 0.10 slippage.
COST_LEVELS = [0.0, 0.5, 1.0, 2.0]


def load_ohlc(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.columns = [str(c).strip().lower() for c in df.columns]

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce", utc=True)

    df = df[df.index.notna()].sort_index()
    df = df[~df.index.duplicated(keep="first")]

    return df[["open", "high", "low", "close"]].astype(float).dropna()


def closed_trades(result):
    return [
        t
        for t in result.trades
        if t.result_status in ("WIN", "LOSS") and t.initial_risk > 0
    ]


def expectancy(trades, cost):
    if not trades:
        return float("nan")
    return sum(t.r_multiple - cost / t.initial_risk for t in trades) / len(trades)


def print_mode(title, result):
    trades = closed_trades(result)

    print()
    print("=" * 60)
    print(title)
    print("=" * 60)
    print(f"signals approved:     {result.total_signals_generated}")
    print(f"rejected by MTF:      {result.total_mtf_rejected}")
    print(f"triggered:            {result.total_trades_triggered}")
    print(f"invalidated:          {result.total_invalidated}")
    print(f"expired:              {result.total_expired}")
    print(f"closed trades:        {len(trades)}")

    if not trades:
        return

    wins = sum(1 for t in trades if t.result_status == "WIN")
    risks = sorted(t.initial_risk for t in trades)

    print(f"win rate (gross):     {100.0 * wins / len(trades):.1f}%")
    print(f"median stop distance: {risks[len(risks) // 2]:.2f} price units")
    print("expectancy by round-trip cost:")
    for cost in COST_LEVELS:
        print(f"   cost {cost:4.2f}: {expectancy(trades, cost):+.3f} R")

    print("by quarter at cost 0.50 (trades, expectancy):")
    quarters = {}
    for t in trades:
        ts = pd.Timestamp(t.entry_time)
        if ts.tzinfo is not None:
            ts = ts.tz_convert("UTC").tz_localize(None)
        quarters.setdefault(str(ts.to_period("Q")), []).append(t)

    for quarter in sorted(quarters):
        group = quarters[quarter]
        print(f"   {quarter}: {len(group):3d} trades  {expectancy(group, 0.5):+.3f} R")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--rows", type=int, default=40000)
    args = parser.parse_args()

    df = load_ohlc(Path(args.data)).tail(args.rows)
    print(f"Using {len(df)} candles: {df.index[0]} -> {df.index[-1]}")

    started = time.perf_counter()

    df_enriched = build_mtf_dataset_with_structure(
        df,
        macro_swings_fn=find_swings,
        internal_swings_fn=find_swings,
        config=MTFConfig(macro_tf="1h", internal_tf="15min"),
    )
    mtf_filter = build_mtf_signal_filter(df_enriched)

    for strict, title in (
        (False, "DEFAULT fill rule (fills on zone-edge touch)"),
        (True, "STRICT fill rule (fills only if price reaches the entry)"),
    ):
        result = run_backtest(
            df,
            mtf_filter_fn=mtf_filter,
            strict_entry_fill=strict,
        )
        print_mode(title, result)

        if result.errors:
            print(f"WARNING: {len(result.errors)} replay error(s); first: {result.errors[0]}")

    print()
    print(f"Total runtime: {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
