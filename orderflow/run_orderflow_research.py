"""
run_orderflow_research.py

Place this file at the project root (next to main.py, alongside
strategy/ and orderflow/).

Runs the EXISTING baseline backtest unchanged, then attaches
order-flow proxy features on top and writes a flat research CSV --
no strategy logic is modified, no trades are filtered.

Usage (PowerShell):

    python run_orderflow_research.py --data real_gold_data_mt5_90000.csv

Adjust the CSV path / run_backtest() kwargs below to match whatever
you're currently using for your 90,000-candle baseline run.
"""

import argparse

import pandas as pd

from strategy.backtest import run_backtest
from orderflow.attach import enrich_trades
from orderflow.research import save_csv, summarize_by_pressure_agreement, to_dataframe


def load_ohlcv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    if "timestamp" not in df.columns:
        raise ValueError(f"{path} has no 'timestamp' column: {list(df.columns)}")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline + order-flow research pass.")
    parser.add_argument("--data", required=True, help="Path to OHLCV CSV (must include volume).")
    parser.add_argument("--out", default="orderflow_research.csv", help="Output CSV path.")
    parser.add_argument("--max-bars-to-retest", type=int, default=20)
    parser.add_argument("--reward-multiple", type=float, default=2.0)
    args = parser.parse_args()

    print(f"Loading {args.data} ...")
    df = load_ohlcv(args.data)
    print(f"  {len(df)} bars, {df.index[0]} -> {df.index[-1]}")

    if "volume" not in df.columns:
        raise SystemExit(
            "This dataset has no 'volume' column -- order-flow proxy "
            "features cannot be computed on it."
        )

    print("Running baseline backtest (strategy/backtest.py, unchanged) ...")
    result = run_backtest(
        df,
        max_bars_to_retest=args.max_bars_to_retest,
        reward_multiple=args.reward_multiple,
    )

    closed = [t for t in result.trades if t.result_status in ("WIN", "LOSS")]
    wins = [t for t in closed if t.result_status == "WIN"]

    print("\n--- BASELINE (unchanged) ---")
    print(f"  Closed trades : {len(closed)}")
    print(f"  Wins          : {len(wins)}")
    print(f"  Win rate      : {result.win_rate:.2f}%")
    print(f"  Total R       : {sum(t.r_multiple for t in closed):.2f}")
    print(f"  Avg R         : {result.expectancy:.3f}")

    print("\nAttaching order-flow proxy features (post-hoc, read-only) ...")
    enriched = enrich_trades(df, result)

    save_csv(enriched, args.out)
    print(f"  Wrote {args.out}  ({len(enriched)} rows)")

    print("\n--- Descriptive: pressure_proxy agreement vs outcome ---")
    summary = summarize_by_pressure_agreement(enriched)
    if summary.empty:
        print("  (no closed trades to summarize)")
    else:
        print(summary.to_string(index=False))

    print(
        "\nNote: this is a descriptive research pass only. No trades "
        "were filtered and no thresholds were applied -- see "
        f"{args.out} for the full per-trade feature table."
    )


if __name__ == "__main__":
    main()