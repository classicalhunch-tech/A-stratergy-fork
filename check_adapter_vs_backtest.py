"""
check_adapter_vs_backtest.py

Diagnostic only. Compares the streaming StrategyAdapter's triggered
signals against strategy.backtest.run_backtest()'s triggered trades
on the same CSV, to check whether the adapter's lack of a
update_mitigation() call causes it to trigger signals off zones that
the batch backtest would have excluded as already mitigated.

Does not modify either implementation.

Usage:
    python check_adapter_vs_backtest.py your_data_file_1000.csv
"""

import sys
import pandas as pd

from phase_03_paper.market.engine import Candle
from phase_03_paper.signals.adapter import StrategyAdapter
from strategy.backtest import run_backtest


def main():
    if len(sys.argv) != 2:
        print("Usage: python check_adapter_vs_backtest.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]

    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df.columns = [c.strip().lower() for c in df.columns]

    # --- Run the batch backtest reference ---
    backtest_df = df.set_index("timestamp")
    backtest_result = run_backtest(backtest_df)

    backtest_triggers = sorted(
        (
            round(t.fill_price, 8),
            round(t.initial_risk, 8),
        )
        for t in backtest_result.trades
    )

    # --- Run the streaming adapter ---
    adapter = StrategyAdapter()
    adapter_triggers = []

    for _, row in df.iterrows():
        candle = Candle(
            timestamp=row["timestamp"],
            open=row["open"],
            high=row["high"],
            low=row["low"],
            close=row["close"],
        )
        events = adapter.on_candle(candle)
        for e in events:
            adapter_triggers.append(
                (round(e.fill_price, 8), round(e.initial_risk, 8))
            )

    adapter_triggers_sorted = sorted(adapter_triggers)

    print("=" * 70)
    print(f"ADAPTER vs BACKTEST: {csv_path}")
    print("=" * 70)
    print(f"backtest total triggered : {len(backtest_triggers)}")
    print(f"adapter  total triggered : {len(adapter_triggers)}")
    print()

    if backtest_triggers == adapter_triggers_sorted:
        print("[MATCH] Adapter and backtest triggered identical signal sets.")
    else:
        print("[DIVERGE] Adapter and backtest triggered different signals.")
        extra_in_adapter = [
            t for t in adapter_triggers_sorted if t not in backtest_triggers
        ]
        missing_from_adapter = [
            t for t in backtest_triggers if t not in adapter_triggers_sorted
        ]
        print(f"  in adapter but not backtest: {extra_in_adapter}")
        print(f"  in backtest but not adapter: {missing_from_adapter}")

    print("=" * 70)


if __name__ == "__main__":
    main()