"""
compare_old_vs_new.py

Runs BOTH the pre-liquidity-fix adapter (adapter_old_check.py, a copy
of the original adapter.py.bak) and the current, fixed adapter
(phase_03_paper.signals.adapter) over the same CSV, and diffs their
observable behavior.

This is a correctness check only. It does not touch or modify either
adapter. It exists to answer one question: did the liquidity fix
change any strategy-visible behavior at 1,000 / 2,000 candles?

Usage:
    python compare_old_vs_new.py your_data_file_1000.csv
    python compare_old_vs_new.py your_data_file_2000.csv
"""

import sys
import importlib.util
import pandas as pd

from phase_03_paper.market.engine import Candle


def _load_old_adapter_class():
    """
    Load StrategyAdapter from adapter_old_check.py by file path,
    under its own isolated module name, so it does not collide
    with or shadow the real phase_03_paper.signals.adapter module.
    """

    spec = importlib.util.spec_from_file_location(
        "adapter_old_check",
        "adapter_old_check.py",
    )
    module = importlib.util.module_from_spec(spec)

    # Register in sys.modules BEFORE exec_module. The old adapter
    # file uses `from __future__ import annotations` combined with
    # @dataclass(frozen=True), and dataclass needs to look up its
    # own module in sys.modules while the class body is executing.
    # Without this line that lookup returns None and dataclass
    # creation crashes.
    sys.modules["adapter_old_check"] = module

    spec.loader.exec_module(module)
    return module.StrategyAdapter


def run_adapter(adapter_cls, df: pd.DataFrame):
    adapter = adapter_cls()

    per_candle_triggered = []
    per_candle_pending = []

    for _, row in df.iterrows():
        candle = Candle(
            timestamp=row["timestamp"],
            open=row["open"],
            high=row["high"],
            low=row["low"],
            close=row["close"],
        )
        events = adapter.on_candle(candle)

        # Record a lightweight signature of each triggered event,
        # not the object identity, so old vs new instances compare
        # equal when they represent the same real-world event.
        per_candle_triggered.append(
            [
                (
                    e.setup_bar_index,
                    e.trigger_bar_index,
                    round(e.fill_price, 8),
                    round(e.initial_risk, 8),
                )
                for e in events
            ]
        )
        per_candle_pending.append(adapter.pending_count)

    return {
        "triggered_by_candle": per_candle_triggered,
        "pending_by_candle": per_candle_pending,
        "total_triggered": sum(
            len(x) for x in per_candle_triggered
        ),
        "final_pending": adapter.pending_count,
        "errors": list(adapter.errors),
    }


def main():
    if len(sys.argv) != 2:
        print("Usage: python compare_old_vs_new.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]

    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df.columns = [c.strip().lower() for c in df.columns]

    OldAdapter = _load_old_adapter_class()

    from phase_03_paper.signals.adapter import (
        StrategyAdapter as NewAdapter,
    )

    print(f"Running OLD adapter over {csv_path} ...")
    old_result = run_adapter(OldAdapter, df)

    print(f"Running NEW adapter over {csv_path} ...")
    new_result = run_adapter(NewAdapter, df)

    print()
    print("=" * 70)
    print(f"COMPARISON: {csv_path}")
    print("=" * 70)

    mismatches = []

    if old_result["total_triggered"] != new_result["total_triggered"]:
        mismatches.append(
            "total_triggered differs: "
            f"old={old_result['total_triggered']} "
            f"new={new_result['total_triggered']}"
        )

    if old_result["final_pending"] != new_result["final_pending"]:
        mismatches.append(
            "final_pending differs: "
            f"old={old_result['final_pending']} "
            f"new={new_result['final_pending']}"
        )

    if old_result["errors"] != new_result["errors"]:
        mismatches.append(
            "adapter errors differ: "
            f"old={old_result['errors']} "
            f"new={new_result['errors']}"
        )

    if old_result["triggered_by_candle"] != new_result["triggered_by_candle"]:
        # Find the first candle index where they diverge, for a
        # useful pointer instead of just "they differ somewhere".
        first_diff = None
        old_list = old_result["triggered_by_candle"]
        new_list = new_result["triggered_by_candle"]
        for i, (o, n) in enumerate(zip(old_list, new_list)):
            if o != n:
                first_diff = i
                break
        mismatches.append(
            "per-candle triggered events differ, first divergence "
            f"at candle index {first_diff}: "
            f"old={old_list[first_diff] if first_diff is not None else None} "
            f"new={new_list[first_diff] if first_diff is not None else None}"
        )

    if old_result["pending_by_candle"] != new_result["pending_by_candle"]:
        first_diff = None
        old_list = old_result["pending_by_candle"]
        new_list = new_result["pending_by_candle"]
        for i, (o, n) in enumerate(zip(old_list, new_list)):
            if o != n:
                first_diff = i
                break
        mismatches.append(
            "per-candle pending_count differs, first divergence "
            f"at candle index {first_diff}: "
            f"old={old_list[first_diff] if first_diff is not None else None} "
            f"new={new_list[first_diff] if first_diff is not None else None}"
        )

    if not mismatches:
        print("[PASS] OLD and NEW adapters produced IDENTICAL behavior.")
        print(f"  total_triggered : {old_result['total_triggered']}")
        print(f"  final_pending   : {old_result['final_pending']}")
        print(f"  errors          : {old_result['errors']}")
    else:
        print("[FAIL] OLD and NEW adapters DIVERGED:")
        for m in mismatches:
            print(f"  - {m}")

    print("=" * 70)


if __name__ == "__main__":
    main()