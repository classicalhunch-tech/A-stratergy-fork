"""
phase_02_optimization/test_structure_state_equivalence.py

Proves StructureState.step(), fed rows and swings one at a time in
causal order, produces IDENTICAL breaks / classified_swings / final
trend to analyze_structure(df, swings) called once on the complete
history.

Mirrors phase_02_optimization/test_swing_state_equivalence.py's
approach and phase_02_optimization/test_zone_equivalence.py's staged
real-dataset checks.
"""

import pandas as pd

from strategy.swings import find_swings, SwingState
from strategy.structure import analyze_structure, StructureState


def _run_reference(df: pd.DataFrame):
    """The correctness reference: unchanged find_swings + analyze_structure."""
    swings = find_swings(df)
    breaks, classified, trend = analyze_structure(df, swings)
    return breaks, classified, trend


def _run_incremental(df: pd.DataFrame):
    """
    The incremental path: SwingState + StructureState, fed one row
    at a time, exactly the way StrategyAdapter.on_candle() will feed
    them live.
    """
    swing_state = SwingState()
    structure_state = StructureState()

    confirmed_swings_seen = 0

    for i in range(len(df)):
        row = df.iloc[i]
        ts = df.index[i]

        confirmed = swing_state.step(
            ts=ts,
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
        )

        new_swings = []
        if confirmed is not None:
            new_swings.append(confirmed)

        if new_swings:
            structure_state.add_swings(new_swings)

        structure_state.step(ts=ts, close=float(row["close"]))

    return (
        structure_state.breaks,
        structure_state.classified_swings,
        structure_state.current_trend,
    )


def _breaks_equal(breaks_a, breaks_b) -> bool:
    if len(breaks_a) != len(breaks_b):
        return False
    for a, b in zip(breaks_a, breaks_b):
        if (
            a.break_type != b.break_type
            or a.direction != b.direction
            or a.broken_at != b.broken_at
            or round(a.break_price, 8) != round(b.break_price, 8)
            or a.broken_swing.price != b.broken_swing.price
            or a.broken_swing.swing_type != b.broken_swing.swing_type
        ):
            return False
    return True


def _classified_equal(classified_a, classified_b) -> bool:
    if len(classified_a) != len(classified_b):
        return False
    for a, b in zip(classified_a, classified_b):
        if (
            a.label != b.label
            or a.swing.price != b.swing.price
            or a.swing.swing_type != b.swing.swing_type
            or a.swing.confirmed_at != b.swing.confirmed_at
        ):
            return False
    return True


def run_equivalence_check(csv_path: str, n: int = None, label: str = ""):
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    df = df.set_index("timestamp")
    if n is not None:
        df = df.head(n)

    ref_breaks, ref_classified, ref_trend = _run_reference(df)
    inc_breaks, inc_classified, inc_trend = _run_incremental(df)

    breaks_ok = _breaks_equal(ref_breaks, inc_breaks)
    classified_ok = _classified_equal(ref_classified, inc_classified)
    trend_ok = (ref_trend == inc_trend)

    status = "PASS" if (breaks_ok and classified_ok and trend_ok) else "FAIL"

    print(f"[{status}] {label or csv_path} (n={n or len(df)})")
    print(f"    breaks: ref={len(ref_breaks)} inc={len(inc_breaks)} match={breaks_ok}")
    print(f"    classified_swings: ref={len(ref_classified)} inc={len(inc_classified)} match={classified_ok}")
    print(f"    final_trend: ref={ref_trend} inc={inc_trend} match={trend_ok}")
    print()

    return status == "PASS"


if __name__ == "__main__":
    results = []

    for n in [500, 1000, 2000, 5000]:
        results.append(
            run_equivalence_check("real_gold_data_mt5_10000.csv", n=n, label=f"real MT5 data, {n} candles")
        )

    all_passed = all(results)
    print("=" * 60)
    print("ALL PASSED" if all_passed else "SOME FAILED -- DO NOT WIRE INTO ADAPTER YET")
    print("=" * 60)