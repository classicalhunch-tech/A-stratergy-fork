"""
phase_02_optimization/stop_buffer_diagnostic.py

Diagnostic for the stop_buffer cliff found in Milestone 5, Parameter C.

Purpose
-------
stop_buffer=0.0    -> expectancy 1.2500R
stop_buffer=0.0005 -> expectancy -0.0625R

Trade count was IDENTICAL (16) at every tested value, meaning this is
not a signal-generation difference -- the same trades are being taken,
just with a different stop-loss level. This script identifies exactly
which trades flip WIN<->LOSS between the two values and by how much,
so the instability can be understood before deciding how (or whether)
to proceed to overfit detection.

This module calls strategy.backtest.run_backtest() unmodified and
performs no strategy changes -- it is read-only diagnostics.

Added: an intermediate-value sweep across
0.0000, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005
to determine whether the expectancy collapse is a smooth gradient or a
sudden cliff at a specific threshold.
"""
from __future__ import annotations

from strategy.backtest import run_backtest
from phase_02_optimization.walk_forward import load_dataset, DATA_FILE

BASELINE_BUFFER = 0.0
COMPARISON_BUFFER = 0.0005

SWEEP_VALUES = [0.0000, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005]

COMMON_KWARGS = dict(
    max_bars_to_retest=20,
    reward_multiple=2.0,
    entry_mode="midpoint",
)


def run_pairwise_diagnostic() -> None:
    df = load_dataset(DATA_FILE)

    print()
    print("=" * 120)
    print("STOP BUFFER CLIFF DIAGNOSTIC")
    print("=" * 120)
    print(f"Comparing stop_buffer={BASELINE_BUFFER} vs stop_buffer={COMPARISON_BUFFER}")
    print()

    result_a = run_backtest(df, stop_buffer=BASELINE_BUFFER, **COMMON_KWARGS)
    result_b = run_backtest(df, stop_buffer=COMPARISON_BUFFER, **COMMON_KWARGS)

    closed_a = {
        t.entry_time: t
        for t in result_a.trades
        if t.result_status in ("WIN", "LOSS")
    }
    closed_b = {
        t.entry_time: t
        for t in result_b.trades
        if t.result_status in ("WIN", "LOSS")
    }

    # --- NEW: show any entry_time that is closed in one run but not
    # the other, so the 16-vs-15 discrepancy is visible rather than
    # silently dropped by the set intersection below. ---
    only_in_a = sorted(set(closed_a.keys()) - set(closed_b.keys()))
    only_in_b = sorted(set(closed_b.keys()) - set(closed_a.keys()))
    if only_in_a or only_in_b:
        print("Closed-trade mismatch between runs (likely open-at-end-of-data boundary effect):")
        for et in only_in_a:
            t = closed_a[et]
            print(f"  entry_time={et}: closed in run A only (status_a={t.result_status})")
            # find its status in run B, whatever it is (e.g. still OPEN)
            match_b = next((t2 for t2 in result_b.trades if t2.entry_time == et), None)
            if match_b is not None:
                print(f"    -> in run B this trade has result_status={match_b.result_status}")
        for et in only_in_b:
            t = closed_b[et]
            print(f"  entry_time={et}: closed in run B only (status_b={t.result_status})")
            match_a = next((t2 for t2 in result_a.trades if t2.entry_time == et), None)
            if match_a is not None:
                print(f"    -> in run A this trade has result_status={match_a.result_status}")
        print()

    common_entry_times = sorted(
        set(closed_a.keys()) & set(closed_b.keys())
    )

    print(
        f"{'Entry Time':<22} "
        f"{'Direction':>9} "
        f"{'SL (a)':>10} "
        f"{'SL (b)':>10} "
        f"{'Status (a)':>10} "
        f"{'Status (b)':>10} "
        f"{'R (a)':>8} "
        f"{'R (b)':>8} "
        f"{'FLIPPED':>8}"
    )
    print("-" * 120)

    flips = 0
    for entry_time in common_entry_times:
        trade_a = closed_a[entry_time]
        trade_b = closed_b[entry_time]
        flipped = trade_a.result_status != trade_b.result_status
        if flipped:
            flips += 1
        print(
            f"{str(entry_time):<22} "
            f"{str(trade_a.direction):>9} "
            f"{trade_a.stop_loss:>10.5f} "
            f"{trade_b.stop_loss:>10.5f} "
            f"{trade_a.result_status:>10} "
            f"{trade_b.result_status:>10} "
            f"{trade_a.r_multiple:>8.4f} "
            f"{trade_b.r_multiple:>8.4f} "
            f"{'YES' if flipped else '':>8}"
        )

    print("-" * 120)
    print(f"Total common closed trades : {len(common_entry_times)}")
    print(f"Trades that flipped result : {flips}")
    print("=" * 120)
    print()


def run_sweep() -> None:
    """
    Sweep stop_buffer across SWEEP_VALUES using the same canonical
    run_backtest() call, and report expectancy/win-count/trade-count at
    each value. This reveals whether the 0.0 -> 0.0005 collapse is a
    smooth gradient or a sudden cliff.
    """
    df = load_dataset(DATA_FILE)

    print()
    print("=" * 90)
    print("STOP BUFFER INTERMEDIATE-VALUE SWEEP")
    print("=" * 90)
    print(
        f"{'stop_buffer':>12} "
        f"{'closed':>8} "
        f"{'wins':>6} "
        f"{'losses':>8} "
        f"{'win_rate':>10} "
        f"{'expectancy_R':>14}"
    )
    print("-" * 90)

    for buffer_value in SWEEP_VALUES:
        result = run_backtest(df, stop_buffer=buffer_value, **COMMON_KWARGS)

        closed_trades = [
            t for t in result.trades if t.result_status in ("WIN", "LOSS")
        ]
        wins = [t for t in closed_trades if t.result_status == "WIN"]
        losses = [t for t in closed_trades if t.result_status == "LOSS"]

        n_closed = len(closed_trades)
        n_wins = len(wins)
        n_losses = len(losses)
        win_rate = (n_wins / n_closed * 100) if n_closed else 0.0

        expectancy = (
            sum(t.r_multiple for t in closed_trades) / n_closed
            if n_closed
            else 0.0
        )

        print(
            f"{buffer_value:>12.4f} "
            f"{n_closed:>8} "
            f"{n_wins:>6} "
            f"{n_losses:>8} "
            f"{win_rate:>9.1f}% "
            f"{expectancy:>14.4f}"
        )

    print("-" * 90)
    print(
        "Read this top-to-bottom: a SMOOTH decline means stop_buffer is a "
        "sensitive-but-continuous parameter (no single bad threshold). A "
        "CLIFF (expectancy holds, then drops sharply between two adjacent "
        "values) points to a specific price-excursion threshold affecting "
        "several trades at once."
    )
    print("=" * 90)
    print()


def main() -> None:
    run_pairwise_diagnostic()
    run_sweep()


if __name__ == "__main__":
    main()