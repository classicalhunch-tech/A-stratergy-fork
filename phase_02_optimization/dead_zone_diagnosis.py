"""
phase_02_optimization/dead_zone_diagnosis.py

Read-only diagnostic: determines whether the Feb-May 2026 dead zone
(walk-forward windows 99-129, near-zero trade counts) is caused by
the base 5M signal engine finding no candidates, or by the MTF
confluence filter rejecting candidates that were generated.

Does NOT modify strategy/backtest.py, strategy/confluence.py, or any
other strategy module. Compares an unfiltered backtest run against an
MTF-filtered run over the same full history, and inspects the raw
macro_trend / internal_trend columns for the suspect date range.
"""

import pandas as pd

from strategy.backtest import run_backtest
from strategy.mtf_structure import build_mtf_dataset_with_structure
from strategy.swings import find_swings
from strategy.confluence import build_mtf_signal_filter
from phase_02_optimization.walk_forward import load_dataset, DATA_FILE

# Approximate dead-zone boundaries (walk-forward windows 99-129).
DEAD_ZONE_START = pd.Timestamp("2026-02-18")
DEAD_ZONE_END = pd.Timestamp("2026-05-18")


def count_trades_in_range(trades, start, end):
    count = 0
    for trade in trades:
        entry_time = getattr(trade, "entry_time", None)
        if entry_time is None:
            continue
        try:
            entry_time = pd.Timestamp(entry_time)
        except (TypeError, ValueError):
            continue
        if start <= entry_time <= end:
            count += 1
    return count


def main():
    df = load_dataset(DATA_FILE)

    print("Building MTF context...")
    df_enriched = build_mtf_dataset_with_structure(
        df,
        macro_swings_fn=find_swings,
        internal_swings_fn=find_swings,
    )
    mtf_filter_fn = build_mtf_signal_filter(df_enriched)

    print("Running UNFILTERED backtest (mtf_filter_fn=None)...")
    result_unfiltered = run_backtest(
        df,
        max_bars_to_retest=20,
        reward_multiple=2.0,
        stop_buffer=0.0,
        entry_mode="midpoint",
        mtf_filter_fn=None,
    )

    print("Running MTF-FILTERED backtest...")
    result_filtered = run_backtest(
        df,
        max_bars_to_retest=20,
        reward_multiple=2.0,
        stop_buffer=0.0,
        entry_mode="midpoint",
        mtf_filter_fn=mtf_filter_fn,
    )

    unfiltered_trades = getattr(result_unfiltered, "trades", result_unfiltered)
    filtered_trades = getattr(result_filtered, "trades", result_filtered)

    unfiltered_count = count_trades_in_range(
        unfiltered_trades, DEAD_ZONE_START, DEAD_ZONE_END
    )
    filtered_count = count_trades_in_range(
        filtered_trades, DEAD_ZONE_START, DEAD_ZONE_END
    )

    print()
    print("=" * 80)
    print(f"Dead zone range: {DEAD_ZONE_START} to {DEAD_ZONE_END}")
    print("=" * 80)
    print(f"Unfiltered (raw 5M engine) trades in range : {unfiltered_count}")
    print(f"MTF-filtered trades in range                : {filtered_count}")
    print()

    if unfiltered_count == 0:
        print(
            "DIAGNOSIS: base engine found ~no candidates in this period. "
            "This looks like a genuine base-engine dead period, not MTF "
            "filter starvation. Check volatility/ATR next."
        )
    elif unfiltered_count > 0 and filtered_count == 0:
        print(
            "DIAGNOSIS: base engine found candidates, but the MTF filter "
            "rejected ALL of them. This points to filter starvation, "
            "likely macro_trend disagreement -- see trend distribution below."
        )
    else:
        print(
            "DIAGNOSIS: mixed result -- some candidates passed the filter. "
            "Compare this ratio against an active window as a baseline."
        )

    in_zone = (df_enriched.index >= DEAD_ZONE_START) & (df_enriched.index <= DEAD_ZONE_END)

    print()
    print("macro_trend distribution INSIDE dead zone:")
    print(df_enriched.loc[in_zone, "macro_trend"].value_counts(dropna=False))

    print()
    print("internal_trend distribution INSIDE dead zone:")
    print(df_enriched.loc[in_zone, "internal_trend"].value_counts(dropna=False))

    print()
    print("macro_trend distribution OUTSIDE dead zone (whole rest of history):")
    print(df_enriched.loc[~in_zone, "macro_trend"].value_counts(dropna=False))

    print()
    print("internal_trend distribution OUTSIDE dead zone (whole rest of history):")
    print(df_enriched.loc[~in_zone, "internal_trend"].value_counts(dropna=False))


if __name__ == "__main__":
    main()