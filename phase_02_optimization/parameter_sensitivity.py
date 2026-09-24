"""
phase_02_optimization/parameter_sensitivity.py

PHASE 2 -- PARAMETER SENSITIVITY ANALYSIS
----------------------------------------

Purpose
-------
Determine whether the strategy's historical performance remains
reasonably stable when legitimate backtest parameters are varied
around the current baseline.

This module is NOT a parameter optimizer.

It does not search for the "best" configuration and does not
automatically select or modify any parameter.

It answers:

    "Does the strategy continue to behave reasonably when its
     legitimate backtest parameters are varied around baseline?"

Design rules
------------
- Calls strategy.backtest.run_backtest() unmodified.
- Never rewrites strategy logic.
- Varies exactly ONE parameter at a time.
- All other parameters remain at BASELINE.
- Each tested value is run exactly once on the full dataset.
- The best historical result is informational only.
- No parameter is automatically selected.

Parameters currently tested
----------------------------
A. max_bars_to_retest
       [10, 15, 20, 25, 30]

B. reward_multiple
       [1.5, 1.75, 2.0, 2.25, 2.5]

C. stop_buffer
       [0.0, 0.0005, 0.001, 0.002, 0.003]

       stop_buffer is an ABSOLUTE PRICE DISTANCE in signals.py.
       These values were calibrated against your_data_file.csv's
       actual price scale (close stdev ~= 0.004673, EUR/USD-like
       instrument around 1.076) rather than guessed arbitrarily.

D. entry_mode
       ["midpoint", "extreme"]

Baseline
--------
    max_bars_to_retest = 20
    reward_multiple    = 2.0
    stop_buffer        = 0.0
    entry_mode         = "midpoint"

Metrics
-------
    expectancy_r
    win_rate
    closed_trades
    total_r
    profit_factor

Important
---------
A positive or stable historical result is NOT proof of live
profitability. Results must be interpreted together with:

    - Monte Carlo
    - Warmup sensitivity
    - Walk-forward analysis
    - Window/step sensitivity
    - Overfit detection
    - Further out-of-sample validation
"""

from __future__ import annotations

import argparse
import statistics
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from phase_02_optimization.monte_carlo import extract_closed_r_multiples
from phase_02_optimization.walk_forward import DATA_FILE, load_dataset
from strategy.backtest import BacktestResult, run_backtest


# ============================================================================
# BASELINE
# ============================================================================

BASELINE: dict[str, object] = {
    "max_bars_to_retest": 20,
    "reward_multiple": 2.0,
    "stop_buffer": 0.0,
    "entry_mode": "midpoint",
}


# ============================================================================
# PARAMETER GRIDS
# ============================================================================
#
# stop_buffer is an absolute price-distance (confirmed in signals.py):
#
# LONG:
#     stop_loss = price_bottom - stop_buffer
#
# SHORT:
#     stop_loss = price_top + stop_buffer
#
# Grid values below were calibrated against your_data_file.csv's actual
# price scale (close stdev ~= 0.004673) rather than guessed arbitrarily.
# 0.003 is ~64% of one candle's stdev -- the upper bound before the
# buffer starts meaningfully distorting stop placement relative to the
# zone boundaries themselves.
# ============================================================================

PARAMETER_GRIDS: dict[str, list[object]] = {
    "max_bars_to_retest": [10, 15, 20, 25, 30],

    "reward_multiple": [1.5, 1.75, 2.0, 2.25, 2.5],

    "stop_buffer": [0.0, 0.0005, 0.001, 0.002, 0.003],

    "entry_mode": ["midpoint", "extreme"],
}


# ============================================================================
# RESULT OBJECT
# ============================================================================

@dataclass(frozen=True)
class ParameterValueResult:
    """Metrics captured for one tested value of one parameter."""

    parameter_name: str
    value: object
    is_baseline: bool

    closed_trades: int
    expectancy_r: Optional[float]
    win_rate: Optional[float]
    total_r: Optional[float]
    profit_factor: Optional[float]

    backtest_result: Optional[BacktestResult]
    error: Optional[str] = None


# ============================================================================
# METRIC CALCULATION
# ============================================================================

def _compute_metrics(
    result: BacktestResult,
) -> tuple[
    int,
    Optional[float],
    Optional[float],
    Optional[float],
    Optional[float],
]:
    """
    Calculate metrics from closed-trade R multiples.

    Returns:
        (
            closed_trades,
            expectancy_r,
            win_rate,
            total_r,
            profit_factor,
        )

    If there are no closed trades, the metric values are None.
    """

    r_multiples = extract_closed_r_multiples(result)

    closed_trades = len(r_multiples)

    if closed_trades == 0:
        return (
            0,
            None,
            None,
            None,
            None,
        )

    expectancy_r = statistics.fmean(r_multiples)

    total_r = sum(r_multiples)

    wins = [
        r for r in r_multiples
        if r > 0
    ]

    losses = [
        r for r in r_multiples
        if r < 0
    ]

    win_rate = len(wins) / closed_trades

    gross_win = sum(wins)

    gross_loss = abs(sum(losses))

    if gross_loss > 0:
        profit_factor = gross_win / gross_loss
    elif gross_win > 0:
        # No losing trades.
        #
        # We deliberately return None instead of infinity so that
        # downstream statistics do not treat infinity as a number.
        profit_factor = None
    else:
        profit_factor = 0.0

    return (
        closed_trades,
        expectancy_r,
        win_rate,
        total_r,
        profit_factor,
    )


# ============================================================================
# SINGLE-PARAMETER SWEEP
# ============================================================================

def run_parameter_sweep(
    df: pd.DataFrame,
    parameter_name: str,
    values: list[object],
) -> list[ParameterValueResult]:
    """
    Run one full-dataset backtest for every supplied value.

    Exactly ONE parameter changes.

    Every other parameter remains at BASELINE.
    """

    if parameter_name not in BASELINE:
        raise ValueError(
            f"Unknown parameter: {parameter_name!r}. "
            f"Must be one of: {sorted(BASELINE)}"
        )

    results: list[ParameterValueResult] = []

    for value in values:

        # Start from a fresh copy of the baseline.
        kwargs = dict(BASELINE)

        # Change ONLY the parameter being swept.
        kwargs[parameter_name] = value

        is_baseline = (
            value == BASELINE[parameter_name]
        )

        print(
            f"  Running {parameter_name}={value}..."
        )

        try:
            result = run_backtest(
                df,
                **kwargs,
            )

        except Exception as exc:

            results.append(
                ParameterValueResult(
                    parameter_name=parameter_name,
                    value=value,
                    is_baseline=is_baseline,
                    closed_trades=0,
                    expectancy_r=None,
                    win_rate=None,
                    total_r=None,
                    profit_factor=None,
                    backtest_result=None,
                    error=(
                        f"run_backtest() raised "
                        f"{type(exc).__name__}: {exc}"
                    ),
                )
            )

            continue

        (
            closed_trades,
            expectancy_r,
            win_rate,
            total_r,
            profit_factor,
        ) = _compute_metrics(result)

        results.append(
            ParameterValueResult(
                parameter_name=parameter_name,
                value=value,
                is_baseline=is_baseline,
                closed_trades=closed_trades,
                expectancy_r=expectancy_r,
                win_rate=win_rate,
                total_r=total_r,
                profit_factor=profit_factor,
                backtest_result=result,
                error=None,
            )
        )

    return results


# ============================================================================
# FORMATTING
# ============================================================================

def _fmt(
    value: Optional[float],
    suffix: str = "",
    digits: int = 4,
) -> str:
    """Format an optional numeric value for terminal output."""

    if value is None:
        return "N/A"

    return f"{value:.{digits}f}{suffix}"


# ============================================================================
# REPORT
# ============================================================================

def print_parameter_sensitivity_report(
    results: list[ParameterValueResult],
) -> None:
    """Print the complete sensitivity report."""

    if not results:
        print("No results to report.")
        return

    parameter_name = results[0].parameter_name

    print()
    print("=" * 120)
    print(
        f"PHASE 2 -- PARAMETER SENSITIVITY: {parameter_name}"
    )
    print("=" * 120)

    print()
    print("BASELINE CONFIGURATION")
    print("-" * 120)

    for key, value in BASELINE.items():

        marker = ""

        if key == parameter_name:
            marker = "  <-- CURRENTLY SWEPT"

        print(
            f"  {key:<22}: {value}{marker}"
        )

    print()
    print("RESULTS")
    print("-" * 120)

    print(
        f"{'Value':>12} "
        f"{'Baseline':>10} "
        f"{'Closed':>8} "
        f"{'Expectancy R':>15} "
        f"{'Win Rate':>11} "
        f"{'Total R':>11} "
        f"{'Profit Factor':>15}"
    )

    print("-" * 120)

    for row in results:

        if row.win_rate is not None:
            win_rate_text = (
                f"{row.win_rate * 100.0:.2f}%"
            )
        else:
            win_rate_text = "N/A"

        print(
            f"{str(row.value):>12} "
            f"{'YES' if row.is_baseline else '':>10} "
            f"{row.closed_trades:>8} "
            f"{_fmt(row.expectancy_r, 'R'):>15} "
            f"{win_rate_text:>11} "
            f"{_fmt(row.total_r, 'R'):>11} "
            f"{_fmt(row.profit_factor):>15}"
        )

        if row.error:
            print(
                f"      ERROR: {row.error}"
            )

    # ------------------------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------------------------

    usable = [
        row
        for row in results
        if row.expectancy_r is not None
    ]

    print()
    print("SUMMARY")
    print("-" * 120)

    if not usable:
        print(
            "No usable expectancy values across this parameter grid."
        )

        print()
        print("STABILITY OBSERVATIONS")
        print("-" * 120)
        print(
            "No stability conclusion can be made because there "
            "were no usable closed-trade results."
        )

        print("=" * 120)
        print()

        return

    expectancies = [
        row.expectancy_r
        for row in usable
        if row.expectancy_r is not None
    ]

    trade_counts = [
        row.closed_trades
        for row in usable
    ]

    baseline_result = next(
        (
            row
            for row in results
            if row.is_baseline
        ),
        None,
    )

    best_result = max(
        usable,
        key=lambda row: row.expectancy_r,
    )

    worst_result = min(
        usable,
        key=lambda row: row.expectancy_r,
    )

    median_expectancy = statistics.median(
        expectancies
    )

    minimum_expectancy = min(
        expectancies
    )

    maximum_expectancy = max(
        expectancies
    )

    expectancy_range = (
        maximum_expectancy
        - minimum_expectancy
    )

    baseline_expectancy_text = _fmt(
        baseline_result.expectancy_r if baseline_result else None,
        "R",
    )

    print(
        f"Baseline expectancy   : {baseline_expectancy_text}"
    )

    print(
        "Best expectancy       : "
        f"{_fmt(best_result.expectancy_r, 'R')} "
        f"(value={best_result.value})"
    )

    print(
        "Worst expectancy      : "
        f"{_fmt(worst_result.expectancy_r, 'R')} "
        f"(value={worst_result.value})"
    )

    print(
        "Median expectancy     : "
        f"{_fmt(median_expectancy, 'R')}"
    )

    print(
        "Min expectancy        : "
        f"{_fmt(minimum_expectancy, 'R')}"
    )

    print(
        "Max expectancy        : "
        f"{_fmt(maximum_expectancy, 'R')}"
    )

    print(
        "Expectancy range      : "
        f"{_fmt(expectancy_range, 'R')}"
    )

    print(
        "Trade-count range     : "
        f"{min(trade_counts)} to {max(trade_counts)}"
    )

    # ------------------------------------------------------------------------
    # STABILITY OBSERVATIONS
    # ------------------------------------------------------------------------

    print()
    print("STABILITY OBSERVATIONS")
    print("-" * 120)

    print(
        "1. This is a one-parameter-at-a-time robustness test."
    )

    print(
        "2. The best historical value is reported for information only."
    )

    print(
        "   It is NOT automatically selected as the new configuration."
    )

    print(
        "3. A broad region of similar results is more informative"
    )

    print(
        "   than one isolated historical peak."
    )

    print(
        "4. A weaker parameter value does not automatically prove"
    )

    print(
        "   overfitting. Interpret it together with walk-forward,"
    )

    print(
        "   Monte Carlo, and later overfit-detection tests."
    )

    print(
        "5. No swing, structure, zone, liquidity, signal, or"
    )

    print(
        "   lifecycle logic was modified by this module."
    )

    if minimum_expectancy > 0:

        print(
            "6. Every tested value with usable trades remained"
        )

        print(
            "   positive in historical expectancy."
        )

    else:

        print(
            "6. At least one tested value was non-positive."
        )

        print(
            "   This parameter shows sensitivity that requires"
        )

        print(
            "   further investigation."
        )

    if parameter_name == "entry_mode":

        print()
        print(
            "ENTRY MODE NOTE:"
        )

        print(
            "Extreme mode has special handling in signals.py:"
        )

        print(
            "when stop_buffer <= 0, the signal engine uses"
        )

        print(
            "a 0.01 minimum stop distance for extreme mode."
        )

        print(
            "Therefore the extreme-mode result is not a pure"
        )

        print(
            "entry-location comparison against a literal zero"
        )

        print(
            "stop distance."
        )

    if parameter_name == "stop_buffer":

        print()
        print(
            "STOP BUFFER NOTE:"
        )

        print(
            "Grid values were calibrated against this dataset's"
        )

        print(
            "actual price scale (close stdev ~= 0.004673), not"
        )

        print(
            "guessed arbitrarily -- see module docstring."
        )

    print("=" * 120)
    print()


# ============================================================================
# COMMAND-LINE ARGUMENTS
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 2 parameter sensitivity analysis. "
            "Tests ONE parameter dimension at a time."
        )
    )

    parser.add_argument(
        "--param",
        type=str,
        default="max_bars_to_retest",
        choices=sorted(
            PARAMETER_GRIDS.keys()
        ),
        help=(
            "Parameter to sweep. "
            "Default: max_bars_to_retest."
        ),
    )

    return parser.parse_args()


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:

    args = parse_args()

    print()
    print("=" * 120)
    print("PHASE 2 -- PARAMETER SENSITIVITY ANALYSIS")
    print("=" * 120)

    print()
    print("LOADING DATASET")
    print("-" * 120)

    df = load_dataset(
        DATA_FILE
    )

    print(
        f"Dataset : {DATA_FILE}"
    )

    print(
        f"Candles : {len(df):,}"
    )

    print(
        f"Start   : {df.index[0]}"
    )

    print(
        f"End     : {df.index[-1]}"
    )

    values = PARAMETER_GRIDS[
        args.param
    ]

    print()
    print(
        f"Sweeping parameter : {args.param}"
    )

    print(
        f"Values             : {values}"
    )

    print(
        "Other parameters   : BASELINE"
    )

    print()
    print(
        "Baseline:"
    )

    for key, value in BASELINE.items():

        print(
            f"  {key:<22} = {value}"
        )

    print()

    results = run_parameter_sweep(
        df,
        args.param,
        values,
    )

    print_parameter_sensitivity_report(
        results
    )


if __name__ == "__main__":
    main()