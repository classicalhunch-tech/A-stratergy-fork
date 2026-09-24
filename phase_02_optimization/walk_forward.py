"""
phase_02_optimization/walk_forward.py

Walk-Forward Stability Analysis for Phase 2
--------------------------------------------

Purpose
-------
Determine whether the already-backtested Machet Mechanics strategy
remains reasonably stable across different historical periods.

This module is a STABILITY TEST.

It is NOT:
    - a parameter optimizer
    - a strategy fitter
    - a signal generator
    - a Monte Carlo simulator


Design
------
The canonical strategy.backtest.run_backtest() engine is called once
for every fixed-size sliding window.

Every window uses exactly the same strategy parameters.

Window architecture:

    [ WARMUP ][ EVALUATION ]
         |
         v
    run_backtest()
         |
         v
    retain only trades that:
        1. entered during evaluation
        2. closed during evaluation
        3. have WIN/LOSS status

Warmup trades are discarded from evaluation metrics.


IMPORTANT
---------
A trade that enters during evaluation but remains OPEN at the end of
the evaluation window is NOT counted.

A trade that enters during evaluation but closes after the evaluation
window is also NOT counted.

This avoids leaking information from future evaluation periods into
the metrics for the current window.


Warmup
------
The warmup probe was run against the real 5,000-candle dataset.

Observed candidate warmups:

    500 candles
    750 candles

Both are retained as candidates for walk-forward validation.

The probe does NOT determine the final warmup automatically.

The first validation run uses 500 candles.
A separate run will then use 750 candles.

The final warmup should be selected only after comparing the
walk-forward stability results of both candidates.


Window design
-------------
Default evaluation configuration:

    window_size = 1000
    step_size   = 500

The final partial window is discarded.


CLI overrides
-------------
window_size, step_size, and warmup_size are overridable via
--window / --step / --warmup CLI arguments to support Phase 2
window/step sensitivity testing. Running with no arguments
reproduces the original baseline (window=1000, step=500, warmup=750).


Metrics
-------
Primary:
    expectancy_r

Secondary:
    win_rate

A window's expectancy is the arithmetic mean of realized R-multiples
for trades fully contained inside that evaluation region.

In addition to the per-window distribution (each window weighted
equally regardless of trade count), a trade-weighted aggregate is
also computed by pooling every individual closed trade's R-multiple
across all windows. This second view avoids letting low-trade-count
windows (e.g. a single-trade window, which can only land at 100% or
0% win rate) distort the picture the same way they can distort a
per-window mean/median.

No stability threshold is hard-coded here.

Thresholds should be decided only after observing the actual
walk-forward distribution.


Known scope
-----------
run_backtest() currently represents the canonical single-timeframe
backtest engine.

run_backtest() now accepts an optional mtf_filter_fn hook (see
strategy.backtest and strategy.confluence.build_mtf_signal_filter).
This module passes that hook straight through to run_backtest() for
every window when the caller supplies one via run_walk_forward()'s
mtf_filter_fn parameter.

When mtf_filter_fn is None (the default), behavior is byte-for-byte
identical to the original single-timeframe-only validation this
module always performed. Passing a filter built from
build_mtf_signal_filter() extends this same stability analysis to
the MTF-gated trade stream, without duplicating any confluence or
execution logic here.
"""

from __future__ import annotations

import argparse
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

from phase_02_optimization.monte_carlo import (
    _normalize_status,
    _percentile_summary,
    extract_closed_r_multiples,
)
from strategy.backtest import (
    BacktestResult,
    run_backtest,
)
from strategy.mtf_structure import build_mtf_dataset_with_structure
from strategy.swings import find_swings
from strategy.confluence import build_mtf_signal_filter


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass(frozen=True)
class WalkForwardConfig:
    """
    Configuration for fixed-size sliding walk-forward analysis.
    """

    window_size: int
    step_size: int
    warmup_size: int

    max_bars_to_retest: int = 20
    reward_multiple: float = 2.0
    stop_buffer: float = 0.0
    entry_mode: str = "midpoint"

    def __post_init__(self) -> None:
        if self.window_size <= 0:
            raise ValueError(
                "window_size must be positive"
            )

        if self.step_size <= 0:
            raise ValueError(
                "step_size must be positive"
            )

        if self.warmup_size < 0:
            raise ValueError(
                "warmup_size must be >= 0"
            )

        if self.max_bars_to_retest <= 0:
            raise ValueError(
                "max_bars_to_retest must be positive"
            )

        if self.reward_multiple <= 0:
            raise ValueError(
                "reward_multiple must be positive"
            )

        if self.stop_buffer < 0:
            raise ValueError(
                "stop_buffer must be >= 0"
            )

        if self.entry_mode not in {
            "midpoint",
            "extreme",
        }:
            raise ValueError(
                "entry_mode must be 'midpoint' or 'extreme'"
            )


# ============================================================================
# RESULT OBJECTS
# ============================================================================

@dataclass(frozen=True)
class WindowResult:
    """Metrics captured for one walk-forward evaluation window."""

    window_index: int

    slice_start_idx: int
    eval_start_idx: int
    end_idx: int

    eval_start_timestamp: pd.Timestamp
    eval_end_timestamp: pd.Timestamp

    num_eval_candles: int
    num_warmup_candles: int

    # Closed WIN/LOSS trades fully contained in evaluation.
    num_closed_trades: int

    # Trades whose entry occurred before evaluation.
    num_warmup_trades_discarded: int

    # Evaluation trades that did not produce a valid closed result
    # within the evaluation region.
    num_eval_open_or_cross_boundary: int

    # Win rate represented internally as 0.0 - 1.0.
    win_rate: Optional[float]

    # Mean realized R.
    expectancy_r: Optional[float]

    # Complete unfiltered backtest result.
    backtest_result: Optional[BacktestResult]

    # Error from run_backtest(), if any.
    error: Optional[str] = None


@dataclass(frozen=True)
class WalkForwardResult:
    """Aggregate result across all walk-forward windows."""

    config: WalkForwardConfig

    windows: tuple[WindowResult, ...]

    num_windows: int

    summary: dict[str, dict[str, float]]

    errors: tuple[str, ...] = ()


# ============================================================================
# INPUT VALIDATION
# ============================================================================

def _validate_walk_forward_inputs(
    df: pd.DataFrame,
    config: WalkForwardConfig,
) -> None:
    """Validate dataframe and ensure at least one full window exists."""

    if not isinstance(
        df,
        pd.DataFrame,
    ):
        raise ValueError(
            "df must be a pandas DataFrame"
        )

    if not isinstance(
        df.index,
        pd.DatetimeIndex,
    ):
        raise ValueError(
            "df must have a DatetimeIndex"
        )

    if df.index.has_duplicates:
        raise ValueError(
            "df index must not contain duplicate timestamps"
        )

    if not df.index.is_monotonic_increasing:
        raise ValueError(
            "df index must be sorted in ascending chronological order"
        )

    total_slice = (
        config.warmup_size
        + config.window_size
    )

    if len(df) < total_slice:
        raise ValueError(
            f"df has {len(df)} rows, fewer than "
            f"warmup_size + window_size = {total_slice}; "
            "no full walk-forward window can be formed"
        )


# ============================================================================
# WINDOW GENERATION
# ============================================================================

def _window_starts(
    total_len: int,
    total_slice: int,
    step_size: int,
) -> list[int]:
    """
    Generate full-size sliding windows.

    Each returned start index represents:

        [ WARMUP ][ EVALUATION ]

    Partial trailing windows are discarded.
    """

    starts: list[int] = []

    start = 0

    while (
        start + total_slice
        <= total_len
    ):
        starts.append(start)
        start += step_size

    return starts


# ============================================================================
# TRADE FILTERING
# ============================================================================

def _window_stats(
    result: BacktestResult,
    eval_start_timestamp: pd.Timestamp,
    eval_end_timestamp: pd.Timestamp,
) -> tuple[
    Optional[float],
    Optional[float],
    int,
    int,
    int,
    list[float],
    int,
]:
    """
    Calculate metrics using ONLY trades fully contained inside evaluation.

    A trade qualifies only when:

        result_status in {"WIN", "LOSS"}

        AND

        entry_time >= eval_start_timestamp

        AND

        exit_time <= eval_end_timestamp

    Returns:

        win_rate
        expectancy_r
        number of closed evaluation trades
        number of warmup trades discarded
        number of evaluation trades still open/crossing boundary
        list of individual realized R-multiples for this window's
            closed evaluation trades (for trade-weighted pooling)
        number of wins among this window's closed evaluation trades
            (for trade-weighted pooling)
    """

    all_trades = getattr(
        result,
        "trades",
        result,
    )

    eval_closed_trades = []

    num_warmup_discarded = 0
    num_eval_open_or_cross_boundary = 0

    for trade in all_trades:

        entry_time = getattr(
            trade,
            "entry_time",
            None,
        )

        exit_time = getattr(
            trade,
            "exit_time",
            None,
        )

        status = _normalize_status(
            getattr(
                trade,
                "result_status",
                None,
            )
        )

        # ------------------------------------------------------------
        # Cannot attribute a trade without entry time.
        # ------------------------------------------------------------

        if entry_time is None:
            continue

        try:
            entry_time = pd.Timestamp(
                entry_time
            )
        except (
            TypeError,
            ValueError,
        ):
            continue

        # ------------------------------------------------------------
        # Trade belongs to warmup.
        # ------------------------------------------------------------

        if entry_time < eval_start_timestamp:
            num_warmup_discarded += 1
            continue

        # ------------------------------------------------------------
        # Evaluation trade must be closed WIN/LOSS.
        # ------------------------------------------------------------

        if status not in {
            "WIN",
            "LOSS",
        }:
            num_eval_open_or_cross_boundary += 1
            continue

        # ------------------------------------------------------------
        # A closed trade must have an exit timestamp.
        # ------------------------------------------------------------

        if exit_time is None:
            num_eval_open_or_cross_boundary += 1
            continue

        try:
            exit_time = pd.Timestamp(
                exit_time
            )
        except (
            TypeError,
            ValueError,
        ):
            num_eval_open_or_cross_boundary += 1
            continue

        # ------------------------------------------------------------
        # Trade must close inside evaluation.
        # ------------------------------------------------------------

        if exit_time > eval_end_timestamp:
            num_eval_open_or_cross_boundary += 1
            continue

        eval_closed_trades.append(
            trade
        )

    # ----------------------------------------------------------------
    # Reuse Monte Carlo's validated R extraction logic.
    # ----------------------------------------------------------------

    r_multiples = extract_closed_r_multiples(
        eval_closed_trades
    )

    if not r_multiples:
        return (
            None,
            None,
            0,
            num_warmup_discarded,
            num_eval_open_or_cross_boundary,
            [],
            0,
        )

    wins = sum(
        1
        for trade in eval_closed_trades
        if _normalize_status(
            getattr(
                trade,
                "result_status",
                None,
            )
        ) == "WIN"
    )

    win_rate = (
        wins
        / len(r_multiples)
    )

    expectancy_r = statistics.fmean(
        r_multiples
    )

    return (
        win_rate,
        expectancy_r,
        len(r_multiples),
        num_warmup_discarded,
        num_eval_open_or_cross_boundary,
        r_multiples,
        wins,
    )


# ============================================================================
# MAIN WALK-FORWARD ENGINE
# ============================================================================

def run_walk_forward(
    df: pd.DataFrame,
    config: WalkForwardConfig,
    mtf_filter_fn: Optional[Callable] = None,
) -> WalkForwardResult:
    """
    Run deterministic walk-forward stability analysis.

    The SAME fixed strategy parameters are passed to run_backtest()
    for every window.

    No parameter search occurs here.

    mtf_filter_fn:
        Optional callable(signal, current_time) -> bool, passed
        straight through to run_backtest() for every window (see
        strategy.confluence.build_mtf_signal_filter). Build it ONCE
        from the full dataset's MTF-enriched context before calling
        run_walk_forward() -- do not rebuild it per window, since
        build_mtf_dataset_with_structure() is already lookahead-safe
        across the whole series. When None (the default), every
        window runs exactly as this module always has.
    """

    _validate_walk_forward_inputs(
        df,
        config,
    )

    total_slice = (
        config.warmup_size
        + config.window_size
    )

    starts = _window_starts(
        total_len=len(df),
        total_slice=total_slice,
        step_size=config.step_size,
    )

    errors: list[str] = []
    windows: list[WindowResult] = []
    pooled_r_multiples: list[float] = []
    pooled_wins: int = 0

    for (
        window_index,
        slice_start,
    ) in enumerate(starts):

        eval_start = (
            slice_start
            + config.warmup_size
        )

        end = (
            eval_start
            + config.window_size
        )

        # ------------------------------------------------------------
        # Warmup + evaluation slice.
        # ------------------------------------------------------------

        slice_df = df.iloc[
            slice_start:end
        ].copy()

        eval_start_timestamp = pd.Timestamp(
            df.index[eval_start]
        )

        eval_end_timestamp = pd.Timestamp(
            df.index[end - 1]
        )

        # ------------------------------------------------------------
        # Run canonical backtest using identical parameters.
        # ------------------------------------------------------------

        try:

            result = run_backtest(
                slice_df,
                max_bars_to_retest=(
                    config.max_bars_to_retest
                ),
                reward_multiple=(
                    config.reward_multiple
                ),
                stop_buffer=(
                    config.stop_buffer
                ),
                entry_mode=(
                    config.entry_mode
                ),
                mtf_filter_fn=mtf_filter_fn,
            )

        except Exception as exc:
            error_msg = (
                f"window {window_index} "
                f"(warmup {slice_start}:{eval_start}, "
                f"eval {eval_start}:{end}, "
                f"{eval_start_timestamp} to "
                f"{eval_end_timestamp}): "
                f"run_backtest() raised "
                f"{type(exc).__name__}: {exc}"
            )

            errors.append(
                error_msg
            )

            windows.append(
                WindowResult(
                    window_index=window_index,
                    slice_start_idx=slice_start,
                    eval_start_idx=eval_start,
                    end_idx=end,
                    eval_start_timestamp=(
                        eval_start_timestamp
                    ),
                    eval_end_timestamp=(
                        eval_end_timestamp
                    ),
                    num_eval_candles=(
                        config.window_size
                    ),
                    num_warmup_candles=(
                        config.warmup_size
                    ),
                    num_closed_trades=0,
                    num_warmup_trades_discarded=0,
                    num_eval_open_or_cross_boundary=0,
                    win_rate=None,
                    expectancy_r=None,
                    backtest_result=None,
                    error=error_msg,
                )
            )

            continue

        # ------------------------------------------------------------
        # Calculate evaluation-only metrics.
        # ------------------------------------------------------------

        (
            win_rate,
            expectancy_r,
            num_closed,
            num_discarded,
            num_cross_boundary,
            window_r_multiples,
            window_wins,
        ) = _window_stats(
            result=result,
            eval_start_timestamp=(
                eval_start_timestamp
            ),
            eval_end_timestamp=(
                eval_end_timestamp
            ),
        )

        pooled_r_multiples.extend(
            window_r_multiples
        )
        pooled_wins += window_wins

        windows.append(
            WindowResult(
                window_index=window_index,
                slice_start_idx=slice_start,
                eval_start_idx=eval_start,
                end_idx=end,
                eval_start_timestamp=(
                    eval_start_timestamp
                ),
                eval_end_timestamp=(
                    eval_end_timestamp
                ),
                num_eval_candles=(
                    config.window_size
                ),
                num_warmup_candles=(
                    config.warmup_size
                ),
                num_closed_trades=(
                    num_closed
                ),
                num_warmup_trades_discarded=(
                    num_discarded
                ),
                num_eval_open_or_cross_boundary=(
                    num_cross_boundary
                ),
                win_rate=win_rate,
                expectancy_r=expectancy_r,
                backtest_result=result,
                error=None,
            )
        )

    # =========================================================================
    # AGGREGATE SUMMARY
    # =========================================================================

    win_rates = [
        float(window.win_rate)
        for window in windows
        if window.win_rate is not None
    ]

    expectancies = [
        float(window.expectancy_r)
        for window in windows
        if window.expectancy_r is not None
    ]

    summary: dict[
        str,
        dict[str, float],
    ] = {}

    if win_rates:
        summary["win_rate"] = (
            _percentile_summary(
                win_rates
            )
        )

    if expectancies:
        summary["expectancy_r"] = (
            _percentile_summary(
                expectancies
            )
        )

    # -------------------------------------------------------------------------
    # Trade-weighted aggregate: pool every individual closed trade's
    # R-multiple across all windows, rather than treating each window
    # (regardless of its trade count) as a single equally-weighted
    # data point. This avoids letting low-trade-count windows (a
    # single-trade window can only land at 100% or 0% win rate)
    # distort the picture the per-window mean/median can show.
    # -------------------------------------------------------------------------

    if pooled_r_multiples:
        summary["trade_weighted"] = {
            "count": float(
                len(pooled_r_multiples)
            ),
            "win_rate": (
                pooled_wins
                / len(pooled_r_multiples)
            ),
            "expectancy_r": statistics.fmean(
                pooled_r_multiples
            ),
        }

    # -------------------------------------------------------------------------
    # Windows with no usable closed trades.
    # -------------------------------------------------------------------------

    windows_with_no_trades = sum(
        1
        for window in windows
        if (
            window.error is None
            and window.num_closed_trades == 0
        )
    )

    if windows_with_no_trades:
        errors.append(
            f"{windows_with_no_trades}/"
            f"{len(windows)} windows had 0 trades "
            "fully closed inside the evaluation region "
            "and were excluded from the "
            "win_rate/expectancy_r summary."
        )

    return WalkForwardResult(
        config=config,
        windows=tuple(
            windows
        ),
        num_windows=len(
            windows
        ),
        summary=summary,
        errors=tuple(
            errors
        ),
    )


# ============================================================================
# REPORTING
# ============================================================================

def print_walk_forward_report(
    result: WalkForwardResult,
) -> None:
    """Print a readable walk-forward stability report."""

    config = result.config

    print()
    print("=" * 120)
    print("PHASE 2 WALK-FORWARD STABILITY ANALYSIS")
    print("=" * 120)

    print()
    print("CONFIGURATION")
    print("-" * 120)

    print(
        f"Evaluation window : "
        f"{config.window_size:,} candles"
    )

    print(
        f"Step size         : "
        f"{config.step_size:,} candles"
    )

    print(
        f"Warmup size       : "
        f"{config.warmup_size:,} candles"
    )

    print(
        f"Max retest bars   : "
        f"{config.max_bars_to_retest}"
    )

    print(
        f"Reward multiple   : "
        f"{config.reward_multiple}"
    )

    print(
        f"Stop buffer       : "
        f"{config.stop_buffer}"
    )

    print(
        f"Entry mode        : "
        f"{config.entry_mode}"
    )

    print(
        f"Windows generated : "
        f"{result.num_windows}"
    )

    print()
    print("WINDOW RESULTS")
    print("-" * 120)

    print(
        f"{'Win':>4} "
        f"{'Eval Start':<20} "
        f"{'Eval End':<20} "
        f"{'Trades':>7} "
        f"{'Warmup':>7} "
        f"{'Open/Cross':>10} "
        f"{'Win Rate':>10} "
        f"{'Expectancy R':>14}"
    )

    print("-" * 120)

    for window in result.windows:

        if window.win_rate is None:
            win_rate_text = "N/A"
        else:
            win_rate_text = (
                f"{window.win_rate * 100.0:8.2f}%"
            )

        if window.expectancy_r is None:
            expectancy_text = "N/A"
        else:
            expectancy_text = (
                f"{window.expectancy_r:10.4f}R"
            )

        print(
            f"{window.window_index:>4} "
            f"{str(window.eval_start_timestamp):<20} "
            f"{str(window.eval_end_timestamp):<20} "
            f"{window.num_closed_trades:>7} "
            f"{window.num_warmup_trades_discarded:>7} "
            f"{window.num_eval_open_or_cross_boundary:>10} "
            f"{win_rate_text:>10} "
            f"{expectancy_text:>14}"
        )

        if window.error:
            print(
                f"      ERROR: {window.error}"
            )

    # =========================================================================
    # SUMMARY
    # =========================================================================

    print()
    print("AGGREGATE SUMMARY")
    print("-" * 120)

    if "expectancy_r" in result.summary:

        stats = result.summary[
            "expectancy_r"
        ]

        print(
            "Expectancy R distribution:"
        )

        for key, value in stats.items():
            print(
                f"  {key:<8}: "
                f"{value:.4f}R"
            )

    else:

        print(
            "Expectancy R distribution: N/A"
        )

    if "win_rate" in result.summary:

        stats = result.summary[
            "win_rate"
        ]

        print()
        print(
            "Win-rate distribution:"
        )

        for key, value in stats.items():
            print(
                f"  {key:<8}: "
                f"{value * 100.0:.2f}%"
            )

    else:

        print(
            "Win-rate distribution: N/A"
        )

    if "trade_weighted" in result.summary:

        stats = result.summary[
            "trade_weighted"
        ]

        print()
        print(
            "Trade-weighted aggregate (pooled across all closed trades):"
        )

        print(
            f"  count      : "
            f"{int(stats['count'])}"
        )

        print(
            f"  win_rate   : "
            f"{stats['win_rate'] * 100.0:.2f}%"
        )

        print(
            f"  expectancy : "
            f"{stats['expectancy_r']:.4f}R"
        )

    # =========================================================================
    # DIAGNOSTICS
    # =========================================================================

    print()
    print("DIAGNOSTICS")
    print("-" * 120)

    total_eval_closed = sum(
        window.num_closed_trades
        for window in result.windows
    )

    total_warmup_discarded = sum(
        window.num_warmup_trades_discarded
        for window in result.windows
    )

    total_cross_boundary = sum(
        window.num_eval_open_or_cross_boundary
        for window in result.windows
    )

    successful_windows = sum(
        1
        for window in result.windows
        if window.error is None
    )

    print(
        f"Successful backtests       : "
        f"{successful_windows}/{result.num_windows}"
    )

    print(
        f"Evaluation closed trades   : "
        f"{total_eval_closed}"
    )

    print(
        f"Warmup trades discarded    : "
        f"{total_warmup_discarded}"
    )

    print(
        f"Open/cross-boundary trades : "
        f"{total_cross_boundary}"
    )

    if result.errors:

        print()
        print(
            "WARNINGS / ERRORS"
        )

        for error in result.errors:
            print(
                f"  - {error}"
            )

    # =========================================================================
    # INTERPRETATION
    # =========================================================================

    print()
    print("INTERPRETATION")
    print("-" * 120)

    print(
        "1. Expectancy R is the primary stability metric."
    )

    print(
        "2. Win rate is secondary context."
    )

    print(
        "3. No pass/fail threshold is hard-coded."
    )

    print(
        "4. Compare the distribution across windows rather than "
        "judging one window in isolation."
    )

    print(
        "5. A window with 0 usable closed trades provides no "
        "expectancy estimate."
    )

    print(
        "6. Trades that cross the evaluation boundary are excluded "
        "to prevent future-period leakage."
    )

    print(
        "7. This is stability analysis, not parameter optimization."
    )

    print(
        "8. This result describes whatever trade stream was passed "
        "in: the canonical single-timeframe backtest when "
        "mtf_filter_fn is None, or the MTF-gated trade stream when "
        "an mtf_filter_fn was supplied to run_walk_forward()."
    )

    print(
        "9. The per-window distribution weights every window "
        "equally regardless of trade count; the trade-weighted "
        "aggregate weights every individual trade equally instead, "
        "reducing distortion from low-trade-count windows."
    )

    print("=" * 120)
    print()


# ============================================================================
# DATA LOADING
# ============================================================================

def load_dataset(
    path: Path,
) -> pd.DataFrame:
    """Load the historical OHLC dataset."""

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    df = pd.read_csv(
        path,
        index_col=0,
        parse_dates=True,
    )

    if df.empty:
        raise ValueError(
            "Dataset is empty."
        )

    if not isinstance(
        df.index,
        pd.DatetimeIndex,
    ):
        raise ValueError(
            "Dataset index must be a DatetimeIndex."
        )

    if df.index.has_duplicates:
        raise ValueError(
            "Dataset contains duplicate timestamps."
        )

    if not df.index.is_monotonic_increasing:
        df = df.sort_index(
            kind="stable"
        )

    return df


# ============================================================================
# COMMAND-LINE ENTRY POINT
# ============================================================================

DATA_FILE = Path(
    "real_gold_data_mt5_90000.csv"
)


def parse_args() -> argparse.Namespace:
    """
    Parse CLI overrides for window/step/warmup sensitivity testing.

    Defaults reproduce the original baseline run
    (window=1000, step=500, warmup=750) when no arguments are supplied.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Phase 2 walk-forward stability analysis. "
            "Run with no arguments to reproduce the original "
            "window=1000, step=500, warmup=750 baseline."
        )
    )

    parser.add_argument(
        "--window",
        type=int,
        default=1000,
        help="Evaluation window size in candles (default: 1000).",
    )

    parser.add_argument(
        "--step",
        type=int,
        default=500,
        help="Step size in candles (default: 500).",
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=750,
        help="Warmup size in candles (default: 750).",
    )

    return parser.parse_args()


def main() -> None:
    """Run walk-forward stability analysis with MTF confluence filtering."""

    args = parse_args()

    print()
    print("=" * 120)
    print("LOADING DATASET & BUILDING MTF CONTEXT")
    print("=" * 120)

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

    # Build lookahead-safe MTF context and filter function globally
    print()
    print("Building multi-timeframe dataset structures...")
    df_enriched = build_mtf_dataset_with_structure(
        df,
        macro_swings_fn=find_swings,
        internal_swings_fn=find_swings,
    )
    mtf_filter_fn = build_mtf_signal_filter(df_enriched)
    print("MTF confluence filter successfully built.")

    config = WalkForwardConfig(
        window_size=args.window,
        step_size=args.step,
        warmup_size=args.warmup,
        max_bars_to_retest=20,
        reward_multiple=2.0,
        stop_buffer=0.0,
        entry_mode="midpoint",
    )

    print()
    print(
        f"Warmup validation candidate : "
        f"{config.warmup_size}"
    )

    print(
        f"Evaluation window            : "
        f"{config.window_size}"
    )

    print(
        f"Step size                    : "
        f"{config.step_size}"
    )

    print()
    print(
        "Running canonical MTF-gated backtest "
        "across walk-forward windows..."
    )

    result = run_walk_forward(
        df,
        config,
        mtf_filter_fn=mtf_filter_fn,
    )

    print_walk_forward_report(
        result
    )


if __name__ == "__main__":
    main()