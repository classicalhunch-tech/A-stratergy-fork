"""
strategy/mtf_structure.py

Orchestrates top-down multi-timeframe structure integration:

1. Resamples base 5M data to higher timeframes (4H and 15M).
2. Runs swing detection on each higher timeframe.
3. Runs the canonical strategy/structure.py engine.
4. Projects structural breaks into causal trend timelines.
5. Attaches the higher-timeframe OHLC and structural trend state
   back onto the 5M execution timeline.

Architecture
------------
5M execution data
    ↓
MTF OHLC resampling
    ├── 4H macro timeframe
    └── 15M internal timeframe
    ↓
Swing detection
    ↓
Canonical structure engine
    ↓
Structural trend timelines
    ↓
Lookahead-safe attachment to 5M
"""

from typing import Callable, Optional

import pandas as pd

from dashboard.mtf_context import (
    MTFConfig,
    attach_htf_context,
    resample_ohlc,
)
from strategy.structure import analyze_structure


def get_structural_trend_timeline(
    df: pd.DataFrame,
    swings: list,
) -> pd.Series:
    """
    Project the canonical analyze_structure() result into a
    bar-by-bar structural trend timeline.

    The returned Series contains:

        None       -> no structural trend established yet
        "BULLISH"  -> latest confirmed bullish structural state
        "BEARISH"  -> latest confirmed bearish structural state

    The function does not detect structure itself. The canonical
    analyze_structure() engine remains the single source of truth.

    A structural direction becomes visible only at the break's
    `broken_at` timestamp and remains active until a later structural
    break changes the direction.
    """

    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("df must have a DatetimeIndex.")

    if not df.index.is_monotonic_increasing:
        raise ValueError(
            "df index must be sorted in ascending chronological order."
        )

    breaks, _, _ = analyze_structure(df, swings)

    trend_series = pd.Series(
        None,
        index=df.index,
        dtype=object,
    )

    sorted_breaks = sorted(
        breaks,
        key=lambda break_event: break_event.broken_at,
    )

    current_trend = None
    break_idx = 0

    for ts in df.index:
        while (
            break_idx < len(sorted_breaks)
            and sorted_breaks[break_idx].broken_at <= ts
        ):
            current_trend = (
                sorted_breaks[break_idx]
                .direction
                .value
                .upper()
            )

            break_idx += 1

        trend_series.loc[ts] = current_trend

    trend_series.index.name = df.index.name

    return trend_series


def build_mtf_dataset_with_structure(
    df_5m: pd.DataFrame,
    macro_swings_fn: Callable[[pd.DataFrame], list],
    internal_swings_fn: Callable[[pd.DataFrame], list],
    config: Optional[MTFConfig] = None,
    timestamp_is_open: bool = True,
) -> pd.DataFrame:
    """
    Enrich the 5M execution dataset with:

        - 4H macro OHLC
        - 4H macro structural trend
        - 15M internal OHLC
        - 15M internal structural trend

    All higher-timeframe information is attached using the existing
    lookahead-safe attach_htf_context() bridge.
    """

    if not isinstance(df_5m.index, pd.DatetimeIndex):
        raise TypeError("df_5m must have a DatetimeIndex.")

    if df_5m.empty:
        raise ValueError("df_5m is empty.")

    if not df_5m.index.is_monotonic_increasing:
        raise ValueError(
            "df_5m index must be sorted in ascending chronological order."
        )

    config = config or MTFConfig()

    # ---------------------------------------------------------
    # 1. Build higher-timeframe OHLC datasets
    # ---------------------------------------------------------

    macro_htf = resample_ohlc(
        df_5m,
        config.macro_tf,
        origin=config.origin,
    )

    internal_htf = resample_ohlc(
        df_5m,
        config.internal_tf,
        origin=config.origin,
    )

    # ---------------------------------------------------------
    # 2. Detect macro swings and calculate 4H structure
    # ---------------------------------------------------------

    macro_swings = macro_swings_fn(macro_htf)

    macro_trend_series = get_structural_trend_timeline(
        macro_htf,
        macro_swings,
    )

    macro_htf = macro_htf.copy()
    macro_htf["macro_trend"] = macro_trend_series

    # ---------------------------------------------------------
    # 3. Detect internal swings and calculate 15M structure
    # ---------------------------------------------------------

    internal_swings = internal_swings_fn(internal_htf)

    internal_trend_series = get_structural_trend_timeline(
        internal_htf,
        internal_swings,
    )

    internal_htf = internal_htf.copy()
    internal_htf["internal_trend"] = internal_trend_series

    # ---------------------------------------------------------
    # 4. Attach 4H context to the 5M execution timeline
    # ---------------------------------------------------------

    df_enriched = attach_htf_context(
        df_5m,
        macro_htf,
        prefix="macro",
        timestamp_is_open=timestamp_is_open,
        base_interval=config.base_interval,
    )

    # ---------------------------------------------------------
    # 5. Attach 15M context to the already-enriched 5M data
    # ---------------------------------------------------------

    df_enriched = attach_htf_context(
        df_enriched,
        internal_htf,
        prefix="internal",
        timestamp_is_open=timestamp_is_open,
        base_interval=config.base_interval,
    )

    return df_enriched