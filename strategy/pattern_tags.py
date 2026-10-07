"""
strategy/pattern_tags.py

Labels a signal by how its direction relates to the 1H (macro) and 15M
(internal) structural trend at the moment it was discovered.

This only LABELS. It never approves or rejects a signal.

Labels (signal direction vs trend):
    B_with_trend     macro agrees, internal agrees
    A_counter_trend  macro opposes, internal agrees (15M has shifted)
    X_macro_only     macro agrees, internal does not (opposes/unknown)
    Z_against_both   macro opposes, internal does not agree
    U_unknown        no macro trend yet, or unusable direction

A_counter_trend is a PROXY for Pattern A. It does not check that price
is inside a 1H demand/supply zone or that 15M swept liquidity.
"""

from typing import Any, Callable

import pandas as pd

from strategy.confluence import (
    BEARISH,
    BULLISH,
    LONG,
    SHORT,
    _normalize_state,
)

PATTERN_LABELS = [
    "A_counter_trend",
    "B_with_trend",
    "X_macro_only",
    "Z_against_both",
    "U_unknown",
]


def classify_pattern(base_signal, macro_trend, internal_trend) -> str:
    if base_signal not in (LONG, SHORT):
        return "U_unknown"

    if macro_trend not in (BULLISH, BEARISH):
        return "U_unknown"

    want = BULLISH if base_signal == LONG else BEARISH

    macro_agrees = macro_trend == want
    internal_agrees = internal_trend == want

    if macro_agrees and internal_agrees:
        return "B_with_trend"

    if macro_agrees:
        return "X_macro_only"

    if internal_agrees:
        return "A_counter_trend"

    return "Z_against_both"


def build_pattern_tagger(
    df_enriched: pd.DataFrame,
) -> Callable[[Any, pd.Timestamp], str]:
    """
    Build tag(signal, current_time) -> label, using the same
    lookahead-safe as-of lookup as strategy.confluence.
    """

    required = {"macro_trend", "internal_trend"}
    missing = required - set(df_enriched.columns)

    if missing:
        raise ValueError(
            f"df_enriched is missing required columns: {sorted(missing)}"
        )

    if not isinstance(df_enriched.index, pd.DatetimeIndex):
        raise TypeError("df_enriched must have a DatetimeIndex.")

    trend_lookup = {
        ts: (_normalize_state(macro), _normalize_state(internal))
        for ts, macro, internal in zip(
            df_enriched.index,
            df_enriched["macro_trend"],
            df_enriched["internal_trend"],
        )
    }

    sorted_index = df_enriched.index.sort_values()

    def _lookup(current_time: pd.Timestamp):
        if current_time in trend_lookup:
            return trend_lookup[current_time]

        position = sorted_index.searchsorted(current_time, side="right") - 1

        if position < 0:
            return None, None

        return trend_lookup.get(sorted_index[position], (None, None))

    def tag(signal: Any, current_time: pd.Timestamp) -> str:
        base_signal = _normalize_state(getattr(signal, "signal_type", None))
        macro_trend, internal_trend = _lookup(current_time)
        return classify_pattern(base_signal, macro_trend, internal_trend)

    return tag
