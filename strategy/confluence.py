"""
strategy/confluence.py

MTF Trend Guard & Confluence Filter.

Acts as the final MTF structural gate between a 5M candidate setup
and the higher-timeframe structural context.

Responsibilities
----------------
- Read the candidate direction produced by the 5M SMC engine.
- Validate that the 4H macro structural trend agrees.
- Validate that the 15M internal structural trend agrees.
- Optionally allow an unestablished internal trend.
- Reject immediately when higher-timeframe structure opposes the
  candidate direction.
- Preserve all source DataFrame columns and metadata.
- Produce a separate `final_signal` column (batch mode) or a
  per-signal boolean predicate (live/backtest mode).

This module does NOT:
- detect swings
- calculate BOS / CHoCH
- detect liquidity
- detect zones
- generate the original 5M candidate
- execute trades
"""

from typing import Any, Callable, Optional

import pandas as pd


LONG = "LONG"
SHORT = "SHORT"

BULLISH = "BULLISH"
BEARISH = "BEARISH"

NEUTRAL_INTERNAL_STATES = {
    None,
    "UNKNOWN",
    "NONE",
    "",
}


def _normalize_state(value: Any) -> Optional[str]:
    """
    Normalize a signal or structural-state value.

    Supported inputs include:
    - plain strings
    - Enum-like objects with `.value`
    - objects exposing `.direction`
    - objects exposing `.type`
    - None
    - NaN / pd.NA

    Returns
    -------
    Optional[str]
        Uppercase normalized state, or None when unavailable.
    """

    if value is None:
        return None

    # Handle scalar missing values safely.
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False

    if isinstance(missing, bool):
        if missing:
            return None
    else:
        # Non-scalar result from pd.isna(), e.g. an array-like object.
        # It is not a valid state value for this filter.
        return None

    # ---------------------------------------------------------
    # 1. Enum-like values
    # ---------------------------------------------------------

    value = getattr(value, "value", value)

    if value is None:
        return None

    # ---------------------------------------------------------
    # 2. Custom signal/state objects
    # ---------------------------------------------------------

    if not isinstance(value, (str, int, float)):
        direction = getattr(value, "direction", None)

        if direction is not None:
            value = getattr(direction, "value", direction)

        else:
            signal_type = getattr(value, "type", None)

            if signal_type is not None:
                value = getattr(signal_type, "value", signal_type)

    if value is None:
        return None

    cleaned = str(value).strip().upper()

    if cleaned in {"", "NAN", "<NA>", "NONE"}:
        return None

    return cleaned


def _mtf_confluence_approved(
    base_signal: Optional[str],
    macro_trend: Optional[str],
    internal_trend: Optional[str],
    allow_neutral_internal: bool,
) -> bool:
    """
    Single source of truth for the LONG/SHORT trend-agreement rule.

    Used by both:
    - apply_mtf_confluence_filter() (batch DataFrame mode)
    - build_mtf_signal_filter()'s returned closure (per-signal,
      live backtest mode)

    so the two call paths can never silently drift apart.

    LONG requires macro_trend == BULLISH, and either
    internal_trend == BULLISH or (allow_neutral_internal and
    internal_trend is unestablished).

    SHORT is the mirror image with BEARISH.
    """

    if base_signal == LONG:
        macro_ok = macro_trend == BULLISH

        if allow_neutral_internal:
            internal_ok = (
                internal_trend == BULLISH
                or internal_trend in NEUTRAL_INTERNAL_STATES
            )
        else:
            internal_ok = internal_trend == BULLISH

        return macro_ok and internal_ok

    if base_signal == SHORT:
        macro_ok = macro_trend == BEARISH

        if allow_neutral_internal:
            internal_ok = (
                internal_trend == BEARISH
                or internal_trend in NEUTRAL_INTERNAL_STATES
            )
        else:
            internal_ok = internal_trend == BEARISH

        return macro_ok and internal_ok

    return False


def apply_mtf_confluence_filter(
    df_enriched: pd.DataFrame,
    allow_neutral_internal: bool = True,
) -> pd.DataFrame:
    """
    Apply the MTF structural trend guard to 5M candidate setups
    already present as a `base_signal` column on df_enriched.

    Any candidate failing the MTF guard receives None in
    `final_signal`.

    The original DataFrame is never modified in place.
    """

    required_columns = {
        "base_signal",
        "macro_trend",
        "internal_trend",
    }

    missing = required_columns - set(df_enriched.columns)

    if missing:
        raise ValueError(
            "df_enriched is missing required columns: "
            f"{sorted(missing)}"
        )

    df = df_enriched.copy()

    final_signals = []

    for row in df.itertuples(index=False):
        row_data = row._asdict()

        base_signal = _normalize_state(row_data.get("base_signal"))
        macro_trend = _normalize_state(row_data.get("macro_trend"))
        internal_trend = _normalize_state(row_data.get("internal_trend"))

        if base_signal not in (LONG, SHORT):
            final_signals.append(None)
            continue

        approved = _mtf_confluence_approved(
            base_signal,
            macro_trend,
            internal_trend,
            allow_neutral_internal,
        )

        final_signals.append(base_signal if approved else None)

    df["final_signal"] = final_signals

    return df


def build_mtf_signal_filter(
    df_enriched: pd.DataFrame,
    allow_neutral_internal: bool = True,
) -> Callable[[Any, pd.Timestamp], bool]:
    """
    Build a per-signal MTF confluence predicate for use as
    strategy.backtest.run_backtest()'s `mtf_filter_fn` hook.

    Unlike apply_mtf_confluence_filter() (which processes a whole
    DataFrame that already has a `base_signal` column), this builds
    a live callable:

        mtf_filter_fn(signal, current_time) -> bool

    matching backtest.py's expected signature:
        mtf_filter_fn: Optional[Callable[[TradeSignal, pd.Timestamp], bool]]

    It reuses the exact same trend-agreement rule as
    apply_mtf_confluence_filter() via _mtf_confluence_approved(), so
    the two code paths cannot disagree.

    Parameters
    ----------
    df_enriched : pd.DataFrame
        Output of build_mtf_dataset_with_structure(). Must contain
        `macro_trend` and `internal_trend` columns, indexed by the
        same 5M timestamps used during the backtest replay.
    allow_neutral_internal : bool
        Whether an unestablished 15M internal trend is treated as
        non-blocking.

    Returns
    -------
    Callable[[Any, pd.Timestamp], bool]
        True  -> signal is MTF-approved, allowed to proceed.
        False -> signal is rejected by the MTF guard.
    """

    required_columns = {"macro_trend", "internal_trend"}
    missing = required_columns - set(df_enriched.columns)

    if missing:
        raise ValueError(
            "df_enriched is missing required columns: "
            f"{sorted(missing)}"
        )

    if not isinstance(df_enriched.index, pd.DatetimeIndex):
        raise TypeError("df_enriched must have a DatetimeIndex.")

    # Precompute an O(1) lookup table once, rather than re-scanning
    # df_enriched on every signal check during the bar-by-bar replay.
    trend_lookup = {
        ts: (
            _normalize_state(macro),
            _normalize_state(internal),
        )
        for ts, macro, internal in zip(
            df_enriched.index,
            df_enriched["macro_trend"],
            df_enriched["internal_trend"],
        )
    }

    sorted_index = df_enriched.index.sort_values()

    def _lookup_trends(current_time: pd.Timestamp):
        # Fast path: exact bar match (expected on every real call,
        # since df_enriched is built directly from the same 5M index
        # the backtest replays over).
        if current_time in trend_lookup:
            return trend_lookup[current_time]

        # Fallback: as-of lookup (last known trend at or before
        # current_time). Guards against a timestamp mismatch without
        # ever looking ahead into future bars.
        position = sorted_index.searchsorted(current_time, side="right") - 1

        if position < 0:
            return None, None

        as_of_ts = sorted_index[position]
        return trend_lookup.get(as_of_ts, (None, None))

    def mtf_filter_fn(signal: Any, current_time: pd.Timestamp) -> bool:
        base_signal = _normalize_state(getattr(signal, "signal_type", None))

        if base_signal not in (LONG, SHORT):
            return False

        macro_trend, internal_trend = _lookup_trends(current_time)

        return _mtf_confluence_approved(
            base_signal,
            macro_trend,
            internal_trend,
            allow_neutral_internal,
        )

    return mtf_filter_fn