"""
strategy/confluence.py

MTF Trend Guard & Confluence Filter.

Acts as the MTF structural gate between a 5M candidate setup and the
higher-timeframe structural context.

Responsibilities
----------------
- Read the candidate direction produced by the 5M SMC engine.
- Optionally validate that the macro (1H) structural trend agrees.
- Optionally validate that the internal (15M) structural trend agrees.
- Optionally allow an unestablished internal trend.
- Optionally allow macro agreement to override an internal conflict
  (soft_internal_conflict).
- Optionally DISABLE the direction gate entirely (gate_direction=False).
  The strategy is a counter-trend reversal from higher-timeframe
  zones, so the 1H/15M trend direction is context, not a gate. With
  the gate off every valid LONG/SHORT candidate is approved and the
  trend labels remain available as reporting tags through
  build_mtf_tag_fn().
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

Defaults (gate_direction=True) reproduce the original behavior
exactly, so existing tests and baselines are unchanged.
"""

from typing import Any, Callable, Optional, Tuple

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
    if value is None:
        return None

    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False

    if isinstance(missing, bool):
        if missing:
            return None
    else:
        return None

    value = getattr(value, "value", value)

    if value is None:
        return None

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
    soft_internal_conflict: bool = False,
    gate_direction: bool = True,
) -> bool:
    """
    Single source of truth for the LONG/SHORT trend-agreement rule.

    Used by both:
    - apply_mtf_confluence_filter() (batch DataFrame mode)
    - build_mtf_signal_filter()'s returned closure (per-signal,
      live backtest mode)

    so the two call paths can never silently drift apart.

    gate_direction=False: the trend direction is NOT a gate. Any valid
    LONG/SHORT candidate is approved. The trends stay available as
    tags for reporting.

    gate_direction=True (default, original behavior):

    Macro (1H) must always agree with the candidate direction; a
    macro conflict is always a hard reject regardless of internal
    state or soft_internal_conflict.

    When soft_internal_conflict is False (default): internal (15M)
    must also not conflict. LONG requires macro_trend == BULLISH, and
    either internal_trend == BULLISH or (allow_neutral_internal and
    internal_trend is unestablished). SHORT is the mirror image with
    BEARISH.

    When soft_internal_conflict is True: once macro agrees, an
    actively conflicting internal trend no longer vetoes the signal
    on its own. allow_neutral_internal has no additional effect in
    this mode since macro agreement alone already approves.
    """

    if base_signal not in (LONG, SHORT):
        return False

    if not gate_direction:
        return True

    want = BULLISH if base_signal == LONG else BEARISH

    macro_ok = macro_trend == want

    if not macro_ok:
        return False

    if soft_internal_conflict:
        return True

    if allow_neutral_internal:
        internal_ok = (
            internal_trend == want
            or internal_trend in NEUTRAL_INTERNAL_STATES
        )
    else:
        internal_ok = internal_trend == want

    return internal_ok


def apply_mtf_confluence_filter(
    df_enriched: pd.DataFrame,
    allow_neutral_internal: bool = True,
    soft_internal_conflict: bool = False,
    gate_direction: bool = True,
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
            soft_internal_conflict,
            gate_direction,
        )

        final_signals.append(base_signal if approved else None)

    df["final_signal"] = final_signals

    return df


def _build_trend_lookup(
    df_enriched: pd.DataFrame,
) -> Callable[[pd.Timestamp], Tuple[Optional[str], Optional[str]]]:
    """
    Build an as-of lookup: timestamp -> (macro_trend, internal_trend).

    Uses the most recent 5M row at or before the requested time, so
    it never reads a trend label from the future.
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
        if current_time in trend_lookup:
            return trend_lookup[current_time]

        position = sorted_index.searchsorted(current_time, side="right") - 1

        if position < 0:
            return None, None

        as_of_ts = sorted_index[position]
        return trend_lookup.get(as_of_ts, (None, None))

    return _lookup_trends


def build_mtf_tag_fn(
    df_enriched: pd.DataFrame,
) -> Callable[[pd.Timestamp], Tuple[Optional[str], Optional[str]]]:
    """
    Reporting helper: returns a function mapping a 5M timestamp to
    (macro_trend, internal_trend) labels, e.g. ("BEARISH", "BULLISH").

    Used to tag trades with the 1H/15M trend at signal time when the
    direction gate is off. Tags are for reporting only; nothing gates
    on them.
    """

    return _build_trend_lookup(df_enriched)


def build_mtf_signal_filter(
    df_enriched: pd.DataFrame,
    allow_neutral_internal: bool = True,
    soft_internal_conflict: bool = False,
    gate_direction: bool = True,
) -> Callable[[Any, pd.Timestamp], bool]:
    """
    Build a per-signal MTF confluence predicate for use as
    strategy.backtest.run_backtest()'s `mtf_filter_fn` hook.

    See _mtf_confluence_approved() for the exact rule.

    Parameters
    ----------
    df_enriched : pd.DataFrame
        Output of build_mtf_dataset_with_structure(). Must contain
        `macro_trend` and `internal_trend` columns, indexed by the
        same 5M timestamps used during the backtest replay.
    allow_neutral_internal : bool
        Whether an unestablished 15M internal trend is treated as
        non-blocking. Only relevant when gate_direction is True and
        soft_internal_conflict is False.
    soft_internal_conflict : bool
        When True, macro (1H) agreement alone approves the signal;
        an actively conflicting internal (15M) trend no longer
        vetoes it. Macro conflict is still a hard reject.
    gate_direction : bool
        When False, trend direction is not a gate: every valid
        LONG/SHORT candidate is approved. Default True keeps the
        original behavior.

    Returns
    -------
    Callable[[Any, pd.Timestamp], bool]
        True  -> signal is MTF-approved, allowed to proceed.
        False -> signal is rejected by the MTF guard.
    """

    lookup_trends = _build_trend_lookup(df_enriched)

    def mtf_filter_fn(signal: Any, current_time: pd.Timestamp) -> bool:
        base_signal = _normalize_state(getattr(signal, "signal_type", None))

        if base_signal not in (LONG, SHORT):
            return False

        macro_trend, internal_trend = lookup_trends(current_time)

        return _mtf_confluence_approved(
            base_signal,
            macro_trend,
            internal_trend,
            allow_neutral_internal,
            soft_internal_conflict,
            gate_direction,
        )

    return mtf_filter_fn
