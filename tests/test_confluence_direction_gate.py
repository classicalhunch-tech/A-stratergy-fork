"""
Tests for the MTF direction gate in strategy/confluence.py.

The default (gate_direction=True) must keep the original behavior.
gate_direction=False must approve any valid LONG/SHORT candidate and
still reject invalid signal types.
"""

from types import SimpleNamespace

import pandas as pd

from strategy.confluence import (
    _mtf_confluence_approved,
    build_mtf_signal_filter,
    build_mtf_tag_fn,
)


def _enriched_frame():
    index = pd.date_range("2026-01-01 00:00", periods=4, freq="5min")
    return pd.DataFrame(
        {
            "macro_trend": ["BEARISH", "BEARISH", "BULLISH", "BULLISH"],
            "internal_trend": [None, "BEARISH", "BEARISH", "BULLISH"],
        },
        index=index,
    )


def test_default_gate_rejects_counter_trend_long():
    assert not _mtf_confluence_approved(
        "LONG", "BEARISH", "BULLISH", allow_neutral_internal=True
    )


def test_default_gate_approves_with_trend_long():
    assert _mtf_confluence_approved(
        "LONG", "BULLISH", "BULLISH", allow_neutral_internal=True
    )


def test_gate_off_approves_counter_trend_both_directions():
    assert _mtf_confluence_approved(
        "LONG", "BEARISH", "BEARISH",
        allow_neutral_internal=True, gate_direction=False,
    )
    assert _mtf_confluence_approved(
        "SHORT", "BULLISH", "BULLISH",
        allow_neutral_internal=True, gate_direction=False,
    )


def test_gate_off_still_rejects_invalid_signal_type():
    assert not _mtf_confluence_approved(
        None, "BULLISH", "BULLISH",
        allow_neutral_internal=True, gate_direction=False,
    )
    assert not _mtf_confluence_approved(
        "HOLD", "BULLISH", "BULLISH",
        allow_neutral_internal=True, gate_direction=False,
    )


def test_filter_closure_respects_gate_flag():
    df = _enriched_frame()
    t = df.index[1]  # macro BEARISH, internal BEARISH
    signal = SimpleNamespace(signal_type="LONG")

    gated = build_mtf_signal_filter(df)
    ungated = build_mtf_signal_filter(df, gate_direction=False)

    assert gated(signal, t) is False
    assert ungated(signal, t) is True


def test_filter_gate_off_rejects_non_trade_signal():
    df = _enriched_frame()
    ungated = build_mtf_signal_filter(df, gate_direction=False)
    assert ungated(SimpleNamespace(signal_type="HOLD"), df.index[1]) is False


def test_tag_fn_returns_trends_as_of_time_without_looking_ahead():
    df = _enriched_frame()
    tag_fn = build_mtf_tag_fn(df)

    # exact timestamp
    assert tag_fn(df.index[2]) == ("BULLISH", "BEARISH")

    # between rows: uses the latest row at or before the time
    between = df.index[1] + pd.Timedelta(minutes=2)
    assert tag_fn(between) == ("BEARISH", "BEARISH")

    # before the first row: nothing known
    before = df.index[0] - pd.Timedelta(minutes=5)
    assert tag_fn(before) == (None, None)
