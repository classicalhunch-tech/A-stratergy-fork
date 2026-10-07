from types import SimpleNamespace

import pandas as pd
import pytest

from strategy.pattern_tags import build_pattern_tagger, classify_pattern
from strategy.signals import SignalType


@pytest.mark.parametrize(
    "signal,macro,internal,expected",
    [
        ("LONG", "BULLISH", "BULLISH", "B_with_trend"),
        ("LONG", "BEARISH", "BULLISH", "A_counter_trend"),
        ("LONG", "BULLISH", "BEARISH", "X_macro_only"),
        ("LONG", "BULLISH", None, "X_macro_only"),
        ("LONG", "BEARISH", "BEARISH", "Z_against_both"),
        ("LONG", "BEARISH", None, "Z_against_both"),
        ("SHORT", "BEARISH", "BEARISH", "B_with_trend"),
        ("SHORT", "BULLISH", "BEARISH", "A_counter_trend"),
        ("SHORT", "BEARISH", "BULLISH", "X_macro_only"),
        ("SHORT", "BULLISH", "BULLISH", "Z_against_both"),
        ("LONG", None, "BULLISH", "U_unknown"),
        ("FLAT", "BULLISH", "BULLISH", "U_unknown"),
    ],
)
def test_classify_pattern(signal, macro, internal, expected):
    assert classify_pattern(signal, macro, internal) == expected


def test_tagger_uses_latest_known_trend_and_never_looks_ahead():
    index = pd.date_range("2026-01-01", periods=3, freq="1h")
    enriched = pd.DataFrame(
        {
            "macro_trend": ["BEARISH", "BEARISH", "BULLISH"],
            "internal_trend": [None, "BULLISH", "BULLISH"],
        },
        index=index,
    )
    tag = build_pattern_tagger(enriched)
    long_signal = SimpleNamespace(signal_type=SignalType.LONG)

    # between row 1 and row 2: uses row 1 (macro BEARISH, internal BULLISH)
    assert tag(long_signal, index[1] + pd.Timedelta(minutes=20)) == "A_counter_trend"

    # exactly on row 2: macro BULLISH, internal BULLISH
    assert tag(long_signal, index[2]) == "B_with_trend"

    # before any data: unknown
    assert tag(long_signal, index[0] - pd.Timedelta(hours=1)) == "U_unknown"


def test_tagger_rejects_bad_input():
    with pytest.raises(ValueError):
        build_pattern_tagger(pd.DataFrame({"x": [1]}, index=pd.date_range("2026-01-01", periods=1)))
    with pytest.raises(TypeError):
        build_pattern_tagger(pd.DataFrame({"macro_trend": [1], "internal_trend": [1]}))
