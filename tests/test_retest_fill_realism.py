"""
tests/test_retest_fill_realism.py

Fill-realism checks for strategy/retest_engine.advance_pending_signal.

Setup used below (BUY):  zone 2048-2052, midpoint entry 2050, stop 2046.

A limit order resting at 2050 can only fill if price actually trades
down to 2050. A bar whose low is only 2051 touches the zone top but never
reaches 2050. The default behaviour still triggers on the zone edge (kept
so existing results do not change silently). strict_entry_fill=True
requires price to reach the entry.
"""

from types import SimpleNamespace

import pandas as pd

from strategy.retest_engine import PendingOutcomeType, advance_pending_signal
from strategy.signals import SignalStatus, SignalType


def _signal(signal_type, entry, stop):
    signal = SimpleNamespace(
        signal_type=signal_type,
        entry_price=entry,
        stop_loss=stop,
        status=SignalStatus.PENDING_RETEST,
        zone_stable_key=None,
        zone_id=1,
    )
    signal.recalculate_risk_reward = lambda: None
    return signal


def _zone(top=2052.0, bottom=2048.0):
    return SimpleNamespace(
        zone_id=1,
        price_top=top,
        price_bottom=bottom,
        mitigated=False,
        is_mitigated=False,
        mitigated_at=None,
    )


def _advance(signal, bar_open, bar_high, bar_low, strict=False):
    return advance_pending_signal(
        signal=signal,
        setup_idx=10,
        current_index=12,
        current_time=pd.Timestamp("2026-01-05 10:00"),
        current_open=bar_open,
        current_high=bar_high,
        current_low=bar_low,
        visible_zones=[_zone()],
        max_bars_to_retest=20,
        strict_entry_fill=strict,
    )


def test_default_triggers_on_zone_edge_even_if_entry_not_reached():
    # Documents the optimistic default: the zone top (2052) is touched,
    # so the setup triggers although price never traded at 2050.
    signal = _signal(SignalType.LONG, entry=2050.0, stop=2046.0)

    outcome = _advance(signal, bar_open=2056.0, bar_high=2057.0, bar_low=2051.0)

    assert outcome.outcome == PendingOutcomeType.TRIGGERED
    assert outcome.fill_price == 2050.0


def test_default_buy_fills_at_entry_when_price_reaches_it():
    signal = _signal(SignalType.LONG, entry=2050.0, stop=2046.0)

    outcome = _advance(signal, bar_open=2056.0, bar_high=2057.0, bar_low=2049.5)

    assert outcome.outcome == PendingOutcomeType.TRIGGERED
    assert outcome.fill_price == 2050.0
    assert outcome.initial_risk == 4.0


def test_default_buy_gapping_below_entry_fills_at_the_open():
    signal = _signal(SignalType.LONG, entry=2050.0, stop=2046.0)

    outcome = _advance(signal, bar_open=2049.0, bar_high=2051.0, bar_low=2048.5)

    assert outcome.outcome == PendingOutcomeType.TRIGGERED
    assert outcome.fill_price == 2049.0
    assert outcome.initial_risk == 3.0


def test_strict_buy_not_filled_when_price_never_reaches_entry():
    signal = _signal(SignalType.LONG, entry=2050.0, stop=2046.0)

    outcome = _advance(
        signal, bar_open=2056.0, bar_high=2057.0, bar_low=2051.0, strict=True
    )

    assert outcome.outcome == PendingOutcomeType.STILL_PENDING


def test_strict_sell_not_filled_when_price_never_reaches_entry():
    signal = _signal(SignalType.SHORT, entry=2050.0, stop=2054.0)

    outcome = _advance(
        signal, bar_open=2044.0, bar_high=2049.0, bar_low=2043.0, strict=True
    )

    assert outcome.outcome == PendingOutcomeType.STILL_PENDING


def test_strict_buy_fills_at_entry_when_price_reaches_it():
    signal = _signal(SignalType.LONG, entry=2050.0, stop=2046.0)

    outcome = _advance(
        signal, bar_open=2056.0, bar_high=2057.0, bar_low=2049.5, strict=True
    )

    assert outcome.outcome == PendingOutcomeType.TRIGGERED
    assert outcome.fill_price == 2050.0
    assert outcome.initial_risk == 4.0


def test_strict_buy_gapping_below_entry_fills_at_the_open():
    signal = _signal(SignalType.LONG, entry=2050.0, stop=2046.0)

    outcome = _advance(
        signal, bar_open=2049.0, bar_high=2051.0, bar_low=2048.5, strict=True
    )

    assert outcome.outcome == PendingOutcomeType.TRIGGERED
    assert outcome.fill_price == 2049.0
    assert outcome.initial_risk == 3.0


def test_strict_sell_fills_at_entry_when_price_reaches_it():
    signal = _signal(SignalType.SHORT, entry=2050.0, stop=2054.0)

    outcome = _advance(
        signal, bar_open=2044.0, bar_high=2050.5, bar_low=2043.0, strict=True
    )

    assert outcome.outcome == PendingOutcomeType.TRIGGERED
    assert outcome.fill_price == 2050.0
    assert outcome.initial_risk == 4.0
