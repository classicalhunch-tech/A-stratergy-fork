"""
tests/test_warmup.py

Startup warm-up helpers and QualityFilter.seed().
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

from phase_03_paper.signals.adapter import StrategyAdapter
from phase_04_live.market.models import Candle
from phase_04_live.market.quality import QualityFilter
from phase_04_live.market.warmup import (
    DEFAULT_WARMUP_CANDLES,
    to_paper_candle,
    warm_up_adapter,
)

DATA = Path(__file__).resolve().parents[1] / "data" / "master_historical_data.csv"
START = datetime(2026, 1, 5, 10, 0, tzinfo=timezone.utc)


def _live_candle(index: int) -> Candle:
    return Candle(
        timestamp=START + timedelta(minutes=5 * index),
        open=2050.0,
        high=2051.0,
        low=2049.0,
        close=2050.5,
        volume=100.0,
        symbol="XAUUSD",
        timeframe="M5",
    )


class _FakeAdapter:
    """Stands in for StrategyAdapter: records candles, returns canned events."""

    def __init__(self, events_by_call=None, error_on_call=None):
        self.seen = []
        self.errors = []
        self._events_by_call = events_by_call or {}
        self._error_on_call = error_on_call

    def on_candle(self, candle):
        self.seen.append(candle.timestamp)
        call = len(self.seen)
        if call == self._error_on_call:
            self.errors.append("boom")
        return list(self._events_by_call.get(call, []))


def test_to_paper_candle_makes_naive_timestamps_utc():
    naive = Candle(
        timestamp=datetime(2026, 1, 5, 10, 0),
        open=2050.0,
        high=2051.0,
        low=2049.0,
        close=2050.5,
        volume=100.0,
        symbol="XAUUSD",
        timeframe="M5",
    )

    paper = to_paper_candle(naive)

    assert paper.timestamp.tzinfo is not None
    assert paper.timestamp == datetime(2026, 1, 5, 10, 0, tzinfo=timezone.utc)


def test_warm_up_feeds_every_candle_in_order():
    adapter = _FakeAdapter()
    candles = [_live_candle(i) for i in range(10)]

    result = warm_up_adapter(adapter, candles)

    assert result.candles_fed == 10
    assert adapter.seen == sorted(adapter.seen)
    assert result.first_timestamp == candles[0].timestamp
    assert result.last_timestamp == candles[-1].timestamp


def test_warm_up_discards_triggered_events():
    adapter = _FakeAdapter(events_by_call={3: ["x"], 7: ["y", "z"]})

    result = warm_up_adapter(adapter, [_live_candle(i) for i in range(10)])

    assert result.signals_discarded == 3


def test_warm_up_rejects_unordered_candles_before_feeding_any():
    adapter = _FakeAdapter()
    candles = [_live_candle(0), _live_candle(1), _live_candle(1)]

    with pytest.raises(ValueError):
        warm_up_adapter(adapter, candles)

    assert adapter.seen == []


def test_warm_up_with_no_candles_does_nothing():
    adapter = _FakeAdapter()

    result = warm_up_adapter(adapter, [])

    assert result.candles_fed == 0
    assert result.signals_discarded == 0
    assert adapter.seen == []


def test_warm_up_reports_new_adapter_errors():
    adapter = _FakeAdapter(error_on_call=2)

    result = warm_up_adapter(adapter, [_live_candle(i) for i in range(5)])

    assert result.adapter_errors == 1


def test_real_adapter_accepts_warm_up_candles():
    df = pd.read_csv(DATA, index_col=0, parse_dates=True)
    df.columns = [str(column).strip().lower() for column in df.columns]
    df = df.tail(300)

    candles = [
        Candle(
            timestamp=timestamp.to_pydatetime(),
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=0.0,
            symbol="XAUUSD",
            timeframe="M5",
        )
        for timestamp, row in df.iterrows()
    ]

    adapter = StrategyAdapter()
    result = warm_up_adapter(adapter, candles)

    assert result.candles_fed == 300
    assert adapter.candle_count == 300


def test_quality_filter_seed_continues_from_history():
    quality = QualityFilter()
    quality.seed(_live_candle(0))

    accepted, issue = quality.check(_live_candle(0))
    assert accepted is False
    assert issue is not None

    assert quality.check(_live_candle(1)) == (True, None)


def test_default_warm_up_is_large_enough():
    assert DEFAULT_WARMUP_CANDLES >= 1000
