"""
tests/test_quality_gap_policy.py

Gap policy of phase_04_live.market.quality.QualityFilter.

With the default policy (reject_gaps=True) a gap candle is rejected and
the baseline does not move, so the feed stays blocked. The live source
uses reject_gaps=False, which reports the gap but keeps the feed going.
"""

from datetime import datetime, timedelta, timezone

from phase_04_live.market.models import Candle, QualityIssueType
from phase_04_live.market.quality import QualityFilter


def _candle(minutes_from_start: int, timeframe: str = "M5") -> Candle:
    ts = datetime(2026, 1, 5, 10, 0, tzinfo=timezone.utc) + timedelta(
        minutes=minutes_from_start
    )
    return Candle(
        timestamp=ts,
        open=2050.0,
        high=2051.0,
        low=2049.0,
        close=2050.5,
        volume=100.0,
        symbol="XAUUSD",
        timeframe=timeframe,
    )


def test_gap_candle_is_rejected_by_default():
    quality = QualityFilter()

    assert quality.check(_candle(0)) == (True, None)

    accepted, issue = quality.check(_candle(30))

    assert accepted is False
    assert issue is not None
    assert issue.issue_type == QualityIssueType.TIMEFRAME_GAP


def test_default_policy_stays_blocked_after_a_gap():
    # Documents the stall: the rejected candle does not move the baseline,
    # so the next candle is still measured against the old one.
    quality = QualityFilter()

    quality.check(_candle(0))
    quality.check(_candle(30))

    accepted, issue = quality.check(_candle(35))

    assert accepted is False
    assert issue is not None
    assert issue.issue_type == QualityIssueType.TIMEFRAME_GAP


def test_live_policy_accepts_candle_after_gap_and_reports_it():
    reported = []
    quality = QualityFilter(on_issue=reported.append, reject_gaps=False)

    quality.check(_candle(0))
    accepted, issue = quality.check(_candle(30))

    assert accepted is True
    assert issue is not None
    assert issue.issue_type == QualityIssueType.TIMEFRAME_GAP
    assert len(reported) == 1
    assert reported[0].issue_type == QualityIssueType.TIMEFRAME_GAP


def test_live_policy_recovers_after_a_gap():
    quality = QualityFilter(reject_gaps=False)

    quality.check(_candle(0))
    quality.check(_candle(30))

    assert quality.check(_candle(35)) == (True, None)


def test_live_policy_still_rejects_duplicates_and_out_of_order():
    quality = QualityFilter(reject_gaps=False)

    quality.check(_candle(0))

    accepted, issue = quality.check(_candle(0))
    assert accepted is False
    assert issue.issue_type == QualityIssueType.DUPLICATE_TIMESTAMP

    accepted, issue = quality.check(_candle(-5))
    assert accepted is False
    assert issue.issue_type == QualityIssueType.OUT_OF_ORDER
