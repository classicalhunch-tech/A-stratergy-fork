
"""
phase_04_live/market/quality.py

Filters bad, missing, duplicate, or out-of-order candles out of the
live feed before they ever reach StrategyAdapter/strategy/.

Policy (per project decision): a bad candle is DROPPED, never passed
through, and a QualityIssue is raised for monitoring to pick up.
This module never halts the feed and never silently swallows a
problem -- every drop produces a QualityIssue.
"""

from datetime import datetime, timezone
from typing import Callable, Optional

from phase_04_live.market.models import Candle, QualityIssue, QualityIssueType


TIMEFRAME_SECONDS = {
    "M1": 60,
    "M5": 300,
    "M15": 900,
    "H1": 3600,
}


class QualityFilter:
    """
    Stateful filter -- tracks the last accepted candle's timestamp per
    symbol/timeframe so it can detect duplicates, out-of-order candles,
    and gaps across successive calls.

    Usage: call check(candle) for each incoming candle, in the order
    the feed delivers them. Returns (accepted: bool, issue: QualityIssue | None).
    """

    def __init__(self, on_issue: Optional[Callable[[QualityIssue], None]] = None):
        self._last_accepted: dict[tuple[str, str], Candle] = {}
        self._on_issue = on_issue

    def check(self, candle: Candle) -> tuple[bool, Optional[QualityIssue]]:
        issue = (
            self._check_price_validity(candle)
            or self._check_ohlc_consistency(candle)
            or self._check_ordering(candle)
        )

        if issue is not None:
            self._raise(issue)
            return False, issue

        gap_issue = self._check_gap(candle)
        if gap_issue is not None:
            self._raise(gap_issue)

        key = (candle.symbol, candle.timeframe)
        self._last_accepted[key] = candle
        return True, None

    def _check_price_validity(self, candle: Candle) -> Optional[QualityIssue]:
        if candle.open <= 0 or candle.high <= 0 or candle.low <= 0 or candle.close <= 0:
            return self._make_issue(
                QualityIssueType.ZERO_OR_NEGATIVE_PRICE,
                candle,
                f"Non-positive price in OHLC: "
                f"O={candle.open} H={candle.high} L={candle.low} C={candle.close}",
            )
        return None

    def _check_ohlc_consistency(self, candle: Candle) -> Optional[QualityIssue]:
        if candle.high < candle.low:
            return self._make_issue(
                QualityIssueType.INVALID_OHLC,
                candle,
                f"high ({candle.high}) < low ({candle.low})",
            )
        if not (candle.low <= candle.open <= candle.high):
            return self._make_issue(
                QualityIssueType.INVALID_OHLC,
                candle,
                f"open ({candle.open}) outside [low, high] = [{candle.low}, {candle.high}]",
            )
        if not (candle.low <= candle.close <= candle.high):
            return self._make_issue(
                QualityIssueType.INVALID_OHLC,
                candle,
                f"close ({candle.close}) outside [low, high] = [{candle.low}, {candle.high}]",
            )
        return None

    def _check_ordering(self, candle: Candle) -> Optional[QualityIssue]:
        key = (candle.symbol, candle.timeframe)
        last = self._last_accepted.get(key)
        if last is None:
            return None

        if candle.timestamp == last.timestamp:
            return self._make_issue(
                QualityIssueType.DUPLICATE_TIMESTAMP,
                candle,
                f"Duplicate timestamp {candle.timestamp} (already accepted)",
            )
        if candle.timestamp < last.timestamp:
            return self._make_issue(
                QualityIssueType.OUT_OF_ORDER,
                candle,
                f"Candle timestamp {candle.timestamp} is older than "
                f"last accepted {last.timestamp}",
            )
        return None

    def _check_gap(self, candle: Candle) -> Optional[QualityIssue]:
        key = (candle.symbol, candle.timeframe)
        last = self._last_accepted.get(key)
        if last is None:
            return None

        expected_step = TIMEFRAME_SECONDS.get(candle.timeframe)
        if expected_step is None:
            return None

        actual_step = (candle.timestamp - last.timestamp).total_seconds()
        if actual_step > expected_step:
            missing = int(actual_step / expected_step) - 1
            return self._make_issue(
                QualityIssueType.TIMEFRAME_GAP,
                candle,
                f"Gap detected: ~{missing} candle(s) missing between "
                f"{last.timestamp} and {candle.timestamp}",
            )
        return None

    def _make_issue(self, issue_type: QualityIssueType, candle: Candle, detail: str) -> QualityIssue:
        return QualityIssue(
            issue_type=issue_type,
            symbol=candle.symbol,
            timeframe=candle.timeframe,
            detail=detail,
            candle=candle,
            detected_at=datetime.now(timezone.utc),
        )

    def _raise(self, issue: QualityIssue) -> None:
        if self._on_issue is not None:
            self._on_issue(issue)
