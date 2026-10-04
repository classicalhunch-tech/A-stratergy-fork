"""
phase_04_live/market/quality.py

Filters bad, missing, duplicate, or out-of-order candles out of the
live feed before they ever reach StrategyAdapter/strategy/.

Policy (per project decision): a bad candle is DROPPED, never passed
through, and a QualityIssue is raised for monitoring to pick up.
This module never halts the feed and never silently swallows a
problem -- every drop produces a QualityIssue.

GAP POLICY: by default (reject_gaps=True) a candle that follows a gap
is dropped. NOTE that a dropped candle does not update the last
accepted candle, so with this default the feed stays blocked after the
first gap (weekend, daily market break, outage) until restart. The
live source therefore uses reject_gaps=False: the gap is still
reported as a QualityIssue, but the candle is accepted and becomes the
new baseline. Real market data contains weekend and break gaps too,
and the backtest data does as well.
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

    def __init__(
        self,
        on_issue: Optional[Callable[[QualityIssue], None]] = None,
        reject_gaps: bool = True,
    ):
        self._last_accepted: dict[tuple[str, str], Candle] = {}
        self._on_issue = on_issue
        self._reject_gaps = reject_gaps

    def check(self, candle: Candle) -> tuple[bool, Optional[QualityIssue]]:
        """
        Check if a candle is acceptable for the live feed.
        
        Returns (accepted: bool, issue: QualityIssue | None):
            - (True, None): candle passed all checks
            - (False, issue): candle failed a check and is rejected
            - (True, issue): only when reject_gaps is False and the
              candle follows a gap; the gap is reported (also through
              the on_issue callback) but the candle is accepted

        All checks are applied in order. The first failure causes rejection,
        except gap issues when reject_gaps is False.
        """
        # Price, OHLC consistency and ordering problems always cause rejection.
        issue = (
            self._check_price_validity(candle)
            or self._check_ohlc_consistency(candle)
            or self._check_ordering(candle)
        )

        if issue is not None:
            self._raise(issue)
            return False, issue

        # Gap check. With reject_gaps=True the candle is dropped (and the
        # baseline is NOT updated). With reject_gaps=False the gap is
        # reported but the candle is accepted.
        gap_issue = self._check_gap(candle)

        if gap_issue is not None:
            self._raise(gap_issue)

            if self._reject_gaps:
                return False, gap_issue

        # Candle passed (or its gap was tolerated) -- update state and accept it
        key = (candle.symbol, candle.timeframe)
        self._last_accepted[key] = candle
        return True, gap_issue

    def _check_price_validity(self, candle: Candle) -> Optional[QualityIssue]:
        """Reject candles with zero or negative prices."""
        if candle.open <= 0 or candle.high <= 0 or candle.low <= 0 or candle.close <= 0:
            return self._make_issue(
                QualityIssueType.ZERO_OR_NEGATIVE_PRICE,
                candle,
                f"Non-positive price in OHLC: "
                f"O={candle.open} H={candle.high} L={candle.low} C={candle.close}",
            )
        return None

    def _check_ohlc_consistency(self, candle: Candle) -> Optional[QualityIssue]:
        """Reject candles with invalid OHLC relationships."""
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
        """Reject duplicate or out-of-order candles."""
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
        """
        Detect missing data between the last accepted candle and this one.
        
        A gap means one or more candles are missing in the feed.
        Gaps indicate:
        - Normal market closures (weekend, daily break)
        - Feed interruption (network issue, broker outage)
        - Subscription/connection problem
        - Data synchronization failure

        Whether a gap candle is rejected or accepted is decided by the
        reject_gaps setting (see the module docstring).
        """
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
                f"{last.timestamp} and {candle.timestamp} "
                f"(expected step {expected_step}s, got {actual_step:.0f}s)",
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
