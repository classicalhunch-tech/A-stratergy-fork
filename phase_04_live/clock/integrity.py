"""
phase_04_live/clock/integrity.py

Two independent checks:
1. Clock drift -- local PC clock vs MT5 server time.
2. Feed staleness -- latest candle older than expected for its timeframe.

Neither duplicates market/quality.py''s out-of-order/duplicate checks --
those compare candles to EACH OTHER. These compare candles (and the
clock itself) to the actual current time.
"""

from datetime import datetime, timezone
from typing import Callable, Optional

from phase_04_live.clock.models import ClockIssue, ClockIssueType
from phase_04_live.clock.synchronizer import compute_drift_seconds
from phase_04_live.market.quality import TIMEFRAME_SECONDS


class ClockIntegrityChecker:
    def __init__(
        self,
        symbol: str,
        max_drift_seconds: float = 2.0,
        stale_after_missed_candles: int = 2,
        on_issue: Optional[Callable[[ClockIssue], None]] = None,
    ):
        self.symbol = symbol
        self.max_drift_seconds = max_drift_seconds
        self.stale_after_missed_candles = stale_after_missed_candles
        self._on_issue = on_issue

    def check_drift(self) -> Optional[ClockIssue]:
        drift = compute_drift_seconds(self.symbol)

        if abs(drift) > self.max_drift_seconds:
            issue = ClockIssue(
                issue_type=ClockIssueType.CLOCK_DRIFT,
                detail=(
                    f"Local clock drift {drift:+.2f}s exceeds "
                    f"threshold {self.max_drift_seconds}s for '{self.symbol}'"
                ),
                measured_seconds=drift,
                threshold_seconds=self.max_drift_seconds,
                detected_at=datetime.now(timezone.utc),
            )
            self._raise(issue)
            return issue

        return None

    def check_staleness(
        self,
        latest_candle_timestamp: Optional[datetime],
        timeframe: str,
    ) -> Optional[ClockIssue]:
        if latest_candle_timestamp is None:
            return None

        expected_step = TIMEFRAME_SECONDS.get(timeframe)
        if expected_step is None:
            return None

        threshold = expected_step * self.stale_after_missed_candles
        age = (datetime.now(timezone.utc) - latest_candle_timestamp).total_seconds()

        if age > threshold:
            issue = ClockIssue(
                issue_type=ClockIssueType.FEED_STALE,
                detail=(
                    f"Latest candle for '{self.symbol}' ({timeframe}) is "
                    f"{age:.0f}s old, exceeding staleness threshold {threshold:.0f}s "
                    f"({self.stale_after_missed_candles} missed candles)"
                ),
                measured_seconds=age,
                threshold_seconds=threshold,
                detected_at=datetime.now(timezone.utc),
            )
            self._raise(issue)
            return issue

        return None

    def _raise(self, issue: ClockIssue) -> None:
        if self._on_issue is not None:
            self._on_issue(issue)
