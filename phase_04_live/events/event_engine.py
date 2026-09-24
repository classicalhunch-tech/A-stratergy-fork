"""
phase_04_live/events/event_engine.py

Thin live-loop wrapper around the EXISTING phase_03_paper
RuntimeCoordinator -- this does not reimplement tick orchestration.
RuntimeCoordinator.tick() already does that deterministically:
market data -> session eval -> notifications -> StrategyAdapter ->
session gate -> audit.

What this adds, which RuntimeCoordinator has no knowledge of:
    - clock drift/staleness checks, on their own fixed wall-clock
      interval, independent of tick count (per project decision --
      NOT tied to how many ticks have run)
    - capturing QualityIssues raised during that tick (via the
      on_quality_issue callback already wired into LiveMarketSource)
    - a safe-mode gate (recovery/safe_mode.py) checked once per cycle:
      when the kill switch is engaged, a signal RuntimeCoordinator
      already triggered (session-permitted) is re-classified as
      SIGNAL_SUPPRESSED here instead of SIGNAL_TRIGGERED, so nothing
      downstream ever sees it as actionable. This mirrors
      RuntimeCoordinator's own session-permission suppression
      pattern -- same idea, one layer up, for kill-switch state
      RuntimeCoordinator has no knowledge of. StrategyAdapter still
      receives every candle regardless (Phase 3's own rule -- safe
      mode never interrupts that)
    - collecting all of the above, plus RuntimeCoordinator's own
      triggered/suppressed signals and tick errors, into one ordered
      LiveEvent stream per cycle
"""

import time
from datetime import datetime, timezone
from typing import List, Optional

from phase_03_paper.runtime.coordinator import RuntimeCoordinator

from phase_04_live.clock.integrity import ClockIntegrityChecker
from phase_04_live.clock.models import ClockIssue, ClockIssueType
from phase_04_live.events.event_types import LiveEvent, LiveEventType
from phase_04_live.events.ordering import order_events
from phase_04_live.market.models import QualityIssue
from phase_04_live.recovery.safe_mode import SafeModeGate


class LiveEventEngine:
    """
    Drives RuntimeCoordinator.tick() in a loop and layers Phase 4
    live-specific checks (clock, quality, safe mode) on top,
    producing one ordered LiveEvent list per cycle.
    """

    def __init__(
        self,
        coordinator: RuntimeCoordinator,
        clock_checker: ClockIntegrityChecker,
        timeframe: str,
        poll_interval_seconds: float = 5.0,
        clock_check_interval_seconds: float = 30.0,
        safe_mode_gate: Optional[SafeModeGate] = None,
    ):
        self._coordinator = coordinator
        self._clock_checker = clock_checker
        self._timeframe = timeframe
        self.poll_interval_seconds = poll_interval_seconds
        self.clock_check_interval_seconds = clock_check_interval_seconds
        self._safe_mode_gate = safe_mode_gate

        self._last_clock_check_at: Optional[datetime] = None
        self._pending_quality_issues: List[QualityIssue] = []

    def on_quality_issue(self, issue: QualityIssue) -> None:
        """
        Wire this as LiveMarketSource(on_quality_issue=...) so issues
        raised during MarketDataEngine.next_candle() (inside
        RuntimeCoordinator.tick()) are captured for this cycle's
        event stream instead of being silently dropped.
        """
        self._pending_quality_issues.append(issue)

    def _should_check_clock(self, now: datetime) -> bool:
        if self._last_clock_check_at is None:
            return True
        elapsed = (now - self._last_clock_check_at).total_seconds()
        return elapsed >= self.clock_check_interval_seconds

    def _run_clock_checks(self, now: datetime) -> List[LiveEvent]:
        events: List[LiveEvent] = []

        drift_issue = self._clock_checker.check_drift()
        if drift_issue is not None:
            events.append(self._clock_issue_to_event(drift_issue))

        latest = self._coordinator.latest_candle()
        latest_ts = latest.timestamp if latest is not None else None

        stale_issue = self._clock_checker.check_staleness(
            latest_ts, self._timeframe
        )
        if stale_issue is not None:
            events.append(self._clock_issue_to_event(stale_issue))

        self._last_clock_check_at = now
        return events

    @staticmethod
    def _clock_issue_to_event(issue: ClockIssue) -> LiveEvent:
        event_type = (
            LiveEventType.CLOCK_DRIFT
            if issue.issue_type == ClockIssueType.CLOCK_DRIFT
            else LiveEventType.CLOCK_STALE
        )
        return LiveEvent(
            event_type=event_type,
            occurred_at=issue.detected_at,
            payload=issue,
        )

    def run_once(self) -> List[LiveEvent]:
        """
        Execute exactly one live cycle: optional clock check (on its
        own interval), one RuntimeCoordinator.tick(), a safe-mode
        check applied to whatever signals that tick triggered, and
        collection of every event produced during that cycle.
        Returns them in fixed deterministic order.
        """
        now = datetime.now(timezone.utc)
        events: List[LiveEvent] = []

        if self._should_check_clock(now):
            events.extend(self._run_clock_checks(now))

        self._pending_quality_issues = []
        status = self._coordinator.tick(now)

        for issue in self._pending_quality_issues:
            events.append(
                LiveEvent(
                    event_type=LiveEventType.QUALITY_ISSUE,
                    occurred_at=issue.detected_at,
                    payload=issue,
                )
            )

        if not status.healthy:
            events.append(
                LiveEvent(
                    event_type=LiveEventType.TICK_ERROR,
                    occurred_at=now,
                    payload=status.last_error,
                )
            )

        # ----------------------------------------------------------
        # Safe-mode gate: RuntimeCoordinator has already applied
        # SESSION permission (see its own tick() docstring). This is
        # a second, independent gate for KILL-SWITCH state, which
        # RuntimeCoordinator has no knowledge of. A signal that
        # passed session permission can still be blocked here.
        # ----------------------------------------------------------
        for event in self._coordinator.triggered_events():
            if self._safe_mode_gate is not None:
                status_check = self._safe_mode_gate.check()
                if not status_check.allowed:
                    events.append(
                        LiveEvent(
                            event_type=LiveEventType.SIGNAL_SUPPRESSED,
                            occurred_at=now,
                            payload=(event, status_check.reason),
                        )
                    )
                    continue

            events.append(
                LiveEvent(
                    event_type=LiveEventType.SIGNAL_TRIGGERED,
                    occurred_at=now,
                    payload=event,
                )
            )

        return order_events(events)

    def run_forever(self) -> None:
        """
        Run run_once() in a loop, sleeping poll_interval_seconds
        between cycles. Caller is responsible for catching
        KeyboardInterrupt / wiring this to operations/ shutdown
        controls -- not yet built.
        """
        while True:
            events = self.run_once()
            for event in events:
                print(event)  # placeholder until monitoring/ exists
            time.sleep(self.poll_interval_seconds)
