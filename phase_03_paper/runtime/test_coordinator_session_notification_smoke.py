"""
phase_03_paper/runtime/test_coordinator_session_notification_smoke.py

Smoke test for:

    RuntimeCoordinator
        |
        v
    SessionEngine
        |
        v
    NotificationEngine

This proves that when RuntimeCoordinator.tick() evaluates sessions,
the correct NotificationEngine method is routed for each relevant
session state (WARNING, ACTIVE, SKIPPED, CLOSED).

This test does NOT cover market-data wiring (see
test_coordinator_market_smoke.py) and does NOT cover error handling
(a separate smoke test covers that, per the project roadmap).

TWO INDEPENDENT SESSION OCCURRENCES ARE USED:

    TestSession  -> a TRADE decision is recorded during its WARNING
                    window, so it progresses:
                        UPCOMING -> WARNING -> ACTIVE -> CLOSED

    SkipSession  -> no decision is ever recorded, so it progresses:
                        UPCOMING -> WARNING -> SKIPPED

All times are supplied explicitly to tick() so the test is fully
deterministic and does not depend on wall-clock time.

IMPORTANT NOTE ON ASSERTIONS
-----------------------------
SessionEngine materializes a rolling 5-day window of occurrences
(yesterday .. +3 days) every time it evaluates, and
RuntimeCoordinator.tick() routes a notification for every
materialized occurrence's current status -- not just the occurrence
under test. In particular, a "yesterday" occurrence of each window is
born already CLOSED and will generate its own session_closed
notification on the very first tick.

Rather than asserting on the *total* notification count (which would
be sensitive to that unrelated background behavior), this test
filters notifications by related_id (the occurrence_id under test),
which is robust regardless of how many other occurrences exist.
"""

from __future__ import annotations

from datetime import datetime, time, timezone

from phase_03_paper.config import Phase3Config, SessionSettings, SessionWindow
from phase_03_paper.events.audit import AuditLog
from phase_03_paper.models import SessionDecision
from phase_03_paper.notifications.engine import (
    Notification,
    NotificationCategory,
    NotificationEngine,
)
from phase_03_paper.runtime.coordinator import RuntimeCoordinator
from phase_03_paper.sessions.engine import SessionEngine


def _utc(year: int, month: int, day: int, hour: int, minute: int) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=timezone.utc)


def _for_occurrence(
    notifications: list[Notification],
    related_id: str,
) -> list[Notification]:
    return [n for n in notifications if n.related_id == related_id]


def run_smoke_test() -> None:
    # ------------------------------------------------------------
    # CONFIG: two independent test session windows.
    #
    # Africa/Nairobi is UTC+3 with no DST, so local times convert to
    # UTC by simply subtracting 3 hours.
    #
    #   TestSession local 10:00-10:30 -> UTC 07:00-07:30
    #   SkipSession local 11:00-11:30 -> UTC 08:00-08:30
    # ------------------------------------------------------------

    config = Phase3Config(
        sessions=SessionSettings(
            sessions=(
                SessionWindow(
                    name="TestSession",
                    start_local=time(10, 0),
                    end_local=time(10, 30),
                ),
                SessionWindow(
                    name="SkipSession",
                    start_local=time(11, 0),
                    end_local=time(11, 30),
                ),
            ),
            warning_minutes_before=10,
        )
    )

    audit_log = AuditLog()
    session_engine = SessionEngine(config=config, audit_log=audit_log)
    notification_engine = NotificationEngine(config=config, audit_log=audit_log)

    coordinator = RuntimeCoordinator(
        session_engine=session_engine,
        notification_engine=notification_engine,
        market_data_engine=None,
        audit_log=audit_log,
    )

    test_occurrence_id = "TestSession-2026-01-01T07:00:00Z"
    skip_occurrence_id = "SkipSession-2026-01-01T08:00:00Z"

    # ------------------------------------------------------------
    # STAGE 1: well before either occurrence's warning window.
    # Neither of OUR occurrences should have fired or be enabled yet.
    # ------------------------------------------------------------

    now = _utc(2026, 1, 1, 6, 0)
    status = coordinator.tick(now=now)

    assert status.healthy, status.last_error

    all_notifications = notification_engine.all_notifications()
    assert _for_occurrence(all_notifications, test_occurrence_id) == []
    assert _for_occurrence(all_notifications, skip_occurrence_id) == []

    enabled = coordinator.enabled_occurrence_ids(now=now)
    assert test_occurrence_id not in enabled
    assert skip_occurrence_id not in enabled

    # ------------------------------------------------------------
    # STAGE 2: TestSession enters its WARNING window.
    #
    # A TRADE decision is recorded directly on SessionEngine here,
    # simulating what the dashboard will eventually do. Recording a
    # decision does not itself notify anyone -- only tick() routes
    # notifications.
    # ------------------------------------------------------------

    now = _utc(2026, 1, 1, 6, 55)
    session_engine.record_decision("TestSession", SessionDecision.TRADE, now=now)

    status = coordinator.tick(now=now)
    assert status.healthy, status.last_error

    test_notifs = _for_occurrence(
        notification_engine.all_notifications(), test_occurrence_id
    )
    assert len(test_notifs) == 1
    assert test_notifs[0].category == NotificationCategory.SESSION

    assert coordinator.is_signal_permitted("TestSession", now=now) is False

    # ------------------------------------------------------------
    # STAGE 3: TestSession becomes ACTIVE (decision was TRADE).
    # ------------------------------------------------------------

    now = _utc(2026, 1, 1, 7, 5)
    status = coordinator.tick(now=now)
    assert status.healthy, status.last_error

    test_notifs = _for_occurrence(
        notification_engine.all_notifications(), test_occurrence_id
    )
    assert len(test_notifs) == 2  # approaching + started

    assert coordinator.is_signal_permitted("TestSession", now=now) is True
    assert test_occurrence_id in coordinator.enabled_occurrence_ids(now=now)

    # ------------------------------------------------------------
    # STAGE 4: TestSession closes. SkipSession is still UPCOMING
    # (its own warning window doesn't start until 07:50).
    # ------------------------------------------------------------

    now = _utc(2026, 1, 1, 7, 35)
    status = coordinator.tick(now=now)
    assert status.healthy, status.last_error

    test_notifs = _for_occurrence(
        notification_engine.all_notifications(), test_occurrence_id
    )
    assert len(test_notifs) == 3  # approaching + started + closed

    assert coordinator.is_signal_permitted("TestSession", now=now) is False
    assert test_occurrence_id not in coordinator.enabled_occurrence_ids(now=now)

    # ------------------------------------------------------------
    # STAGE 5: SkipSession enters its WARNING window. No decision is
    # ever recorded for it.
    # ------------------------------------------------------------

    now = _utc(2026, 1, 1, 7, 55)
    status = coordinator.tick(now=now)
    assert status.healthy, status.last_error

    skip_notifs = _for_occurrence(
        notification_engine.all_notifications(), skip_occurrence_id
    )
    assert len(skip_notifs) == 1  # approaching only

    # ------------------------------------------------------------
    # STAGE 6: SkipSession reaches its start time with no decision
    # and must automatically become SKIPPED -- never ACTIVE.
    # ------------------------------------------------------------

    now = _utc(2026, 1, 1, 8, 5)
    status = coordinator.tick(now=now)
    assert status.healthy, status.last_error

    skip_notifs = _for_occurrence(
        notification_engine.all_notifications(), skip_occurrence_id
    )
    assert len(skip_notifs) == 2  # approaching + skipped

    assert coordinator.is_signal_permitted("SkipSession", now=now) is False
    assert skip_occurrence_id not in coordinator.enabled_occurrence_ids(now=now)

    # ------------------------------------------------------------
    # FINAL CHECKS
    # ------------------------------------------------------------

    assert status.tick_count == 6

    event_types = {event.event_type for event in audit_log.all_events()}
    assert "session_status_change" in event_types
    assert "session_decision" in event_types
    assert "notification_generated" in event_types


if __name__ == "__main__":
    run_smoke_test()