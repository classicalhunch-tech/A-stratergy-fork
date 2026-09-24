"""
tests/test_phase3_sessions.py

Phase 3 session-engine test suite.

Covers:

* UPCOMING -> WARNING -> ACTIVE/CLOSED lifecycle
* exact 10-minute warning boundary
* explicit TRADE decision
* explicit SKIP decision
* undecided sessions never auto-enable
* London/New York overlap
* independent decisions per session occurrence
* next-day occurrence isolation
* post-close decisions target the next occurrence
* audit events for decisions and status changes
* deterministic UTC timestamps
* naive and aware datetime normalization
* invalid/unknown session protection
* invalid decision protection
* warning-state idempotency
* current/active/next session helpers
* session signal gating

All tests use explicit now= timestamps and therefore never depend
on the real wall clock.
"""

from __future__ import annotations

import unittest
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from phase_03_paper.config import DEFAULT_CONFIG
from phase_03_paper.events.audit import AuditLog
from phase_03_paper.models import SessionDecision, SessionStatus
from phase_03_paper.sessions.engine import SessionEngine


# ============================================================
# FIXED TEST CLOCK
# ============================================================

BASE_DATE = date(2026, 9, 9)


def dt(
    hour: int,
    minute: int,
    day_offset: int = 0,
) -> datetime:
    """Return a deterministic timezone-aware UTC datetime."""
    day = BASE_DATE + timedelta(days=day_offset)

    return datetime(
        day.year,
        day.month,
        day.day,
        hour,
        minute,
        tzinfo=timezone.utc,
    )


def naive_dt(
    hour: int,
    minute: int,
    day_offset: int = 0,
) -> datetime:
    """Return a deterministic naive datetime for normalization tests."""
    day = BASE_DATE + timedelta(days=day_offset)

    return datetime(
        day.year,
        day.month,
        day.day,
        hour,
        minute,
    )


def occurrence_id(
    session_name: str,
    start: datetime,
) -> str:
    """Build the SessionEngine occurrence ID format."""
    return (
        f"{session_name}-"
        f"{start.isoformat().replace('+00:00', 'Z')}"
    )


LONDON_ID = occurrence_id(
    "London",
    dt(7, 0),
)

NEW_YORK_ID = occurrence_id(
    "New York",
    dt(12, 0),
)

LONDON_ID_DAY2 = occurrence_id(
    "London",
    dt(7, 0, day_offset=1),
)


# ============================================================
# TEST FACTORY
# ============================================================

def new_engine() -> tuple[SessionEngine, AuditLog]:
    """Create a completely fresh engine and audit log."""
    audit_log = AuditLog()

    engine = SessionEngine(
        config=DEFAULT_CONFIG,
        audit_log=audit_log,
    )

    return engine, audit_log


# ============================================================
# LIFECYCLE STATUS TESTS
# ============================================================

class TestSessionLifecycleStatus(unittest.TestCase):
    """Verify the lifecycle of an individual session occurrence."""

    def test_upcoming_before_warning(self) -> None:
        engine, _ = new_engine()

        state = engine.get_state(
            LONDON_ID,
            now=dt(6, 0),
        )

        self.assertEqual(state.status, SessionStatus.UPCOMING)
        self.assertEqual(
            state.decision,
            SessionDecision.UNDECIDED,
        )
        self.assertFalse(state.enabled)
        self.assertFalse(state.warning_emitted)

    def test_warning_window(self) -> None:
        engine, _ = new_engine()

        state = engine.get_state(
            LONDON_ID,
            now=dt(6, 55),
        )

        self.assertEqual(state.status, SessionStatus.WARNING)
        self.assertEqual(
            state.decision,
            SessionDecision.UNDECIDED,
        )
        self.assertTrue(state.warning_emitted)
        self.assertFalse(state.enabled)

    def test_warning_exactly_ten_minutes_before_start(self) -> None:
        engine, _ = new_engine()

        state = engine.get_state(
            LONDON_ID,
            now=dt(6, 50),
        )

        self.assertEqual(state.status, SessionStatus.WARNING)
        self.assertTrue(state.warning_emitted)
        self.assertFalse(state.enabled)

    def test_warning_inside_warning_window(self) -> None:
        engine, _ = new_engine()

        state = engine.get_state(
            LONDON_ID,
            now=dt(6, 51),
        )

        self.assertEqual(state.status, SessionStatus.WARNING)
        self.assertTrue(state.warning_emitted)
        self.assertFalse(state.enabled)

    def test_trade_decision_before_start_enables_at_session_start(
        self,
    ) -> None:
        engine, _ = new_engine()

        engine.record_decision(
            "London",
            SessionDecision.TRADE,
            now=dt(6, 55),
        )

        before_start = engine.get_state(
            LONDON_ID,
            now=dt(6, 59),
        )

        self.assertEqual(
            before_start.status,
            SessionStatus.WARNING,
        )
        self.assertFalse(before_start.enabled)

        at_start = engine.get_state(
            LONDON_ID,
            now=dt(7, 0),
        )

        self.assertEqual(
            at_start.status,
            SessionStatus.ACTIVE,
        )
        self.assertEqual(
            at_start.decision,
            SessionDecision.TRADE,
        )
        self.assertTrue(at_start.enabled)

    def test_trade_decision_during_active_session_remains_enabled(
        self,
    ) -> None:
        engine, _ = new_engine()

        engine.record_decision(
            "London",
            SessionDecision.TRADE,
            now=dt(6, 55),
        )

        state = engine.get_state(
            LONDON_ID,
            now=dt(8, 0),
        )

        self.assertEqual(
            state.status,
            SessionStatus.ACTIVE,
        )
        self.assertEqual(
            state.decision,
            SessionDecision.TRADE,
        )
        self.assertTrue(state.enabled)

    def test_explicit_skip_becomes_skipped_at_session_start(
        self,
    ) -> None:
        engine, _ = new_engine()

        engine.record_decision(
            "London",
            SessionDecision.SKIP,
            now=dt(6, 55),
        )

        before_start = engine.get_state(
            LONDON_ID,
            now=dt(6, 59),
        )

        self.assertEqual(
            before_start.status,
            SessionStatus.WARNING,
        )
        self.assertFalse(before_start.enabled)

        after_start = engine.get_state(
            LONDON_ID,
            now=dt(7, 1),
        )

        self.assertEqual(
            after_start.status,
            SessionStatus.SKIPPED,
        )
        self.assertEqual(
            after_start.decision,
            SessionDecision.SKIP,
        )
        self.assertFalse(after_start.enabled)

    def test_undecided_session_never_auto_enables(self) -> None:
        """
        Safety-critical rule:

        The engine must never assume the user wants to trade a
        session. Without an explicit TRADE decision, the occurrence
        must remain disabled.
        """
        engine, _ = new_engine()

        warning_state = engine.get_state(
            LONDON_ID,
            now=dt(6, 55),
        )

        self.assertEqual(
            warning_state.decision,
            SessionDecision.UNDECIDED,
        )
        self.assertFalse(warning_state.enabled)

        session_state = engine.get_state(
            LONDON_ID,
            now=dt(7, 30),
        )

        self.assertEqual(
            session_state.status,
            SessionStatus.SKIPPED,
        )
        self.assertEqual(
            session_state.decision,
            SessionDecision.UNDECIDED,
        )
        self.assertFalse(session_state.enabled)

    def test_session_closes_after_end(self) -> None:
        engine, _ = new_engine()

        engine.record_decision(
            "London",
            SessionDecision.TRADE,
            now=dt(6, 55),
        )

        state = engine.get_state(
            LONDON_ID,
            now=dt(16, 1),
        )

        self.assertEqual(
            state.status,
            SessionStatus.CLOSED,
        )
        self.assertEqual(
            state.decision,
            SessionDecision.TRADE,
        )
        self.assertFalse(state.enabled)

    def test_full_lifecycle_progression(self) -> None:
        """Walk through the complete lifecycle."""
        engine, _ = new_engine()

        # UPCOMING
        state = engine.get_state(
            LONDON_ID,
            now=dt(5, 0),
        )

        self.assertEqual(
            state.status,
            SessionStatus.UPCOMING,
        )

        # WARNING
        state = engine.get_state(
            LONDON_ID,
            now=dt(6, 55),
        )

        self.assertEqual(
            state.status,
            SessionStatus.WARNING,
        )
        self.assertTrue(state.warning_emitted)

        # TRADE DECISION
        engine.record_decision(
            "London",
            SessionDecision.TRADE,
            now=dt(6, 56),
        )

        # ACTIVE
        state = engine.get_state(
            LONDON_ID,
            now=dt(8, 0),
        )

        self.assertEqual(
            state.status,
            SessionStatus.ACTIVE,
        )
        self.assertTrue(state.enabled)

        # CLOSED
        state = engine.get_state(
            LONDON_ID,
            now=dt(17, 0),
        )

        self.assertEqual(
            state.status,
            SessionStatus.CLOSED,
        )
        self.assertFalse(state.enabled)


# ============================================================
# DECISION TESTS
# ============================================================

class TestSessionDecisions(unittest.TestCase):
    """Verify decision recording and signal-gate behavior."""

    def test_trade_decision_is_recorded(self) -> None:
        engine, _ = new_engine()

        state = engine.record_decision(
            "London",
            SessionDecision.TRADE,
            now=dt(6, 55),
        )

        self.assertEqual(
            state.decision,
            SessionDecision.TRADE,
        )
        self.assertEqual(
            state.decision_timestamp,
            dt(6, 55),
        )

    def test_skip_decision_is_recorded(self) -> None:
        engine, _ = new_engine()

        state = engine.record_decision(
            "London",
            SessionDecision.SKIP,
            now=dt(6, 55),
        )

        self.assertEqual(
            state.decision,
            SessionDecision.SKIP,
        )
        self.assertEqual(
            state.decision_timestamp,
            dt(6, 55),
        )

    def test_decision_note_is_preserved(self) -> None:
        engine, _ = new_engine()

        state = engine.record_decision(
            "London",
            SessionDecision.TRADE,
            now=dt(6, 55),
            note="Testing session decision persistence",
        )

        self.assertEqual(
            state.decision_note,
            "Testing session decision persistence",
        )

    def test_trade_gate_is_false_before_session_start(self) -> None:
        engine, _ = new_engine()

        engine.record_decision(
            "London",
            SessionDecision.TRADE,
            now=dt(6, 55),
        )

        state = engine.get_state(
            LONDON_ID,
            now=dt(6, 59),
        )

        self.assertFalse(state.enabled)

    def test_trade_gate_becomes_true_when_active(self) -> None:
        engine, _ = new_engine()

        engine.record_decision(
            "London",
            SessionDecision.TRADE,
            now=dt(6, 55),
        )

        state = engine.get_state(
            LONDON_ID,
            now=dt(7, 30),
        )

        self.assertEqual(
            state.status,
            SessionStatus.ACTIVE,
        )
        self.assertTrue(state.enabled)

    def test_trade_gate_becomes_false_after_close(self) -> None:
        engine, _ = new_engine()

        engine.record_decision(
            "London",
            SessionDecision.TRADE,
            now=dt(6, 55),
        )

        state = engine.get_state(
            LONDON_ID,
            now=dt(16, 1),
        )

        self.assertEqual(
            state.status,
            SessionStatus.CLOSED,
        )
        self.assertFalse(state.enabled)

    def test_skip_gate_never_becomes_true(self) -> None:
        engine, _ = new_engine()

        engine.record_decision(
            "London",
            SessionDecision.SKIP,
            now=dt(6, 55),
        )

        state = engine.get_state(
            LONDON_ID,
            now=dt(7, 30),
        )

        self.assertFalse(state.enabled)

    def test_is_session_enabled_matches_state(self) -> None:
        engine, _ = new_engine()

        engine.record_decision(
            "London",
            SessionDecision.TRADE,
            now=dt(6, 55),
        )

        self.assertFalse(
            engine.is_session_enabled(
                LONDON_ID,
                now=dt(6, 59),
            )
        )

        self.assertTrue(
            engine.is_session_enabled(
                LONDON_ID,
                now=dt(7, 30),
            )
        )

        self.assertFalse(
            engine.is_session_enabled(
                LONDON_ID,
                now=dt(16, 1),
            )
        )

    def test_invalid_session_name_is_rejected(self) -> None:
        engine, _ = new_engine()

        with self.assertRaises((KeyError, ValueError)):
            engine.record_decision(
                "InvalidSessionName",
                SessionDecision.TRADE,
                now=dt(6, 55),
            )

    def test_invalid_decision_is_rejected(self) -> None:
        engine, _ = new_engine()

        with self.assertRaises(ValueError):
            engine.record_decision(
                "London",
                SessionDecision.UNDECIDED,
                now=dt(6, 55),
            )

    def test_decision_after_close_targets_next_occurrence(self) -> None:
        """
        After today's London occurrence closes, a session-name
        decision must target tomorrow's eligible London occurrence.

        The closed occurrence must remain unchanged.
        """
        engine, _ = new_engine()

        engine.record_decision(
            "London",
            SessionDecision.TRADE,
            now=dt(6, 55),
        )

        closed_state = engine.get_state(
            LONDON_ID,
            now=dt(16, 1),
        )

        self.assertEqual(
            closed_state.status,
            SessionStatus.CLOSED,
        )
        self.assertEqual(
            closed_state.decision,
            SessionDecision.TRADE,
        )

        next_state = engine.record_decision(
            "London",
            SessionDecision.SKIP,
            now=dt(16, 1),
        )

        self.assertEqual(
            next_state.occurrence_id,
            LONDON_ID_DAY2,
        )
        self.assertEqual(
            next_state.decision,
            SessionDecision.SKIP,
        )
        self.assertFalse(next_state.enabled)

        closed_state_after = engine.get_state(
            LONDON_ID,
            now=dt(16, 1),
        )

        self.assertEqual(
            closed_state_after.status,
            SessionStatus.CLOSED,
        )
        self.assertEqual(
            closed_state_after.decision,
            SessionDecision.TRADE,
        )


# ============================================================
# WARNING IDEMPOTENCY TESTS
# ============================================================

class TestWarningIdempotency(unittest.TestCase):
    """Verify warning state remains stable across repeated evaluation."""

    def test_warning_emission_flag_is_idempotent(self) -> None:
        engine, _ = new_engine()

        first = engine.get_state(
            LONDON_ID,
            now=dt(6, 55),
        )

        self.assertEqual(
            first.status,
            SessionStatus.WARNING,
        )
        self.assertTrue(first.warning_emitted)

        second = engine.get_state(
            LONDON_ID,
            now=dt(6, 56),
        )

        self.assertEqual(
            second.status,
            SessionStatus.WARNING,
        )
        self.assertTrue(second.warning_emitted)

        third = engine.get_state(
            LONDON_ID,
            now=dt(6, 57),
        )

        self.assertEqual(
            third.status,
            SessionStatus.WARNING,
        )
        self.assertTrue(third.warning_emitted)

    def test_warning_does_not_reappear_after_active(self) -> None:
        engine, _ = new_engine()

        engine.record_decision(
            "London",
            SessionDecision.TRADE,
            now=dt(6, 55),
        )

        active = engine.get_state(
            LONDON_ID,
            now=dt(7, 30),
        )

        self.assertEqual(
            active.status,
            SessionStatus.ACTIVE,
        )
        self.assertTrue(active.warning_emitted)

        closed = engine.get_state(
            LONDON_ID,
            now=dt(16, 1),
        )

        self.assertEqual(
            closed.status,
            SessionStatus.CLOSED,
        )
        self.assertTrue(closed.warning_emitted)


# ============================================================
# OVERLAPPING SESSION TESTS
# ============================================================

class TestOverlappingSessions(unittest.TestCase):
    """
    London and New York overlap.

    Their decisions and signal permissions must remain independent.
    """

    def test_london_and_new_york_are_distinct_occurrences(
        self,
    ) -> None:
        engine, _ = new_engine()

        london = engine.get_state(
            LONDON_ID,
            now=dt(6, 55),
        )

        new_york = engine.get_state(
            NEW_YORK_ID,
            now=dt(11, 55),
        )

        self.assertEqual(
            london.session_name,
            "London",
        )
        self.assertEqual(
            new_york.session_name,
            "New York",
        )
        self.assertNotEqual(
            london.occurrence_id,
            new_york.occurrence_id,
        )

    def test_london_trade_does_not_enable_new_york(self) -> None:
        engine, _ = new_engine()

        engine.record_decision(
            "London",
            SessionDecision.TRADE,
            now=dt(6, 55),
        )

        london = engine.get_state(
            LONDON_ID,
            now=dt(12, 30),
        )

        new_york = engine.get_state(
            NEW_YORK_ID,
            now=dt(12, 30),
        )

        self.assertTrue(london.enabled)
        self.assertEqual(
            new_york.decision,
            SessionDecision.UNDECIDED,
        )
        self.assertFalse(new_york.enabled)

    def test_new_york_skip_does_not_skip_london(self) -> None:
        engine, _ = new_engine()

        engine.record_decision(
            "New York",
            SessionDecision.SKIP,
            now=dt(11, 55),
        )

        london = engine.get_state(
            LONDON_ID,
            now=dt(12, 30),
        )

        new_york = engine.get_state(
            NEW_YORK_ID,
            now=dt(12, 30),
        )

        self.assertEqual(
            london.decision,
            SessionDecision.UNDECIDED,
        )
        self.assertFalse(london.enabled)

        self.assertEqual(
            new_york.decision,
            SessionDecision.SKIP,
        )
        self.assertEqual(
            new_york.status,
            SessionStatus.SKIPPED,
        )
        self.assertFalse(new_york.enabled)

    def test_both_sessions_can_be_enabled_independently(
        self,
    ) -> None:
        engine, _ = new_engine()

        engine.record_decision(
            "London",
            SessionDecision.TRADE,
            now=dt(6, 55),
        )

        engine.record_decision(
            "New York",
            SessionDecision.TRADE,
            now=dt(11, 55),
        )

        london = engine.get_state(
            LONDON_ID,
            now=dt(12, 30),
        )

        new_york = engine.get_state(
            NEW_YORK_ID,
            now=dt(12, 30),
        )

        self.assertTrue(london.enabled)
        self.assertTrue(new_york.enabled)


# ============================================================
# NEXT-DAY OCCURRENCE TESTS
# ============================================================

class TestSessionOccurrenceIsolation(unittest.TestCase):
    """
    Verify that decisions belong to one exact occurrence.

    Decisions must never leak from one day into another.
    """

    def test_next_day_london_has_new_occurrence_id(self) -> None:
        engine, _ = new_engine()

        day1 = engine.get_state(
            LONDON_ID,
            now=dt(7, 30),
        )

        day2 = engine.get_state(
            LONDON_ID_DAY2,
            now=dt(7, 30, day_offset=1),
        )

        self.assertNotEqual(
            day1.occurrence_id,
            day2.occurrence_id,
        )

    def test_trade_decision_does_not_leak_to_next_day(self) -> None:
        engine, _ = new_engine()

        engine.record_decision(
            "London",
            SessionDecision.TRADE,
            now=dt(6, 55),
        )

        day1 = engine.get_state(
            LONDON_ID,
            now=dt(7, 30),
        )

        day2 = engine.get_state(
            LONDON_ID_DAY2,
            now=dt(7, 30, day_offset=1),
        )

        self.assertEqual(
            day1.decision,
            SessionDecision.TRADE,
        )
        self.assertTrue(day1.enabled)

        self.assertEqual(
            day2.decision,
            SessionDecision.UNDECIDED,
        )
        self.assertFalse(day2.enabled)

    def test_skip_decision_does_not_leak_to_next_day(self) -> None:
        engine, _ = new_engine()

        engine.record_decision(
            "London",
            SessionDecision.SKIP,
            now=dt(6, 55),
        )

        day1 = engine.get_state(
            LONDON_ID,
            now=dt(7, 30),
        )

        day2 = engine.get_state(
            LONDON_ID_DAY2,
            now=dt(7, 30, day_offset=1),
        )

        self.assertEqual(
            day1.decision,
            SessionDecision.SKIP,
        )
        self.assertEqual(
            day1.status,
            SessionStatus.SKIPPED,
        )

        self.assertEqual(
            day2.decision,
            SessionDecision.UNDECIDED,
        )
        self.assertFalse(day2.enabled)


# ============================================================
# SESSION HELPER TESTS
# ============================================================

class TestSessionHelpers(unittest.TestCase):
    """Verify current, active, next, and timing helper methods."""

    def test_current_session_returns_active_session(self) -> None:
        engine, _ = new_engine()

        engine.record_decision(
            "London",
            SessionDecision.TRADE,
            now=dt(6, 55),
        )

        current = engine.current_session(
            now=dt(8, 0),
        )

        self.assertIsNotNone(current)
        self.assertEqual(
            current.session_name,
            "London",
        )
        self.assertEqual(
            current.status,
            SessionStatus.ACTIVE,
        )
        self.assertTrue(current.enabled)

    def test_current_session_is_none_when_nothing_is_active(
        self,
    ) -> None:
        engine, _ = new_engine()

        current = engine.current_session(
            now=dt(5, 0),
        )

        self.assertIsNone(current)

    def test_active_sessions_support_overlap(self) -> None:
        engine, _ = new_engine()

        engine.record_decision(
            "London",
            SessionDecision.TRADE,
            now=dt(6, 55),
        )

        engine.record_decision(
            "New York",
            SessionDecision.TRADE,
            now=dt(11, 55),
        )

        active = engine.active_sessions(
            now=dt(12, 30),
        )

        names = [
            state.session_name
            for state in active
        ]

        self.assertIn(
            "London",
            names,
        )
        self.assertIn(
            "New York",
            names,
        )

    def test_next_session_returns_new_york_before_its_start(
        self,
    ) -> None:
        engine, _ = new_engine()

        next_state = engine.next_session(
            now=dt(8, 0),
        )

        self.assertIsNotNone(next_state)
        self.assertEqual(
            next_state.session_name,
            "New York",
        )
        self.assertEqual(
            next_state.occurrence_id,
            NEW_YORK_ID,
        )
        self.assertEqual(
            next_state.status,
            SessionStatus.UPCOMING,
        )

    def test_minutes_until_next_session_is_correct(self) -> None:
        engine, _ = new_engine()

        minutes = engine.minutes_until_next_session(
            now=dt(8, 0),
        )

        self.assertIsNotNone(minutes)
        self.assertEqual(
            minutes,
            240.0,
        )

    def test_closed_occurrence_is_not_returned_as_next_session(
        self,
    ) -> None:
        engine, _ = new_engine()

        next_state = engine.next_session(
            now=dt(16, 30),
        )

        self.assertIsNotNone(next_state)
        self.assertEqual(
            next_state.session_name,
            "London",
        )
        self.assertEqual(
            next_state.occurrence_id,
            LONDON_ID_DAY2,
        )


# ============================================================
# DATETIME NORMALIZATION TESTS
# ============================================================

class TestDatetimeNormalization(unittest.TestCase):
    """Verify deterministic UTC normalization."""

    def test_naive_datetime_is_interpreted_as_utc(self) -> None:
        engine, _ = new_engine()

        state = engine.get_state(
            LONDON_ID,
            now=naive_dt(6, 55),
        )

        self.assertEqual(
            state.status,
            SessionStatus.WARNING,
        )

    def test_aware_non_utc_datetime_is_converted_to_utc(
        self,
    ) -> None:
        engine, _ = new_engine()

        nairobi_time = datetime(
            2026,
            9,
            9,
            9,
            55,
            tzinfo=ZoneInfo("Africa/Nairobi"),
        )

        state = engine.get_state(
            LONDON_ID,
            now=nairobi_time,
        )

        self.assertEqual(
            state.status,
            SessionStatus.WARNING,
        )

    def test_utc_state_timestamps_remain_utc(self) -> None:
        engine, _ = new_engine()

        state = engine.get_state(
            LONDON_ID,
            now=dt(6, 55),
        )

        self.assertIsNotNone(
            state.scheduled_start.tzinfo,
        )
        self.assertEqual(
            state.scheduled_start.utcoffset(),
            timedelta(0),
        )

        self.assertIsNotNone(
            state.scheduled_end.tzinfo,
        )
        self.assertEqual(
            state.scheduled_end.utcoffset(),
            timedelta(0),
        )


# ============================================================
# AUDIT TESTS
# ============================================================

class TestSessionAudit(unittest.TestCase):
    """Verify important session changes are audited."""

    def test_decision_creates_audit_event(self) -> None:
        engine, audit_log = new_engine()

        engine.record_decision(
            "London",
            SessionDecision.TRADE,
            now=dt(6, 55),
        )

        events = audit_log.events_for(
            LONDON_ID,
        )

        self.assertTrue(events)

        decision_events = [
            event
            for event in events
            if event.event_type == "session_decision"
        ]

        self.assertTrue(decision_events)

        event = decision_events[-1]

        self.assertEqual(
            event.source,
            "SessionEngine",
        )
        self.assertEqual(
            event.related_id,
            LONDON_ID,
        )

    def test_status_change_creates_audit_event(self) -> None:
        engine, audit_log = new_engine()

        # Materialize the occurrence while it's still UPCOMING, BEFORE
        # touching the warning window. If its first-ever observation
        # happens directly inside WARNING, the engine intentionally
        # "births" it already in WARNING (see _build_occurrence's
        # true_now handling -- this is what stops a restart/recovery
        # discovery from firing a spurious transition event), so no
        # session_status_change event would ever exist to find.
        engine.get_state(
            LONDON_ID,
            now=dt(6, 0),
        )

        engine.get_state(
            LONDON_ID,
            now=dt(6, 55),
        )

        events = audit_log.events_for(
            LONDON_ID,
        )

        status_events = [
            event
            for event in events
            if event.event_type == "session_status_change"
        ]

        self.assertTrue(status_events)
        self.assertIn("WARNING", status_events[-1].message)

    def test_signal_gate_change_creates_audit_event(
        self,
    ) -> None:
        engine, audit_log = new_engine()

        engine.record_decision(
            "London",
            SessionDecision.TRADE,
            now=dt(6, 55),
        )

        engine.get_state(
            LONDON_ID,
            now=dt(7, 30),
        )

        events = audit_log.events_for(
            LONDON_ID,
        )

        gate_events = [
            event
            for event in events
            if event.event_type
            == "session_signal_gate_change"
        ]

        self.assertTrue(gate_events)

    def test_audit_events_have_utc_timestamps(self) -> None:
        engine, audit_log = new_engine()

        engine.record_decision(
            "London",
            SessionDecision.TRADE,
            now=dt(6, 55),
        )

        events = audit_log.events_for(
            LONDON_ID,
        )

        self.assertTrue(events)

        for event in events:
            self.assertIsNotNone(
                event.timestamp,
            )
            self.assertIsNotNone(
                event.timestamp.tzinfo,
            )
            self.assertEqual(
                event.timestamp.utcoffset(),
                timedelta(0),
            )


# ============================================================
# TEST RUNNER
# ============================================================

if __name__ == "__main__":
    unittest.main()