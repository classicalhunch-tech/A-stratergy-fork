"""
phase_03_paper/sessions/engine.py

Centralized Phase 3 session engine.

This module is the single source of truth for:

    - configured session windows
    - session occurrences
    - warning timing
    - session lifecycle
    - per-occurrence YES / NO decisions
    - session signal permission
    - session transition auditing

IMPORTANT
---------
A configured session name is NOT a unique runtime identity.

For example:

    London - 2026-09-09
    London - 2026-09-10

are two different session occurrences and must have independent
decisions and lifecycle state.

Therefore SessionState objects are keyed by occurrence_id.

TIMEZONE MODEL
--------------
Session times are configured in the Phase 3 session timezone
(Africa/Nairobi by default).

The engine converts those local session times into timezone-aware
UTC datetimes internally.

This means:

    CONFIGURATION / DASHBOARD
        Africa/Nairobi local time

                â†“

        SessionEngine conversion

                â†“

    RUNTIME / LIFECYCLE
        UTC

The engine does NOT perform strategy calculations.

The engine does NOT own:

    - dashboard rendering
    - notification delivery
    - market-data calculations
    - strategy calculations
    - order execution
"""

from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from typing import Final
from zoneinfo import ZoneInfo

from phase_03_paper.config import (
    DEFAULT_CONFIG,
    Phase3Config,
    SessionWindow,
)
from phase_03_paper.events.audit import AuditLog
from phase_03_paper.models import (
    SessionDecision,
    SessionState,
    SessionStatus,
)


# ============================================================
# CONSTANTS
# ============================================================

_SOURCE_NAME: Final[str] = "SessionEngine"

_EVENT_DECISION: Final[str] = "session_decision"
_EVENT_STATUS_CHANGE: Final[str] = "session_status_change"
_EVENT_SIGNAL_GATE_CHANGE: Final[str] = "session_signal_gate_change"


# ============================================================
# SESSION ENGINE
# ============================================================


class SessionEngine:
    """
    Centralized, stateful Phase 3 session lifecycle engine.

    Each enabled configured session produces a SessionState for each
    concrete calendar occurrence that the engine has materialized.

    Example occurrence IDs:

        London-2026-09-09T07:00:00Z
        New York-2026-09-09T12:00:00Z

    Decisions are stored against occurrence_id, never merely against
    the session name.

    Lifecycle:

        UPCOMING
            |
            v
        WARNING
            |
            +---- TRADE ----> ACTIVE ----> CLOSED
            |
            +---- SKIP  ----> SKIPPED

    A session that reaches its start without an explicit TRADE
    decision is automatically marked SKIPPED and remains disabled.

    This is intentional: Phase 3 must never assume that the user
    wants to trade a session.
    """

    def __init__(
        self,
        config: Phase3Config = DEFAULT_CONFIG,
        audit_log: AuditLog | None = None,
    ) -> None:
        self._config = config

        self._audit_log = (
            audit_log
            if audit_log is not None
            else AuditLog()
        )

        # Runtime state is keyed by unique session occurrence ID.
        self._states: dict[str, SessionState] = {}

    # ============================================================
    # PUBLIC - DECISION MANAGEMENT
    # ============================================================

    def record_decision(
        self,
        session_name: str,
        decision: SessionDecision,
        note: str | None = None,
        now: datetime | None = None,
    ) -> SessionState:
        """
        Record a user's decision for the relevant session occurrence.

        The occurrence selected is the occurrence that is currently
        in the WARNING window or otherwise the next occurrence that
        has not yet closed.

        Decisions are allowed only for an occurrence that has not
        already closed or been skipped.

        TRADE means the session may enable signal processing once
        the session becomes ACTIVE.

        SKIP means the session must remain disabled.

        Returns:
            A defensive copy of the updated SessionState.

        Raises:
            ValueError:
                If the decision is invalid, no valid occurrence can
                be selected, or the occurrence is already terminal.
        """

        now_utc = self._normalize_now(now)

        if decision not in {
            SessionDecision.TRADE,
            SessionDecision.SKIP,
        }:
            raise ValueError(
                "record_decision requires SessionDecision.TRADE "
                "or SessionDecision.SKIP."
            )

        self._ensure_occurrences(now_utc)
        self.evaluate(now_utc)

        state = self._find_decision_target(
            session_name=session_name,
            now=now_utc,
        )

        if state is None:
            raise ValueError(
                f"No decision-eligible occurrence found for "
                f"session={session_name!r}."
            )

        if state.status in {
            SessionStatus.CLOSED,
            SessionStatus.SKIPPED,
        }:
            raise ValueError(
                f"Session occurrence {state.occurrence_id!r} "
                f"is already terminal with status={state.status.value}."
            )

        if now_utc >= state.scheduled_end:
            raise ValueError(
                f"Session occurrence {state.occurrence_id!r} "
                "has already ended."
            )

        state.decision = decision
        state.decision_timestamp = now_utc
        state.decision_note = note

        # A TRADE decision enables the occurrence only when it is
        # actually ACTIVE. If it is still UPCOMING/WARNING, enabled
        # remains False until the session begins.
        state.enabled = (
            state.status == SessionStatus.ACTIVE
            and decision == SessionDecision.TRADE
        )

        self._audit_log.record(
            event_type=_EVENT_DECISION,
            source=_SOURCE_NAME,
            message=(
                f"{state.session_name} session decision recorded: "
                f"{decision.value}."
            ),
            related_id=state.occurrence_id,
        )

        # Re-evaluate so the status immediately reflects the decision.
        self.evaluate(now_utc)

        return replace(self._states[state.occurrence_id])

    # ============================================================
    # PUBLIC - EVALUATION
    # ============================================================

    def evaluate(
        self,
        now: datetime | None = None,
    ) -> list[SessionState]:
        """
        Advance all materialized session occurrences to their correct
        state for the supplied time.

        Returns:
            Defensive copies of all materialized states.

        The method is deterministic for a supplied datetime, making
        it suitable for unit testing and paper-runtime operation.
        """

        now_utc = self._normalize_now(now)

        self._ensure_occurrences(
            now_utc,
            true_now=now_utc,
        )

        for occurrence_id, state in list(self._states.items()):
            previous_status = state.status
            previous_enabled = state.enabled

            new_status = self._compute_status(
                state=state,
                now=now_utc,
            )

            state.status = new_status

            state.enabled = (
                state.status == SessionStatus.ACTIVE
                and state.decision == SessionDecision.TRADE
            )

            if new_status != previous_status:
                if new_status == SessionStatus.WARNING:
                    state.warning_emitted = True

                self._audit_log.record(
                    event_type=_EVENT_STATUS_CHANGE,
                    source=_SOURCE_NAME,
                    message=(
                        f"{state.session_name} session changed from "
                        f"{previous_status.value} to "
                        f"{new_status.value}."
                    ),
                    related_id=occurrence_id,
                )

            if state.enabled != previous_enabled:
                self._audit_log.record(
                    event_type=_EVENT_SIGNAL_GATE_CHANGE,
                    source=_SOURCE_NAME,
                    message=(
                        f"{state.session_name} signal gate "
                        f"{'enabled' if state.enabled else 'disabled'}."
                    ),
                    related_id=occurrence_id,
                )

        # Keep the rolling horizon populated.
        #
        # reference_time is shifted forward only so that tomorrow's
        # occurrences are materialized. true_now remains the ACTUAL
        # current time so freshly-created future occurrences do not
        # incorrectly believe that tomorrow has already arrived.
        self._ensure_occurrences(
            now_utc + timedelta(days=1),
            true_now=now_utc,
        )

        return self.all_states()

    # ============================================================
    # PUBLIC - STATE ACCESS
    # ============================================================

    def get_state(
        self,
        session_name_or_occurrence_id: str,
        now: datetime | None = None,
    ) -> SessionState:
        """
        Retrieve one session state.

        The argument may be either:

            - a full occurrence_id
            - a configured session name such as "London"

        When a session name is supplied, the nearest relevant
        occurrence is returned.

        Raises:
            KeyError:
                If no matching state exists.
        """

        now_utc = self._normalize_now(now)

        self._ensure_occurrences(now_utc)
        self.evaluate(now_utc)

        if session_name_or_occurrence_id in self._states:
            return replace(
                self._states[session_name_or_occurrence_id]
            )

        candidates = [
            state
            for state in self._states.values()
            if state.session_name == session_name_or_occurrence_id
        ]

        if not candidates:
            raise KeyError(
                f"Unknown session or occurrence: "
                f"{session_name_or_occurrence_id!r}"
            )

        relevant = self._select_relevant_occurrence(
            candidates,
            now_utc,
        )

        if relevant is None:
            relevant = min(
                candidates,
                key=lambda item: item.scheduled_start,
            )

        return replace(relevant)

    def all_states(
        self,
        now: datetime | None = None,
    ) -> list[SessionState]:
        """
        Return defensive copies of all materialized session
        occurrences.

        States are returned in chronological order.

        Terminal occurrences are retained long enough for audit and
        recovery consumers to inspect them.
        """

        if now is not None:
            self.evaluate(now)

        return [
            replace(state)
            for state in sorted(
                self._states.values(),
                key=lambda item: item.scheduled_start,
            )
        ]

    # ============================================================
    # PUBLIC - CURRENT / NEXT SESSION
    # ============================================================

    def current_session(
        self,
        now: datetime | None = None,
    ) -> SessionState | None:
        """
        Return the currently ACTIVE session occurrence.

        If multiple configured sessions overlap, the earliest active
        occurrence is returned.
        """

        now_utc = self._normalize_now(now)

        self.evaluate(now_utc)

        active = [
            state
            for state in self._states.values()
            if state.status == SessionStatus.ACTIVE
        ]

        if not active:
            return None

        active.sort(
            key=lambda state: state.scheduled_start
        )

        return replace(active[0])

    def active_sessions(
        self,
        now: datetime | None = None,
    ) -> list[SessionState]:
        """
        Return every currently ACTIVE session occurrence.

        This is important because London and New York can overlap.
        """

        now_utc = self._normalize_now(now)

        self.evaluate(now_utc)

        active = [
            replace(state)
            for state in self._states.values()
            if state.status == SessionStatus.ACTIVE
        ]

        active.sort(
            key=lambda state: state.scheduled_start
        )

        return active

    def next_session(
        self,
        now: datetime | None = None,
    ) -> SessionState | None:
        """
        Return the next session occurrence that has not yet started.

        SKIPPED and CLOSED occurrences are ignored.
        """

        now_utc = self._normalize_now(now)

        self.evaluate(now_utc)

        candidates = [
            state
            for state in self._states.values()
            if state.scheduled_start > now_utc
            and state.status not in {
                SessionStatus.SKIPPED,
                SessionStatus.CLOSED,
            }
        ]

        if not candidates:
            return None

        candidates.sort(
            key=lambda state: state.scheduled_start
        )

        return replace(candidates[0])

    def minutes_until_next_session(
        self,
        now: datetime | None = None,
    ) -> float | None:
        """
        Return minutes until the next session starts.

        Returns None when no next session exists.
        """

        now_utc = self._normalize_now(now)

        next_state = self.next_session(now_utc)

        if next_state is None:
            return None

        seconds = (
            next_state.scheduled_start - now_utc
        ).total_seconds()

        return max(0.0, seconds / 60.0)

    # ============================================================
    # PUBLIC - SIGNAL GATING
    # ============================================================

    def is_session_enabled(
        self,
        session_name_or_occurrence_id: str,
        now: datetime | None = None,
    ) -> bool:
        """
        Return whether the specified session occurrence is allowed
        to generate strategy signals.

        Signal permission is TRUE only when:

            status == ACTIVE
            AND
            decision == TRADE
        """

        state = self.get_state(
            session_name_or_occurrence_id,
            now=now,
        )

        return state.enabled

    # ============================================================
    # PUBLIC - AUDIT
    # ============================================================

    @property
    def audit_log(self) -> AuditLog:
        """Return the audit log used by this engine."""

        return self._audit_log

    # ============================================================
    # INTERNAL - OCCURRENCE CREATION
    # ============================================================

    def _ensure_occurrences(
        self,
        reference_time: datetime,
        true_now: datetime | None = None,
    ) -> None:
        """
        Materialize enabled session occurrences around reference_time.

        We keep:

            - yesterday
            - today
            - tomorrow
            - day after tomorrow
            - two days after tomorrow

        This provides enough horizon for:

            - current session lookup
            - next session lookup
            - overnight/cross-midnight windows
            - restart/recovery
            - dashboard rendering

        Existing occurrences are never recreated, so decisions and
        warning state survive repeated evaluate() calls.

        reference_time controls which calendar dates are materialized.

        true_now is the ACTUAL current instant used to determine the
        initial status of a newly-created occurrence.

        These are deliberately separate because reference_time may
        be shifted forward to extend the rolling horizon.
        """

        if true_now is None:
            true_now = reference_time

        reference_date = reference_time.date()

        for day_offset in range(-1, 4):
            occurrence_date = (
                reference_date
                + timedelta(days=day_offset)
            )

            for window in self._config.sessions.sessions:
                if not window.enabled:
                    continue

                occurrence = self._build_occurrence(
                    window=window,
                    occurrence_date=occurrence_date,
                    true_now=true_now,
                )

                if occurrence is None:
                    continue

                occurrence_id = occurrence.occurrence_id

                if occurrence_id not in self._states:
                    self._states[occurrence_id] = occurrence

    def _build_occurrence(
        self,
        window: SessionWindow,
        occurrence_date: date,
        true_now: datetime,
    ) -> SessionState | None:
        """
        Build one concrete SessionState from a configured local-time
        session window.

        Session times are configured in the Phase 3 session timezone
        (Africa/Nairobi by default).

        The engine converts those local times to timezone-aware UTC
        datetimes internally.

        If end_local is earlier than start_local, the session crosses
        midnight in the configured local timezone.

        A zero-length window is rejected.
        """

        session_timezone = ZoneInfo(
            self._config.market_data.session_timezone
        )

        # --------------------------------------------------------
        # LOCAL START
        # --------------------------------------------------------

        start_local_dt = datetime.combine(
            occurrence_date,
            window.start_local,
            tzinfo=session_timezone,
        )

        # --------------------------------------------------------
        # LOCAL END
        # --------------------------------------------------------

        if window.end_local > window.start_local:
            end_local_date = occurrence_date

        elif window.end_local < window.start_local:
            # Cross-midnight session.
            end_local_date = occurrence_date + timedelta(days=1)

        else:
            # Equal start/end means no meaningful session window.
            return None

        end_local_dt = datetime.combine(
            end_local_date,
            window.end_local,
            tzinfo=session_timezone,
        )

        # --------------------------------------------------------
        # CONVERT LOCAL SESSION TIMES TO UTC
        # --------------------------------------------------------

        start_dt = start_local_dt.astimezone(timezone.utc)
        end_dt = end_local_dt.astimezone(timezone.utc)

        # --------------------------------------------------------
        # WARNING TIME
        # --------------------------------------------------------

        warning_minutes = (
            self._config.sessions.warning_minutes_before
        )

        warning_dt = (
            start_dt
            - timedelta(minutes=warning_minutes)
        )

        # --------------------------------------------------------
        # UNIQUE OCCURRENCE ID
        # --------------------------------------------------------

        occurrence_id = (
            f"{window.name}-"
            f"{start_dt.isoformat().replace('+00:00', 'Z')}"
        )

        # --------------------------------------------------------
        # INITIAL STATE
        # --------------------------------------------------------

        state = SessionState(
            session_name=window.name,
            scheduled_start=start_dt,
            scheduled_end=end_dt,
            warning_time=warning_dt,
            occurrence_id=occurrence_id,
            status=SessionStatus.UPCOMING,
            decision=SessionDecision.UNDECIDED,
            decision_timestamp=None,
            decision_note=None,
            warning_emitted=False,
            enabled=False,
        )

        # Give the occurrence its TRUE current status at birth.
        #
        # This prevents a newly-discovered occurrence whose window
        # already passed from being born UPCOMING and then producing
        # a false lifecycle transition on the next evaluation.
        state.status = self._compute_status(
            state=state,
            now=true_now,
        )

        # WARNING notifications are transition-driven.
        #
        # A freshly materialized occurrence must not pretend that
        # its warning event was emitted merely because its current
        # status happens to be WARNING.
        state.warning_emitted = state.status != SessionStatus.UPCOMING

        return state

    # ============================================================
    # INTERNAL - STATUS CALCULATION
    # ============================================================

    def _compute_status(
        self,
        state: SessionState,
        now: datetime,
    ) -> SessionStatus:
        """
        Compute lifecycle status from occurrence timing + decision.

        Before warning:
            UPCOMING

        Warning window:
            WARNING

        At/after start:
            ACTIVE if TRADE
            SKIPPED if SKIP
            SKIPPED if no decision was made

        At/after end:
            CLOSED

        The engine never assumes an undecided session should be
        traded.
        """

        if now >= state.scheduled_end:
            return SessionStatus.CLOSED

        if now >= state.scheduled_start:
            if state.decision == SessionDecision.TRADE:
                return SessionStatus.ACTIVE

            # SKIP or UNDECIDED at/after start both mean that the
            # session has no signal permission.
            return SessionStatus.SKIPPED

        if now >= state.warning_time:
            return SessionStatus.WARNING

        return SessionStatus.UPCOMING

    # ============================================================
    # INTERNAL - DECISION TARGET SELECTION
    # ============================================================

    def _find_decision_target(
        self,
        session_name: str,
        now: datetime,
    ) -> SessionState | None:
        """
        Select the occurrence to which a YES/NO decision belongs.

        Priority:

            1. Current WARNING occurrence.
            2. Current ACTIVE occurrence.
            3. Nearest future occurrence.

        Closed/skipped occurrences are never selected.
        """

        candidates = [
            state
            for state in self._states.values()
            if state.session_name == session_name
            and state.status not in {
                SessionStatus.CLOSED,
                SessionStatus.SKIPPED,
            }
            and now < state.scheduled_end
        ]

        if not candidates:
            return None

        warning_candidates = [
            state
            for state in candidates
            if state.status == SessionStatus.WARNING
        ]

        if warning_candidates:
            warning_candidates.sort(
                key=lambda state: state.scheduled_start
            )
            return warning_candidates[0]

        active_candidates = [
            state
            for state in candidates
            if state.status == SessionStatus.ACTIVE
        ]

        if active_candidates:
            active_candidates.sort(
                key=lambda state: state.scheduled_start
            )
            return active_candidates[0]

        future_candidates = [
            state
            for state in candidates
            if state.scheduled_start > now
        ]

        if not future_candidates:
            return None

        future_candidates.sort(
            key=lambda state: state.scheduled_start
        )

        return future_candidates[0]

    # ============================================================
    # INTERNAL - RELEVANT OCCURRENCE SELECTION
    # ============================================================

    @staticmethod
    def _select_relevant_occurrence(
        candidates: list[SessionState],
        now: datetime,
    ) -> SessionState | None:
        """
        Select the most relevant occurrence for a session name.

        Priority:

            1. ACTIVE
            2. WARNING
            3. nearest future occurrence
        """

        active = [
            state
            for state in candidates
            if state.status == SessionStatus.ACTIVE
        ]

        if active:
            return min(
                active,
                key=lambda state: state.scheduled_start,
            )

        warning = [
            state
            for state in candidates
            if state.status == SessionStatus.WARNING
        ]

        if warning:
            return min(
                warning,
                key=lambda state: state.scheduled_start,
            )

        future = [
            state
            for state in candidates
            if state.scheduled_start > now
        ]

        if future:
            return min(
                future,
                key=lambda state: state.scheduled_start,
            )

        return None

    # ============================================================
    # INTERNAL - DATETIME NORMALIZATION
    # ============================================================

    @staticmethod
    def _normalize_now(
        value: datetime | None,
    ) -> datetime:
        """
        Normalize all runtime timestamps to timezone-aware UTC.

        Naive datetimes are interpreted as UTC for deterministic
        Phase 3 behavior.

        Aware datetimes are converted to UTC.
        """

        if value is None:
            return datetime.now(timezone.utc)

        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)

        return value.astimezone(timezone.utc)


__all__ = ["SessionEngine"]
