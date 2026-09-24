from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Final, Optional, Any

from phase_03_paper.events.audit import AuditLog
from phase_03_paper.market.engine import Candle, MarketDataEngine
from phase_03_paper.models import (
    NotificationSeverity,
    SessionState,
    SessionStatus,
)
from phase_03_paper.notifications.engine import NotificationEngine
from phase_03_paper.sessions.engine import SessionEngine


_SOURCE_NAME: Final[str] = "RuntimeCoordinator"

_EVENT_TICK_ERROR: Final[str] = "runtime_tick_error"
_EVENT_SIGNAL_TRIGGERED: Final[str] = "strategy_signal_triggered"
_EVENT_SIGNAL_SUPPRESSED: Final[str] = (
    "strategy_signal_suppressed_no_session"
)
_EVENT_ADAPTER_ERROR: Final[str] = "strategy_adapter_error"


@dataclass
class RuntimeStatus:
    """Snapshot of the coordinator's runtime state."""

    tick_count: int
    last_tick_at: Optional[datetime]
    last_error: Optional[str]
    latest_candle: Optional[Candle]

    @property
    def healthy(self) -> bool:
        """Return True when the most recent tick had no error."""
        return self.last_error is None


class RuntimeCoordinator:
    """
    Coordinate Phase 3 runtime components.

    Responsibilities:

        MarketDataEngine
              |
              v
        session evaluation
              |
              v
        notification routing
              |
              v
        StrategyAdapter
              |
              v
        session permission gate
              |
              v
        triggered-event queue
    """

    def __init__(
        self,
        session_engine: SessionEngine,
        notification_engine: NotificationEngine,
        market_data_engine: MarketDataEngine | None = None,
        audit_log: AuditLog | None = None,
        strategy_adapter: Any | None = None,
    ) -> None:
        self._session_engine = session_engine
        self._notification_engine = notification_engine
        self._market_data_engine = market_data_engine
        self._strategy_adapter = strategy_adapter

        self._audit_log = (
            audit_log
            if audit_log is not None
            else session_engine.audit_log
        )

        self._tick_count: int = 0
        self._last_tick_at: Optional[datetime] = None
        self._last_error: Optional[str] = None

        # Stage 1 event queue.
        # Stage 2 will consume these events for paper execution.
        self._triggered_events: list[Any] = []

        # Number of adapter errors already written to the audit log.
        self._adapter_errors_seen: int = 0

    def tick(self, now: datetime | None = None) -> RuntimeStatus:
        """
        Execute exactly one runtime cycle.

        A successful tick:

            1. advances market data by one candle
            2. evaluates session state once
            3. routes session notifications
            4. feeds the NEW candle to the strategy adapter
            5. applies session permission to triggered events
            6. audits new adapter errors
            7. updates runtime status

        Runtime exceptions are captured rather than propagated.
        """

        now_utc = self._normalize_now(now)

        try:
            # ------------------------------------------------------
            # 1. Advance market data.
            # ------------------------------------------------------
            #
            # IMPORTANT:
            # Use the return value directly.
            #
            # Do not call latest() and assume it is a new candle.
            # latest() may still contain the previous candle when the
            # source is exhausted.
            # ------------------------------------------------------

            new_candle: Candle | None = None

            if self._market_data_engine is not None:
                new_candle = self._market_data_engine.next_candle()

            # ------------------------------------------------------
            # 2. Evaluate session state exactly once.
            # ------------------------------------------------------

            states = self._session_engine.evaluate(now_utc)

            # ------------------------------------------------------
            # 3. Route session notifications.
            # ------------------------------------------------------

            for state in states:
                self._notify_for_state(state)

            # ------------------------------------------------------
            # 4. Feed every NEW candle to the strategy adapter.
            # ------------------------------------------------------

            if (
                self._strategy_adapter is not None
                and new_candle is not None
            ):
                new_events = self._strategy_adapter.on_candle(
                    new_candle
                )

                # --------------------------------------------------
                # 5. Apply session permission AFTER strategy
                #    processing.
                #
                # Reuse the already-evaluated states rather than
                # calling SessionEngine.evaluate() a second time.
                # --------------------------------------------------

                session_permitted = any(
                    state.enabled
                    for state in states
                )

                for event in new_events:
                    if session_permitted:
                        self._queue_triggered_event(event)
                    else:
                        self._suppress_triggered_event(event)

                # --------------------------------------------------
                # 6. Audit only NEW adapter errors.
                # --------------------------------------------------

                self._audit_new_adapter_errors()

            # A successful tick clears a previous transient error.
            self._last_error = None

        except Exception as exc:  # noqa: BLE001
            self._record_tick_error(exc)

        finally:
            self._tick_count += 1
            self._last_tick_at = now_utc

        return self.status()

    def triggered_events(self) -> list[Any]:
        """
        Drain and return triggered AdapterSignalEvents.

        The returned events are removed from the coordinator queue.

        Stage 2 will consume these events and pass them to
        PaperTradeEngine.
        """

        events = self._triggered_events
        self._triggered_events = []
        return events

    def enabled_occurrence_ids(
        self,
        now: datetime | None = None,
    ) -> list[str]:
        """
        Return IDs of currently enabled session occurrences.

        This is a read/query helper and evaluates SessionEngine at
        the requested time.
        """

        states = self._session_engine.evaluate(
            self._normalize_now(now)
        )

        return [
            state.occurrence_id
            for state in states
            if state.enabled
        ]

    def is_signal_permitted(
        self,
        session_name_or_occurrence_id: str,
        now: datetime | None = None,
    ) -> bool:
        """Return whether the requested session is currently enabled."""

        return self._session_engine.is_session_enabled(
            session_name_or_occurrence_id,
            now=self._normalize_now(now),
        )

    def latest_candle(self) -> Optional[Candle]:
        """Return the latest market candle known by the data engine."""

        if self._market_data_engine is None:
            return None

        return self._market_data_engine.latest()

    def status(self) -> RuntimeStatus:
        """Return the current runtime status snapshot."""

        return RuntimeStatus(
            tick_count=self._tick_count,
            last_tick_at=self._last_tick_at,
            last_error=self._last_error,
            latest_candle=self.latest_candle(),
        )

    @property
    def audit_log(self) -> AuditLog:
        """Return the coordinator audit log."""

        return self._audit_log

    def _queue_triggered_event(self, event: Any) -> None:
        """Queue a session-permitted strategy event and audit it."""

        self._triggered_events.append(event)

        self._audit_log.record(
            event_type=_EVENT_SIGNAL_TRIGGERED,
            source=_SOURCE_NAME,
            message=(
                f"Strategy signal triggered at bar "
                f"{event.trigger_bar_index} "
                f"(setup bar {event.setup_bar_index}), "
                f"fill={event.fill_price}, "
                f"risk={event.initial_risk}"
            ),
            severity=NotificationSeverity.INFO,
        )

    def _suppress_triggered_event(self, event: Any) -> None:
        """Audit a triggered strategy event suppressed by session rules."""

        self._audit_log.record(
            event_type=_EVENT_SIGNAL_SUPPRESSED,
            source=_SOURCE_NAME,
            message=(
                f"Strategy signal at bar "
                f"{event.trigger_bar_index} suppressed: "
                f"no session occurrence currently enabled"
            ),
            severity=NotificationSeverity.INFO,
        )

    def _audit_new_adapter_errors(self) -> None:
        """Audit adapter errors that have not yet been recorded."""

        adapter_errors = self._strategy_adapter.errors

        if len(adapter_errors) <= self._adapter_errors_seen:
            return

        for error in adapter_errors[self._adapter_errors_seen:]:
            self._audit_log.record(
                event_type=_EVENT_ADAPTER_ERROR,
                source=_SOURCE_NAME,
                message=f"Strategy adapter error: {error}",
                severity=NotificationSeverity.ERROR,
            )

        self._adapter_errors_seen = len(adapter_errors)

    def _record_tick_error(self, exc: Exception) -> None:
        """Store and audit a runtime tick error."""

        self._last_error = repr(exc)

        self._audit_log.record(
            event_type=_EVENT_TICK_ERROR,
            source=_SOURCE_NAME,
            message=(
                f"Unhandled error during coordinator tick: "
                f"{exc!r}"
            ),
            severity=NotificationSeverity.ERROR,
        )

    def _notify_for_state(self, state: SessionState) -> None:
        """Route a session state to the appropriate notification."""

        if state.status == SessionStatus.WARNING:
            self._notification_engine.session_approaching(
                state.session_name,
                state.occurrence_id,
            )

        elif state.status == SessionStatus.ACTIVE:
            self._notification_engine.session_started(
                state.session_name,
                state.occurrence_id,
            )

        elif state.status == SessionStatus.SKIPPED:
            self._notification_engine.session_skipped(
                state.session_name,
                state.occurrence_id,
            )

        elif state.status == SessionStatus.CLOSED:
            self._notification_engine.session_closed(
                state.session_name,
                state.occurrence_id,
            )

    @staticmethod
    def _normalize_now(value: datetime | None) -> datetime:
        """Normalize timestamps to timezone-aware UTC datetimes."""

        if value is None:
            return datetime.now(timezone.utc)

        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)

        return value.astimezone(timezone.utc)


__all__ = [
    "RuntimeCoordinator",
    "RuntimeStatus",
]
