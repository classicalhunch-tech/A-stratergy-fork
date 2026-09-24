"""
phase_03_paper/events/audit.py

Central event/audit log (§26).

Every important Phase 3 state transition is recorded here. This
module holds events in memory only — disk persistence is the
responsibility of the persistence layer (§24), built later in the
sequence. Session engine, notification engine, and everything
downstream depend on this module existing first, not the reverse.
"""

from __future__ import annotations

from phase_03_paper.models import AuditEvent, NotificationSeverity


class AuditLog:
    """Append-only in-memory event log. No silent state changes
    anywhere in Phase 3 should occur without a corresponding record()
    call here."""

    def __init__(self) -> None:
        self._events: list[AuditEvent] = []

    def record(
        self,
        event_type: str,
        source: str,
        message: str,
        severity: NotificationSeverity = NotificationSeverity.INFO,
        related_id: str | None = None,
    ) -> AuditEvent:
        """Create and store one audit event. Returns the stored event."""

        event = AuditEvent(
            event_type=event_type,
            severity=severity,
            source=source,
            message=message,
            related_id=related_id,
        )

        self._events.append(event)

        return event

    def all_events(self) -> list[AuditEvent]:
        """Return every recorded event, oldest first."""

        return list(self._events)

    def events_for(self, related_id: str) -> list[AuditEvent]:
        """Return events tied to a specific session/trade/order ID."""

        return [e for e in self._events if e.related_id == related_id]

    def events_of_type(self, event_type: str) -> list[AuditEvent]:
        """Return events matching one event_type."""

        return [e for e in self._events if e.event_type == event_type]