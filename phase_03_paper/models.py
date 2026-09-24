from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from uuid import uuid4


class SessionStatus(str, Enum):
    UPCOMING = "UPCOMING"
    WARNING = "WARNING"
    ACTIVE = "ACTIVE"
    SKIPPED = "SKIPPED"
    CLOSED = "CLOSED"


class SessionDecision(str, Enum):
    UNDECIDED = "UNDECIDED"
    TRADE = "TRADE"
    SKIP = "SKIP"


class NotificationSeverity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class NotificationCategory(str, Enum):
    SESSION = "SESSION"
    RECOVERY = "RECOVERY"


@dataclass
class Notification:
    category: NotificationCategory
    severity: NotificationSeverity
    title: str
    message: str
    related_id: Optional[str] = None
    notification_id: str = field(default_factory=lambda: str(uuid4()))
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    is_read: bool = False


@dataclass(frozen=True)
class AuditEvent:
    """
    One immutable audit-log entry.

    Every important Phase 3 state transition must produce one of
    these via AuditLog.record(). timestamp is when the event was
    recorded, not necessarily when the underlying condition occurred.
    """

    event_type: str
    source: str
    message: str
    severity: NotificationSeverity = NotificationSeverity.INFO
    related_id: Optional[str] = None
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


@dataclass
class SessionState:
    """
    Complete state for ONE specific session occurrence.

    A London session on one day and a London session on another
    day are different SessionState objects with different occurrence_id
    values.

    This prevents decisions from leaking between occurrences.
    """

    session_name: str
    scheduled_start: datetime
    scheduled_end: datetime
    warning_time: datetime

    occurrence_id: str

    status: SessionStatus = SessionStatus.UPCOMING

    decision: SessionDecision = SessionDecision.UNDECIDED
    decision_timestamp: Optional[datetime] = None
    decision_note: Optional[str] = None

    warning_emitted: bool = False

    enabled: bool = False


__all__ = [
    "SessionStatus",
    "SessionDecision",
    "NotificationSeverity",
    "NotificationCategory",
    "Notification",
    "SessionState",
    "AuditEvent",
]
