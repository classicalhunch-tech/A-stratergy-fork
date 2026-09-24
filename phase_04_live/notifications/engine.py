"""
phase_04_live/notifications/engine.py

Central Phase 4 notification engine.

Same architecture as phase_03_paper/notifications/engine.py
(in-memory storage, duplicate protection, read/unread tracking,
audit integration) but with recovery-specific emit methods instead
of session-specific ones. This is a SEPARATE class, not a subclass
of Phase 3's NotificationEngine -- Phase 3's public surface is
entirely session-shaped, and forcing recovery concepts into it would
blur the same paper/live separation the project already enforces
elsewhere (dashboard never owns trading state, etc).

Reuses phase_03_paper.models.Notification / NotificationCategory /
NotificationSeverity directly -- these are generic value types with
no Phase-3 coupling, so duplicating them here would violate the
project's own "one source of truth" rule for no benefit.

This module does NOT:
    - decide when a recovery event occurred (restart.py does)
    - persist notifications (a later persistence layer's job, same
      as Phase 3's notification engine)
    - render anything (dashboard/UI layer's job)
"""

from __future__ import annotations

from typing import Final

from phase_03_paper.events.audit import AuditLog
from phase_03_paper.models import (
    Notification,
    NotificationCategory,
    NotificationSeverity,
)

from phase_04_live.config import DEFAULT_CONFIG, Phase4Config
from phase_04_live.notifications import templates


_SOURCE_NAME: Final[str] = "Phase4NotificationEngine"
_EVENT_NOTIFICATION_GENERATED: Final[str] = "notification_generated"


class Phase4NotificationEngine:
    """
    Central in-memory Phase 4 notification engine.

    Deliberately UI-independent, same as Phase 3's notification
    engine. Runtime/recovery code tells this engine that an event
    occurred; the engine builds, stores, and audits the Notification.
    """

    def __init__(
        self,
        config: Phase4Config = DEFAULT_CONFIG,
        audit_log: AuditLog | None = None,
    ) -> None:
        self._config = config
        self._audit_log = (
            audit_log
            if audit_log is not None
            else AuditLog()
        )
        self._notifications: list[Notification] = []

    # ============================================================
    # PUBLIC -- RECOVERY NOTIFICATIONS
    # ============================================================

    def recovery_pause(
        self,
        reason: str,
        related_id: str | None = None,
    ) -> Notification | None:
        """Emit a notification when restart/recovery engages the kill switch."""

        if not self._config.notifications.recovery_notifications_enabled:
            return None

        title, message = templates.recovery_pause(reason)

        return self._emit_if_new(
            severity=NotificationSeverity.CRITICAL,
            title=title,
            message=message,
            related_id=related_id,
        )

    def recovery_review_required(
        self,
        reason: str,
        related_id: str | None = None,
    ) -> Notification | None:
        """Emit a notification for a REQUIRE_MANUAL_REVIEW recovery decision."""

        if not self._config.notifications.recovery_notifications_enabled:
            return None

        title, message = templates.recovery_review_required(reason)

        return self._emit_if_new(
            severity=NotificationSeverity.WARNING,
            title=title,
            message=message,
            related_id=related_id,
        )

    def recovery_alert(
        self,
        reason: str,
        related_id: str | None = None,
    ) -> Notification | None:
        """Emit a general recovery ALERT notification."""

        if not self._config.notifications.recovery_notifications_enabled:
            return None

        title, message = templates.recovery_alert(reason)

        return self._emit_if_new(
            severity=NotificationSeverity.WARNING,
            title=title,
            message=message,
            related_id=related_id,
        )

    def order_attempt_correlated(
        self,
        ticket: int,
        identity_key: str,
        related_id: str | None = None,
    ) -> Notification | None:
        """
        Emit a notification when an UNEXPECTED_AT_BROKER position is
        correlated against a pending durable order attempt.
        """

        if not self._config.notifications.recovery_notifications_enabled:
            return None

        title, message = templates.order_attempt_correlated(
            ticket,
            identity_key,
        )

        return self._emit_if_new(
            severity=NotificationSeverity.WARNING,
            title=title,
            message=message,
            related_id=related_id or str(ticket),
        )

    # ============================================================
    # PUBLIC -- QUERY
    # ============================================================

    def all_notifications(self) -> list[Notification]:
        """Return every notification, oldest first."""

        return list(self._notifications)

    def unread(self) -> list[Notification]:
        """Return all unread notifications, oldest first."""

        return [
            notification
            for notification in self._notifications
            if not notification.is_read
        ]

    def unread_count(self) -> int:
        """Return the number of unread notifications."""

        return sum(
            1
            for notification in self._notifications
            if not notification.is_read
        )

    def mark_read(self, notification_id: str) -> bool:
        """Mark one notification as read. Returns True if found."""

        for notification in self._notifications:
            if notification.notification_id == notification_id:
                notification.is_read = True
                return True

        return False

    def mark_all_read(self) -> int:
        """Mark every unread notification as read. Returns count changed."""

        changed = 0

        for notification in self._notifications:
            if not notification.is_read:
                notification.is_read = True
                changed += 1

        return changed

    # ============================================================
    # PUBLIC -- AUDIT ACCESS
    # ============================================================

    @property
    def audit_log(self) -> AuditLog:
        return self._audit_log

    # ============================================================
    # PUBLIC -- CLEARING
    # ============================================================

    def clear(self) -> None:
        """
        Clear the in-memory notification collection. Primarily for
        tests. Audit log is left untouched (permanent record).
        """

        self._notifications.clear()

    # ============================================================
    # INTERNAL -- DUPLICATE PROTECTION
    # ============================================================

    def _already_emitted(
        self,
        title: str,
        related_id: str | None,
    ) -> bool:
        """
        Duplicate protection checked against in-memory state, same
        as Phase 3's engine -- the audit log is a permanent record
        and must not be what decides re-emission eligibility.
        """

        for notification in self._notifications:
            if (
                notification.category == NotificationCategory.RECOVERY
                and notification.title == title
                and notification.related_id == related_id
            ):
                return True

        return False

    # ============================================================
    # INTERNAL -- EMISSION
    # ============================================================

    def _emit_if_new(
        self,
        severity: NotificationSeverity,
        title: str,
        message: str,
        related_id: str | None,
    ) -> Notification | None:
        if self._already_emitted(title=title, related_id=related_id):
            return None

        notification = Notification(
            category=NotificationCategory.RECOVERY,
            severity=severity,
            title=title,
            message=message,
            related_id=related_id,
        )

        self._notifications.append(notification)

        self._audit_log.record(
            event_type=_EVENT_NOTIFICATION_GENERATED,
            source=_SOURCE_NAME,
            message=f"{NotificationCategory.RECOVERY.value}: {title}",
            severity=severity,
            related_id=related_id,
        )

        return notification


__all__ = ["Phase4NotificationEngine"]
