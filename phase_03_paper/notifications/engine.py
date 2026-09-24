"""
phase_03_paper/notifications/engine.py

Central notification engine.

Builds Notification objects for Phase 3 session events.

This module contains no UI code. The dashboard reads Notification
objects from here and renders them; it never constructs notification
text itself.

DESIGN
------
The notification engine is responsible for:

1. Building notifications from centralized templates.
2. Storing notifications in memory.
3. Preventing duplicate notifications for the same session occurrence.
4. Tracking read/unread state.
5. Recording notification generation in the central AuditLog.

SESSION OCCURRENCES
-------------------
SessionEngine treats each occurrence of a session as a distinct state.

For example:

    London / 2026-09-09
    London / 2026-09-10

must not be treated as the same notification target.

Therefore callers should provide occurrence_id whenever it is available.

The fallback related_id remains session_name so the engine remains
usable before the runtime coordinator is fully wired.

CONFIGURATION
-------------
The session warning interval comes from:

    config.sessions.warning_minutes_before

This is the same configuration value used by SessionEngine and is
therefore the single source of truth for the warning window.
"""

from __future__ import annotations

from typing import Final

from phase_03_paper.config import DEFAULT_CONFIG, Phase3Config
from phase_03_paper.events.audit import AuditLog
from phase_03_paper.models import (
    Notification,
    NotificationCategory,
    NotificationSeverity,
)
from phase_03_paper.notifications import templates


# ============================================================
# AUDIT CONSTANTS
# ============================================================

_SOURCE_NAME: Final[str] = "NotificationEngine"
_EVENT_NOTIFICATION_GENERATED: Final[str] = "notification_generated"


class NotificationEngine:
    """
    Central in-memory notification engine.

    The engine is deliberately UI-independent.

    Runtime/coordinator code tells this engine that an event occurred.
    The engine builds the Notification object, stores it, and audits
    its creation.

    Persistence is intentionally not handled here. A later persistence
    layer is responsible for saving and restoring notifications.
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
        self._notifications: list[Notification] = []

    # ============================================================
    # PUBLIC — SESSION NOTIFICATIONS
    # ============================================================

    def session_approaching(
        self,
        session_name: str,
        occurrence_id: str | None = None,
    ) -> Notification | None:
        """
        Build the mandatory session-approaching notification.

        Args:
            session_name:
                Human-readable session name, for example "London".

            occurrence_id:
                Unique identifier for this specific session occurrence.
                Runtime code should provide this whenever available.

        Returns:
            The newly created Notification, or None when session
            notifications are disabled or the notification was already
            emitted for this occurrence.
        """

        if not self._config.notifications.session_notifications_enabled:
            return None

        minutes = self._config.sessions.warning_minutes_before

        title, message = templates.session_approaching(
            session_name,
            minutes,
        )

        related_id = occurrence_id or session_name

        if self._already_emitted(
            category=NotificationCategory.SESSION,
            title=title,
            related_id=related_id,
        ):
            return None

        return self._emit(
            category=NotificationCategory.SESSION,
            severity=NotificationSeverity.WARNING,
            title=title,
            message=message,
            related_id=related_id,
        )

    def session_started(
        self,
        session_name: str,
        occurrence_id: str | None = None,
    ) -> Notification | None:
        """Build a session-started notification."""

        if not self._config.notifications.session_notifications_enabled:
            return None

        title, message = templates.session_started(session_name)

        related_id = occurrence_id or session_name

        if self._already_emitted(
            category=NotificationCategory.SESSION,
            title=title,
            related_id=related_id,
        ):
            return None

        return self._emit(
            category=NotificationCategory.SESSION,
            severity=NotificationSeverity.INFO,
            title=title,
            message=message,
            related_id=related_id,
        )

    def session_closed(
        self,
        session_name: str,
        occurrence_id: str | None = None,
    ) -> Notification | None:
        """Build a session-closed notification."""

        if not self._config.notifications.session_notifications_enabled:
            return None

        title, message = templates.session_closed(session_name)

        related_id = occurrence_id or session_name

        if self._already_emitted(
            category=NotificationCategory.SESSION,
            title=title,
            related_id=related_id,
        ):
            return None

        return self._emit(
            category=NotificationCategory.SESSION,
            severity=NotificationSeverity.INFO,
            title=title,
            message=message,
            related_id=related_id,
        )

    def session_skipped(
        self,
        session_name: str,
        occurrence_id: str | None = None,
    ) -> Notification | None:
        """Build a session-skipped notification."""

        if not self._config.notifications.session_notifications_enabled:
            return None

        title, message = templates.session_skipped(session_name)

        related_id = occurrence_id or session_name

        if self._already_emitted(
            category=NotificationCategory.SESSION,
            title=title,
            related_id=related_id,
        ):
            return None

        return self._emit(
            category=NotificationCategory.SESSION,
            severity=NotificationSeverity.INFO,
            title=title,
            message=message,
            related_id=related_id,
        )

    # ============================================================
    # PUBLIC — QUERY
    # ============================================================

    def all_notifications(self) -> list[Notification]:
        """
        Return every notification, oldest first.

        A copy of the internal list is returned so callers cannot
        modify the engine's collection directly.
        """

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
        """
        Mark one notification as read.

        Returns:
            True when the notification exists and was marked read.
            False when no matching notification exists.
        """

        for notification in self._notifications:
            if notification.notification_id == notification_id:
                notification.is_read = True
                return True

        return False

    def mark_all_read(self) -> int:
        """
        Mark every unread notification as read.

        Returns:
            Number of notifications changed from unread to read.
        """

        changed = 0

        for notification in self._notifications:
            if not notification.is_read:
                notification.is_read = True
                changed += 1

        return changed

    # ============================================================
    # PUBLIC — AUDIT ACCESS
    # ============================================================

    @property
    def audit_log(self) -> AuditLog:
        """Return the AuditLog instance used by this engine."""

        return self._audit_log

    # ============================================================
    # PUBLIC — CLEARING
    # ============================================================

    def clear(self) -> None:
        """
        Clear the in-memory notification collection.

        This is primarily useful for tests and controlled resets.

        Duplicate-detection is based on self._notifications (not the
        audit log), so clearing here also correctly re-opens the door
        for the same session_name/occurrence_id to emit again -- the
        audit log itself is left untouched, since it is a permanent
        historical record and clearing it is not this method's job.

        Persistent notification storage is handled later by the
        persistence layer and is intentionally not touched here.
        """

        self._notifications.clear()

    # ============================================================
    # INTERNAL — DUPLICATE PROTECTION
    # ============================================================

    def _already_emitted(
        self,
        category: NotificationCategory,
        title: str,
        related_id: str | None,
    ) -> bool:
        """
        Determine whether an equivalent notification was already emitted.

        Duplicate protection is checked against the engine's own
        in-memory notification list (the same collection clear()
        empties), not the audit log -- the audit log is a permanent
        record and must not be the thing that decides whether a
        notification can be re-emitted after a reset.

        Because the session engine supplies unique occurrence IDs,
        different London days remain independent:

            London / occurrence-A
            London / occurrence-B

        are treated as separate notification targets.
        """

        for notification in self._notifications:
            if (
                notification.category == category
                and notification.title == title
                and notification.related_id == related_id
            ):
                return True

        return False

    # ============================================================
    # INTERNAL — EMISSION
    # ============================================================

    def _emit(
        self,
        category: NotificationCategory,
        severity: NotificationSeverity,
        title: str,
        message: str,
        related_id: str | None,
    ) -> Notification:
        """
        Create, store, and audit one notification.

        Notification generation is recorded immediately after the
        Notification is stored so the audit trail reflects successful
        creation.
        """

        notification = Notification(
            category=category,
            severity=severity,
            title=title,
            message=message,
            related_id=related_id,
        )

        self._notifications.append(notification)

        self._audit_log.record(
            event_type=_EVENT_NOTIFICATION_GENERATED,
            source=_SOURCE_NAME,
            message=f"{category.value}: {title}",
            severity=severity,
            related_id=related_id,
        )

        return notification


__all__ = [
    "NotificationEngine",
]