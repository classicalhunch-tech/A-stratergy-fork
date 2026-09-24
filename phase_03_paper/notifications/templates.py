"""
phase_03_paper/notifications/templates.py

Exact notification message templates.

Kept separate from notifications/engine.py so wording changes never
require touching notification logic.

Every session notification should use these templates to maintain
consistent wording throughout Phase 3.
"""

from __future__ import annotations


# ============================================================
# PUBLIC — SESSION NOTIFICATION TEMPLATES
# ============================================================

def session_approaching(
    session_name: str,
    minutes: int,
) -> tuple[str, str]:
    """
    Return (title, message) for the mandatory session warning.
    """

    title = "◆ SESSION APPROACHING"

    message = (
        f"{session_name} session begins in approximately "
        f"{minutes} minutes.\n\n"
        "Do you plan to trade this session?"
    )

    return title, message


def session_started(
    session_name: str,
) -> tuple[str, str]:
    """
    Return (title, message) when a session starts.
    """

    title = "◆ SESSION STARTED"
    message = f"{session_name} session is now active."

    return title, message


def session_closed(
    session_name: str,
) -> tuple[str, str]:
    """
    Return (title, message) when a session ends.
    """

    title = "◆ SESSION CLOSED"
    message = f"{session_name} session has ended."

    return title, message


def session_skipped(
    session_name: str,
) -> tuple[str, str]:
    """
    Return (title, message) when a session is skipped.
    """

    title = "◆ SESSION SKIPPED"
    message = f"{session_name} session was disabled for this session."

    return title, message


__all__ = [
    "session_approaching",
    "session_started",
    "session_closed",
    "session_skipped",
]