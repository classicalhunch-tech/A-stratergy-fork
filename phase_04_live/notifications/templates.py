"""
phase_04_live/notifications/templates.py

Exact notification message templates for Phase 4 recovery events.

Kept separate from notifications/engine.py, mirroring the same split
used by phase_03_paper/notifications/templates.py -- wording changes
never require touching notification logic.
"""

from __future__ import annotations


def recovery_pause(reason: str) -> tuple[str, str]:
    """Return (title, message) when the kill switch is engaged by restart/recovery."""

    title = "\u26a0 TRADING PAUSED"
    message = f"Live signal processing has been paused.\n\n{reason}"

    return title, message


def recovery_review_required(reason: str) -> tuple[str, str]:
    """Return (title, message) for a REQUIRE_MANUAL_REVIEW recovery decision."""

    title = "\u26a0 MANUAL REVIEW REQUIRED"
    message = f"A recovery decision needs review before proceeding.\n\n{reason}"

    return title, message


def recovery_alert(reason: str) -> tuple[str, str]:
    """Return (title, message) for a general recovery ALERT."""

    title = "\U0001f514 RECOVERY ALERT"
    message = reason

    return title, message


def order_attempt_correlated(ticket: int, identity_key: str) -> tuple[str, str]:
    """
    Return (title, message) when an UNEXPECTED_AT_BROKER position is
    correlated against a pending durable order attempt.
    """

    title = "\U0001f517 ORDER ATTEMPT CORRELATED"
    message = (
        f"Broker ticket {ticket} correlates with a pending order "
        f"attempt (identity {identity_key}). The broker's response "
        "to this attempt was likely lost. This is a probable match, "
        "not a confirmed identity -- review before adopting."
    )

    return title, message


__all__ = [
    "recovery_pause",
    "recovery_review_required",
    "recovery_alert",
    "order_attempt_correlated",
]
