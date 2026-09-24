"""
phase_04_live/config.py

Centralized Phase 4 settings.

DESIGN PRINCIPLE
-----------------
Phase 4 does NOT import phase_03_paper.config.Phase3Config directly.
That config's SafetySettings hardcodes paper_mode=True and
real_orders_disabled=True, and its own validate_config() raises if
mode != "paper" -- carrying that object into live code would mean
live code holding a config whose own safety section asserts it is
paper-only. Phase 3's own docstring says explicitly: "Phase 4 will
use a separate LiveExecutor implementation."

What IS reused directly (genuinely generic, no Phase-3 coupling):

    phase_03_paper.config.SessionSettings
    phase_03_paper.config.SessionWindow

SessionEngine itself (phase_03_paper/sessions/engine.py) is also
reused directly for Phase 4 -- as a SEPARATE instance constructed
with this Phase4Config's session settings, not a shared instance.
Paper and live trading must be able to reach independent TRADE/SKIP
decisions for the same session occurrence (e.g. paper-trading London
while not yet trusting it live).

This module does NOT:
    - implement session lifecycle logic (SessionEngine does)
    - implement notification delivery (phase_04_live/notifications does)
    - contain any live-execution safety toggle
"""

from __future__ import annotations

from dataclasses import dataclass, field

from phase_03_paper.config import SessionSettings, SessionWindow


@dataclass(frozen=True)
class Phase4SessionTimingSettings:
    """
    The minimal slice of Phase 3's MarketDataSettings that
    SessionEngine actually reads (config.market_data.session_timezone)
    -- kept separate from Phase 3's MarketDataSettings, which also
    carries unrelated fields (symbol, data_source, staleness
    thresholds) that have no place in a session-timing-only config.
    """

    session_timezone: str = "Africa/Nairobi"


# ============================================================
# SESSIONS
# ============================================================

# Phase 4 reuses SessionSettings/SessionWindow as-is (generic,
# no Phase-3 lock-in). Live session windows may eventually diverge
# from paper's -- kept as a separate default here rather than
# importing Phase 3's DEFAULT_CONFIG.sessions, so changing paper's
# defaults never silently changes live's.
def _default_phase4_sessions() -> SessionSettings:
    return SessionSettings(
        sessions=(
            SessionWindow(
                name="London",
                start_local=__import__("datetime").time(10, 0),
                end_local=__import__("datetime").time(19, 0),
            ),
            SessionWindow(
                name="New York",
                start_local=__import__("datetime").time(15, 0),
                end_local=__import__("datetime").time(0, 0),
            ),
        ),
        warning_minutes_before=10,
    )


# ============================================================
# NOTIFICATIONS
# ============================================================


@dataclass(frozen=True)
class Phase4NotificationSettings:
    """
    Minimal Phase 4 notification configuration.

    Deliberately small -- only what phase_04_live/notifications
    currently needs. Extend as new recovery/session notification
    types are added, rather than pre-building unused fields.
    """

    session_notifications_enabled: bool = True
    recovery_notifications_enabled: bool = True


# ============================================================
# COMPLETE PHASE 4 CONFIGURATION
# ============================================================


@dataclass(frozen=True)
class Phase4Config:
    """
    Complete centralized Phase 4 configuration.

    Phase 4 components should receive this rather than independently
    defining constants -- same design principle as Phase3Config, kept
    as a fully separate object so Phase 4 never inherits Phase 3's
    paper-only safety assumptions.

    market_data is a duck-typed stand-in for the slice of
    phase_03_paper.sessions.engine.SessionEngine's config parameter
    it actually reads (config.market_data.session_timezone) --
    SessionEngine's constructor type-hints Phase3Config but does not
    enforce it at runtime, so this object can be passed directly:
        SessionEngine(config=DEFAULT_CONFIG)
    """

    sessions: SessionSettings = field(
        default_factory=_default_phase4_sessions
    )

    market_data: Phase4SessionTimingSettings = field(
        default_factory=Phase4SessionTimingSettings
    )

    notifications: Phase4NotificationSettings = field(
        default_factory=Phase4NotificationSettings
    )


DEFAULT_CONFIG = Phase4Config()


__all__ = [
    "Phase4SessionTimingSettings",
    "Phase4NotificationSettings",
    "Phase4Config",
    "DEFAULT_CONFIG",
]
