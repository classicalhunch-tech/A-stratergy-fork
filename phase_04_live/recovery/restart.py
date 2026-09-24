"""
phase_04_live/recovery/restart.py

Top-level Phase 4 restart entrypoint.

Ties broker_resync.py's position recovery together with the existing
EmergencyKillSwitch: connects, resyncs, and turns PAUSE-level
RecoveryDecisions into an actual kill-switch engagement so trading is
blocked until a human resets it. REQUIRE_MANUAL_REVIEW decisions are
surfaced in the return value but do NOT engage the kill switch --
per project decision, this restart sequence handles broker/position
recovery only; rebuilding StrategyAdapter's in-memory strategy state
via candle replay is a separate, not-yet-built piece (flagged, not
handled here).

Correlated adoption handling
----------------------------
resync_positions() only DETECTS correlations between pending order
attempts and UNEXPECTED_AT_BROKER positions (read-only, no writes --
see broker_resync.py). This function is where the actual persistence
writes for a correlated adoption happen: for every
ResyncResult.correlated_adoptions entry, restart() durably saves the
now-confirmed ExpectedPosition and marks the corresponding order
attempt resolved, then (if a notification engine was supplied) emits
an order_attempt_correlated notification. This keeps broker_resync.py
purely read-only while still letting restart() be the single place
that performs Phase 4 recovery side effects, alongside the existing
kill-switch engagement.

This module does NOT:
    - rebuild StrategyAdapter state
    - place, close, or modify any order/position
    - decide when restart() should be called (that's the process
      entrypoint / live runtime's job)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Tuple

from phase_04_live.broker.connection import BrokerConnection
from phase_04_live.notifications.engine import Phase4NotificationEngine
from phase_04_live.persistence.store import Phase4Persistence
from phase_04_live.recovery.broker_resync import ResyncResult, resync_positions
from phase_04_live.recovery.recovery_state import RecoveryAction, RecoveryDecision
from phase_04_live.risk.kill_switch import EmergencyKillSwitch


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class RestartResult:
    """
    Immutable outcome of one restart sequence.

    resync:
        The full ResyncResult from broker_resync.resync_positions().

    kill_switch_engaged:
        True if THIS restart engaged the kill switch (because at
        least one decision required PAUSE, or the resync snapshot
        was disconnected). Does NOT reflect a kill switch that was
        already engaged before this restart ran -- check
        kill_switch.is_engaged() separately for that.

    review_required:
        RecoveryDecisions with action == REQUIRE_MANUAL_REVIEW,
        surfaced for the caller/monitoring to act on. The kill
        switch is deliberately NOT engaged for these -- pausing all
        trading over every minor mismatch would be overly aggressive;
        a human needs to look at these, not necessarily halt
        everything. Correlated UNEXPECTED_AT_BROKER positions are
        NOT included here -- they were resolved via
        resync.correlated_adoptions instead (action
        ADOPT_EXISTING_STATE, not REQUIRE_MANUAL_REVIEW).

    safe_to_proceed:
        True only when resync succeeded (connected) AND no PAUSE
        decision was produced AND the kill switch is not currently
        engaged (whether from this restart or a prior one). This is
        a convenience summary -- callers should still inspect
        review_required even when safe_to_proceed is True.

    as_of:
        UTC timestamp this restart sequence completed.
    """

    resync: ResyncResult
    kill_switch_engaged: bool
    review_required: Tuple[RecoveryDecision, ...]
    adopted_tickets: Tuple[int, ...]
    safe_to_proceed: bool
    as_of: datetime


def restart(
    connection: BrokerConnection,
    persistence: Phase4Persistence,
    kill_switch: EmergencyKillSwitch,
    notification_engine: Optional[Phase4NotificationEngine] = None,
) -> RestartResult:
    """
    Run one full Phase 4 restart sequence: connect/resync broker
    position state, persist any correlated order-attempt adoptions,
    and engage the kill switch for any PAUSE-level recovery decision
    or a disconnected resync.

    If resync_positions() itself raises (e.g. BrokerConnection.connect()
    fails outright), that exception propagates -- a restart that
    can't even reach the broker is not something this function should
    paper over by returning a "safe" result.

    notification_engine:
        Optional. When supplied, an order_attempt_correlated
        notification is emitted for each correlated adoption. None
        (the default) skips notification -- persistence writes for
        correlated adoptions still happen regardless.
    """
    resync = resync_positions(connection, persistence)

    for adoption in resync.correlated_adoptions:
        persistence.save_expected_position(adoption.expected_position)

        # Attach the pre-send R denominator to the adopted broker
        # position (adoption.ticket is the position ID). Skipped when
        # none was recorded; never guessed.
        attempt_risk = persistence.get_attempt_risk(adoption.identity_key)
        if attempt_risk is not None:
            persistence.record_trade_risk(adoption.ticket, attempt_risk)

        persistence.resolve_order_attempt(adoption.identity_key)

        if notification_engine is not None:
            notification_engine.order_attempt_correlated(
                ticket=adoption.ticket,
                identity_key=adoption.identity_key,
            )

    pause_decisions = tuple(
        d for d in resync.decisions if d.action == RecoveryAction.PAUSE
    )
    review_decisions = tuple(
        d for d in resync.decisions
        if d.action == RecoveryAction.REQUIRE_MANUAL_REVIEW
    )

    kill_switch_engaged_this_restart = False

    if pause_decisions:
        reasons = "; ".join(d.reason for d in pause_decisions)
        kill_switch.engage(
            f"Restart resync produced {len(pause_decisions)} PAUSE "
            f"decision(s): {reasons}"
        )
        kill_switch_engaged_this_restart = True

    if not resync.connected:
        # Broker snapshot itself was untrusted -- even if no individual
        # discrepancy happened to produce a PAUSE decision (e.g. zero
        # expected positions were on file), a disconnected resync is
        # never safe to proceed on.
        kill_switch.engage(
            "Restart resync could not obtain a connected broker "
            "position snapshot -- broker state is unknown."
        )
        kill_switch_engaged_this_restart = True

    safe_to_proceed = (
        resync.connected
        and not pause_decisions
        and not kill_switch.is_engaged()
    )

    adopted_tickets = tuple(
        adoption.ticket for adoption in resync.correlated_adoptions
    )

    return RestartResult(
        resync=resync,
        kill_switch_engaged=kill_switch_engaged_this_restart,
        review_required=review_decisions,
        adopted_tickets=adopted_tickets,
        safe_to_proceed=safe_to_proceed,
        as_of=_utc_now(),
    )


__all__ = ["RestartResult", "restart"]
