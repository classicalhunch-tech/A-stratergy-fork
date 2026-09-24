"""
phase_04_live/recovery/broker_resync.py

Startup/reconnect broker resync for Phase 4.

Ties together four already-built, independently-tested pieces:

    BrokerConnection              (broker/connection.py)
    get_broker_positions()        (broker/mt5.py)
    Phase4Persistence              (persistence/store.py)
    reconcile_positions()          (reconciliation/positions.py)
    decide_position_recovery()     (recovery/recovery_state.py)
    correlate_all_pending_attempts()(recovery/correlation.py)

into one function: connect (if needed), load what we durably believe
should exist, compare against what the broker confirms, attempt to
correlate any UNEXPECTED_AT_BROKER positions against pending order
attempts, and decide what should happen about any differences.

This module does NOT:
    - persist anything (correlated adoptions are surfaced via
      ResyncResult.correlated_adoptions for the CALLER -- restart.py
      -- to persist via Phase4Persistence.save_expected_position()
      and Phase4Persistence.resolve_order_attempt(); this module
      performs reads only, same as before)
    - act on any RecoveryDecision (no pausing, no alerting) -- it
      only produces decisions for a caller (restart.py / the live
      runtime) to act on
    - decide WHEN resync should run (startup, post-disconnect, on a
      timer -- all caller decisions)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Tuple

from phase_04_live.broker.connection import BrokerConnection
from phase_04_live.broker.mt5 import get_broker_positions
from phase_04_live.persistence.store import Phase4Persistence
from phase_04_live.reconciliation.discrepancies import PositionDiscrepancy
from phase_04_live.reconciliation.positions import reconcile_positions
from phase_04_live.recovery.correlation import (
    CorrelatedAdoption,
    CorrelationResult,
    correlate_all_pending_attempts,
)
from phase_04_live.recovery.recovery_state import (
    RecoveryDecision,
    decide_position_recovery,
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class ResyncResult:
    """
    Immutable outcome of one broker resync attempt.

    connected:
        Whether the broker snapshot used for this resync was
        actually connected (mirrors BrokerPositionsSnapshot.connected).
        False means discrepancies/decisions reflect an UNKNOWN state
        for every expected position, not a confirmed comparison.

    discrepancies:
        Raw PositionDiscrepancy results from reconcile_positions().

    decisions:
        One RecoveryDecision per discrepancy, same order, produced by
        decide_position_recovery(). Zero-length when there were zero
        expected positions AND zero broker positions (nothing to
        reconcile) -- distinct from a disconnected snapshot, which
        still produces one UNKNOWN discrepancy/decision per expected
        position. Any discrepancy that correlated with a pending
        order attempt (see correlated_adoptions) produces an
        ADOPT_EXISTING_STATE decision here instead of the default
        REQUIRE_MANUAL_REVIEW.

    correlated_adoptions:
        UNEXPECTED_AT_BROKER positions matched to a pending order
        attempt by correlate_all_pending_attempts(). This is a
        READ-ONLY output of this function -- resync_positions() does
        NOT persist these (no writes here, preserving this module's
        read-only guarantee); the caller (restart.py) is responsible
        for calling Phase4Persistence.save_expected_position() and
        Phase4Persistence.resolve_order_attempt() for each one.

    as_of:
        UTC timestamp this resync was performed.
    """

    connected: bool
    discrepancies: Tuple[PositionDiscrepancy, ...]
    decisions: Tuple[RecoveryDecision, ...]
    correlated_adoptions: Tuple[CorrelatedAdoption, ...]
    as_of: datetime


def resync_positions(
    connection: BrokerConnection,
    persistence: Phase4Persistence,
) -> ResyncResult:
    """
    Perform one broker position resync.

    If `connection` is not already connected, this attempts to
    connect it. A failed connection attempt is NOT swallowed --
    BrokerConnection.connect() raises RuntimeError on failure, and
    that propagates to the caller rather than being silently treated
    as "disconnected, proceed anyway". A genuinely disconnected
    snapshot (connect() succeeded locally but MT5 still can't be
    reached) is a different, already-handled case: get_broker_positions()
    returns BrokerPositionsSnapshot.disconnected(...) for that,
    which flows through as connected=False here.

    Correlation step: any discrepancy with outcome
    UNEXPECTED_AT_BROKER is checked against every currently-pending
    order attempt (Phase4Persistence.load_pending_order_attempts())
    via correlate_all_pending_attempts(). A pending-attempt lookup is
    only performed at all when there is at least one
    UNEXPECTED_AT_BROKER discrepancy, to avoid an unnecessary
    persistence read on the common (nothing unexpected) path.
    """
    if not connection.is_connected():
        connection.connect()

    now = _utc_now()

    expected = persistence.load_expected_positions()
    broker_snapshot = get_broker_positions(connection)

    discrepancies = reconcile_positions(expected, broker_snapshot)

    unexpected = tuple(
        d for d in discrepancies if d.outcome == "UNEXPECTED_AT_BROKER"
    )

    correlated_adoptions: Tuple[CorrelatedAdoption, ...] = ()

    if unexpected:
        pending_attempts = persistence.load_pending_order_attempts()
        if pending_attempts:
            correlated_adoptions = correlate_all_pending_attempts(
                pending_attempts=pending_attempts,
                unexpected_positions=unexpected,
            )

    correlation_by_ticket = {
        adoption.ticket: adoption for adoption in correlated_adoptions
    }

    decisions = []
    for discrepancy in discrepancies:
        adoption = correlation_by_ticket.get(discrepancy.ticket)

        if adoption is not None:
            correlation = CorrelationResult(
                matched=discrepancy,
                candidate_count=1,
                detail=adoption.detail,
            )
            decisions.append(
                decide_position_recovery(discrepancy, correlation)
            )
        else:
            decisions.append(decide_position_recovery(discrepancy))

    return ResyncResult(
        connected=broker_snapshot.connected,
        discrepancies=discrepancies,
        decisions=tuple(decisions),
        correlated_adoptions=correlated_adoptions,
        as_of=now,
    )


__all__ = ["ResyncResult", "resync_positions"]
