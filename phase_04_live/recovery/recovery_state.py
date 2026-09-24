"""
phase_04_live/recovery/recovery_state.py

Pure recovery-decision logic for Phase 4.

Given a reconciliation result (ExecutionReconciliation or
PositionDiscrepancy), decide what SHOULD happen -- never DOES
anything itself. This mirrors reconciliation/orders.py and
reconciliation/positions.py's own style exactly: pure functions, no
I/O, no broker calls, no side effects. Those two modules explicitly
say deciding recovery actions is NOT their job; this is that job.

Core safety rule (carried through from every reconciliation module
read so far): UNKNOWN is never auto-resolved into an assumption.
Where the correct action genuinely depends on evidence this module
does not have (e.g. whether a MISSING_AT_BROKER position closed
normally via SL/TP, or something went wrong), the safe default is
REQUIRE_MANUAL_REVIEW, never a guess in either direction.

Correlation update
------------------
decide_position_recovery() now accepts an OPTIONAL `correlation`
argument (a CorrelationResult from
phase_04_live.recovery.correlation). When the discrepancy's outcome
is UNEXPECTED_AT_BROKER and the caller has already determined this
exact discrepancy correlates to a pending order attempt (via
correlate_unknown_execution() or the batch
correlate_all_pending_attempts()), passing that CorrelationResult in
here changes the decision from REQUIRE_MANUAL_REVIEW to
ADOPT_EXISTING_STATE. Omitting it (the default) preserves the
original fail-safe behavior exactly -- every existing call site that
doesn't pass correlation is unaffected.

This module does NOT:
    - query MT5 or any broker state
    - PERFORM the correlation between an UNKNOWN_NO_TICKET execution
      and an UNEXPECTED_AT_BROKER position itself (that is
      recovery/correlation.py's job; this module only accepts an
      already-computed CorrelationResult as an optional input)
    - persist anything
    - place, close, or modify orders/positions
    - decide WHEN to run reconciliation (that's the live runtime's job)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from phase_04_live.reconciliation.discrepancies import (
    ExecutionReconciliation,
    PositionDiscrepancy,
)
from phase_04_live.recovery.correlation import CorrelationResult


class RecoveryAction(str, Enum):
    NO_ACTION = "NO_ACTION"
    ADOPT_EXISTING_STATE = "ADOPT_EXISTING_STATE"
    PAUSE = "PAUSE"
    ALERT = "ALERT"
    REQUIRE_MANUAL_REVIEW = "REQUIRE_MANUAL_REVIEW"


@dataclass(frozen=True)
class RecoveryDecision:
    """
    Immutable recovery decision for ONE reconciliation result.

    action:
        What should happen. See RecoveryAction.

    should_track_as_expected_position:
        True only when a broker-confirmed position now exists and
        should be durably tracked (via
        Phase4Persistence.save_expected_position()) regardless of
        whether `action` also flags it for review -- e.g. a
        PARTIAL_FILL is both tracked (a real position exists) AND
        flagged REQUIRE_MANUAL_REVIEW (sizing deviated from intent).
        Tracking and flagging are independent, not mutually exclusive.

    source:
        "execution" or "position" -- which reconciliation type this
        decision was derived from.

    ticket:
        Broker ticket this decision concerns, when one exists.

    reason:
        Human-readable explanation for journal/monitoring/alerts.
        Never parsed programmatically.

    as_of:
        UTC timestamp this decision was produced.
    """

    action: RecoveryAction
    should_track_as_expected_position: bool
    source: str
    ticket: Optional[int]
    reason: str
    as_of: datetime


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


# ============================================================
# Execution reconciliation -> recovery decision
# ============================================================

def decide_execution_recovery(
    reconciliation: ExecutionReconciliation,
) -> RecoveryDecision:
    """
    Map one ExecutionReconciliation outcome to a RecoveryDecision.

    Does not correlate UNKNOWN_NO_TICKET against position state --
    that correlation lives in recovery/correlation.py and is applied
    on the POSITION side, via decide_position_recovery()'s optional
    `correlation` argument. This function only decides based on the
    execution outcome alone.
    """
    now = _utc_now()
    outcome = reconciliation.outcome
    ticket = reconciliation.ticket

    if outcome == "NOT_APPLICABLE":
        return RecoveryDecision(
            action=RecoveryAction.NO_ACTION,
            should_track_as_expected_position=False,
            source="execution",
            ticket=ticket,
            reason="Dry run -- no broker order was ever submitted.",
            as_of=now,
        )

    if outcome in ("NEVER_SUBMITTED", "REJECTED_CONFIRMED", "CANCELLED", "EXPIRED"):
        return RecoveryDecision(
            action=RecoveryAction.NO_ACTION,
            should_track_as_expected_position=False,
            source="execution",
            ticket=ticket,
            reason=(
                f"Outcome {outcome}: broker confirms no live exposure "
                "resulted from this order attempt. Nothing to recover."
            ),
            as_of=now,
        )

    if outcome == "PENDING":
        return RecoveryDecision(
            action=RecoveryAction.NO_ACTION,
            should_track_as_expected_position=False,
            source="execution",
            ticket=ticket,
            reason=(
                "Order still PENDING at broker. No terminal outcome "
                "yet -- re-check on the next reconciliation cycle."
            ),
            as_of=now,
        )

    if outcome == "MATCHED":
        return RecoveryDecision(
            action=RecoveryAction.ADOPT_EXISTING_STATE,
            should_track_as_expected_position=True,
            source="execution",
            ticket=ticket,
            reason=(
                "Broker confirms full fill at expected volume. "
                "Adopt as a durably tracked expected position."
            ),
            as_of=now,
        )

    if outcome == "PARTIAL_FILL":
        return RecoveryDecision(
            action=RecoveryAction.REQUIRE_MANUAL_REVIEW,
            should_track_as_expected_position=True,
            source="execution",
            ticket=ticket,
            reason=(
                "Broker confirms a PARTIAL fill. The filled volume is "
                "a real position and must still be tracked, but the "
                "deviation from intended size needs human review "
                "(risk exposure differs from what was planned)."
            ),
            as_of=now,
        )

    if outcome == "MISMATCH":
        return RecoveryDecision(
            action=RecoveryAction.REQUIRE_MANUAL_REVIEW,
            should_track_as_expected_position=True,
            source="execution",
            ticket=ticket,
            reason=(
                "Broker confirms FILLED but the confirmed volume does "
                "not match what was expected. This is unusual and "
                "must not be silently accepted -- track the real "
                "position but require review of the discrepancy."
            ),
            as_of=now,
        )

    if outcome == "UNKNOWN_NO_TICKET":
        return RecoveryDecision(
            action=RecoveryAction.PAUSE,
            should_track_as_expected_position=False,
            source="execution",
            ticket=None,
            reason=(
                "order_send() returned no ticket and no retcode -- "
                "the broker's true response is unknown. Do NOT "
                "resubmit. Pause new signal processing until this is "
                "resolved by correlating against broker position "
                "state (an UNEXPECTED_AT_BROKER position may be the "
                "same trade -- see recovery/correlation.py)."
            ),
            as_of=now,
        )

    # UNKNOWN (explicit) and any unrecognized future outcome both
    # fail closed the same way.
    return RecoveryDecision(
        action=RecoveryAction.PAUSE,
        should_track_as_expected_position=False,
        source="execution",
        ticket=ticket,
        reason=(
            f"Execution outcome {outcome!r} is unresolved/unrecognized. "
            "Broker state cannot be safely interpreted -- pause until "
            "investigated."
        ),
        as_of=now,
    )


# ============================================================
# Position reconciliation -> recovery decision
# ============================================================

def decide_position_recovery(
    discrepancy: PositionDiscrepancy,
    correlation: Optional[CorrelationResult] = None,
) -> RecoveryDecision:
    """
    Map one PositionDiscrepancy outcome to a RecoveryDecision.

    correlation:
        When discrepancy.outcome == "UNEXPECTED_AT_BROKER" and the
        caller has already run correlate_unknown_execution() or
        correlate_all_pending_attempts() and confirmed THIS exact
        discrepancy is the one that correlated (correlation.matched
        is discrepancy), pass that CorrelationResult here to adopt
        the position instead of requiring manual review. None (the
        default) preserves the original fail-safe behavior exactly.
        Ignored for every outcome other than UNEXPECTED_AT_BROKER.
    """
    now = _utc_now()
    outcome = discrepancy.outcome
    ticket = discrepancy.ticket

    if outcome == "MATCHED":
        return RecoveryDecision(
            action=RecoveryAction.NO_ACTION,
            should_track_as_expected_position=False,
            source="position",
            ticket=ticket,
            reason="Broker-confirmed position matches local expectation.",
            as_of=now,
        )

    if outcome == "MISSING_AT_BROKER":
        return RecoveryDecision(
            action=RecoveryAction.REQUIRE_MANUAL_REVIEW,
            should_track_as_expected_position=False,
            source="position",
            ticket=ticket,
            reason=(
                "Expected position not found at broker. This may be a "
                "normal SL/TP close, or may indicate a problem -- "
                "this module has no evidence to distinguish the two, "
                "so it does not assume either. Requires review "
                "(e.g. against broker deal/history) before the "
                "expected-position record is removed."
            ),
            as_of=now,
        )

    if outcome == "UNEXPECTED_AT_BROKER":
        if correlation is not None and correlation.matched is discrepancy:
            return RecoveryDecision(
                action=RecoveryAction.ADOPT_EXISTING_STATE,
                should_track_as_expected_position=True,
                source="position",
                ticket=ticket,
                reason=(
                    "Correlated with a pending order attempt whose "
                    "broker response was lost (UNKNOWN_NO_TICKET): "
                    f"{correlation.detail} Adopting as a durably "
                    "tracked expected position; the pending order "
                    "attempt will be resolved."
                ),
                as_of=now,
            )

        return RecoveryDecision(
            action=RecoveryAction.REQUIRE_MANUAL_REVIEW,
            should_track_as_expected_position=False,
            source="position",
            ticket=ticket,
            reason=(
                "Broker reports an open position with no local "
                "expectation. May correlate with a pending "
                "UNKNOWN_NO_TICKET execution, or may be unrelated "
                "(e.g. manual trade) -- no matching correlation was "
                "provided/found. Requires review before any action."
            ),
            as_of=now,
        )

    if outcome in (
        "SYMBOL_MISMATCH",
        "DIRECTION_MISMATCH",
        "VOLUME_MISMATCH",
        "SLTP_MISMATCH",
    ):
        return RecoveryDecision(
            action=RecoveryAction.REQUIRE_MANUAL_REVIEW,
            should_track_as_expected_position=False,
            source="position",
            ticket=ticket,
            reason=(
                f"{outcome}: broker-confirmed position state diverges "
                "from local expectation. Never silently auto-corrected "
                "-- SL/TP and sizing fields directly affect risk "
                "exposure and require explicit review."
            ),
            as_of=now,
        )

    # UNKNOWN (broker snapshot untrusted/disconnected) and any
    # unrecognized future outcome both fail closed.
    return RecoveryDecision(
        action=RecoveryAction.PAUSE,
        should_track_as_expected_position=False,
        source="position",
        ticket=ticket,
        reason=(
            f"Position outcome {outcome!r}: broker state cannot be "
            "trusted or is unrecognized. Pause until broker "
            "connectivity/state is confirmed reliable again."
        ),
        as_of=now,
    )


__all__ = ["RecoveryAction", "RecoveryDecision", "decide_execution_recovery", "decide_position_recovery"]
