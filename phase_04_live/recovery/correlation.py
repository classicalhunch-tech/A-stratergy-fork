"""
phase_04_live/recovery/correlation.py

Correlates an UNKNOWN_NO_TICKET execution outcome against
UNEXPECTED_AT_BROKER position discrepancies, so a broker position
with no local expectation can be recognized as the SAME trade as an
order whose broker response was lost -- rather than remaining two
separate, seemingly-unrelated unknowns.

Explicitly deferred by:
    reconciliation/orders.py    (does not attempt this)
    reconciliation/positions.py (does not attempt this)
    recovery/recovery_state.py  (module docstring flags this as a
                                  separate, not-yet-built piece)

Design principle: multiple candidate matches is NOT a match. A wrong
correlation here means treating an unrelated broker position as a
known trade (or vice versa) -- ambiguity must return no match, never
a best guess.

This module does NOT:
    - query MT5 (only uses MT5 order-type constants, no connection)
    - place, close, or modify anything
    - persist anything
    - decide what to DO with a correlated (or uncorrelated) result
      (the caller decides: adopt the position, keep flagging for
      review, etc.)

Two entry points:

    correlate_unknown_execution()
        Original single-attempt-vs-many-positions correlator.

    correlate_all_pending_attempts()
        Batch wrapper: correlates EVERY pending order attempt against
        a shared pool of UNEXPECTED_AT_BROKER discrepancies, ensuring
        no single broker position is claimed by more than one
        attempt. Added for restart/resync flows where multiple
        pending attempts and multiple unexpected positions may exist
        simultaneously. Still pure -- no I/O, no persistence, no
        broker calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional, Sequence, Tuple

import MetaTrader5 as mt5

from phase_04_live.orders.order_manager import OrderRequest
from phase_04_live.persistence.store import OrderAttempt
from phase_04_live.reconciliation.discrepancies import (
    ExpectedPosition,
    PositionDiscrepancy,
)


def _order_type_to_direction(order_type: int) -> str:
    """
    Normalize OrderRequest.order_type (an MT5 ORDER_TYPE_* constant)
    to the shared "BUY"/"SELL" convention used by PositionState.

    Only BUY/SELL market order types are valid here -- OrderRequest
    is always built from mt5.TRADE_ACTION_DEAL market orders (see
    order_manager._build_order_request()), never pending order types.
    """
    if order_type == mt5.ORDER_TYPE_BUY:
        return "BUY"
    if order_type == mt5.ORDER_TYPE_SELL:
        return "SELL"
    raise ValueError(
        f"OrderRequest.order_type {order_type!r} is not a market "
        "BUY/SELL type -- cannot correlate a non-market order type."
    )


@dataclass(frozen=True)
class CorrelationResult:
    """
    Outcome of attempting to correlate one UNKNOWN_NO_TICKET order
    attempt against a set of UNEXPECTED_AT_BROKER positions.

    matched:
        The single PositionDiscrepancy this order attempt correlates
        to, when exactly one candidate satisfies every criterion.
        None when zero or more than one candidate matched -- both
        cases are treated identically (no safe correlation), never
        as a best-guess pick.

    candidate_count:
        How many UNEXPECTED_AT_BROKER positions satisfied symbol +
        direction + volume criteria before the time-window filter.
        Surfaced for diagnostics/logging even when matched is None.

    detail:
        Human-readable explanation. Never parsed programmatically.
    """
    matched: Optional[PositionDiscrepancy]
    candidate_count: int
    detail: str


def correlate_unknown_execution(
    order_request: OrderRequest,
    attempted_at: datetime,
    unexpected_positions: Sequence[PositionDiscrepancy],
    time_window: timedelta = timedelta(minutes=5),
    volume_tolerance: float = 1e-6,
) -> CorrelationResult:
    """
    Attempt to correlate ONE UNKNOWN_NO_TICKET order attempt against
    a set of PositionDiscrepancy results whose outcome is
    UNEXPECTED_AT_BROKER.

    Match criteria (ALL must hold):
        - same symbol
        - same direction (BUY/SELL)
        - broker-confirmed volume <= requested volume + tolerance,
          and > 0 (a partial fill is still a valid match; an
          over-fill is not, since that would indicate something else
          is wrong)
        - broker position's opened_at falls within `time_window` of
          `attempted_at`

    Callers MUST pass only discrepancies with outcome ==
    "UNEXPECTED_AT_BROKER" -- this function does not filter by
    outcome itself, to keep it simple and testable; passing anything
    else produces meaningless comparisons, since only
    UNEXPECTED_AT_BROKER discrepancies carry a `broker` PositionState
    with no local `expected` counterpart.
    """
    direction = _order_type_to_direction(order_request.order_type)

    same_symbol_direction_volume = [
        d for d in unexpected_positions
        if d.broker is not None
        and d.broker.symbol == order_request.symbol
        and d.broker.direction == direction
        and 0 < d.broker.volume <= order_request.volume + volume_tolerance
    ]

    within_time_window = [
        d for d in same_symbol_direction_volume
        if abs((d.broker.opened_at - attempted_at).total_seconds())
        <= time_window.total_seconds()
    ]

    if len(within_time_window) == 1:
        match = within_time_window[0]
        return CorrelationResult(
            matched=match,
            candidate_count=len(same_symbol_direction_volume),
            detail=(
                f"Matched ticket {match.ticket}: same symbol/direction/"
                f"volume, opened within {time_window} of the order "
                "attempt."
            ),
        )

    if len(within_time_window) == 0:
        return CorrelationResult(
            matched=None,
            candidate_count=len(same_symbol_direction_volume),
            detail=(
                "No UNEXPECTED_AT_BROKER position matched symbol, "
                "direction, volume, AND the time window -- "
                f"{len(same_symbol_direction_volume)} candidate(s) "
                "matched on symbol/direction/volume alone but fell "
                "outside the time window."
            ),
        )

    return CorrelationResult(
        matched=None,
        candidate_count=len(within_time_window),
        detail=(
            f"{len(within_time_window)} UNEXPECTED_AT_BROKER positions "
            "matched symbol, direction, volume, AND time window -- "
            "ambiguous, refusing to guess which one corresponds to "
            "this order attempt. Requires manual review."
        ),
    )


# ============================================================
# Batch correlation across multiple pending attempts
# ============================================================


@dataclass(frozen=True)
class CorrelatedAdoption:
    """
    One UNEXPECTED_AT_BROKER position successfully correlated with a
    pending order attempt whose broker response was lost.

    Pure data produced by correlate_all_pending_attempts() -- carries
    everything a caller needs to ACT (persist the position via
    Phase4Persistence.save_expected_position(), resolve the order
    attempt via Phase4Persistence.resolve_order_attempt(), notify)
    without this module performing any I/O itself.
    """

    ticket: int
    identity_key: str
    expected_position: ExpectedPosition
    detail: str


def correlate_all_pending_attempts(
    pending_attempts: Sequence[OrderAttempt],
    unexpected_positions: Sequence[PositionDiscrepancy],
) -> Tuple[CorrelatedAdoption, ...]:
    """
    Attempt to correlate EVERY pending order attempt against the
    given UNEXPECTED_AT_BROKER discrepancies, earliest attempt first.

    Once a discrepancy is claimed by one attempt, it is removed from
    the candidate pool for every later attempt in this same call --
    a single broker position can be adopted by at most ONE order
    attempt, never claimed twice, even if it would otherwise satisfy
    correlate_unknown_execution()'s criteria for more than one
    attempt.

    Pure function: takes plain sequences, returns plain data.
    Callers are responsible for loading pending_attempts (a
    persistence read, e.g. Phase4Persistence.load_pending_order_attempts())
    and unexpected_positions (the UNEXPECTED_AT_BROKER subset of
    reconcile_positions()'s output), and for acting on the result
    (persistence writes, notifications) -- this function performs
    neither.
    """
    ordered_attempts = sorted(pending_attempts, key=lambda a: a.attempted_at)

    claimed_tickets = set()
    adoptions: List[CorrelatedAdoption] = []

    for attempt in ordered_attempts:
        available = [
            d for d in unexpected_positions if d.ticket not in claimed_tickets
        ]

        if not available:
            break

        result = correlate_unknown_execution(
            order_request=attempt.request,
            attempted_at=attempt.attempted_at,
            unexpected_positions=available,
        )

        if result.matched is None:
            continue

        matched = result.matched
        claimed_tickets.add(matched.ticket)

        broker_state = matched.broker

        adoptions.append(
            CorrelatedAdoption(
                ticket=matched.ticket,
                identity_key=attempt.identity_key,
                expected_position=ExpectedPosition(
                    ticket=matched.ticket,
                    symbol=broker_state.symbol,
                    direction=broker_state.direction,
                    volume=broker_state.volume,
                    stop_loss=broker_state.stop_loss,
                    take_profit=broker_state.take_profit,
                ),
                detail=result.detail,
            )
        )

    return tuple(adoptions)


__all__ = [
    "CorrelationResult",
    "correlate_unknown_execution",
    "CorrelatedAdoption",
    "correlate_all_pending_attempts",
]
