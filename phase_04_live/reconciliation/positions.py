"""
phase_04_live/reconciliation/positions.py

Position reconciliation for Phase 4.

Compares what the local system believes should currently exist at
the broker (ExpectedPosition) against what the broker confirms
currently exists (BrokerPositionsSnapshot).

This is independent of execution reconciliation.

Its purpose includes providing broker-side evidence that can help
resolve an UNKNOWN_NO_TICKET execution result. However, this module
does NOT decide which UNKNOWN_NO_TICKET event a broker position
belongs to. That correlation belongs to the recovery layer.

This module does NOT:

    - query MT5
    - place orders
    - close positions
    - modify positions
    - make risk decisions
    - persist results
    - resolve recovery actions

It answers one question:

    "Does the local expectation match what the broker confirms?"
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Sequence, Tuple

from phase_04_live.broker.models import BrokerPositionsSnapshot
from phase_04_live.reconciliation.discrepancies import (
    ExpectedPosition,
    PositionDiscrepancy,
)


def _utc_now() -> datetime:
    """Return the current timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


def reconcile_positions(
    expected: Sequence[ExpectedPosition],
    broker_snapshot: BrokerPositionsSnapshot,
) -> Tuple[PositionDiscrepancy, ...]:
    """
    Compare expected local positions against broker-confirmed state.

    One PositionDiscrepancy is produced for every position involved
    on either side:

        expected only
            -> MISSING_AT_BROKER

        broker only
            -> UNEXPECTED_AT_BROKER

        both sides
            -> field-by-field comparison

        broker snapshot disconnected/untrusted
            -> UNKNOWN for every expected position

    Duplicate expected or broker tickets are rejected because the same broker
    position must never be represented by multiple entries.

    This function is pure reconciliation logic.

    It does not:
        - contact MT5
        - place or modify orders
        - close positions
        - resolve discrepancies
        - resubmit UNKNOWN orders
    """

    now = _utc_now()

    # ================================================================
    # BUILD EXPECTED POSITION INDEX
    # ================================================================

    expected_by_ticket: dict[int, ExpectedPosition] = {}

    for position in expected:
        if position.ticket in expected_by_ticket:
            raise ValueError(
                f"Duplicate expected position ticket: {position.ticket}. "
                "The same broker ticket must not be tracked by multiple "
                "ExpectedPosition records."
            )

        expected_by_ticket[position.ticket] = position

    # ================================================================
    # BROKER SNAPSHOT TRUST CHECK
    # ================================================================

    if not broker_snapshot.connected:
        return tuple(
            PositionDiscrepancy(
                ticket=position.ticket,
                outcome="UNKNOWN",
                expected=position,
                broker=None,
                detail=(
                    "Broker position snapshot is disconnected or "
                    "otherwise untrusted. The absence of broker "
                    f"ticket {position.ticket} cannot be interpreted "
                    "as confirmation that the position is missing."
                ),
                as_of=now,
            )
            for position in expected
        )

    # ================================================================
    # BUILD BROKER POSITION INDEX
    # ================================================================

    broker_by_ticket = {}

    for position in broker_snapshot.positions:
        if position.ticket in broker_by_ticket:
            raise ValueError(
                f"Duplicate broker position ticket: {position.ticket}. "
                "A broker positions snapshot must not contain duplicate tickets."
            )
        broker_by_ticket[position.ticket] = position

    results: list[PositionDiscrepancy] = []

    # ================================================================
    # EXPECTED POSITIONS
    # ================================================================

    for ticket, expected_position in expected_by_ticket.items():

        broker_position = broker_by_ticket.get(ticket)

        # ------------------------------------------------------------
        # Expected position does not exist at broker.
        # ------------------------------------------------------------

        if broker_position is None:
            results.append(
                PositionDiscrepancy(
                    ticket=ticket,
                    outcome="MISSING_AT_BROKER",
                    expected=expected_position,
                    broker=None,
                    detail=(
                        f"Expected position ticket {ticket} "
                        f"({expected_position.symbol} "
                        f"{expected_position.direction} "
                        f"{expected_position.volume}) was not found "
                        "in the broker's current open positions."
                    ),
                    as_of=now,
                )
            )
            continue

        # ------------------------------------------------------------
        # Symbol
        # ------------------------------------------------------------

        if broker_position.symbol != expected_position.symbol:
            results.append(
                PositionDiscrepancy(
                    ticket=ticket,
                    outcome="SYMBOL_MISMATCH",
                    expected=expected_position,
                    broker=broker_position,
                    detail=(
                        f"Ticket {ticket}: expected symbol "
                        f"{expected_position.symbol!r}, broker reports "
                        f"{broker_position.symbol!r}."
                    ),
                    as_of=now,
                )
            )
            continue

        # ------------------------------------------------------------
        # Direction
        # ------------------------------------------------------------

        if broker_position.direction != expected_position.direction:
            results.append(
                PositionDiscrepancy(
                    ticket=ticket,
                    outcome="DIRECTION_MISMATCH",
                    expected=expected_position,
                    broker=broker_position,
                    detail=(
                        f"Ticket {ticket}: expected direction "
                        f"{expected_position.direction!r}, broker reports "
                        f"{broker_position.direction!r}."
                    ),
                    as_of=now,
                )
            )
            continue

        # ------------------------------------------------------------
        # Volume
        # ------------------------------------------------------------

        if broker_position.volume != expected_position.volume:
            results.append(
                PositionDiscrepancy(
                    ticket=ticket,
                    outcome="VOLUME_MISMATCH",
                    expected=expected_position,
                    broker=broker_position,
                    detail=(
                        f"Ticket {ticket}: expected volume "
                        f"{expected_position.volume}, broker reports "
                        f"{broker_position.volume}."
                    ),
                    as_of=now,
                )
            )
            continue

        # ------------------------------------------------------------
        # Stop Loss / Take Profit
        # ------------------------------------------------------------

        if (
            broker_position.stop_loss != expected_position.stop_loss
            or broker_position.take_profit != expected_position.take_profit
        ):
            results.append(
                PositionDiscrepancy(
                    ticket=ticket,
                    outcome="SLTP_MISMATCH",
                    expected=expected_position,
                    broker=broker_position,
                    detail=(
                        f"Ticket {ticket}: expected SL/TP "
                        f"({expected_position.stop_loss}, "
                        f"{expected_position.take_profit}), "
                        f"broker reports "
                        f"({broker_position.stop_loss}, "
                        f"{broker_position.take_profit})."
                    ),
                    as_of=now,
                )
            )
            continue

        # ------------------------------------------------------------
        # Everything matches.
        # ------------------------------------------------------------

        results.append(
            PositionDiscrepancy(
                ticket=ticket,
                outcome="MATCHED",
                expected=expected_position,
                broker=broker_position,
                detail=(
                    f"Ticket {ticket}: broker-confirmed position "
                    "matches local expectation."
                ),
                as_of=now,
            )
        )

    # ================================================================
    # BROKER POSITIONS WITH NO LOCAL EXPECTATION
    # ================================================================

    for ticket, broker_position in broker_by_ticket.items():

        if ticket in expected_by_ticket:
            continue

        results.append(
            PositionDiscrepancy(
                ticket=ticket,
                outcome="UNEXPECTED_AT_BROKER",
                expected=None,
                broker=broker_position,
                detail=(
                    f"Broker reports open position ticket {ticket} "
                    f"({broker_position.symbol} "
                    f"{broker_position.direction} "
                    f"{broker_position.volume}) with no corresponding "
                    "local expectation. This may be related to an "
                    "UNKNOWN_NO_TICKET execution result, but this "
                    "function does not establish that correlation."
                ),
                as_of=now,
            )
        )

    return tuple(results)