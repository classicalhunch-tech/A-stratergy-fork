"""
phase_04_live/reconciliation/discrepancies.py

Shared reconciliation result types for Phase 4.

This module contains immutable data models used by both:

    reconciliation/orders.py
    reconciliation/positions.py

The purpose is to provide one consistent vocabulary for answering:

    "Does the local system's expectation match what the broker confirms?"

This module contains DATA MODELS ONLY.

It does NOT:

    - query MT5
    - place orders
    - modify orders
    - reconcile orders
    - reconcile positions
    - make risk decisions
    - persist reconciliation results
"""


from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from phase_04_live.broker.models import PositionState


VALID_RECONCILIATION_OUTCOMES = frozenset({
    "MATCHED",
    "PARTIAL_FILL",
    "MISMATCH",
    "REJECTED_CONFIRMED",
    "CANCELLED",
    "EXPIRED",
    "PENDING",
    "NEVER_SUBMITTED",
    "UNKNOWN_NO_TICKET",
    "UNKNOWN",
    "NOT_APPLICABLE",
})


VALID_BROKER_ORDER_STATUSES = frozenset({
    "PENDING",
    "FILLED",
    "PARTIALLY_FILLED",
    "REJECTED",
    "CANCELLED",
    "EXPIRED",
    "UNKNOWN",
})


@dataclass(frozen=True)
class ExecutionReconciliation:
    """
    Immutable result of comparing the system's expected order outcome
    against the broker-confirmed order state.

    ticket:
        Broker order ticket when one exists.

        None is valid for outcomes where no usable broker ticket
        exists, such as:

            NEVER_SUBMITTED
            UNKNOWN_NO_TICKET

    outcome:
        Normalized reconciliation result.

        Must be one of VALID_RECONCILIATION_OUTCOMES.

    expected_volume:
        Volume the system attempted to submit.

        None is valid when the order never reached request
        construction.

    broker_volume_filled:
        Volume the broker confirms was actually filled.

        None when no broker-confirmed fill volume exists.

    broker_status:
        Original normalized broker OrderState.status.

        None when no broker state exists, for example:
            NEVER_SUBMITTED
            UNKNOWN_NO_TICKET

    detail:
        Human-readable explanation for journal, logging, monitoring,
        and alerts.

        This field must never be parsed programmatically. Callers
        should use outcome for machine decisions.

    as_of:
        UTC timestamp representing when this reconciliation result
        was produced.
    """

    ticket: Optional[int]

    outcome: str

    expected_volume: Optional[float]
    broker_volume_filled: Optional[float]

    broker_status: Optional[str]

    detail: str

    as_of: datetime

    def __post_init__(self) -> None:
        """Validate the structural invariants of the result."""

        if self.outcome not in VALID_RECONCILIATION_OUTCOMES:
            raise ValueError(
                f"Invalid reconciliation outcome: {self.outcome!r}"
            )

        if (
            self.broker_status is not None
            and self.broker_status not in VALID_BROKER_ORDER_STATUSES
        ):
            raise ValueError(
                f"Invalid broker order status: "
                f"{self.broker_status!r}"
            )

        if self.as_of.tzinfo is None:
            raise ValueError(
                "ExecutionReconciliation.as_of must be "
                "timezone-aware."
            )

        if self.as_of.utcoffset() is None:
            raise ValueError(
                "ExecutionReconciliation.as_of must have "
                "a valid UTC offset."
            )

        if not isinstance(self.detail, str) or not self.detail.strip():
            raise ValueError(
                "ExecutionReconciliation.detail must be a non-empty string."
            )


# ============================================================
# POSITION RECONCILIATION MODELS
# ============================================================
#
# Added for reconciliation/positions.py. Phase 3's PositionManager
# classifies PaperTrade lifecycle state (OPEN/MANAGED/CLOSED) and is
# NOT reused here -- PaperTrade has no broker ticket or volume field,
# since it never talks to a real broker. ExpectedPosition is the
# Phase 4 equivalent: what we believe should exist at the broker,
# built from a MATCHED or PARTIAL_FILL ExecutionReconciliation plus
# the original OrderRequest that produced it.

VALID_POSITION_OUTCOMES = frozenset({
    "MATCHED",
    "VOLUME_MISMATCH",
    "DIRECTION_MISMATCH",
    "SYMBOL_MISMATCH",
    "SLTP_MISMATCH",
    "MISSING_AT_BROKER",
    "UNEXPECTED_AT_BROKER",
    "UNKNOWN",
})


@dataclass(frozen=True)
class ExpectedPosition:
    """
    One position the local system believes should currently exist
    at the broker.

    Built by the caller from a successfully-reconciled
    ExecutionReconciliation (outcome MATCHED or PARTIAL_FILL, which
    are the only outcomes that carry a real broker ticket alongside
    a confirmed fill) plus the OrderRequest that produced it.

    volume is the CONFIRMED broker-filled volume (not the originally
    requested volume) -- for a MATCHED order these are equal; for a
    PARTIAL_FILL, using the filled volume here is what makes
    VOLUME_MISMATCH detection meaningful rather than trivially always
    true.

    stop_loss / take_profit follow PositionState's convention: None
    means no protective level is set, not 0.0.
    """

    ticket: int
    symbol: str
    direction: str  # "BUY" or "SELL", matching PositionState.direction

    volume: float

    stop_loss: Optional[float]
    take_profit: Optional[float]

    def __post_init__(self) -> None:
        if not isinstance(self.ticket, int) or self.ticket <= 0:
            raise ValueError(
                f"ExpectedPosition.ticket must be a positive integer, got {self.ticket!r}"
            )

        if not isinstance(self.symbol, str) or not self.symbol.strip():
            raise ValueError(
                f"ExpectedPosition.symbol must be a non-empty string, got {self.symbol!r}"
            )

        if self.direction not in {"BUY", "SELL"}:
            raise ValueError(
                f"Invalid ExpectedPosition.direction: {self.direction!r}"
            )

        if not isinstance(self.volume, (int, float)) or self.volume <= 0.0:
            raise ValueError(
                f"ExpectedPosition.volume must be greater than zero, got {self.volume!r}"
            )

        if self.stop_loss is not None and (not isinstance(self.stop_loss, (int, float)) or self.stop_loss < 0.0):
            raise ValueError(
                f"ExpectedPosition.stop_loss must be non-negative or None, got {self.stop_loss!r}"
            )

        if self.take_profit is not None and (not isinstance(self.take_profit, (int, float)) or self.take_profit < 0.0):
            raise ValueError(
                f"ExpectedPosition.take_profit must be non-negative or None, got {self.take_profit!r}"
            )


@dataclass(frozen=True)
class PositionDiscrepancy:
    """
    Immutable result of comparing one ExpectedPosition against
    broker-confirmed position state, OR of a broker position that
    has no corresponding local expectation at all.

    ticket:
        The position ticket this result concerns. Always present --
        every discrepancy is keyed by a real ticket, either from the
        expected side, the broker side, or both.

    outcome:
        Must be one of VALID_POSITION_OUTCOMES.

        UNKNOWN is used when the broker snapshot itself could not be
        trusted (BrokerPositionsSnapshot.connected is False), NOT
        when a comparison was simply not attempted. In that case
        every expected position produces an UNKNOWN discrepancy,
        because a disconnected snapshot proves nothing about what
        currently exists at the broker -- treating it as
        MISSING_AT_BROKER would be an unsafe guess.

    expected:
        The local ExpectedPosition, when one exists for this ticket.

    broker:
        The broker-confirmed PositionState, when one exists for this
        ticket.

    detail:
        Human-readable explanation only. Never parsed programmatically.

    as_of:
        UTC timestamp when this comparison was produced.
    """

    ticket: int
    outcome: str

    expected: Optional["ExpectedPosition"]
    broker: Optional["PositionState"]

    detail: str

    as_of: datetime

    def __post_init__(self) -> None:
        if self.outcome not in VALID_POSITION_OUTCOMES:
            raise ValueError(
                f"Invalid position reconciliation outcome: {self.outcome!r}"
            )

        if self.as_of.tzinfo is None or self.as_of.utcoffset() is None:
            raise ValueError(
                "PositionDiscrepancy.as_of must be timezone-aware."
            )

        if not isinstance(self.detail, str) or not self.detail.strip():
            raise ValueError(
                "PositionDiscrepancy.detail must be a non-empty string."
            )