"""
phase_04_live/broker/models.py

Shared broker-facing state models for Phase 4's live layer.

Purpose
-------
This module contains immutable data models representing broker state
read by the Phase 4 live layer.

The models currently include:

    AccountState
        Point-in-time broker account snapshot.

    PositionState
        One broker-reported open position.

    BrokerPositionsSnapshot
        Complete set of broker-reported open positions from one read.

    OrderState
        Broker-confirmed state for one order ticket.

AccountState is intentionally SEPARATE from:

    phase_04_live.risk.sizing.AccountInfo

AccountInfo is already tested and consumed by compute_position_size().
It must remain unchanged unless a demonstrated defect requires a
separate change.

This module contains DATA MODELS ONLY.

It does NOT:

    - connect to MT5
    - query broker state
    - calculate position size
    - make risk decisions
    - perform reconciliation
    - calculate daily PnL
    - map MT5 position types to BUY/SELL
    - map MT5 order types to normalized order states
    - read or modify orders
    - persist broker state


Data-flow boundary
------------------

    Broker / MT5
        |
        v
    Broker readers
        |
        +-------------------+
        |                   |
        v                   v
    AccountState      BrokerPositionsSnapshot
                            |
                            v
                      PositionState

    Broker / MT5
        |
        v
    Order reader
        |
        v
    OrderState
        |
        +--> Execution reconciliation
        +--> Recovery
        +--> Monitoring
        +--> Journal / persistence
"""


from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple


@dataclass(frozen=True)
class AccountState:
    """
    Immutable point-in-time snapshot of the live broker account.

    connected:
        True only when the broker/account read succeeded and the
        resulting financial fields are considered usable.

        False means the snapshot is disconnected and must not be
        used as confirmed broker state.

    balance:
        Current account balance.

    equity:
        Current account equity.

    margin:
        Currently used margin.

    margin_free:
        Currently available/free margin.

    margin_level:
        Margin level percentage when available.

        None means the value was unavailable or not applicable.

    leverage:
        Account leverage when available.

    currency:
        Account base currency when available.

    realized_pnl_today:
        Realized PnL for the current trading day when the caller
        has a trustworthy value available.

        This model does NOT determine the trading-day boundary and
        does NOT query trade history.

        None means unavailable, not zero.

    as_of:
        Time at which this snapshot was obtained.

        Must be timezone-aware.
    """

    connected: bool

    balance: Optional[float]
    equity: Optional[float]

    margin: Optional[float]
    margin_free: Optional[float]
    margin_level: Optional[float]

    leverage: Optional[int]
    currency: Optional[str]

    realized_pnl_today: Optional[float]

    as_of: datetime

    def __post_init__(self) -> None:
        """Validate the AccountState invariants."""

        if self.as_of.tzinfo is None:
            raise ValueError(
                "AccountState.as_of must be timezone-aware."
            )

        if self.as_of.utcoffset() is None:
            raise ValueError(
                "AccountState.as_of must have a valid UTC offset."
            )

        if self.connected:
            if self.balance is None:
                raise ValueError(
                    "Connected AccountState requires balance."
                )

            if self.equity is None:
                raise ValueError(
                    "Connected AccountState requires equity."
                )

            return

        financial_fields = (
            self.balance,
            self.equity,
            self.margin,
            self.margin_free,
            self.margin_level,
            self.leverage,
            self.currency,
            self.realized_pnl_today,
        )

        if any(value is not None for value in financial_fields):
            raise ValueError(
                "Disconnected AccountState must not contain "
                "broker-derived financial values."
            )

    @staticmethod
    def disconnected(as_of: datetime) -> "AccountState":
        """
        Construct an explicit disconnected account snapshot.

        All broker-derived financial fields are None.

        This prevents unavailable broker state from being confused
        with legitimate zero values.
        """

        if as_of.tzinfo is None or as_of.utcoffset() is None:
            raise ValueError(
                "AccountState.as_of must be timezone-aware."
            )

        return AccountState(
            connected=False,
            balance=None,
            equity=None,
            margin=None,
            margin_free=None,
            margin_level=None,
            leverage=None,
            currency=None,
            realized_pnl_today=None,
            as_of=as_of,
        )


@dataclass(frozen=True)
class PositionState:
    """
    Immutable representation of one broker-reported open position.

    direction is normalized to the literal strings "BUY" or "SELL".
    The MT5-specific integer mapping belongs in broker/mt5.py, not
    in this data model.

    stop_loss and take_profit are Optional because MT5 reports 0.0
    when the corresponding protective level is not set. The broker
    reader normalizes those values to None.

    ticket is the broker's position identifier and is the primary
    cross-reference key for reconciliation.

    Symbol, direction, and volume must not be assumed to uniquely
    identify a position because multiple positions may exist for the
    same symbol.
    """

    ticket: int
    symbol: str
    direction: str

    volume: float

    price_open: float
    price_current: float

    stop_loss: Optional[float]
    take_profit: Optional[float]

    profit: float
    swap: float

    opened_at: datetime

    magic: int
    comment: str


@dataclass(frozen=True)
class BrokerPositionsSnapshot:
    """
    Immutable snapshot of all currently open broker positions from
    one broker read.

    connected=True:
        The broker position read succeeded.

        An empty positions tuple therefore means the broker
        confirmed that there are currently no open positions.

    connected=False:
        The broker position read could not be trusted.

        positions is empty in this state, but callers MUST interpret
        that as "unknown", not "confirmed zero open positions".

    as_of records when this broker snapshot was obtained.
    """

    connected: bool
    positions: Tuple[PositionState, ...]
    as_of: datetime

    @staticmethod
    def disconnected(
        as_of: datetime,
    ) -> "BrokerPositionsSnapshot":
        """
        Construct an explicit disconnected position snapshot.

        An empty tuple here represents unavailable broker state, not
        a confirmed absence of positions.
        """

        if as_of.tzinfo is None or as_of.utcoffset() is None:
            raise ValueError(
                "BrokerPositionsSnapshot.as_of must be timezone-aware."
            )

        return BrokerPositionsSnapshot(
            connected=False,
            positions=(),
            as_of=as_of,
        )


# ---------------------------------------------------------------------------
# Order state
# ---------------------------------------------------------------------------

VALID_ORDER_STATUSES = frozenset(
    {
        "PENDING",
        "FILLED",
        "PARTIALLY_FILLED",
        "REJECTED",
        "CANCELLED",
        "EXPIRED",
        "UNKNOWN",
    }
)


@dataclass(frozen=True)
class OrderState:
    """
    Broker-confirmed state for ONE order ticket, as of one read.

    This is deliberately keyed by ticket, not by any local
    signal/event id.

    The purpose of this model is to answer:

        "What did the broker actually do with this order?"

    using only broker-confirmed data.
    """

    ticket: int

    status: str

    symbol: Optional[str]
    direction: Optional[str]

    volume_requested: Optional[float]
    volume_filled: Optional[float]

    price: Optional[float]

    comment: Optional[str]

    as_of: datetime

    def __post_init__(self) -> None:
        """Validate the OrderState invariants."""

        if self.status not in VALID_ORDER_STATUSES:
            raise ValueError(
                f"Invalid OrderState.status: {self.status!r}"
            )

        if self.as_of.tzinfo is None:
            raise ValueError(
                "OrderState.as_of must be timezone-aware."
            )

        if self.as_of.utcoffset() is None:
            raise ValueError(
                "OrderState.as_of must have a valid UTC offset."
            )

        if self.direction is not None and self.direction not in {
            "BUY",
            "SELL",
        }:
            raise ValueError(
                f"Invalid OrderState.direction: "
                f"{self.direction!r}"
            )

    @staticmethod
    def unknown(
        ticket: int,
        as_of: datetime,
    ) -> "OrderState":
        """
        Construct an explicit UNKNOWN broker order state.
        """

        if as_of.tzinfo is None or as_of.utcoffset() is None:
            raise ValueError(
                "OrderState.as_of must be timezone-aware."
            )

        return OrderState(
            ticket=ticket,
            status="UNKNOWN",
            symbol=None,
            direction=None,
            volume_requested=None,
            volume_filled=None,
            price=None,
            comment=None,
            as_of=as_of,
        )