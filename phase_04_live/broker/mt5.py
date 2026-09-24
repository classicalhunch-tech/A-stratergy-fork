"""
phase_04_live/broker/mt5.py

Broker-side MT5 state reads for Phase 4.

This module is the broker-read boundary between raw MetaTrader 5
responses and the shared broker-facing models in models.py.

Current read paths:

    get_account_state()
        Reads the current broker account state.

    get_broker_positions()
        Reads all currently open broker positions.

    get_order_state()
        Reads the broker-confirmed state of one order ticket.

    get_pending_orders()
        Reads all currently pending broker orders.

    resolve_position_id()
        Resolves the broker POSITION ID created by an accepted order,
        using MT5 deal history.

Responsibilities
----------------
This module:

    - reads account information from MT5
    - reads open positions from MT5
    - reads pending orders from MT5
    - reads historical order state from MT5
    - resolves the position ID of an accepted order from MT5 deals
    - converts raw MT5 responses into shared broker models
    - normalizes MT5-specific position direction values
    - normalizes MT5-specific order direction values
    - normalizes unset SL/TP values from 0.0 to None
    - calculates margin_level from equity and margin
    - represents unavailable broker reads safely
    - preserves UNKNOWN when broker state cannot be confirmed

This module does NOT:

    - manage the MT5 connection lifecycle
    - initialize or shut down MT5
    - select symbols
    - place orders
    - modify orders
    - perform reconciliation
    - calculate position size
    - make risk decisions
    - calculate realized daily P&L
    - persist broker state
    - determine execution fill price from deal history


Connection ownership
--------------------
Connection lifecycle is owned by BrokerConnection.

This module receives an existing BrokerConnection and never calls:

    mt5.initialize()
    mt5.shutdown()

The caller is responsible for establishing the connection before
requesting broker state.


Account-state failure contract
------------------------------
get_account_state() returns AccountState.disconnected(...) when:

    - the supplied connection is not locally connected, or
    - mt5.account_info() returns None.

A disconnected AccountState means the broker account state is
unavailable. It does not mean the account has zero values.

No fabricated financial values are produced.


Position-state failure contract
-------------------------------
get_broker_positions() returns
BrokerPositionsSnapshot.disconnected(...) when:

    - the supplied connection is not locally connected, or
    - mt5.positions_get() returns None.

The distinction between an empty tuple and None is important:

    positions_get() == ()
        -> broker successfully reported zero open positions

    positions_get() is None
        -> broker position state could not be read

Therefore an empty connected snapshot means "confirmed flat",
while a disconnected snapshot means "position state unknown".


Order-state failure contract
----------------------------
get_order_state() uses the following lookup order:

    1. Current pending orders
    2. Historical orders
    3. UNKNOWN when neither provides a confirming record

UNKNOWN is never converted into a guessed outcome.

A disconnected connection therefore produces:

    OrderState(status="UNKNOWN")

rather than pretending the order was rejected, cancelled, or filled.


Position-ID resolution contract
-------------------------------
An MT5 ORDER ticket is not guaranteed to equal the POSITION ID that
the order creates. Realized-R lookups are keyed by position ID
(deal.position_id), so the bridge is:

    order ticket -> deal ticket -> deal.position_id

resolve_position_id() returns a position ID only when MT5 deal
history confirms it.

It returns None when it cannot be confirmed, including:

    - disconnected broker connection
    - deal history not yet available
    - no matching opening deal
    - a deal belonging to a different order
    - ambiguous matching deals with different position IDs

None is never converted into a guessed identifier, and the order
ticket is never substituted for the position ID.

An unresolved execution is left to the durable order-attempt /
recovery mechanism.

The function performs a single broker-history read attempt.
Retry policy belongs to the caller.


MT5 normalization boundary
--------------------------
MT5-specific representations are converted here so downstream
components do not need to import MetaTrader5.

Position direction:

    mt5.POSITION_TYPE_BUY
        -> "BUY"

    mt5.POSITION_TYPE_SELL
        -> "SELL"

Order direction is normalized explicitly for all MT5 order types:

    BUY-family order types
        -> "BUY"

    SELL-family order types
        -> "SELL"

An unsupported order type raises rather than being silently
classified.

Unset protective levels:

    MT5 SL/TP == 0.0
        -> None

This prevents an unset protective level from being confused with
a real price of 0.0.


Order execution price boundary
------------------------------
MT5 order history provides the order's requested/open price, but
that is not necessarily the actual execution price of the trade.

Therefore get_order_state() does NOT pretend that:

    order.price_open

is the actual fill price.

For now, OrderState.price is populated only when the broker's order
record itself provides a meaningful order price.

Actual execution/fill price for filled orders must be established
from MT5 deal history in the execution/reconciliation layer.


Relationship with risk/sizing.py
--------------------------------
risk/sizing.py contains its own get_account_info() function because
position sizing has a different failure contract.

That function raises when MT5 cannot provide account information.
This is appropriate immediately before sizing or order preparation,
where silently continuing would be unsafe.

get_account_state() is intended for monitoring, reconciliation,
broker-state observation, and other components that need to observe
an unavailable broker read without terminating their processing loop.

This module does not replace risk/sizing.py's get_account_info().


Account-state boundary
----------------------
realized_pnl_today intentionally remains None here.

A current account snapshot does not by itself establish the
application's trading-day boundary or provide the complete
historical information required for a trustworthy daily realized
P&L value.

That value must be supplied by the appropriate Journal/Performance
layer when available.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

import MetaTrader5 as mt5

from phase_04_live.broker.connection import BrokerConnection
from phase_04_live.broker.models import (
    AccountState,
    BrokerPositionsSnapshot,
    OrderState,
    PositionState,
)


# ---------------------------------------------------------------------------
# MT5 order-state normalization
# ---------------------------------------------------------------------------

_MT5_ORDER_STATE_MAP = {
    mt5.ORDER_STATE_STARTED: "PENDING",
    mt5.ORDER_STATE_PLACED: "PENDING",
    mt5.ORDER_STATE_CANCELED: "CANCELLED",
    mt5.ORDER_STATE_PARTIAL: "PARTIALLY_FILLED",
    mt5.ORDER_STATE_FILLED: "FILLED",
    mt5.ORDER_STATE_REJECTED: "REJECTED",
    mt5.ORDER_STATE_EXPIRED: "EXPIRED",
    mt5.ORDER_STATE_REQUEST_ADD: "UNKNOWN",
    mt5.ORDER_STATE_REQUEST_MODIFY: "UNKNOWN",
    mt5.ORDER_STATE_REQUEST_CANCEL: "UNKNOWN",
}


# ---------------------------------------------------------------------------
# MT5 order-direction normalization
# ---------------------------------------------------------------------------

_MT5_ORDER_DIRECTION_MAP = {
    mt5.ORDER_TYPE_BUY: "BUY",
    mt5.ORDER_TYPE_SELL: "SELL",
    mt5.ORDER_TYPE_BUY_LIMIT: "BUY",
    mt5.ORDER_TYPE_SELL_LIMIT: "SELL",
    mt5.ORDER_TYPE_BUY_STOP: "BUY",
    mt5.ORDER_TYPE_SELL_STOP: "SELL",
    mt5.ORDER_TYPE_BUY_STOP_LIMIT: "BUY",
    mt5.ORDER_TYPE_SELL_STOP_LIMIT: "SELL",
}


def _normalize_order_direction(order_type: int) -> str:
    """
    Convert an MT5 order type into the shared "BUY"/"SELL" convention.

    Raises:
        ValueError:
            If the MT5 order type is unsupported.
    """

    try:
        return _MT5_ORDER_DIRECTION_MAP[order_type]
    except KeyError:
        raise ValueError(
            f"Unsupported MT5 order type: {order_type}"
        ) from None


# ---------------------------------------------------------------------------
# Account state
# ---------------------------------------------------------------------------

def get_account_state(
    connection: BrokerConnection,
) -> AccountState:
    """
    Read and return a fresh AccountState snapshot.

    Returns AccountState.disconnected(...) when the connection is not
    locally connected or MT5 cannot provide account information.

    The returned snapshot uses a timezone-aware UTC timestamp.

    margin_level is calculated as:

        equity / margin * 100

    When margin is zero, margin_level is None rather than zero.
    """

    now = datetime.now(timezone.utc)

    if not connection.is_connected():
        return AccountState.disconnected(now)

    account = mt5.account_info()

    if account is None:
        return AccountState.disconnected(now)

    balance = float(account.balance)
    equity = float(account.equity)
    margin = float(account.margin)
    margin_free = float(account.margin_free)

    margin_level = (
        equity / margin * 100.0
        if margin > 0
        else None
    )

    return AccountState(
        connected=True,
        balance=balance,
        equity=equity,
        margin=margin,
        margin_free=margin_free,
        margin_level=margin_level,
        leverage=int(account.leverage),
        currency=str(account.currency),
        realized_pnl_today=None,
        as_of=now,
    )


# ---------------------------------------------------------------------------
# Broker positions
# ---------------------------------------------------------------------------

def get_broker_positions(
    connection: BrokerConnection,
) -> BrokerPositionsSnapshot:
    """
    Read every currently open broker position across all symbols.

    Returns BrokerPositionsSnapshot.disconnected(...) when the
    connection is not locally connected or MT5 cannot provide
    position information.

    MT5 returns an empty tuple when the broker successfully reports
    zero open positions.

    Therefore:

        positions_get() == ()
            -> connected=True, confirmed zero positions

        positions_get() is None
            -> connected=False, position state unknown

    MT5-specific position direction values are normalized here so
    downstream components only work with "BUY" or "SELL".
    """

    now = datetime.now(timezone.utc)

    if not connection.is_connected():
        return BrokerPositionsSnapshot.disconnected(now)

    raw_positions = mt5.positions_get()

    if raw_positions is None:
        return BrokerPositionsSnapshot.disconnected(now)

    positions = []

    for position in raw_positions:
        if position.type == mt5.POSITION_TYPE_BUY:
            direction = "BUY"
        elif position.type == mt5.POSITION_TYPE_SELL:
            direction = "SELL"
        else:
            raise ValueError(
                f"Unsupported MT5 position type: {position.type}"
            )

        positions.append(
            PositionState(
                ticket=int(position.ticket),
                symbol=str(position.symbol),
                direction=direction,
                volume=float(position.volume),
                price_open=float(position.price_open),
                price_current=float(position.price_current),
                stop_loss=(
                    float(position.sl)
                    if position.sl != 0.0
                    else None
                ),
                take_profit=(
                    float(position.tp)
                    if position.tp != 0.0
                    else None
                ),
                profit=float(position.profit),
                swap=float(position.swap),
                opened_at=datetime.fromtimestamp(
                    position.time,
                    tz=timezone.utc,
                ),
                magic=int(position.magic),
                comment=str(position.comment),
            )
        )

    return BrokerPositionsSnapshot(
        connected=True,
        positions=tuple(positions),
        as_of=now,
    )


# ---------------------------------------------------------------------------
# One-order broker state
# ---------------------------------------------------------------------------

def get_order_state(
    connection: BrokerConnection,
    ticket: int,
) -> OrderState:
    """
    Determine the broker-confirmed state of ONE order ticket.

    Lookup order:

        1. mt5.orders_get(ticket=ticket)
           -> currently pending order

        2. mt5.history_orders_get(ticket=ticket)
           -> historical broker order record

        3. Neither provides a record
           -> OrderState.unknown(...)

    A pending broker order is normalized to PENDING.

    Historical broker states are mapped through
    _MT5_ORDER_STATE_MAP.

    If the broker gives no confirming record, UNKNOWN is returned.

    If the connection is not locally connected, UNKNOWN is returned.

    This function does not attempt to infer actual execution price
    from the order record. Execution price belongs to deal history.
    """

    now = datetime.now(timezone.utc)

    if not connection.is_connected():
        return OrderState.unknown(ticket, now)

    pending = mt5.orders_get(ticket=ticket)

    if pending:
        order = pending[0]

        return OrderState(
            ticket=int(order.ticket),
            status="PENDING",
            symbol=str(order.symbol),
            direction=_normalize_order_direction(order.type),
            volume_requested=float(order.volume_initial),
            volume_filled=float(
                order.volume_initial - order.volume_current
            ),
            price=(
                float(order.price_open)
                if order.price_open
                else None
            ),
            comment=str(order.comment),
            as_of=now,
        )

    historical = mt5.history_orders_get(ticket=ticket)

    if historical:
        order = historical[0]

        status = _MT5_ORDER_STATE_MAP.get(
            order.state,
            "UNKNOWN",
        )

        return OrderState(
            ticket=int(order.ticket),
            status=status,
            symbol=str(order.symbol),
            direction=_normalize_order_direction(order.type),
            volume_requested=float(order.volume_initial),
            volume_filled=float(
                order.volume_initial - order.volume_current
            ),
            price=None,
            comment=str(order.comment),
            as_of=now,
        )

    return OrderState.unknown(ticket, now)


# ---------------------------------------------------------------------------
# All currently pending orders
# ---------------------------------------------------------------------------

def get_pending_orders(
    connection: BrokerConnection,
) -> tuple[OrderState, ...]:
    """
    Read every currently pending order across all symbols.

    These are limit/stop orders currently sitting on the broker's
    order book and waiting to trigger.

    This does NOT cover market orders sent through order_manager.py
    using TRADE_ACTION_DEAL. Those orders should be checked through
    get_order_state(connection, ticket).

    Return contract:

        connected + no pending orders
            -> ()

        connected + pending orders
            -> tuple[OrderState, ...]

        disconnected
            -> ()

    IMPORTANT:

        When disconnected, the empty tuple means "unavailable",
        NOT "confirmed no pending orders".

        Callers must verify connection state before interpreting
        an empty result as confirmed flat pending-order state.
    """

    if not connection.is_connected():
        return ()

    raw_orders = mt5.orders_get()

    if raw_orders is None:
        return ()

    return tuple(
        OrderState(
            ticket=int(order.ticket),
            status="PENDING",
            symbol=str(order.symbol),
            direction=_normalize_order_direction(order.type),
            volume_requested=float(order.volume_initial),
            volume_filled=float(
                order.volume_initial - order.volume_current
            ),
            price=(
                float(order.price_open)
                if order.price_open
                else None
            ),
            comment=str(order.comment),
            as_of=datetime.now(timezone.utc),
        )
        for order in raw_orders
    )


# ---------------------------------------------------------------------------
# Order -> deal -> position ID resolution
# ---------------------------------------------------------------------------

def _is_opening_deal_for_order(
    deal,
    order_ticket: int,
) -> bool:
    """
    Return True only for an opening deal produced by this order.

    The deal must:

        - belong to the supplied order ticket
        - be an ENTRY_IN deal
        - contain a positive broker position ID

    No identifier is inferred when these broker-confirmed
    relationships are absent.
    """

    return (
        int(deal.order) == int(order_ticket)
        and deal.entry == mt5.DEAL_ENTRY_IN
        and int(deal.position_id) > 0
    )


def resolve_position_id(
    connection: BrokerConnection,
    order_ticket: int,
    deal_ticket: Optional[int] = None,
    lookback: timedelta = timedelta(hours=24),
) -> Optional[int]:
    """
    Resolve the broker POSITION ID created by an accepted order.

    Returns the position ID only when MT5 deal history confirms it.

    Returns None when it cannot be confirmed, including:

        - disconnected broker connection
        - requested deal is unavailable
        - requested deal belongs to another order
        - no matching opening deal exists
        - matching deals produce conflicting position IDs

    None is never converted into a guessed identifier.

    The order ticket is never substituted for the position ID.

    Lookup order:

        1. If deal_ticket is supplied, query that exact deal first.
           It must belong to order_ticket and be an ENTRY_IN deal.

        2. Otherwise, query recent deal history and retain only
           opening deals whose deal.order matches order_ticket.

        3. Return the position ID only when all matching opening
           deals agree on exactly one position ID.

    The history interval is causal:

        now - lookback <= deal time <= now

    No future broker-history window is queried.

    This is a single-shot broker lookup. Retry policy belongs to
    the caller.
    """

    if not connection.is_connected():
        return None

    if deal_ticket is not None:
        by_ticket = mt5.history_deals_get(
            ticket=int(deal_ticket)
        )

        if by_ticket:
            deal = by_ticket[0]

            if _is_opening_deal_for_order(
                deal,
                order_ticket,
            ):
                return int(deal.position_id)

            return None

    now = datetime.now(timezone.utc)
    start = now - lookback

    recent = mt5.history_deals_get(
        start,
        now,
    )

    if not recent:
        return None

    position_ids = {
        int(deal.position_id)
        for deal in recent
        if _is_opening_deal_for_order(
            deal,
            order_ticket,
        )
    }

    if len(position_ids) == 1:
        return next(iter(position_ids))

    return None