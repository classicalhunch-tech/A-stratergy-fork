"""
phase_04_live/reconciliation/orders.py

Execution reconciliation for Phase 4.

Given an OrderResult produced by:

    phase_04_live.orders.order_manager.place_order_for_signal()

this module determines what the broker actually confirms happened.

The broker is the final authority.

This module deliberately does NOT treat:

    OrderResult.accepted
    OrderResult.reason
    OrderResult.retcode

as the final execution verdict when a broker ticket exists.

Instead:

    OrderResult
          |
          +--> local rejection / dry-run / missing broker response
          |
          +--> broker ticket
                    |
                    v
             get_order_state()
                    |
                    v
          ExecutionReconciliation

This is the ONLY module responsible for turning an OrderResult into
an execution-reconciliation verdict.

Callers such as:

    - live runtime
    - monitoring
    - journal integration
    - recovery

should consume ExecutionReconciliation rather than independently
re-deriving order success/failure from OrderResult fields.

Important safety rule
---------------------
If order_send() returned no ticket and no retcode, this module does
NOT assume failure and does NOT recommend resubmission.

The outcome is:

    UNKNOWN_NO_TICKET

because the broker may have received and processed the request even
though the local process received no usable response.

That situation must later be resolved through broker state and
position reconciliation.

This module does NOT:

    - submit orders
    - resend orders
    - modify orders
    - close positions
    - make risk decisions
    - persist results
    - perform position reconciliation
"""

from __future__ import annotations

from datetime import datetime, timezone

from phase_04_live.broker.connection import BrokerConnection
from phase_04_live.broker.mt5 import get_order_state
from phase_04_live.orders.order_manager import OrderResult
from phase_04_live.reconciliation.discrepancies import (
    ExecutionReconciliation,
)


def _utc_now() -> datetime:
    """Return the current timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


def _expected_volume(order_result: OrderResult) -> float | None:
    """
    Return the volume requested by the order attempt when a request
    was actually constructed.

    A missing request means no broker order request existed.
    """
    if order_result.request is None:
        return None

    return order_result.request.volume


def reconcile_order_execution(
    connection: BrokerConnection,
    order_result: OrderResult,
) -> ExecutionReconciliation:
    """
    Reconcile one OrderResult against broker-confirmed state.

    Decision flow
    ------------

    1. dry_run
        Nothing reached the broker.

    2. no ticket + no request
        The order was rejected locally before request construction.

    3. no ticket + request + explicit retcode
        The broker returned an explicit rejection.

    4. no ticket + request + no retcode
        order_send() returned None. The true broker outcome is unknown.

    5. ticket exists
        Query the broker and reconcile the confirmed broker state.

    This function never resubmits an uncertain order.
    """

    now = _utc_now()
    expected_volume = _expected_volume(order_result)

    # ------------------------------------------------------------------
    # 1. DRY RUN
    # ------------------------------------------------------------------

    if order_result.dry_run:
        return ExecutionReconciliation(
            ticket=None,
            outcome="NOT_APPLICABLE",
            expected_volume=expected_volume,
            broker_volume_filled=None,
            broker_status=None,
            detail=(
                "dry_run=True: the order request was constructed "
                "locally but was never sent to the broker."
            ),
            as_of=now,
        )

    # ------------------------------------------------------------------
    # 2. NO TICKET
    # ------------------------------------------------------------------

    if order_result.ticket is None:

        # --------------------------------------------------------------
        # 2A. No request was constructed.
        #
        # This means the order failed before broker submission.
        # Examples:
        #
        #   - sizing rejection
        #   - compatibility rejection
        #   - invalid tick
        #   - spread rejection
        #   - stale price
        #   - invalid SL/TP
        # --------------------------------------------------------------

        if order_result.request is None:
            return ExecutionReconciliation(
                ticket=None,
                outcome="NEVER_SUBMITTED",
                expected_volume=None,
                broker_volume_filled=None,
                broker_status=None,
                detail=(
                    order_result.reason
                    or (
                        "Order was rejected before an MT5 request "
                        "was constructed; broker was never contacted."
                    )
                ),
                as_of=now,
            )

        # --------------------------------------------------------------
        # 2B. Request existed and MT5 returned an explicit retcode.
        #
        # OrderManager only reaches this state for a retcode other
        # than TRADE_RETCODE_DONE.
        #
        # Therefore the broker explicitly rejected the submission.
        #
        # We do NOT classify this as UNKNOWN because the broker did
        # return a response.
        # --------------------------------------------------------------

        if order_result.retcode is not None:
            return ExecutionReconciliation(
                ticket=None,
                outcome="REJECTED_CONFIRMED",
                expected_volume=expected_volume,
                broker_volume_filled=0.0,
                broker_status=None,
                detail=(
                    order_result.reason
                    or (
                        "Broker returned an explicit non-success "
                        f"retcode={order_result.retcode}; "
                        "order was not confirmed as submitted."
                    )
                ),
                as_of=now,
            )

        # --------------------------------------------------------------
        # 2C. Request existed, but order_send() returned no usable
        # response.
        #
        # This is NOT a rejection.
        #
        # The broker may have received the order even though the local
        # process received neither a ticket nor a retcode.
        #
        # NEVER resend automatically from this branch.
        # --------------------------------------------------------------

        return ExecutionReconciliation(
            ticket=None,
            outcome="UNKNOWN_NO_TICKET",
            expected_volume=expected_volume,
            broker_volume_filled=None,
            broker_status=None,
            detail=(
                order_result.reason
                or (
                    "order_send() returned no ticket and no retcode. "
                    "Broker outcome is unknown; do not resend. "
                    "Resolve through broker state and position "
                    "reconciliation."
                )
            ),
            as_of=now,
        )

    # ------------------------------------------------------------------
    # 3. TICKET EXISTS
    # ------------------------------------------------------------------

    broker_state = get_order_state(
        connection,
        order_result.ticket,
    )

    broker_status = broker_state.status
    broker_volume_filled = broker_state.volume_filled

    # ------------------------------------------------------------------
    # 3A. UNKNOWN
    # ------------------------------------------------------------------

    if broker_status == "UNKNOWN":
        return ExecutionReconciliation(
            ticket=order_result.ticket,
            outcome="UNKNOWN",
            expected_volume=expected_volume,
            broker_volume_filled=broker_volume_filled,
            broker_status=broker_status,
            detail=(
                f"Broker did not provide a confirming pending or "
                f"historical state for ticket {order_result.ticket}. "
                "Execution outcome remains unknown; do not assume "
                "success or failure."
            ),
            as_of=now,
        )

    # ------------------------------------------------------------------
    # 3B. FULL FILL
    # ------------------------------------------------------------------

    if broker_status == "FILLED":

        if (
            expected_volume is not None
            and broker_volume_filled is not None
            and broker_volume_filled == expected_volume
        ):
            return ExecutionReconciliation(
                ticket=order_result.ticket,
                outcome="MATCHED",
                expected_volume=expected_volume,
                broker_volume_filled=broker_volume_filled,
                broker_status=broker_status,
                detail=(
                    "Broker confirms the order as FILLED with "
                    "the expected requested volume."
                ),
                as_of=now,
            )

        return ExecutionReconciliation(
            ticket=order_result.ticket,
            outcome="MISMATCH",
            expected_volume=expected_volume,
            broker_volume_filled=broker_volume_filled,
            broker_status=broker_status,
            detail=(
                "Broker confirms the order as FILLED, but the "
                "confirmed filled volume does not match the "
                f"expected volume: expected={expected_volume}, "
                f"broker={broker_volume_filled}."
            ),
            as_of=now,
        )

    # ------------------------------------------------------------------
    # 3C. PARTIAL FILL
    # ------------------------------------------------------------------

    if broker_status == "PARTIALLY_FILLED":
        return ExecutionReconciliation(
            ticket=order_result.ticket,
            outcome="PARTIAL_FILL",
            expected_volume=expected_volume,
            broker_volume_filled=broker_volume_filled,
            broker_status=broker_status,
            detail=(
                "Broker confirms a partial fill: "
                f"expected={expected_volume}, "
                f"filled={broker_volume_filled}."
            ),
            as_of=now,
        )

    # ------------------------------------------------------------------
    # 3D. REJECTED
    # ------------------------------------------------------------------

    if broker_status == "REJECTED":
        return ExecutionReconciliation(
            ticket=order_result.ticket,
            outcome="REJECTED_CONFIRMED",
            expected_volume=expected_volume,
            broker_volume_filled=broker_volume_filled,
            broker_status=broker_status,
            detail=(
                f"Broker confirms ticket {order_result.ticket} "
                "as REJECTED."
            ),
            as_of=now,
        )

    # ------------------------------------------------------------------
    # 3E. CANCELLED
    # ------------------------------------------------------------------

    if broker_status == "CANCELLED":
        return ExecutionReconciliation(
            ticket=order_result.ticket,
            outcome="CANCELLED",
            expected_volume=expected_volume,
            broker_volume_filled=broker_volume_filled,
            broker_status=broker_status,
            detail=(
                f"Broker confirms ticket {order_result.ticket} "
                "as CANCELLED."
            ),
            as_of=now,
        )

    # ------------------------------------------------------------------
    # 3F. EXPIRED
    # ------------------------------------------------------------------

    if broker_status == "EXPIRED":
        return ExecutionReconciliation(
            ticket=order_result.ticket,
            outcome="EXPIRED",
            expected_volume=expected_volume,
            broker_volume_filled=broker_volume_filled,
            broker_status=broker_status,
            detail=(
                f"Broker confirms ticket {order_result.ticket} "
                "as EXPIRED."
            ),
            as_of=now,
        )

    # ------------------------------------------------------------------
    # 3G. PENDING
    # ------------------------------------------------------------------

    if broker_status == "PENDING":
        return ExecutionReconciliation(
            ticket=order_result.ticket,
            outcome="PENDING",
            expected_volume=expected_volume,
            broker_volume_filled=broker_volume_filled,
            broker_status=broker_status,
            detail=(
                f"Broker still reports ticket {order_result.ticket} "
                "as PENDING. No final execution verdict is assumed."
            ),
            as_of=now,
        )

    # ------------------------------------------------------------------
    # 3H. DEFENSIVE FALLBACK
    #
    # OrderState currently validates its status against the known
    # broker status vocabulary. Keep this fallback anyway so this
    # function fails closed if that contract changes later.
    # ------------------------------------------------------------------

    return ExecutionReconciliation(
        ticket=order_result.ticket,
        outcome="UNKNOWN",
        expected_volume=expected_volume,
        broker_volume_filled=broker_volume_filled,
        broker_status=broker_status,
        detail=(
            f"Unrecognized broker status {broker_status!r} for "
            f"ticket {order_result.ticket}; execution outcome "
            "cannot be safely determined."
        ),
        as_of=now,
    )