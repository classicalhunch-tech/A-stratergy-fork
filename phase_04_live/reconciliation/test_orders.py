"""
phase_04_live/reconciliation/test_orders.py

Tests for reconcile_order_execution().

These tests use mocks for get_order_state() and a stand-in broker
connection. No real MT5 orders are placed and no real broker state
is queried.

The test suite covers every currently defined execution-reconciliation
branch and verifies the important safety invariants around unknown
execution state.
"""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch
import unittest

import MetaTrader5 as mt5

from phase_04_live.orders.order_manager import (
    OrderRequest,
    OrderResult,
)
from phase_04_live.reconciliation.orders import (
    reconcile_order_execution,
)


# ---------------------------------------------------------------------------
# Test fixtures helpers
# ---------------------------------------------------------------------------

FAKE_CONNECTION = object()


def _make_request(volume: float = 0.01) -> OrderRequest:
    """
    Build a minimal realistic OrderRequest for test fixtures.

    The request does not need to be submitted to MT5. It only needs
    to satisfy the actual OrderRequest contract used by OrderResult.
    """
    return OrderRequest(
        symbol="XAUUSD",
        order_type=mt5.ORDER_TYPE_BUY,
        volume=volume,
        price=4100.0,
        sl=4090.0,
        tp=4120.0,
        deviation=20,
        magic=404_001,
        comment="test",
        filling_mode=mt5.ORDER_FILLING_FOK,
    )


def _make_broker_state(
    status: str,
    volume_filled: float | None,
):
    """
    Build a minimal fake broker OrderState.

    reconcile_order_execution() currently consumes only:

        .status
        .volume_filled

    Therefore SimpleNamespace is sufficient here and avoids coupling
    these tests to unrelated OrderState constructor details.
    """
    return SimpleNamespace(
        status=status,
        volume_filled=volume_filled,
    )


class TestOrderReconciliation(unittest.TestCase):

    # ===========================================================================
    # 1. DRY RUN
    # ===========================================================================

    def test_dry_run_returns_not_applicable(self):
        """
        A dry-run request was never sent to the broker.

        Therefore there is no broker execution to reconcile.
        """
        order_result = OrderResult(
            accepted=True,
            dry_run=True,
            request=_make_request(volume=0.01),
            retcode=None,
            ticket=None,
            reason=None,
        )

        result = reconcile_order_execution(
            FAKE_CONNECTION,
            order_result,
        )

        self.assertEqual(result.outcome, "NOT_APPLICABLE")
        self.assertIsNone(result.ticket)
        self.assertIsNone(result.broker_status)
        self.assertEqual(result.expected_volume, 0.01)
        self.assertIsNone(result.broker_volume_filled)

    # ===========================================================================
    # 2. NEVER_SUBMITTED
    # ===========================================================================

    def test_local_rejection_before_request_returns_never_submitted(self):
        """
        request=None + ticket=None means the order failed locally before
        an MT5 request was constructed.

        The broker was never contacted.
        """
        order_result = OrderResult(
            accepted=False,
            dry_run=False,
            request=None,
            retcode=None,
            ticket=None,
            reason="spread too wide",
        )

        result = reconcile_order_execution(
            FAKE_CONNECTION,
            order_result,
        )

        self.assertEqual(result.outcome, "NEVER_SUBMITTED")
        self.assertIsNone(result.ticket)
        self.assertIsNone(result.expected_volume)
        self.assertIsNone(result.broker_volume_filled)
        self.assertIsNone(result.broker_status)
        self.assertEqual(result.detail, "spread too wide")

    # ===========================================================================
    # 3. REJECTED_CONFIRMED
    # ===========================================================================

    def test_explicit_broker_rejection_returns_rejected_confirmed(self):
        """
        A request existed and MT5 returned an explicit non-success
        retcode without a ticket.

        The broker responded, so this is a confirmed rejection rather
        than UNKNOWN_NO_TICKET.
        """
        order_result = OrderResult(
            accepted=False,
            dry_run=False,
            request=_make_request(volume=0.02),
            retcode=mt5.TRADE_RETCODE_REQUOTE,
            ticket=None,
            reason=None,
        )

        result = reconcile_order_execution(
            FAKE_CONNECTION,
            order_result,
        )

        self.assertEqual(result.outcome, "REJECTED_CONFIRMED")
        self.assertIsNone(result.ticket)
        self.assertEqual(result.expected_volume, 0.02)
        self.assertEqual(result.broker_volume_filled, 0.0)
        self.assertIsNone(result.broker_status)
        self.assertIn(str(mt5.TRADE_RETCODE_REQUOTE), result.detail)

    # ===========================================================================
    # 4. UNKNOWN_NO_TICKET
    # ===========================================================================

    def test_order_send_returning_none_returns_unknown_no_ticket(self):
        """
        request exists + no ticket + no retcode means order_send()
        returned no usable broker response.

        The true broker outcome is unknown.

        The system must not assume rejection or success.
        """
        order_result = OrderResult(
            accepted=False,
            dry_run=False,
            request=_make_request(volume=0.03),
            retcode=None,
            ticket=None,
            reason=None,
        )

        result = reconcile_order_execution(
            FAKE_CONNECTION,
            order_result,
        )

        self.assertEqual(result.outcome, "UNKNOWN_NO_TICKET")
        self.assertIsNone(result.ticket)
        self.assertEqual(result.expected_volume, 0.03)
        self.assertIsNone(result.broker_volume_filled)
        self.assertIsNone(result.broker_status)
        self.assertIn("do not resend", result.detail.lower())

    # ===========================================================================
    # 5. MATCHED
    # ===========================================================================

    def test_broker_filled_matching_volume_returns_matched(self):
        """
        Broker confirms FILLED and the filled volume exactly matches
        the requested volume.
        """
        order_result = OrderResult(
            accepted=True,
            dry_run=False,
            request=_make_request(volume=0.05),
            retcode=mt5.TRADE_RETCODE_DONE,
            ticket=555111,
            reason=None,
        )

        with patch(
            "phase_04_live.reconciliation.orders.get_order_state",
            return_value=_make_broker_state("FILLED", 0.05),
        ) as mock_get_order_state:

            result = reconcile_order_execution(
                FAKE_CONNECTION,
                order_result,
            )

        mock_get_order_state.assert_called_once_with(
            FAKE_CONNECTION,
            555111,
        )

        self.assertEqual(result.outcome, "MATCHED")
        self.assertEqual(result.ticket, 555111)
        self.assertEqual(result.broker_status, "FILLED")
        self.assertEqual(result.expected_volume, 0.05)
        self.assertEqual(result.broker_volume_filled, 0.05)

    # ===========================================================================
    # 6. MISMATCH
    # ===========================================================================

    def test_broker_filled_mismatched_volume_returns_mismatch(self):
        """
        Broker confirms FILLED but the confirmed filled volume differs
        from the requested volume.
        """
        order_result = OrderResult(
            accepted=True,
            dry_run=False,
            request=_make_request(volume=0.05),
            retcode=mt5.TRADE_RETCODE_DONE,
            ticket=555112,
            reason=None,
        )

        with patch(
            "phase_04_live.reconciliation.orders.get_order_state",
            return_value=_make_broker_state("FILLED", 0.03),
        ):
            result = reconcile_order_execution(
                FAKE_CONNECTION,
                order_result,
            )

        self.assertEqual(result.outcome, "MISMATCH")
        self.assertEqual(result.ticket, 555112)
        self.assertEqual(result.broker_status, "FILLED")
        self.assertEqual(result.expected_volume, 0.05)
        self.assertEqual(result.broker_volume_filled, 0.03)

    # ===========================================================================
    # 7. PARTIAL_FILL
    # ===========================================================================

    def test_broker_partially_filled_returns_partial_fill(self):
        """
        Broker confirms that only part of the requested volume was filled.
        """
        order_result = OrderResult(
            accepted=True,
            dry_run=False,
            request=_make_request(volume=0.10),
            retcode=mt5.TRADE_RETCODE_DONE,
            ticket=555113,
            reason=None,
        )

        with patch(
            "phase_04_live.reconciliation.orders.get_order_state",
            return_value=_make_broker_state(
                "PARTIALLY_FILLED",
                0.04,
            ),
        ):
            result = reconcile_order_execution(
                FAKE_CONNECTION,
                order_result,
            )

        self.assertEqual(result.outcome, "PARTIAL_FILL")
        self.assertEqual(result.ticket, 555113)
        self.assertEqual(result.broker_status, "PARTIALLY_FILLED")
        self.assertEqual(result.expected_volume, 0.10)
        self.assertEqual(result.broker_volume_filled, 0.04)

    # ===========================================================================
    # 8. BROKER REJECTED
    # ===========================================================================

    def test_broker_rejected_returns_rejected_confirmed(self):
        """
        A ticket exists, but broker history confirms the order as REJECTED.
        """
        order_result = OrderResult(
            accepted=True,
            dry_run=False,
            request=_make_request(volume=0.02),
            retcode=mt5.TRADE_RETCODE_DONE,
            ticket=555114,
            reason=None,
        )

        with patch(
            "phase_04_live.reconciliation.orders.get_order_state",
            return_value=_make_broker_state(
                "REJECTED",
                None,
            ),
        ):
            result = reconcile_order_execution(
                FAKE_CONNECTION,
                order_result,
            )

        self.assertEqual(result.outcome, "REJECTED_CONFIRMED")
        self.assertEqual(result.ticket, 555114)
        self.assertEqual(result.broker_status, "REJECTED")

    # ===========================================================================
    # 9. CANCELLED
    # ===========================================================================

    def test_broker_cancelled_returns_cancelled(self):
        """
        Broker confirms the order as CANCELLED.
        """
        order_result = OrderResult(
            accepted=True,
            dry_run=False,
            request=_make_request(volume=0.02),
            retcode=mt5.TRADE_RETCODE_DONE,
            ticket=555115,
            reason=None,
        )

        with patch(
            "phase_04_live.reconciliation.orders.get_order_state",
            return_value=_make_broker_state(
                "CANCELLED",
                None,
            ),
        ):
            result = reconcile_order_execution(
                FAKE_CONNECTION,
                order_result,
            )

        self.assertEqual(result.outcome, "CANCELLED")
        self.assertEqual(result.ticket, 555115)
        self.assertEqual(result.broker_status, "CANCELLED")

    # ===========================================================================
    # 10. EXPIRED
    # ===========================================================================

    def test_broker_expired_returns_expired(self):
        """
        Broker confirms the order as EXPIRED.
        """
        order_result = OrderResult(
            accepted=True,
            dry_run=False,
            request=_make_request(volume=0.02),
            retcode=mt5.TRADE_RETCODE_DONE,
            ticket=555116,
            reason=None,
        )

        with patch(
            "phase_04_live.reconciliation.orders.get_order_state",
            return_value=_make_broker_state(
                "EXPIRED",
                None,
            ),
        ):
            result = reconcile_order_execution(
                FAKE_CONNECTION,
                order_result,
            )

        self.assertEqual(result.outcome, "EXPIRED")
        self.assertEqual(result.ticket, 555116)
        self.assertEqual(result.broker_status, "EXPIRED")

    # ===========================================================================
    # 11. PENDING
    # ===========================================================================

    def test_broker_pending_returns_pending(self):
        """
        Broker still reports the order as PENDING.

        No final execution verdict is guessed.
        """
        order_result = OrderResult(
            accepted=True,
            dry_run=False,
            request=_make_request(volume=0.02),
            retcode=mt5.TRADE_RETCODE_DONE,
            ticket=555117,
            reason=None,
        )

        with patch(
            "phase_04_live.reconciliation.orders.get_order_state",
            return_value=_make_broker_state(
                "PENDING",
                None,
            ),
        ):
            result = reconcile_order_execution(
                FAKE_CONNECTION,
                order_result,
            )

        self.assertEqual(result.outcome, "PENDING")
        self.assertEqual(result.ticket, 555117)
        self.assertEqual(result.broker_status, "PENDING")
        self.assertEqual(result.expected_volume, 0.02)

    # ===========================================================================
    # 12. UNKNOWN
    # ===========================================================================

    def test_broker_unknown_returns_unknown(self):
        """
        A broker ticket exists, but the broker reader cannot currently
        confirm a pending or historical state.

        The system must not assume success or failure.
        """
        order_result = OrderResult(
            accepted=True,
            dry_run=False,
            request=_make_request(volume=0.02),
            retcode=mt5.TRADE_RETCODE_DONE,
            ticket=555118,
            reason=None,
        )

        with patch(
            "phase_04_live.reconciliation.orders.get_order_state",
            return_value=_make_broker_state(
                "UNKNOWN",
                None,
            ),
        ):
            result = reconcile_order_execution(
                FAKE_CONNECTION,
                order_result,
            )

        self.assertEqual(result.outcome, "UNKNOWN")
        self.assertEqual(result.ticket, 555118)
        self.assertEqual(result.broker_status, "UNKNOWN")
        self.assertIsNone(result.broker_volume_filled)
        self.assertIn("do not assume", result.detail.lower())

    # ===========================================================================
    # CROSS-CUTTING INVARIANTS
    # ===========================================================================

    def test_as_of_is_timezone_aware_utc(self):
        """
        Every reconciliation result must contain a timezone-aware UTC
        timestamp.
        """
        order_result = OrderResult(
            accepted=True,
            dry_run=True,
            request=_make_request(),
            retcode=None,
            ticket=None,
            reason=None,
        )

        result = reconcile_order_execution(
            FAKE_CONNECTION,
            order_result,
        )

        self.assertIsInstance(result.as_of, datetime)
        self.assertIsNotNone(result.as_of.tzinfo)
        self.assertEqual(
            result.as_of.utcoffset(),
            timezone.utc.utcoffset(None),
        )

    def test_unknown_no_ticket_remains_unconfirmed(self):
        """
        UNKNOWN_NO_TICKET must remain explicitly unconfirmed.

        This branch must never fabricate a broker ticket or broker status.
        """
        order_result = OrderResult(
            accepted=False,
            dry_run=False,
            request=_make_request(),
            retcode=None,
            ticket=None,
            reason=None,
        )

        result = reconcile_order_execution(
            FAKE_CONNECTION,
            order_result,
        )

        self.assertEqual(result.outcome, "UNKNOWN_NO_TICKET")
        self.assertIsNone(result.ticket)
        self.assertIsNone(result.broker_status)
        self.assertIsNone(result.broker_volume_filled)

    def test_ticket_path_queries_broker_with_exact_connection_and_ticket(self):
        """
        When an OrderResult contains a ticket, reconciliation must query
        the broker using the exact connection and ticket supplied.

        This verifies the core boundary:

            OrderResult
                ->
            get_order_state(connection, ticket)
                ->
            broker-confirmed reconciliation
        """
        order_result = OrderResult(
            accepted=True,
            dry_run=False,
            request=_make_request(volume=0.05),
            retcode=mt5.TRADE_RETCODE_DONE,
            ticket=555119,
            reason=None,
        )

        with patch(
            "phase_04_live.reconciliation.orders.get_order_state",
            return_value=_make_broker_state(
                "FILLED",
                0.05,
            ),
        ) as mock_get_order_state:

            result = reconcile_order_execution(
                FAKE_CONNECTION,
                order_result,
            )

        mock_get_order_state.assert_called_once_with(
            FAKE_CONNECTION,
            555119,
        )

        self.assertEqual(result.outcome, "MATCHED")
        self.assertEqual(result.ticket, 555119)


if __name__ == "__main__":
    unittest.main()