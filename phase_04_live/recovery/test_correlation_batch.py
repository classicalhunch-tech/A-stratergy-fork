"""
phase_04_live/recovery/test_correlation_batch.py

Tests for correlate_all_pending_attempts() -- pure, no I/O.
"""

import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import MetaTrader5 as mt5

from phase_04_live.orders.order_manager import OrderRequest
from phase_04_live.persistence.store import OrderAttempt
from phase_04_live.reconciliation.discrepancies import PositionDiscrepancy
from phase_04_live.recovery.correlation import correlate_all_pending_attempts


def _order_request(symbol="EURUSD", order_type=mt5.ORDER_TYPE_BUY, volume=0.01):
    return OrderRequest(
        symbol=symbol,
        order_type=order_type,
        volume=volume,
        price=1.0850,
        sl=1.0800,
        tp=1.0900,
        deviation=20,
        magic=404001,
        comment="test",
        filling_mode=mt5.ORDER_FILLING_IOC,
    )


def _attempt(identity_key, request, attempted_at):
    return OrderAttempt(
        identity_key=identity_key,
        request=request,
        attempted_at=attempted_at,
        resolved_at=None,
    )


def _unexpected_discrepancy(ticket, symbol, direction, volume, opened_at):
    broker = SimpleNamespace(
        symbol=symbol,
        direction=direction,
        volume=volume,
        opened_at=opened_at,
        stop_loss=1.0800,
        take_profit=1.0900,
    )
    return PositionDiscrepancy(
        ticket=ticket,
        outcome="UNEXPECTED_AT_BROKER",
        expected=None,
        broker=broker,
        detail="no local expectation",
        as_of=datetime.now(timezone.utc),
    )


class TestCorrelateAllPendingAttempts(unittest.TestCase):
    def test_single_attempt_single_match(self):
        now = datetime.now(timezone.utc)
        attempt = _attempt("id-1", _order_request(), now)
        disc = _unexpected_discrepancy(
            ticket=555, symbol="EURUSD", direction="BUY",
            volume=0.01, opened_at=now,
        )

        adoptions = correlate_all_pending_attempts([attempt], [disc])

        self.assertEqual(len(adoptions), 1)
        self.assertEqual(adoptions[0].ticket, 555)
        self.assertEqual(adoptions[0].identity_key, "id-1")
        self.assertEqual(adoptions[0].expected_position.symbol, "EURUSD")

    def test_no_pending_attempts_produces_no_adoptions(self):
        now = datetime.now(timezone.utc)
        disc = _unexpected_discrepancy(555, "EURUSD", "BUY", 0.01, now)
        self.assertEqual(correlate_all_pending_attempts([], [disc]), ())

    def test_no_unexpected_positions_produces_no_adoptions(self):
        now = datetime.now(timezone.utc)
        attempt = _attempt("id-1", _order_request(), now)
        self.assertEqual(correlate_all_pending_attempts([attempt], []), ())

    def test_a_claimed_discrepancy_is_not_reused_by_a_later_attempt(self):
        now = datetime.now(timezone.utc)
        earlier = _attempt("id-early", _order_request(), now)
        later = _attempt(
            "id-late", _order_request(), now + timedelta(seconds=1)
        )
        disc = _unexpected_discrepancy(555, "EURUSD", "BUY", 0.01, now)

        adoptions = correlate_all_pending_attempts([later, earlier], [disc])

        self.assertEqual(len(adoptions), 1)
        self.assertEqual(adoptions[0].identity_key, "id-early")

    def test_ambiguous_match_produces_no_adoption(self):
        now = datetime.now(timezone.utc)
        attempt = _attempt("id-1", _order_request(), now)
        disc_a = _unexpected_discrepancy(101, "EURUSD", "BUY", 0.01, now)
        disc_b = _unexpected_discrepancy(102, "EURUSD", "BUY", 0.01, now)

        adoptions = correlate_all_pending_attempts([attempt], [disc_a, disc_b])
        self.assertEqual(adoptions, ())

    def test_mismatched_symbol_produces_no_adoption(self):
        now = datetime.now(timezone.utc)
        attempt = _attempt("id-1", _order_request(symbol="EURUSD"), now)
        disc = _unexpected_discrepancy(555, "GBPUSD", "BUY", 0.01, now)
        self.assertEqual(correlate_all_pending_attempts([attempt], [disc]), ())


if __name__ == "__main__":
    unittest.main(verbosity=2)
