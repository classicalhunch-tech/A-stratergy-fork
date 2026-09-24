"""
phase_04_live/reconciliation/test_positions.py

Tests for reconcile_positions().

No real MT5 calls -- BrokerPositionsSnapshot and PositionState are
constructed directly as test fixtures.
"""

from datetime import datetime, timezone
import unittest

from phase_04_live.broker.models import BrokerPositionsSnapshot, PositionState
from phase_04_live.reconciliation.discrepancies import ExpectedPosition
from phase_04_live.reconciliation.positions import reconcile_positions


NOW = datetime(2026, 9, 18, 12, 0, tzinfo=timezone.utc)


def _expected(
    ticket=1001,
    symbol="XAUUSD",
    direction="BUY",
    volume=0.05,
    stop_loss=4090.0,
    take_profit=4120.0,
):
    return ExpectedPosition(
        ticket=ticket,
        symbol=symbol,
        direction=direction,
        volume=volume,
        stop_loss=stop_loss,
        take_profit=take_profit,
    )


def _broker_position(
    ticket=1001,
    symbol="XAUUSD",
    direction="BUY",
    volume=0.05,
    stop_loss=4090.0,
    take_profit=4120.0,
    price_open=4100.0,
    price_current=4105.0,
    profit=25.0,
):
    return PositionState(
        ticket=ticket,
        symbol=symbol,
        direction=direction,
        volume=volume,
        price_open=price_open,
        price_current=price_current,
        stop_loss=stop_loss,
        take_profit=take_profit,
        profit=profit,
        swap=0.0,
        opened_at=NOW,
        magic=404_001,
        comment="test",
    )


def _snapshot(*positions, connected=True):
    return BrokerPositionsSnapshot(
        connected=connected,
        positions=tuple(positions),
        as_of=NOW,
    )


class TestPositionReconciliation(unittest.TestCase):

    # ============================================================
    # MATCHED
    # ============================================================

    def test_matched_when_everything_agrees(self):
        results = reconcile_positions(
            [_expected()],
            _snapshot(_broker_position()),
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].outcome, "MATCHED")
        self.assertEqual(results[0].ticket, 1001)
        self.assertIsNotNone(results[0].expected)
        self.assertIsNotNone(results[0].broker)

    # ============================================================
    # EMPTY STATE
    # ============================================================

    def test_empty_expected_and_empty_connected_broker_returns_nothing(self):
        results = reconcile_positions(
            [],
            _snapshot(),
        )

        self.assertEqual(results, ())

    def test_empty_expected_and_empty_disconnected_broker_returns_nothing(self):
        disconnected = BrokerPositionsSnapshot.disconnected(as_of=NOW)

        results = reconcile_positions(
            [],
            disconnected,
        )

        self.assertEqual(results, ())

    # ============================================================
    # MISSING_AT_BROKER
    # ============================================================

    def test_missing_at_broker_when_expected_ticket_not_present(self):
        results = reconcile_positions(
            [_expected(ticket=2002)],
            _snapshot(),
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].outcome, "MISSING_AT_BROKER")
        self.assertEqual(results[0].ticket, 2002)
        self.assertIsNotNone(results[0].expected)
        self.assertIsNone(results[0].broker)

    # ============================================================
    # UNEXPECTED_AT_BROKER
    # ============================================================

    def test_unexpected_at_broker_when_no_matching_expectation(self):
        results = reconcile_positions(
            [],
            _snapshot(_broker_position(ticket=3003)),
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].outcome, "UNEXPECTED_AT_BROKER")
        self.assertEqual(results[0].ticket, 3003)
        self.assertIsNone(results[0].expected)
        self.assertIsNotNone(results[0].broker)
        self.assertIn("UNKNOWN_NO_TICKET", results[0].detail)

    # ============================================================
    # VOLUME_MISMATCH
    # ============================================================

    def test_volume_mismatch(self):
        results = reconcile_positions(
            [_expected(volume=0.05)],
            _snapshot(_broker_position(volume=0.03)),
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].outcome, "VOLUME_MISMATCH")

    # ============================================================
    # DIRECTION_MISMATCH
    # ============================================================

    def test_direction_mismatch(self):
        results = reconcile_positions(
            [_expected(direction="BUY")],
            _snapshot(_broker_position(direction="SELL")),
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].outcome, "DIRECTION_MISMATCH")

    # ============================================================
    # SYMBOL_MISMATCH
    # ============================================================

    def test_symbol_mismatch(self):
        results = reconcile_positions(
            [_expected(symbol="XAUUSD")],
            _snapshot(_broker_position(symbol="EURUSD")),
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].outcome, "SYMBOL_MISMATCH")

    # ============================================================
    # SLTP_MISMATCH
    # ============================================================

    def test_stop_loss_mismatch(self):
        results = reconcile_positions(
            [_expected(stop_loss=4090.0)],
            _snapshot(_broker_position(stop_loss=4080.0)),
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].outcome, "SLTP_MISMATCH")

    def test_take_profit_mismatch(self):
        results = reconcile_positions(
            [_expected(take_profit=4120.0)],
            _snapshot(_broker_position(take_profit=4150.0)),
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].outcome, "SLTP_MISMATCH")

    def test_none_stop_loss_and_take_profit_match(self):
        results = reconcile_positions(
            [_expected(stop_loss=None, take_profit=None)],
            _snapshot(
                _broker_position(
                    stop_loss=None,
                    take_profit=None,
                )
            ),
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].outcome, "MATCHED")

    def test_one_missing_protective_level_causes_sltp_mismatch(self):
        results = reconcile_positions(
            [_expected(stop_loss=None, take_profit=4120.0)],
            _snapshot(
                _broker_position(
                    stop_loss=4090.0,
                    take_profit=4120.0,
                )
            ),
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].outcome, "SLTP_MISMATCH")

    # ============================================================
    # DISCONNECTED SNAPSHOT: FAIL CLOSED
    # ============================================================

    def test_disconnected_snapshot_returns_unknown_not_missing(self):
        disconnected = BrokerPositionsSnapshot.disconnected(
            as_of=NOW
        )

        results = reconcile_positions(
            [
                _expected(ticket=1001),
                _expected(ticket=1002),
            ],
            disconnected,
        )

        self.assertEqual(len(results), 2)
        self.assertTrue(
            all(result.outcome == "UNKNOWN" for result in results)
        )
        self.assertTrue(
            all(result.broker is None for result in results)
        )

    # ============================================================
    # MULTIPLE POSITIONS
    # ============================================================

    def test_multiple_positions_mixed_outcomes(self):
        expected = [
            _expected(ticket=1, volume=0.05),
            _expected(ticket=2, volume=0.05),
        ]

        broker_snapshot = _snapshot(
            _broker_position(ticket=1, volume=0.05),
            _broker_position(ticket=3, volume=0.10),
        )

        results = reconcile_positions(
            expected,
            broker_snapshot,
        )

        by_ticket = {
            result.ticket: result.outcome
            for result in results
        }

        self.assertEqual(by_ticket[1], "MATCHED")
        self.assertEqual(by_ticket[2], "MISSING_AT_BROKER")
        self.assertEqual(by_ticket[3], "UNEXPECTED_AT_BROKER")
        self.assertEqual(len(results), 3)

    def test_multiple_expected_positions_are_processed(self):
        expected = [
            _expected(ticket=1001),
            _expected(ticket=1002),
            _expected(ticket=1003),
        ]

        broker_snapshot = _snapshot(
            _broker_position(ticket=1001),
            _broker_position(ticket=1002),
            _broker_position(ticket=1003),
        )

        results = reconcile_positions(
            expected,
            broker_snapshot,
        )

        self.assertEqual(len(results), 3)
        self.assertEqual(
            [result.ticket for result in results],
            [1001, 1002, 1003],
        )
        self.assertTrue(
            all(result.outcome == "MATCHED" for result in results)
        )

    # ============================================================
    # DUPLICATE EXPECTED TICKET
    # ============================================================

    def test_duplicate_expected_ticket_raises(self):
        duplicated = [
            _expected(ticket=1001, volume=0.05),
            _expected(ticket=1001, volume=0.02),
        ]

        with self.assertRaisesRegex(
            ValueError,
            "Duplicate expected position ticket",
        ):
            reconcile_positions(
                duplicated,
                _snapshot(),
            )

    # ============================================================
    # DUPLICATE BROKER TICKET
    # ============================================================

    def test_duplicate_broker_ticket_raises(self):
        duplicated = _snapshot(
            _broker_position(ticket=1001, volume=0.05),
            _broker_position(ticket=1001, volume=0.02),
        )

        with self.assertRaisesRegex(
            ValueError,
            "Duplicate broker position ticket",
        ):
            reconcile_positions(
                [],
                duplicated,
            )

    # ============================================================
    # RESULT CONTENT
    # ============================================================

    def test_matched_result_preserves_expected_and_broker(self):
        expected = _expected()
        broker = _broker_position()

        results = reconcile_positions(
            [expected],
            _snapshot(broker),
        )

        self.assertIs(results[0].expected, expected)
        self.assertIs(results[0].broker, broker)

    def test_missing_result_contains_expected_and_no_broker(self):
        expected = _expected(ticket=5005)

        results = reconcile_positions(
            [expected],
            _snapshot(),
        )

        self.assertIs(results[0].expected, expected)
        self.assertIsNone(results[0].broker)

    def test_unexpected_result_contains_broker_and_no_expected(self):
        broker = _broker_position(ticket=6006)

        results = reconcile_positions(
            [],
            _snapshot(broker),
        )

        self.assertIsNone(results[0].expected)
        self.assertIs(results[0].broker, broker)

    # ============================================================
    # ORDERING
    # ============================================================

    def test_results_preserve_expected_then_broker_only_order(self):
        expected = [
            _expected(ticket=10),
            _expected(ticket=20),
        ]

        broker_snapshot = _snapshot(
            _broker_position(ticket=20),
            _broker_position(ticket=30),
            _broker_position(ticket=40),
        )

        results = reconcile_positions(
            expected,
            broker_snapshot,
        )

        self.assertEqual(
            [result.ticket for result in results],
            [10, 20, 30, 40],
        )

    # ============================================================
    # TIMEZONE
    # ============================================================

    def test_as_of_is_timezone_aware(self):
        results = reconcile_positions(
            [_expected()],
            _snapshot(_broker_position()),
        )

        self.assertIsNotNone(results[0].as_of.tzinfo)
        self.assertIsNotNone(results[0].as_of.utcoffset())


if __name__ == "__main__":
    unittest.main()