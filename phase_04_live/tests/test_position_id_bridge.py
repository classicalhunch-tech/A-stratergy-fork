import unittest
from types import SimpleNamespace
from unittest.mock import ANY, MagicMock, patch

from phase_04_live.broker import mt5 as broker_mt5
from phase_04_live.execution.position_risk import (
    record_risk_for_accepted_order,
)
from phase_04_live.orders.order_manager import OrderResult

ENTRY_IN, ENTRY_OUT = 0, 1


class _Conn:
    def __init__(self, connected=True):
        self._connected = connected

    def is_connected(self):
        return self._connected


def _deal(order, position_id, entry=ENTRY_IN):
    return SimpleNamespace(
        order=order, position_id=position_id, entry=entry
    )


def _mt5_mock():
    m = MagicMock()
    m.DEAL_ENTRY_IN = ENTRY_IN
    m.DEAL_ENTRY_OUT = ENTRY_OUT
    return m


class ResolvePositionIdTests(unittest.TestCase):

    def test_deal_ticket_resolves_position_id_not_order_ticket(self):
        m = _mt5_mock()
        m.history_deals_get.return_value = (_deal(100, 555),)
        with patch.object(broker_mt5, "mt5", m):
            pid = broker_mt5.resolve_position_id(
                _Conn(), 100, deal_ticket=900
            )
        self.assertEqual(pid, 555)
        self.assertNotEqual(pid, 100)
        m.history_deals_get.assert_called_once_with(ticket=900)

    def test_deal_belonging_to_other_order_returns_none(self):
        m = _mt5_mock()
        m.history_deals_get.return_value = (_deal(999, 555),)
        with patch.object(broker_mt5, "mt5", m):
            pid = broker_mt5.resolve_position_id(
                _Conn(), 100, deal_ticket=900
            )
        self.assertIsNone(pid)
        self.assertEqual(m.history_deals_get.call_count, 1)

    def test_exit_deal_is_not_accepted(self):
        m = _mt5_mock()
        m.history_deals_get.return_value = (
            _deal(100, 555, entry=ENTRY_OUT),
        )
        with patch.object(broker_mt5, "mt5", m):
            pid = broker_mt5.resolve_position_id(
                _Conn(), 100, deal_ticket=900
            )
        self.assertIsNone(pid)

    def test_zero_position_id_is_not_accepted(self):
        m = _mt5_mock()
        m.history_deals_get.return_value = (_deal(100, 0),)
        with patch.object(broker_mt5, "mt5", m):
            pid = broker_mt5.resolve_position_id(
                _Conn(), 100, deal_ticket=900
            )
        self.assertIsNone(pid)

    def test_falls_back_to_recent_history_by_order_ticket(self):
        m = _mt5_mock()
        m.history_deals_get.side_effect = [
            (),
            (_deal(100, 555), _deal(101, 777)),
        ]
        with patch.object(broker_mt5, "mt5", m):
            pid = broker_mt5.resolve_position_id(
                _Conn(), 100, deal_ticket=900
            )
        self.assertEqual(pid, 555)

    def test_conflicting_position_ids_return_none(self):
        m = _mt5_mock()
        m.history_deals_get.return_value = (
            _deal(100, 555),
            _deal(100, 556),
        )
        with patch.object(broker_mt5, "mt5", m):
            pid = broker_mt5.resolve_position_id(_Conn(), 100)
        self.assertIsNone(pid)

    def test_no_history_returns_none(self):
        m = _mt5_mock()
        m.history_deals_get.return_value = None
        with patch.object(broker_mt5, "mt5", m):
            pid = broker_mt5.resolve_position_id(_Conn(), 100)
        self.assertIsNone(pid)

    def test_disconnected_returns_none_without_broker_call(self):
        m = _mt5_mock()
        with patch.object(broker_mt5, "mt5", m):
            pid = broker_mt5.resolve_position_id(
                _Conn(connected=False), 100, deal_ticket=900
            )
        self.assertIsNone(pid)
        m.history_deals_get.assert_not_called()


def _result(**over):
    base = dict(
        accepted=True,
        dry_run=False,
        request=None,
        retcode=10009,
        ticket=100,
        deal=900,
        reason=None,
    )
    base.update(over)
    return OrderResult(**base)


class OrderResultDealFieldTests(unittest.TestCase):

    def test_deal_defaults_to_none_and_can_be_set(self):
        r = OrderResult(
            accepted=False,
            dry_run=True,
            request=None,
            retcode=None,
            ticket=None,
        )
        self.assertIsNone(r.deal)
        self.assertEqual(_result(deal=7).deal, 7)


class RecordRiskBridgeTests(unittest.TestCase):

    def setUp(self):
        self.persistence = MagicMock()
        self.sleep = MagicMock()

    def _run(self, result, resolver):
        return record_risk_for_accepted_order(
            _Conn(),
            result,
            25.0,
            self.persistence,
            attempts=3,
            delay_seconds=0.1,
            sleep=self.sleep,
            resolver=resolver,
        )

    def test_records_risk_under_position_id_not_order_ticket(self):
        resolver = MagicMock(return_value=555)
        pid = self._run(_result(), resolver)
        self.assertEqual(pid, 555)
        self.persistence.record_trade_risk.assert_called_once_with(
            555, 25.0
        )
        resolver.assert_called_once_with(ANY, 100, deal_ticket=900)

    def test_unresolved_records_nothing_and_never_uses_order_ticket(self):
        resolver = MagicMock(return_value=None)
        pid = self._run(_result(), resolver)
        self.assertIsNone(pid)
        self.assertEqual(resolver.call_count, 3)
        self.assertEqual(self.sleep.call_count, 2)
        self.persistence.record_trade_risk.assert_not_called()

    def test_resolves_on_later_attempt(self):
        resolver = MagicMock(side_effect=[None, None, 555])
        pid = self._run(_result(), resolver)
        self.assertEqual(pid, 555)
        self.persistence.record_trade_risk.assert_called_once_with(
            555, 25.0
        )

    def test_dry_run_is_ignored(self):
        resolver = MagicMock(return_value=555)
        self.assertIsNone(self._run(_result(dry_run=True), resolver))
        resolver.assert_not_called()
        self.persistence.record_trade_risk.assert_not_called()

    def test_rejected_order_is_ignored(self):
        resolver = MagicMock(return_value=555)
        rejected = _result(accepted=False, ticket=None, deal=None)
        self.assertIsNone(self._run(rejected, resolver))
        resolver.assert_not_called()
        self.persistence.record_trade_risk.assert_not_called()


if __name__ == "__main__":
    unittest.main()
