"""
phase_04_live/orders/test_on_before_send.py

Direct unit test of place_order_for_signal()'s on_before_send hook --
no live MT5 data, no session/staleness dependency. Confirms:

    1. on_before_send is called with the exact OrderRequest, BEFORE
       any broker call, when dry_run=False and all gates pass.
    2. on_before_send is NOT called when dry_run=True (no broker
       contact happens in dry-run, so nothing needs recording).
    3. Omitting on_before_send (the default) does not break anything.
"""

import unittest
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import MetaTrader5 as mt5

from phase_04_live.orders.order_manager import (
    OrderManagerConfig,
    place_order_for_signal,
)
from phase_04_live.broker.compatibility import SymbolCompatibilityReport
from strategy.signals import SignalType


def _fake_compatibility_report(symbol="EURUSD"):
    return SymbolCompatibilityReport(
        symbol=symbol,
        exists=True,
        visible=True,
        trade_mode=4,
        trade_mode_name="FULL",
        execution_mode=mt5.SYMBOL_TRADE_EXECUTION_MARKET,
        execution_mode_name="MARKET",
        volume_min=0.01,
        volume_max=100.0,
        volume_step=0.01,
        point=0.0001,
        digits=5,
        trade_stops_level=0,
        trade_freeze_level=0,
        raw_filling_mode_bitmask=2,
        supported_filling_modes=frozenset({"IOC"}),
        resolved_order_filling_mode=mt5.ORDER_FILLING_IOC,
        resolved_order_filling_mode_name="IOC",
        blocking_issues=(),
        warnings=(),
    )


def _fake_event_sizing_specs():
    event = MagicMock()
    event.fill_price = 100.0
    event.signal.stop_loss = 95.0
    event.signal.take_profit = 110.0

    sizing = MagicMock()
    sizing.accepted = True
    sizing.lots = 0.1
    sizing.direction = SignalType.LONG

    specs = MagicMock()
    specs.symbol = "EURUSD"
    specs.point = 0.0001

    return event, sizing, specs


class TestOnBeforeSendHook(unittest.TestCase):

    def setUp(self):
        self._stack = ExitStack()
        self.addCleanup(self._stack.close)

        self._stack.enter_context(
            patch(
                "phase_04_live.orders.order_manager.check_symbol_compatibility",
                return_value=_fake_compatibility_report(),
            )
        )
        self.tick_mock = self._stack.enter_context(
            patch("phase_04_live.orders.order_manager.mt5.symbol_info_tick")
        )
        self.send_mock = self._stack.enter_context(
            patch("phase_04_live.orders.order_manager.mt5.order_send")
        )

    def test_on_before_send_called_before_broker_call_when_live(self):
        event, sizing, specs = _fake_event_sizing_specs()
        config = OrderManagerConfig(dry_run=False, max_deviation_points=1000)

        self.tick_mock.return_value = MagicMock(bid=99.99, ask=100.01)
        self.send_mock.return_value = MagicMock(
            retcode=mt5.TRADE_RETCODE_DONE, order=12345,
        )

        calls = []

        def on_before_send(request):
            calls.append(request)
            self.send_mock.assert_not_called()

        result = place_order_for_signal(
            event, sizing, specs, config,
            on_before_send=on_before_send,
        )

        self.assertEqual(len(calls), 1, msg=f"result.reason={result.reason!r}")
        self.assertEqual(calls[0].symbol, "EURUSD")
        self.assertTrue(result.accepted)
        self.assertEqual(result.ticket, 12345)
        self.send_mock.assert_called_once()

    def test_on_before_send_not_called_in_dry_run(self):
        event, sizing, specs = _fake_event_sizing_specs()
        config = OrderManagerConfig(dry_run=True, max_deviation_points=1000)

        self.tick_mock.return_value = MagicMock(bid=99.9, ask=100.1)

        called = []
        result = place_order_for_signal(
            event, sizing, specs, config,
            on_before_send=lambda r: called.append(r),
        )

        self.assertEqual(called, [])
        self.assertTrue(result.dry_run)
        self.send_mock.assert_not_called()

    def test_omitting_on_before_send_still_works(self):
        event, sizing, specs = _fake_event_sizing_specs()
        config = OrderManagerConfig(dry_run=False, max_deviation_points=1000)

        self.tick_mock.return_value = MagicMock(bid=99.99, ask=100.01)
        self.send_mock.return_value = MagicMock(
            retcode=mt5.TRADE_RETCODE_DONE, order=999,
        )

        result = place_order_for_signal(event, sizing, specs, config)

        self.assertTrue(result.accepted, msg=f"result.reason={result.reason!r}")
        self.assertEqual(result.ticket, 999)


if __name__ == "__main__":
    unittest.main(verbosity=2)


