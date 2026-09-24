"""
phase_04_live/orders/test_order_manager.py

Unit tests for phase_04_live/orders/order_manager.py.

MetaTrader5 is fully mocked here -- these tests do NOT require a live
MT5 terminal connection. `mt5.symbol_info_tick`, `mt5.order_send`, and
check_symbol_compatibility() are patched; every constant used by
order_manager.py (TRADE_ACTION_DEAL, ORDER_TYPE_BUY, TRADE_RETCODE_DONE,
etc.) comes from the real MetaTrader5 package, since those are plain
module-level ints that don't require a connection to access.

check_symbol_compatibility() is patched in the shared setUp() below
because order_manager.place_order_for_signal() resolves a broker-safe
filling mode (Step 3 of the Phase 4 roadmap) before it ever fetches a
tick or sends an order. Without mocking it, every test would hit the
real MT5 symbol_info() call and fail with "No IPC connection".

Per the project roadmap, this test file's job is to PROVE the
OrderManager is correct before dry_run is ever set to False for real.
It does not touch position reconciliation, retries, or strategy logic.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import MetaTrader5 as mt5

from phase_04_live.orders.order_manager import (
    OrderManagerConfig,
    OrderRequest,
    OrderResult,
    place_order_for_signal,
)
from phase_04_live.risk.sizing import (
    PositionSizeResult,
    SymbolTradingSpecs,
)
from phase_03_paper.signals.adapter import AdapterSignalEvent
from strategy.signals import SignalType


# ---------------------------------------------------------------------------
# Test doubles / factories
#
# We deliberately do NOT construct a real strategy.signals.TradeSignal --
# its full constructor isn't part of the OrderManager's contract. OrderManager
# only ever touches signal.stop_loss and signal.take_profit (and, via
# sizing.direction, the trade direction), so a minimal stand-in with just
# those attributes is a faithful, low-coupling test double.
# ---------------------------------------------------------------------------

def make_signal(stop_loss: float, take_profit: float):
    return SimpleNamespace(stop_loss=stop_loss, take_profit=take_profit)


def make_event(
    *,
    signal,
    fill_price: float = 1.10000,
    initial_risk: float = 0.00100,
    setup_bar_index: int = 5,
    trigger_bar_index: int = 6,
) -> AdapterSignalEvent:
    return AdapterSignalEvent(
        signal=signal,
        setup_bar_index=setup_bar_index,
        trigger_bar_index=trigger_bar_index,
        fill_price=fill_price,
        initial_risk=initial_risk,
    )


def make_specs(
    *,
    symbol: str = "EURUSD",
    contract_size: float = 100_000.0,
    volume_min: float = 0.01,
    volume_max: float = 100.0,
    volume_step: float = 0.01,
    point: float = 0.00001,
    digits: int = 5,
) -> SymbolTradingSpecs:
    return SymbolTradingSpecs(
        symbol=symbol,
        contract_size=contract_size,
        volume_min=volume_min,
        volume_max=volume_max,
        volume_step=volume_step,
        point=point,
        digits=digits,
    )


def make_sizing(
    *,
    accepted: bool = True,
    lots: float = 0.10,
    direction: SignalType = SignalType.LONG,
    dollar_risk: float = 10.0,
    sizing_mode: str = "fixed_lot",
    reward_multiple_override=None,
    reason=None,
) -> PositionSizeResult:
    return PositionSizeResult(
        accepted=accepted,
        lots=lots if accepted else None,
        direction=direction,
        dollar_risk=dollar_risk if accepted else None,
        sizing_mode=sizing_mode,
        reward_multiple_override=reward_multiple_override,
        reason=reason,
    )


def make_config(
    *,
    dry_run: bool = True,
    max_spread_points: int = 500,
    max_deviation_points: int = 50,
    broker_deviation_points: int = 20,
    magic_number: int = 404_001,
    comment: str = "phase4-live-test",
) -> OrderManagerConfig:
    return OrderManagerConfig(
        magic_number=magic_number,
        max_deviation_points=max_deviation_points,
        broker_deviation_points=broker_deviation_points,
        max_spread_points=max_spread_points,
        comment=comment,
        dry_run=dry_run,
    )


def make_tick(bid: float, ask: float):
    """A stand-in for the object mt5.symbol_info_tick() returns."""
    return SimpleNamespace(bid=bid, ask=ask)


def make_broker_result(retcode: int, order: int = 123456, comment: str = ""):
    """A stand-in for the object mt5.order_send() returns."""
    return SimpleNamespace(retcode=retcode, order=order, comment=comment)


def make_compatibility_report(
    *,
    is_compatible: bool = True,
    resolved_order_filling_mode=mt5.ORDER_FILLING_IOC,
    blocking_issues=None,
):
    """A stand-in for check_symbol_compatibility()'s return value.

    order_manager.py only ever reads .is_compatible and
    .resolved_order_filling_mode (plus .blocking_issues on the
    rejection path), so a duck-typed SimpleNamespace is a faithful,
    low-coupling test double -- same rationale as make_tick/make_signal
    above. This avoids coupling the test file to
    SymbolCompatibilityReport's exact constructor signature.
    """
    return SimpleNamespace(
        is_compatible=is_compatible,
        resolved_order_filling_mode=resolved_order_filling_mode,
        blocking_issues=blocking_issues or [],
    )


ORDER_MANAGER_MODULE = "phase_04_live.orders.order_manager"


class OrderManagerTestCase(unittest.TestCase):
    """Shared fixtures for a straightforward, all-checks-pass LONG setup."""

    def setUp(self):
        # EURUSD-style specs: point=0.00001 ("5th decimal" pricing).
        self.specs = make_specs(point=0.00001)

        # A LONG signal whose stop/target bracket price=1.10000 correctly.
        self.long_signal = make_signal(stop_loss=1.09000, take_profit=1.12000)
        self.long_event = make_event(signal=self.long_signal, fill_price=1.10000)
        self.long_sizing = make_sizing(direction=SignalType.LONG, lots=0.10)

        # A SHORT signal whose stop/target bracket price=1.10000 correctly.
        self.short_signal = make_signal(stop_loss=1.11000, take_profit=1.08000)
        self.short_event = make_event(signal=self.short_signal, fill_price=1.10000)
        self.short_sizing = make_sizing(direction=SignalType.SHORT, lots=0.10)

        self.config = make_config(dry_run=True)

        # Tight, clean spread around 1.10000 -- well within max_spread_points.
        self.tick = make_tick(bid=1.09999, ask=1.10001)

        # The Step 3 compatibility gate (_resolve_filling_mode) runs
        # before any tick is fetched or order is sent. Patch it here,
        # once, for every test in every subclass, so existing tests
        # written before that gate existed don't hit the real MT5
        # symbol_info() call. Individual tests can still override
        # self.mock_compat.return_value to exercise the rejection path.
        compat_patcher = patch(
            f"{ORDER_MANAGER_MODULE}.check_symbol_compatibility"
        )
        self.mock_compat = compat_patcher.start()
        self.mock_compat.return_value = make_compatibility_report()
        self.addCleanup(compat_patcher.stop)


class TestDryRunSafety(OrderManagerTestCase):
    """1. dry-run never calls mt5.order_send()."""

    @patch(f"{ORDER_MANAGER_MODULE}.mt5.order_send")
    @patch(f"{ORDER_MANAGER_MODULE}.mt5.symbol_info_tick")
    def test_dry_run_never_calls_order_send(self, mock_tick, mock_send):
        mock_tick.return_value = self.tick

        result = place_order_for_signal(
            self.long_event, self.long_sizing, self.specs, self.config
        )

        mock_send.assert_not_called()
        self.assertTrue(result.accepted)
        self.assertTrue(result.dry_run)
        self.assertIsNotNone(result.request)


class TestExecutionSideSelection(OrderManagerTestCase):
    """2. LONG uses ask. 3. SHORT uses bid."""

    @patch(f"{ORDER_MANAGER_MODULE}.mt5.symbol_info_tick")
    def test_long_uses_ask(self, mock_tick):
        mock_tick.return_value = self.tick

        result = place_order_for_signal(
            self.long_event, self.long_sizing, self.specs, self.config
        )

        self.assertTrue(result.accepted)
        self.assertEqual(result.request.price, self.tick.ask)

    @patch(f"{ORDER_MANAGER_MODULE}.mt5.symbol_info_tick")
    def test_short_uses_bid(self, mock_tick):
        mock_tick.return_value = self.tick

        result = place_order_for_signal(
            self.short_event, self.short_sizing, self.specs, self.config
        )

        self.assertTrue(result.accepted)
        self.assertEqual(result.request.price, self.tick.bid)


class TestSingleTickFetch(OrderManagerTestCase):
    """4. Only one tick is fetched per order attempt."""

    @patch(f"{ORDER_MANAGER_MODULE}.mt5.symbol_info_tick")
    def test_only_one_tick_fetched(self, mock_tick):
        mock_tick.return_value = self.tick

        place_order_for_signal(
            self.long_event, self.long_sizing, self.specs, self.config
        )

        mock_tick.assert_called_once()


class TestSpreadGuard(OrderManagerTestCase):
    """5. Excessive spread is rejected."""

    @patch(f"{ORDER_MANAGER_MODULE}.mt5.order_send")
    @patch(f"{ORDER_MANAGER_MODULE}.mt5.symbol_info_tick")
    def test_excessive_spread_rejected(self, mock_tick, mock_send):
        # point=0.00001, max_spread_points=500 -> max allowed spread is
        # 0.005. Use a spread well beyond that.
        wide_tick = make_tick(bid=1.09000, ask=1.10000)  # 1000 points
        mock_tick.return_value = wide_tick

        result = place_order_for_signal(
            self.long_event, self.long_sizing, self.specs, self.config
        )

        self.assertFalse(result.accepted)
        self.assertIsNone(result.request)
        self.assertIn("Spread too wide", result.reason)
        mock_send.assert_not_called()


class TestPriceStaleness(OrderManagerTestCase):
    """6. A stale strategy price is rejected rather than chased."""

    @patch(f"{ORDER_MANAGER_MODULE}.mt5.order_send")
    @patch(f"{ORDER_MANAGER_MODULE}.mt5.symbol_info_tick")
    def test_stale_price_rejected(self, mock_tick, mock_send):
        # max_deviation_points=50, point=0.00001 -> max drift is 0.0005.
        # Move live price far beyond that from event.fill_price=1.10000.
        far_tick = make_tick(bid=1.10999, ask=1.11001)
        mock_tick.return_value = far_tick

        result = place_order_for_signal(
            self.long_event, self.long_sizing, self.specs, self.config
        )

        self.assertFalse(result.accepted)
        self.assertIsNone(result.request)
        self.assertIn("drifted", result.reason)
        mock_send.assert_not_called()


class TestStopAndTargetValidation(OrderManagerTestCase):
    """7-10. Invalid SL/TP relative to current execution price is rejected."""

    @patch(f"{ORDER_MANAGER_MODULE}.mt5.order_send")
    @patch(f"{ORDER_MANAGER_MODULE}.mt5.symbol_info_tick")
    def test_invalid_long_stop_rejected(self, mock_tick, mock_send):
        mock_tick.return_value = self.tick  # execution price (ask) = 1.10001

        bad_signal = make_signal(stop_loss=1.10500, take_profit=1.12000)  # SL above price
        event = make_event(signal=bad_signal, fill_price=1.10000)

        result = place_order_for_signal(event, self.long_sizing, self.specs, self.config)

        self.assertFalse(result.accepted)
        self.assertIn("Invalid LONG stop_loss", result.reason)
        mock_send.assert_not_called()

    @patch(f"{ORDER_MANAGER_MODULE}.mt5.order_send")
    @patch(f"{ORDER_MANAGER_MODULE}.mt5.symbol_info_tick")
    def test_invalid_long_target_rejected(self, mock_tick, mock_send):
        mock_tick.return_value = self.tick  # execution price (ask) = 1.10001

        bad_signal = make_signal(stop_loss=1.09000, take_profit=1.09500)  # TP below price
        event = make_event(signal=bad_signal, fill_price=1.10000)

        result = place_order_for_signal(event, self.long_sizing, self.specs, self.config)

        self.assertFalse(result.accepted)
        self.assertIn("Invalid LONG take_profit", result.reason)
        mock_send.assert_not_called()

    @patch(f"{ORDER_MANAGER_MODULE}.mt5.order_send")
    @patch(f"{ORDER_MANAGER_MODULE}.mt5.symbol_info_tick")
    def test_invalid_short_stop_rejected(self, mock_tick, mock_send):
        mock_tick.return_value = self.tick  # execution price (bid) = 1.09999

        bad_signal = make_signal(stop_loss=1.09500, take_profit=1.08000)  # SL below price
        event = make_event(signal=bad_signal, fill_price=1.10000)

        result = place_order_for_signal(event, self.short_sizing, self.specs, self.config)

        self.assertFalse(result.accepted)
        self.assertIn("Invalid SHORT stop_loss", result.reason)
        mock_send.assert_not_called()

    @patch(f"{ORDER_MANAGER_MODULE}.mt5.order_send")
    @patch(f"{ORDER_MANAGER_MODULE}.mt5.symbol_info_tick")
    def test_invalid_short_target_rejected(self, mock_tick, mock_send):
        mock_tick.return_value = self.tick  # execution price (bid) = 1.09999

        bad_signal = make_signal(stop_loss=1.11000, take_profit=1.10500)  # TP above price
        event = make_event(signal=bad_signal, fill_price=1.10000)

        result = place_order_for_signal(event, self.short_sizing, self.specs, self.config)

        self.assertFalse(result.accepted)
        self.assertIn("Invalid SHORT take_profit", result.reason)
        mock_send.assert_not_called()


class TestRequestConstruction(OrderManagerTestCase):
    """11-15. A valid request is constructed with the correct fields."""

    @patch(f"{ORDER_MANAGER_MODULE}.mt5.symbol_info_tick")
    def test_valid_request_constructed_correctly(self, mock_tick):
        mock_tick.return_value = self.tick

        result = place_order_for_signal(
            self.long_event, self.long_sizing, self.specs, self.config
        )

        self.assertTrue(result.accepted)
        req = result.request
        self.assertIsInstance(req, OrderRequest)
        self.assertEqual(req.symbol, self.specs.symbol)
        self.assertEqual(req.order_type, mt5.ORDER_TYPE_BUY)
        self.assertEqual(req.price, self.tick.ask)
        self.assertEqual(req.comment, self.config.comment.strip())

    @patch(f"{ORDER_MANAGER_MODULE}.mt5.symbol_info_tick")
    def test_correct_volume_used(self, mock_tick):
        mock_tick.return_value = self.tick

        result = place_order_for_signal(
            self.long_event, self.long_sizing, self.specs, self.config
        )

        self.assertEqual(result.request.volume, self.long_sizing.lots)

    @patch(f"{ORDER_MANAGER_MODULE}.mt5.symbol_info_tick")
    def test_correct_sl_used(self, mock_tick):
        mock_tick.return_value = self.tick

        result = place_order_for_signal(
            self.long_event, self.long_sizing, self.specs, self.config
        )

        self.assertEqual(result.request.sl, self.long_signal.stop_loss)

    @patch(f"{ORDER_MANAGER_MODULE}.mt5.symbol_info_tick")
    def test_correct_tp_used(self, mock_tick):
        mock_tick.return_value = self.tick

        result = place_order_for_signal(
            self.long_event, self.long_sizing, self.specs, self.config
        )

        self.assertEqual(result.request.tp, self.long_signal.take_profit)

    @patch(f"{ORDER_MANAGER_MODULE}.mt5.symbol_info_tick")
    def test_broker_deviation_used(self, mock_tick):
        mock_tick.return_value = self.tick

        result = place_order_for_signal(
            self.long_event, self.long_sizing, self.specs, self.config
        )

        # broker_deviation_points (execution tolerance) must be used here,
        # NOT max_deviation_points (strategy staleness threshold) -- the
        # two are deliberately different config fields for different jobs.
        self.assertEqual(result.request.deviation, self.config.broker_deviation_points)
        self.assertNotEqual(self.config.broker_deviation_points, self.config.max_deviation_points)


class TestBrokerSubmission(OrderManagerTestCase):
    """16-17. Broker rejection and success are mapped to OrderResult correctly."""

    @patch(f"{ORDER_MANAGER_MODULE}.mt5.order_send")
    @patch(f"{ORDER_MANAGER_MODULE}.mt5.symbol_info_tick")
    def test_broker_rejection_becomes_order_result_rejected(self, mock_tick, mock_send):
        mock_tick.return_value = self.tick
        mock_send.return_value = make_broker_result(
            retcode=mt5.TRADE_RETCODE_REQUOTE, comment="requote"
        )

        live_config = make_config(dry_run=False)

        result = place_order_for_signal(
            self.long_event, self.long_sizing, self.specs, live_config
        )

        mock_send.assert_called_once()
        self.assertFalse(result.accepted)
        self.assertFalse(result.dry_run)
        self.assertEqual(result.retcode, mt5.TRADE_RETCODE_REQUOTE)
        self.assertIsNone(result.ticket)
        self.assertIsNotNone(result.request)

    @patch(f"{ORDER_MANAGER_MODULE}.mt5.order_send")
    @patch(f"{ORDER_MANAGER_MODULE}.mt5.symbol_info_tick")
    def test_successful_broker_response_becomes_accepted(self, mock_tick, mock_send):
        mock_tick.return_value = self.tick
        mock_send.return_value = make_broker_result(
            retcode=mt5.TRADE_RETCODE_DONE, order=987654
        )

        live_config = make_config(dry_run=False)

        result = place_order_for_signal(
            self.long_event, self.long_sizing, self.specs, live_config
        )

        mock_send.assert_called_once()
        self.assertTrue(result.accepted)
        self.assertFalse(result.dry_run)
        self.assertEqual(result.retcode, mt5.TRADE_RETCODE_DONE)
        self.assertEqual(result.ticket, 987654)
        self.assertIsNone(result.reason)


class TestSizingRejection(OrderManagerTestCase):
    """18. A sizing rejection short-circuits before any tick/order work."""

    @patch(f"{ORDER_MANAGER_MODULE}.mt5.order_send")
    @patch(f"{ORDER_MANAGER_MODULE}.mt5.symbol_info_tick")
    def test_sizing_rejection_prevents_tick_and_order_processing(self, mock_tick, mock_send):
        rejected_sizing = make_sizing(
            accepted=False,
            lots=None,
            direction=SignalType.LONG,
            reason="Calculated size below broker minimum",
        )

        result = place_order_for_signal(
            self.long_event, rejected_sizing, self.specs, self.config
        )

        self.assertFalse(result.accepted)
        self.assertIsNone(result.request)
        self.assertIn("Sizing rejected", result.reason)
        mock_tick.assert_not_called()
        mock_send.assert_not_called()


class TestCompatibilityRejection(OrderManagerTestCase):
    """19. A broker-incompatible symbol is rejected before any tick is fetched."""

    @patch(f"{ORDER_MANAGER_MODULE}.mt5.order_send")
    @patch(f"{ORDER_MANAGER_MODULE}.mt5.symbol_info_tick")
    def test_incompatible_symbol_rejected(self, mock_tick, mock_send):
        self.mock_compat.return_value = make_compatibility_report(
            is_compatible=False,
            resolved_order_filling_mode=None,
            blocking_issues=["Symbol is disabled for trading"],
        )

        result = place_order_for_signal(
            self.long_event, self.long_sizing, self.specs, self.config
        )

        self.assertFalse(result.accepted)
        self.assertIsNone(result.request)
        self.assertIn("failed compatibility check", result.reason)
        mock_tick.assert_not_called()
        mock_send.assert_not_called()

    @patch(f"{ORDER_MANAGER_MODULE}.mt5.order_send")
    @patch(f"{ORDER_MANAGER_MODULE}.mt5.symbol_info_tick")
    def test_no_resolved_filling_mode_rejected(self, mock_tick, mock_send):
        self.mock_compat.return_value = make_compatibility_report(
            is_compatible=True,
            resolved_order_filling_mode=None,
        )

        result = place_order_for_signal(
            self.long_event, self.long_sizing, self.specs, self.config
        )

        self.assertFalse(result.accepted)
        self.assertIsNone(result.request)
        self.assertIn("no resolved order filling mode", result.reason)
        mock_tick.assert_not_called()
        mock_send.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)