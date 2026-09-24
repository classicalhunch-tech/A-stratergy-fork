"""
phase_04_live/broker/test_compatibility.py

Unit tests for phase_04_live/broker/compatibility.py:

    decode_supported_filling_modes()
    return_filling_is_allowed()
    select_order_filling_mode()
    check_symbol_compatibility()

These tests:

    - do not connect to a live MT5 terminal
    - do not place orders
    - do not modify broker state

mt5.symbol_info() is mocked for check_symbol_compatibility() tests only.

The tests use the local SYMBOL_FILLING_FOK / SYMBOL_FILLING_IOC
protocol bit values defined by the compatibility module because the
installed MetaTrader5 Python package does not expose those names.

The MT5 module is used only for actual ORDER_FILLING_* and
SYMBOL_TRADE_EXECUTION_* constants.

Phase 4 compatibility rule:

    NEVER GUESS BROKER CAPABILITIES.

If explicit FOK/IOC support is unavailable, RETURN may only be
selected when the execution mode is known and explicitly permits it.

Otherwise the selector must fail closed.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import MetaTrader5 as mt5

from phase_04_live.broker import compatibility
from phase_04_live.broker.compatibility import (
    SYMBOL_FILLING_FOK,
    SYMBOL_FILLING_IOC,
    check_symbol_compatibility,
    decode_supported_filling_modes,
    return_filling_is_allowed,
    select_order_filling_mode,
)


class TestDecodeSupportedFillingModes(unittest.TestCase):
    """Test decoding of the symbol filling capability bitmask."""

    def test_zero_bitmask_gives_no_explicit_flags(self):
        """A zero bitmask contains no explicit FOK/IOC capability flags."""
        self.assertEqual(
            decode_supported_filling_modes(0),
            frozenset(),
        )

    def test_fok_only_bit(self):
        self.assertEqual(
            decode_supported_filling_modes(
                SYMBOL_FILLING_FOK
            ),
            frozenset({"FOK"}),
        )

    def test_ioc_only_bit(self):
        self.assertEqual(
            decode_supported_filling_modes(
                SYMBOL_FILLING_IOC
            ),
            frozenset({"IOC"}),
        )

    def test_both_fok_and_ioc_bits(self):
        combined = (
            SYMBOL_FILLING_FOK
            | SYMBOL_FILLING_IOC
        )

        self.assertEqual(
            decode_supported_filling_modes(combined),
            frozenset({"FOK", "IOC"}),
        )

    def test_unknown_bits_are_ignored(self):
        """
        Unknown or reserved bits must not be interpreted as
        filling capabilities.
        """
        self.assertEqual(
            decode_supported_filling_modes(1 << 10),
            frozenset(),
        )


class TestReturnFillingIsAllowed(unittest.TestCase):
    """Test execution-mode rules governing ORDER_FILLING_RETURN."""

    def test_market_execution_prohibits_return(self):
        self.assertFalse(
            return_filling_is_allowed(
                mt5.SYMBOL_TRADE_EXECUTION_MARKET
            )
        )

    def test_instant_execution_permits_return(self):
        self.assertTrue(
            return_filling_is_allowed(
                mt5.SYMBOL_TRADE_EXECUTION_INSTANT
            )
        )

    def test_request_execution_permits_return(self):
        self.assertTrue(
            return_filling_is_allowed(
                mt5.SYMBOL_TRADE_EXECUTION_REQUEST
            )
        )

    def test_exchange_execution_permits_return(self):
        self.assertTrue(
            return_filling_is_allowed(
                mt5.SYMBOL_TRADE_EXECUTION_EXCHANGE
            )
        )

    def test_unknown_execution_mode_prohibits_return(self):
        self.assertFalse(
            return_filling_is_allowed(999)
        )

    def test_none_execution_mode_prohibits_return(self):
        """
        Missing execution-mode information must never be interpreted
        as permission for RETURN.
        """
        self.assertFalse(
            return_filling_is_allowed(None)
        )


class TestSelectOrderFillingMode(unittest.TestCase):
    """Test safe resolution of the final ORDER_FILLING_* constant."""

    def test_prefers_ioc_when_explicitly_supported(self):
        """
        IOC is preferred when both FOK and IOC are available.
        """
        combined = (
            SYMBOL_FILLING_FOK
            | SYMBOL_FILLING_IOC
        )

        result = select_order_filling_mode(
            combined,
            execution_mode=mt5.SYMBOL_TRADE_EXECUTION_INSTANT,
        )

        self.assertEqual(
            result,
            mt5.ORDER_FILLING_IOC,
        )

    def test_falls_back_to_fok_when_ioc_is_unsupported(self):
        result = select_order_filling_mode(
            SYMBOL_FILLING_FOK,
            execution_mode=mt5.SYMBOL_TRADE_EXECUTION_INSTANT,
        )

        self.assertEqual(
            result,
            mt5.ORDER_FILLING_FOK,
        )

    def test_selects_return_when_allowed_and_no_explicit_flags(self):
        """
        RETURN is selected when permitted by execution mode and
        the symbol exposes no explicit FOK/IOC flags.
        """
        result = select_order_filling_mode(
            0,
            execution_mode=mt5.SYMBOL_TRADE_EXECUTION_INSTANT,
        )

        self.assertEqual(
            result,
            mt5.ORDER_FILLING_RETURN,
        )

    def test_fails_closed_on_market_execution_with_zero_bitmask(self):
        """
        Market Execution prohibits RETURN and bitmask 0 provides
        no explicit fallback.
        """
        with self.assertRaises(RuntimeError):
            select_order_filling_mode(
                0,
                execution_mode=mt5.SYMBOL_TRADE_EXECUTION_MARKET,
            )

    def test_fails_closed_when_execution_mode_is_omitted(self):
        """
        Missing execution mode with no explicit flags must fail closed.
        """
        with self.assertRaises(RuntimeError):
            select_order_filling_mode(
                0,
                execution_mode=None,
            )

    def test_fails_closed_on_unknown_execution_mode(self):
        """
        Unknown execution mode with no explicit flags must fail closed.
        """
        with self.assertRaises(RuntimeError):
            select_order_filling_mode(
                0,
                execution_mode=999,
            )

    def test_explicit_ioc_is_sufficient_even_with_unknown_execution_mode(self):
        """
        Explicit IOC capability is sufficient even when execution mode
        is unknown.
        """
        result = select_order_filling_mode(
            SYMBOL_FILLING_IOC,
            execution_mode=999,
        )

        self.assertEqual(
            result,
            mt5.ORDER_FILLING_IOC,
        )

    def test_explicit_fok_is_sufficient_even_with_unknown_execution_mode(self):
        """
        Explicit FOK capability is sufficient even when execution mode
        is unknown.
        """
        result = select_order_filling_mode(
            SYMBOL_FILLING_FOK,
            execution_mode=999,
        )

        self.assertEqual(
            result,
            mt5.ORDER_FILLING_FOK,
        )

    def test_explicit_ioc_takes_priority_over_return(self):
        """
        Explicit IOC support wins over a permitted RETURN policy.
        """
        result = select_order_filling_mode(
            SYMBOL_FILLING_IOC,
            execution_mode=mt5.SYMBOL_TRADE_EXECUTION_INSTANT,
        )

        self.assertEqual(
            result,
            mt5.ORDER_FILLING_IOC,
        )

    def test_explicit_fok_takes_priority_over_return(self):
        """
        Explicit FOK support wins over a permitted RETURN policy.
        """
        result = select_order_filling_mode(
            SYMBOL_FILLING_FOK,
            execution_mode=mt5.SYMBOL_TRADE_EXECUTION_INSTANT,
        )

        self.assertEqual(
            result,
            mt5.ORDER_FILLING_FOK,
        )

    def test_result_is_always_a_valid_order_filling_constant(self):
        """
        Every successful selection must return an actual MT5
        ORDER_FILLING_* constant.
        """
        valid_constants = {
            mt5.ORDER_FILLING_FOK,
            mt5.ORDER_FILLING_IOC,
            mt5.ORDER_FILLING_RETURN,
        }

        test_cases = [
            (
                0,
                mt5.SYMBOL_TRADE_EXECUTION_INSTANT,
            ),
            (
                SYMBOL_FILLING_FOK,
                mt5.SYMBOL_TRADE_EXECUTION_MARKET,
            ),
            (
                SYMBOL_FILLING_IOC,
                mt5.SYMBOL_TRADE_EXECUTION_MARKET,
            ),
            (
                SYMBOL_FILLING_FOK | SYMBOL_FILLING_IOC,
                mt5.SYMBOL_TRADE_EXECUTION_EXCHANGE,
            ),
        ]

        for bitmask, execution_mode in test_cases:
            with self.subTest(
                bitmask=bitmask,
                execution_mode=execution_mode,
            ):
                result = select_order_filling_mode(
                    bitmask,
                    execution_mode=execution_mode,
                )

                self.assertIn(
                    result,
                    valid_constants,
                )


def make_symbol_info(**overrides):
    """
    Build a fake MT5 SymbolInfo-like object with sane FULL/compatible
    defaults. Override only the fields relevant to each test case.
    """
    defaults = dict(
        visible=True,
        trade_mode=4,  # FULL
        trade_exemode=mt5.SYMBOL_TRADE_EXECUTION_INSTANT,
        volume_min=0.01,
        volume_max=100.0,
        volume_step=0.01,
        point=0.00001,
        digits=5,
        trade_stops_level=0,
        trade_freeze_level=0,
        filling_mode=SYMBOL_FILLING_IOC | SYMBOL_FILLING_FOK,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class TestCheckSymbolCompatibility(unittest.TestCase):
    """
    Tests for check_symbol_compatibility() using a mocked
    mt5.symbol_info(). No live MT5 connection is used.
    """

    def _check(self, info):
        with patch.object(
            compatibility.mt5, "symbol_info", return_value=info
        ):
            return check_symbol_compatibility("TESTSYMBOL")

    # -- symbol existence ---------------------------------------------

    def test_symbol_does_not_exist(self):
        with patch.object(
            compatibility.mt5, "symbol_info", return_value=None
        ), patch.object(
            compatibility.mt5, "last_error", return_value=(1, "mock error")
        ):
            report = check_symbol_compatibility("NOPE")

        self.assertFalse(report.exists)
        self.assertFalse(report.is_compatible)
        self.assertTrue(report.blocking_issues)

    # -- filling mode scenarios ----------------------------------------

    def test_ioc_only(self):
        report = self._check(
            make_symbol_info(filling_mode=SYMBOL_FILLING_IOC)
        )
        self.assertEqual(
            report.resolved_order_filling_mode, mt5.ORDER_FILLING_IOC
        )
        self.assertTrue(report.is_compatible)

    def test_fok_only(self):
        report = self._check(
            make_symbol_info(filling_mode=SYMBOL_FILLING_FOK)
        )
        self.assertEqual(
            report.resolved_order_filling_mode, mt5.ORDER_FILLING_FOK
        )
        self.assertTrue(report.is_compatible)
        # OrderManager currently hardcodes IOC, so this must warn.
        self.assertTrue(
            any(
                "hardcodes ORDER_FILLING_IOC" in w
                for w in report.warnings
            )
        )

    def test_ioc_and_fok_prefers_ioc(self):
        report = self._check(
            make_symbol_info(
                filling_mode=SYMBOL_FILLING_IOC | SYMBOL_FILLING_FOK
            )
        )
        self.assertEqual(
            report.resolved_order_filling_mode, mt5.ORDER_FILLING_IOC
        )
        self.assertTrue(report.is_compatible)

    def test_return_allowed_when_execution_mode_permits(self):
        report = self._check(
            make_symbol_info(
                filling_mode=0,
                trade_exemode=mt5.SYMBOL_TRADE_EXECUTION_INSTANT,
            )
        )
        self.assertEqual(
            report.resolved_order_filling_mode, mt5.ORDER_FILLING_RETURN
        )
        self.assertTrue(report.is_compatible)
        self.assertTrue(
            any(
                "hardcodes ORDER_FILLING_IOC" in w
                for w in report.warnings
            )
        )

    def test_return_prohibited_fails_closed(self):
        report = self._check(
            make_symbol_info(
                filling_mode=0,
                trade_exemode=mt5.SYMBOL_TRADE_EXECUTION_MARKET,
            )
        )
        self.assertIsNone(report.resolved_order_filling_mode)
        self.assertFalse(report.is_compatible)
        self.assertTrue(report.blocking_issues)

    def test_no_supported_filling_mode_unknown_execution_fails_closed(self):
        report = self._check(
            make_symbol_info(filling_mode=0, trade_exemode=999)
        )
        self.assertIsNone(report.resolved_order_filling_mode)
        self.assertFalse(report.is_compatible)

    # -- trade mode scenarios -------------------------------------------

    def test_disabled_symbol_is_blocking(self):
        report = self._check(make_symbol_info(trade_mode=0))
        self.assertEqual(report.trade_mode_name, "DISABLED")
        self.assertFalse(report.is_compatible)

    def test_long_only_symbol_warns_not_blocks(self):
        report = self._check(make_symbol_info(trade_mode=1))
        self.assertEqual(report.trade_mode_name, "LONGONLY")
        self.assertTrue(report.is_compatible)
        self.assertTrue(
            any("LONGONLY" in w for w in report.warnings)
        )

    def test_short_only_symbol_warns_not_blocks(self):
        report = self._check(make_symbol_info(trade_mode=2))
        self.assertEqual(report.trade_mode_name, "SHORTONLY")
        self.assertTrue(report.is_compatible)
        self.assertTrue(
            any("SHORTONLY" in w for w in report.warnings)
        )

    def test_close_only_symbol_is_blocking(self):
        report = self._check(make_symbol_info(trade_mode=3))
        self.assertEqual(report.trade_mode_name, "CLOSEONLY")
        self.assertFalse(report.is_compatible)

    def test_unknown_trade_mode_is_blocking(self):
        report = self._check(make_symbol_info(trade_mode=99))
        self.assertFalse(report.is_compatible)

    # -- volume validation -----------------------------------------------

    def test_invalid_volume_min_is_blocking(self):
        report = self._check(make_symbol_info(volume_min=0))
        self.assertFalse(report.is_compatible)

    def test_invalid_volume_step_is_blocking(self):
        report = self._check(make_symbol_info(volume_step=0))
        self.assertFalse(report.is_compatible)

    def test_volume_max_below_volume_min_is_blocking(self):
        report = self._check(
            make_symbol_info(volume_min=1.0, volume_max=0.5)
        )
        self.assertFalse(report.is_compatible)

    # -- stops / freeze levels -------------------------------------------

    def test_negative_stops_level_is_blocking(self):
        report = self._check(make_symbol_info(trade_stops_level=-1))
        self.assertFalse(report.is_compatible)

    def test_positive_stops_level_warns_not_blocks(self):
        report = self._check(make_symbol_info(trade_stops_level=10))
        self.assertTrue(report.is_compatible)
        self.assertTrue(
            any("minimum stop distance" in w for w in report.warnings)
        )

    def test_negative_freeze_level_is_blocking(self):
        report = self._check(make_symbol_info(trade_freeze_level=-1))
        self.assertFalse(report.is_compatible)

    def test_positive_freeze_level_warns_not_blocks(self):
        report = self._check(make_symbol_info(trade_freeze_level=5))
        self.assertTrue(report.is_compatible)
        self.assertTrue(
            any("freeze level" in w for w in report.warnings)
        )

    # -- missing execution mode -------------------------------------------

    def test_missing_execution_mode_is_blocking(self):
        report = self._check(make_symbol_info(trade_exemode=None))
        self.assertFalse(report.is_compatible)
        self.assertTrue(
            any(
                "execution" in issue.lower()
                for issue in report.blocking_issues
            )
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)