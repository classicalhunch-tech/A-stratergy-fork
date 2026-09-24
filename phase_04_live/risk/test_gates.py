
"""
phase_04_live/risk/test_gates.py

Tests for the Phase 4 pre-trade risk gate (gates.py).
"""

import unittest

from phase_04_live.risk.gates import (
    DailyLossLimitConfig,
    DrawdownLimitConfig,
    PositionLimitConfig,
    RiskGateInput,
    evaluate_risk_gate,
)
from strategy.signals import SignalType


def _baseline_input(**overrides) -> RiskGateInput:
    """
    A risk-state input with nothing tripped, overridable per test.
    """

    defaults = dict(
        symbol="EURUSD",
        direction=SignalType.LONG,
        open_positions_count=0,
        open_positions_count_for_symbol=0,
        has_opposite_direction_open=False,
        daily_realized_r=0.0,
        daily_realized_amount=0.0,
        current_drawdown_r=0.0,
        current_drawdown_pct=0.0,
        kill_switch_engaged=False,
    )

    defaults.update(overrides)
    return RiskGateInput(**defaults)


_DEFAULT_POSITION_CONFIG = PositionLimitConfig(
    max_open_positions=1,
    max_open_positions_per_symbol=1,
    allow_hedging=False,
)

_DEFAULT_DAILY_LOSS_CONFIG = DailyLossLimitConfig(
    max_daily_loss_r=3.0,
)

_DEFAULT_DRAWDOWN_CONFIG = DrawdownLimitConfig(
    max_drawdown_r=5.0,
)


def _evaluate(data: RiskGateInput):
    return evaluate_risk_gate(
        data=data,
        position_config=_DEFAULT_POSITION_CONFIG,
        daily_loss_config=_DEFAULT_DAILY_LOSS_CONFIG,
        drawdown_config=_DEFAULT_DRAWDOWN_CONFIG,
    )


# ---------------------------------------------------------------------------
# Clean state
# ---------------------------------------------------------------------------


class TestCleanState(unittest.TestCase):
    def test_all_checks_pass_when_nothing_is_tripped(self):
        result = _evaluate(_baseline_input())

        self.assertTrue(result.allowed)
        self.assertIsNone(result.reason)
        self.assertEqual(result.blocking_issues, ())


# ---------------------------------------------------------------------------
# Kill switch
# ---------------------------------------------------------------------------


class TestKillSwitch(unittest.TestCase):
    def test_engaged_kill_switch_blocks_regardless_of_other_state(self):
        result = _evaluate(
            _baseline_input(
                kill_switch_engaged=True,
            )
        )

        self.assertFalse(result.allowed)
        self.assertIsNotNone(result.reason)
        self.assertIn("kill switch", result.reason.lower())

    def test_kill_switch_is_checked_first(self):
        """
        Kill switch AND position-limit breach both present.

        The reported .reason must be the kill switch because it is
        first in the fixed check order.
        """

        result = _evaluate(
            _baseline_input(
                kill_switch_engaged=True,
                open_positions_count=1,
            )
        )

        self.assertFalse(result.allowed)
        self.assertIn("kill switch", result.reason.lower())
        self.assertEqual(len(result.blocking_issues), 2)

    def test_kill_switch_issue_is_in_blocking_issues(self):
        result = _evaluate(
            _baseline_input(
                kill_switch_engaged=True,
            )
        )

        self.assertFalse(result.allowed)
        self.assertEqual(len(result.blocking_issues), 1)
        self.assertIn(
            "kill switch",
            result.blocking_issues[0].lower(),
        )


# ---------------------------------------------------------------------------
# Position limits
# ---------------------------------------------------------------------------


class TestPositionLimits(unittest.TestCase):
    def test_overall_cap_blocks(self):
        result = _evaluate(
            _baseline_input(
                open_positions_count=1,
            )
        )

        self.assertFalse(result.allowed)
        self.assertIn(
            "Open position cap reached",
            result.reason,
        )

    def test_per_symbol_cap_blocks_even_under_overall_cap(self):
        config = PositionLimitConfig(
            max_open_positions=5,
            max_open_positions_per_symbol=1,
            allow_hedging=False,
        )

        data = _baseline_input(
            open_positions_count=1,
            open_positions_count_for_symbol=1,
        )

        result = evaluate_risk_gate(
            data,
            config,
            _DEFAULT_DAILY_LOSS_CONFIG,
            _DEFAULT_DRAWDOWN_CONFIG,
        )

        self.assertFalse(result.allowed)
        self.assertIn(
            "per-symbol max",
            result.reason,
        )

    def test_hedging_disabled_blocks_opposite_direction(self):
        result = _evaluate(
            _baseline_input(
                has_opposite_direction_open=True,
            )
        )

        self.assertFalse(result.allowed)
        self.assertIn(
            "Opposite-direction",
            result.reason,
        )

    def test_hedging_allowed_permits_opposite_direction(self):
        config = PositionLimitConfig(
            max_open_positions=5,
            max_open_positions_per_symbol=5,
            allow_hedging=True,
        )

        data = _baseline_input(
            has_opposite_direction_open=True,
        )

        result = evaluate_risk_gate(
            data,
            config,
            _DEFAULT_DAILY_LOSS_CONFIG,
            _DEFAULT_DRAWDOWN_CONFIG,
        )

        self.assertTrue(result.allowed)

    def test_none_disables_overall_and_per_symbol_caps(self):
        config = PositionLimitConfig(
            max_open_positions=None,
            max_open_positions_per_symbol=None,
            allow_hedging=True,
        )

        data = _baseline_input(
            open_positions_count=999,
            open_positions_count_for_symbol=999,
        )

        result = evaluate_risk_gate(
            data,
            config,
            _DEFAULT_DAILY_LOSS_CONFIG,
            _DEFAULT_DRAWDOWN_CONFIG,
        )

        self.assertTrue(result.allowed)

    def test_invalid_overall_limit_raises_value_error(self):
        with self.assertRaises(ValueError):
            PositionLimitConfig(
                max_open_positions=0,
            )

    def test_invalid_per_symbol_limit_raises_value_error(self):
        with self.assertRaises(ValueError):
            PositionLimitConfig(
                max_open_positions_per_symbol=0,
            )

    def test_negative_overall_limit_raises_value_error(self):
        with self.assertRaises(ValueError):
            PositionLimitConfig(
                max_open_positions=-1,
            )

    def test_negative_per_symbol_limit_raises_value_error(self):
        with self.assertRaises(ValueError):
            PositionLimitConfig(
                max_open_positions_per_symbol=-1,
            )


# ---------------------------------------------------------------------------
# Daily loss limit
# ---------------------------------------------------------------------------


class TestDailyLossLimit(unittest.TestCase):
    def test_loss_at_exactly_the_limit_blocks(self):
        result = _evaluate(
            _baseline_input(
                daily_realized_r=-3.0,
            )
        )

        self.assertFalse(result.allowed)
        self.assertIn(
            "Daily loss limit",
            result.reason,
        )

    def test_loss_under_the_limit_does_not_block(self):
        result = _evaluate(
            _baseline_input(
                daily_realized_r=-2.9,
            )
        )

        self.assertTrue(result.allowed)

    def test_positive_daily_result_never_blocks(self):
        result = _evaluate(
            _baseline_input(
                daily_realized_r=10.0,
            )
        )

        self.assertTrue(result.allowed)

    def test_amount_based_limit_blocks_independently_of_r_based(self):
        config = DailyLossLimitConfig(
            max_daily_loss_r=None,
            max_daily_loss_amount=500.0,
        )

        data = _baseline_input(
            daily_realized_r=-100.0,
            daily_realized_amount=-500.0,
        )

        result = evaluate_risk_gate(
            data,
            _DEFAULT_POSITION_CONFIG,
            config,
            _DEFAULT_DRAWDOWN_CONFIG,
        )

        self.assertFalse(result.allowed)
        self.assertIn(
            "Daily loss limit",
            result.reason,
        )

    def test_both_daily_limits_can_report_simultaneously(self):
        config = DailyLossLimitConfig(
            max_daily_loss_r=3.0,
            max_daily_loss_amount=500.0,
        )

        data = _baseline_input(
            daily_realized_r=-3.0,
            daily_realized_amount=-500.0,
        )

        result = evaluate_risk_gate(
            data,
            _DEFAULT_POSITION_CONFIG,
            config,
            _DEFAULT_DRAWDOWN_CONFIG,
        )

        self.assertFalse(result.allowed)
        self.assertEqual(len(result.blocking_issues), 2)

    def test_none_disables_daily_loss_check(self):
        config = DailyLossLimitConfig(
            max_daily_loss_r=None,
            max_daily_loss_amount=None,
        )

        data = _baseline_input(
            daily_realized_r=-1000.0,
            daily_realized_amount=-1000.0,
        )

        result = evaluate_risk_gate(
            data,
            _DEFAULT_POSITION_CONFIG,
            config,
            _DEFAULT_DRAWDOWN_CONFIG,
        )

        self.assertTrue(result.allowed)

    def test_missing_r_data_blocks_when_r_limit_enabled(self):
        result = _evaluate(
            _baseline_input(
                daily_realized_r=None,
            )
        )

        self.assertFalse(result.allowed)
        self.assertIn(
            "daily_realized_r is unavailable",
            result.reason,
        )

    def test_missing_amount_data_blocks_when_amount_limit_enabled(self):
        config = DailyLossLimitConfig(
            max_daily_loss_r=None,
            max_daily_loss_amount=500.0,
        )

        data = _baseline_input(
            daily_realized_amount=None,
        )

        result = evaluate_risk_gate(
            data,
            _DEFAULT_POSITION_CONFIG,
            config,
            _DEFAULT_DRAWDOWN_CONFIG,
        )

        self.assertFalse(result.allowed)
        self.assertIn(
            "daily_realized_amount is unavailable",
            result.reason,
        )

    def test_invalid_r_limit_raises_value_error(self):
        with self.assertRaises(ValueError):
            DailyLossLimitConfig(
                max_daily_loss_r=0.0,
            )

    def test_negative_r_limit_raises_value_error(self):
        with self.assertRaises(ValueError):
            DailyLossLimitConfig(
                max_daily_loss_r=-3.0,
            )

    def test_invalid_amount_limit_raises_value_error(self):
        with self.assertRaises(ValueError):
            DailyLossLimitConfig(
                max_daily_loss_amount=0.0,
            )

    def test_negative_amount_limit_raises_value_error(self):
        with self.assertRaises(ValueError):
            DailyLossLimitConfig(
                max_daily_loss_amount=-500.0,
            )


# ---------------------------------------------------------------------------
# Drawdown limit
# ---------------------------------------------------------------------------


class TestDrawdownLimit(unittest.TestCase):
    def test_drawdown_at_exactly_the_limit_blocks(self):
        result = _evaluate(
            _baseline_input(
                current_drawdown_r=5.0,
            )
        )

        self.assertFalse(result.allowed)
        self.assertIn(
            "Drawdown limit",
            result.reason,
        )

    def test_drawdown_under_the_limit_does_not_block(self):
        result = _evaluate(
            _baseline_input(
                current_drawdown_r=4.9,
            )
        )

        self.assertTrue(result.allowed)

    def test_pct_based_limit_blocks_independently_of_r_based(self):
        config = DrawdownLimitConfig(
            max_drawdown_r=None,
            max_drawdown_pct=20.0,
        )

        data = _baseline_input(
            current_drawdown_r=1000.0,
            current_drawdown_pct=20.0,
        )

        result = evaluate_risk_gate(
            data,
            _DEFAULT_POSITION_CONFIG,
            _DEFAULT_DAILY_LOSS_CONFIG,
            config,
        )

        self.assertFalse(result.allowed)
        self.assertIn(
            "Drawdown limit",
            result.reason,
        )

    def test_both_drawdown_limits_can_report_simultaneously(self):
        config = DrawdownLimitConfig(
            max_drawdown_r=5.0,
            max_drawdown_pct=20.0,
        )

        data = _baseline_input(
            current_drawdown_r=5.0,
            current_drawdown_pct=20.0,
        )

        result = evaluate_risk_gate(
            data,
            _DEFAULT_POSITION_CONFIG,
            _DEFAULT_DAILY_LOSS_CONFIG,
            config,
        )

        self.assertFalse(result.allowed)
        self.assertEqual(len(result.blocking_issues), 2)

    def test_none_disables_drawdown_check(self):
        config = DrawdownLimitConfig(
            max_drawdown_r=None,
            max_drawdown_pct=None,
        )

        data = _baseline_input(
            current_drawdown_r=1000.0,
            current_drawdown_pct=1000.0,
        )

        result = evaluate_risk_gate(
            data,
            _DEFAULT_POSITION_CONFIG,
            _DEFAULT_DAILY_LOSS_CONFIG,
            config,
        )

        self.assertTrue(result.allowed)

    def test_missing_r_data_blocks_when_r_limit_enabled(self):
        result = _evaluate(
            _baseline_input(
                current_drawdown_r=None,
            )
        )

        self.assertFalse(result.allowed)
        self.assertIn(
            "current_drawdown_r is unavailable",
            result.reason,
        )

    def test_missing_pct_data_blocks_when_pct_limit_enabled(self):
        config = DrawdownLimitConfig(
            max_drawdown_r=None,
            max_drawdown_pct=20.0,
        )

        data = _baseline_input(
            current_drawdown_pct=None,
        )

        result = evaluate_risk_gate(
            data,
            _DEFAULT_POSITION_CONFIG,
            _DEFAULT_DAILY_LOSS_CONFIG,
            config,
        )

        self.assertFalse(result.allowed)
        self.assertIn(
            "current_drawdown_pct is unavailable",
            result.reason,
        )

    def test_invalid_r_limit_raises_value_error(self):
        with self.assertRaises(ValueError):
            DrawdownLimitConfig(
                max_drawdown_r=0.0,
            )

    def test_negative_r_limit_raises_value_error(self):
        with self.assertRaises(ValueError):
            DrawdownLimitConfig(
                max_drawdown_r=-5.0,
            )

    def test_invalid_pct_limit_raises_value_error(self):
        with self.assertRaises(ValueError):
            DrawdownLimitConfig(
                max_drawdown_pct=0.0,
            )

    def test_negative_pct_limit_raises_value_error(self):
        with self.assertRaises(ValueError):
            DrawdownLimitConfig(
                max_drawdown_pct=-20.0,
            )


# ---------------------------------------------------------------------------
# Combined blocking and ordering
# ---------------------------------------------------------------------------


class TestCombinedBlockingReporting(unittest.TestCase):
    def test_multiple_simultaneous_breaches_are_all_reported(self):
        data = _baseline_input(
            open_positions_count=1,
            daily_realized_r=-5.0,
            current_drawdown_r=10.0,
        )

        result = _evaluate(data)

        self.assertFalse(result.allowed)
        self.assertEqual(len(result.blocking_issues), 3)

        # First blocking check is position limits because the kill
        # switch is not engaged.
        self.assertIn(
            "Open position cap reached",
            result.reason,
        )

    def test_all_four_check_groups_are_reported_in_order(self):
        """
        Trigger one failure in every check group.

        Expected order:
            1. kill switch
            2. position limits
            3. daily loss
            4. drawdown
        """

        data = _baseline_input(
            kill_switch_engaged=True,
            open_positions_count=1,
            daily_realized_r=-3.0,
            current_drawdown_r=5.0,
        )

        result = _evaluate(data)

        self.assertFalse(result.allowed)
        self.assertEqual(len(result.blocking_issues), 4)

        self.assertIn(
            "kill switch",
            result.blocking_issues[0].lower(),
        )

        self.assertIn(
            "Open position cap reached",
            result.blocking_issues[1],
        )

        self.assertIn(
            "Daily loss limit",
            result.blocking_issues[2],
        )

        self.assertIn(
            "Drawdown limit",
            result.blocking_issues[3],
        )

        self.assertEqual(
            result.reason,
            result.blocking_issues[0],
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    unittest.main(verbosity=2)

