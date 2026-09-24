"""
Unit tests for phase_02_optimization/monte_carlo.py.

These use lightweight stand-ins for TradeResult / BacktestResult rather
than importing the real strategy.backtest classes, since Phase 2 must
not depend on strategy internals to be testable in isolation. Swap in
the real classes here once this module is dropped into the actual repo,
to confirm the duck-typed field access (result_status, r_multiple,
entry_time, exit_time) lines up.
"""

import datetime as dt
import unittest

from phase_02_optimization.monte_carlo import (
    MonteCarloConfig,
    estimate_avg_trade_duration_days,
    extract_closed_r_multiples,
    run_monte_carlo,
)


class FakeTradeResult:
    def __init__(
        self,
        result_status,
        r_multiple=None,
        entry_time=None,
        exit_time=None,
    ):
        self.result_status = result_status
        self.r_multiple = r_multiple
        self.entry_time = entry_time
        self.exit_time = exit_time


class FakeBacktestResult:
    def __init__(self, trades):
        self.trades = trades


class FakeStatusEnum:
    """Mimics an Enum member exposing .value, like real SignalStatus."""

    def __init__(self, value):
        self.value = value


class TestExtractClosedRMultiples(unittest.TestCase):
    def test_filters_to_win_loss_only(self):
        trades = [
            FakeTradeResult("WIN", r_multiple=2.0),
            FakeTradeResult("LOSS", r_multiple=-1.0),
            FakeTradeResult("OPEN", r_multiple=None),
            FakeTradeResult("EXPIRED", r_multiple=0.0),
        ]
        result = extract_closed_r_multiples(FakeBacktestResult(trades))
        self.assertEqual(result, [2.0, -1.0])

    def test_handles_enum_like_status(self):
        trades = [
            FakeTradeResult(FakeStatusEnum("win"), r_multiple=1.5),
            FakeTradeResult(FakeStatusEnum("loss"), r_multiple=-1.0),
        ]
        result = extract_closed_r_multiples(FakeBacktestResult(trades))
        self.assertEqual(result, [1.5, -1.0])

    def test_skips_missing_or_invalid_r_multiple(self):
        trades = [
            FakeTradeResult("WIN", r_multiple=None),
            FakeTradeResult("WIN", r_multiple="not-a-number"),
            FakeTradeResult("WIN", r_multiple=3.0),
        ]
        result = extract_closed_r_multiples(FakeBacktestResult(trades))
        self.assertEqual(result, [3.0])

    def test_empty_ledger_returns_empty_list(self):
        result = extract_closed_r_multiples(FakeBacktestResult([]))
        self.assertEqual(result, [])


class TestAvgTradeDuration(unittest.TestCase):
    def test_computes_average_days(self):
        t0 = dt.datetime(2026, 1, 1)
        trades = [
            FakeTradeResult("WIN", 1.0, entry_time=t0, exit_time=t0 + dt.timedelta(days=1)),
            FakeTradeResult("LOSS", -1.0, entry_time=t0, exit_time=t0 + dt.timedelta(days=3)),
        ]
        avg = estimate_avg_trade_duration_days(FakeBacktestResult(trades))
        self.assertAlmostEqual(avg, 2.0)

    def test_returns_none_when_timestamps_missing(self):
        trades = [FakeTradeResult("WIN", 1.0)]
        avg = estimate_avg_trade_duration_days(FakeBacktestResult(trades))
        self.assertIsNone(avg)


class TestRunMonteCarlo(unittest.TestCase):
    def setUp(self):
        # A deliberately lopsided ledger: mostly small wins, one big loss,
        # so drawdown/ruin behavior is meaningfully exercised.
        self.trades = [
            FakeTradeResult("WIN", 1.0),
            FakeTradeResult("WIN", 1.5),
            FakeTradeResult("LOSS", -1.0),
            FakeTradeResult("WIN", 2.0),
            FakeTradeResult("LOSS", -3.0),
            FakeTradeResult("WIN", 1.0),
        ]
        self.ledger = FakeBacktestResult(self.trades)

    def test_reproducible_with_seed(self):
        config = MonteCarloConfig(num_simulations=200, random_seed=42)
        result_a = run_monte_carlo(self.ledger, config)
        result_b = run_monte_carlo(self.ledger, config)
        self.assertEqual(
            [r.final_balance for r in result_a.runs],
            [r.final_balance for r in result_b.runs],
        )

    def test_shuffle_preserves_trade_composition(self):
        config = MonteCarloConfig(
            method="shuffle", num_simulations=50, random_seed=1
        )
        result = run_monte_carlo(self.ledger, config)
        self.assertEqual(result.num_trades_per_run, len(self.trades))
        # Every run's equity curve should have exactly one point per trade
        # plus the starting balance, since shuffle never adds/removes trades.
        for run in result.runs:
            self.assertEqual(len(run.equity_curve), len(self.trades) + 1)

    def test_bootstrap_allows_repeated_trades(self):
        config = MonteCarloConfig(
            method="bootstrap", num_simulations=1, random_seed=7
        )
        result = run_monte_carlo(self.ledger, config)
        self.assertEqual(result.num_trades_per_run, len(self.trades))

    def test_probability_of_ruin_is_between_zero_and_one(self):
        config = MonteCarloConfig(num_simulations=100, random_seed=3)
        result = run_monte_carlo(self.ledger, config)
        self.assertGreaterEqual(result.probability_of_ruin, 0.0)
        self.assertLessEqual(result.probability_of_ruin, 1.0)

    def test_raises_on_empty_ledger(self):
        empty = FakeBacktestResult([])
        with self.assertRaises(ValueError):
            run_monte_carlo(empty, MonteCarloConfig(num_simulations=10))

    def test_accepts_raw_r_multiple_list(self):
        config = MonteCarloConfig(num_simulations=20, random_seed=5)
        result = run_monte_carlo([1.0, -1.0, 2.0, -1.0], config)
        self.assertEqual(result.num_trades_per_run, 4)

    def test_summary_contains_expected_keys(self):
        config = MonteCarloConfig(num_simulations=50, random_seed=9)
        result = run_monte_carlo(self.ledger, config)
        for key in (
            "final_balance",
            "total_return_pct",
            "max_drawdown_pct",
            "max_win_streak",
            "max_loss_streak",
        ):
            self.assertIn(key, result.summary)
            self.assertIn("median", result.summary[key])

    def test_cagr_omitted_without_duration_info(self):
        # FakeTradeResult defaults have no entry_time/exit_time.
        config = MonteCarloConfig(num_simulations=10, random_seed=2)
        result = run_monte_carlo(self.ledger, config)
        self.assertNotIn("cagr_pct", result.summary)
        self.assertTrue(any("cagr_pct omitted" in e for e in result.errors))

    def test_cagr_present_with_explicit_duration(self):
        config = MonteCarloConfig(
            num_simulations=10, random_seed=2, avg_trade_duration_days=5.0
        )
        result = run_monte_carlo(self.ledger, config)
        self.assertIn("cagr_pct", result.summary)


if __name__ == "__main__":
    unittest.main()