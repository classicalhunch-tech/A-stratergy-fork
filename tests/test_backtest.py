"""
tests/test_backtest.py

Test suite for strategy/backtest.py -- the event-driven replay engine.

Design:
- Input-validation tests exercise run_backtest() directly.
- Lifecycle tests isolate the backtest state machine by mocking the
  detection pipeline.
- Lifecycle tests use the real Zone dataclass from strategy.zones,
  rather than a fake zone object.
- The test zone is supplied through _visible_zones(), because the
  production backtest resolves pending signals against visible zones.

FIX (2026-09): zone identity now uses the REAL _stable_cache_key()
from strategy.signals directly, instead of a hand-rolled
_zone_stable_key() reproduction. The previous reproduction could
silently drift out of sync with the production key computation --
if even one field, ordering, or type coercion differed, the
"LOCATE ORIGINAL ZONE" match in run_backtest() would fail on every
bar, the signal would sit PENDING forever, and total_trades_triggered
would stay 0 with no error raised anywhere. Importing and calling the
same function backtest.py actually uses makes that class of bug
structurally impossible: the test and the production code cannot
disagree about the key, because they're now the same code.
"""

import unittest
from unittest.mock import patch

import pandas as pd

from strategy.backtest import run_backtest, BacktestResult
from strategy.signals import SignalType, SignalStatus, TradeSignal, _stable_cache_key
from strategy.zones import Zone, ZoneType


def _make_ohlc(rows):
    """
    Build a minimal valid OHLC DataFrame from:
        (timestamp_str, open, high, low, close)
    tuples.
    """
    index = pd.to_datetime([r[0] for r in rows])

    data = {
        "open": [r[1] for r in rows],
        "high": [r[2] for r in rows],
        "low": [r[3] for r in rows],
        "close": [r[4] for r in rows],
    }

    return pd.DataFrame(data, index=index)


def _make_signal(
    signal_type,
    entry_price,
    stop_loss,
    take_profit,
    setup_bar_index,
    setup_timestamp,
    bar_index,
):
    """
    Create a pending signal born from a structural break.

    The zone relationship is attached by the lifecycle fixture so that
    the test signal resolves against the real Zone instance supplied
    to the backtest replay.
    """
    return TradeSignal(
        signal_type=signal_type,
        entry_price=entry_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        setup_bar_index=setup_bar_index,
        setup_timestamp=setup_timestamp,
        bar_index=bar_index,
        status=SignalStatus.PENDING_RETEST,
    )


class TestBacktestInputValidation(unittest.TestCase):

    def test_none_dataframe_returns_empty_result(self):
        """None input must return an empty BacktestResult, not raise."""
        result = run_backtest(None)

        self.assertIsInstance(result, BacktestResult)
        self.assertEqual(result.trades, [])
        self.assertEqual(result.total_signals_generated, 0)

    def test_empty_dataframe_returns_empty_result(self):
        """A structurally valid but empty DataFrame -> empty result."""
        df = pd.DataFrame(
            columns=["open", "high", "low", "close"],
            index=pd.DatetimeIndex([]),
        )

        result = run_backtest(df)

        self.assertIsInstance(result, BacktestResult)
        self.assertEqual(result.trades, [])

    def test_short_dataframe_returns_empty_result(self):
        """Fewer than 5 rows -> empty result, no exception."""
        rows = [
            ("2024-01-01", 100, 101, 99, 100.5),
            ("2024-01-02", 100.5, 102, 100, 101.5),
        ]

        df = _make_ohlc(rows)
        result = run_backtest(df)

        self.assertEqual(result.trades, [])

    def test_non_dataframe_input_raises_value_error(self):
        """Anything that isn't a DataFrame (and isn't None) must raise."""
        with self.assertRaises(ValueError):
            run_backtest([1, 2, 3])

    def test_missing_required_columns_raises_value_error(self):
        df = pd.DataFrame(
            {
                "open": [1, 2, 3, 4, 5],
                "high": [1, 2, 3, 4, 5],
            },
            index=pd.date_range("2024-01-01", periods=5),
        )

        with self.assertRaises(ValueError):
            run_backtest(df)

    def test_non_datetimeindex_raises_value_error(self):
        rows = [
            (i, 100, 101, 99, 100.5)
            for i in range(5)
        ]

        df = pd.DataFrame(
            {
                "open": [r[1] for r in rows],
                "high": [r[2] for r in rows],
                "low": [r[3] for r in rows],
                "close": [r[4] for r in rows],
            },
            index=[r[0] for r in rows],
        )

        with self.assertRaises(ValueError):
            run_backtest(df)

    def test_column_names_normalized_case_insensitively(self):
        """OPEN/High/CLOSE etc. should be accepted."""
        rows = [
            ("2024-01-01", 100, 101, 99, 100.5),
            ("2024-01-02", 100.5, 102, 100, 101.5),
            ("2024-01-03", 101.5, 103, 101, 102.5),
            ("2024-01-04", 102.5, 104, 102, 103.5),
            ("2024-01-05", 103.5, 105, 103, 104.5),
        ]

        df = _make_ohlc(rows)
        df.columns = ["Open", "HIGH", "Low", "close"]

        result = run_backtest(df)

        self.assertIsInstance(result, BacktestResult)

    def test_negative_max_bars_to_retest_raises_value_error(self):
        rows = [
            ("2024-01-0%d" % i, 100, 101, 99, 100.5)
            for i in range(1, 6)
        ]

        df = _make_ohlc(rows)

        with self.assertRaises(ValueError):
            run_backtest(df, max_bars_to_retest=-1)

    def test_zero_max_bars_to_retest_raises_value_error(self):
        """
        max_bars_to_retest must be > 0.
        """
        rows = [
            ("2024-01-0%d" % i, 100, 101, 99, 100.5)
            for i in range(1, 6)
        ]

        df = _make_ohlc(rows)

        with self.assertRaises(ValueError):
            run_backtest(df, max_bars_to_retest=0)

    def test_zero_reward_multiple_raises_value_error(self):
        rows = [
            ("2024-01-0%d" % i, 100, 101, 99, 100.5)
            for i in range(1, 6)
        ]

        df = _make_ohlc(rows)

        with self.assertRaises(ValueError):
            run_backtest(df, reward_multiple=0)

    def test_negative_stop_buffer_raises_value_error(self):
        rows = [
            ("2024-01-0%d" % i, 100, 101, 99, 100.5)
            for i in range(1, 6)
        ]

        df = _make_ohlc(rows)

        with self.assertRaises(ValueError):
            run_backtest(df, stop_buffer=-1)

    def test_invalid_entry_mode_raises_value_error(self):
        rows = [
            ("2024-01-0%d" % i, 100, 101, 99, 100.5)
            for i in range(1, 6)
        ]

        df = _make_ohlc(rows)

        with self.assertRaises(ValueError):
            run_backtest(df, entry_mode="not_a_real_mode")


class BacktestLifecycleTestCase(unittest.TestCase):
    """
    Base fixture for lifecycle tests.

    The detector pipeline is mocked so these tests focus exclusively
    on the backtest replay engine.

    A REAL Zone dataclass is used. It is deliberately kept outside the
    full detector output and injected through _visible_zones(). This
    isolates pending-signal zone resolution without allowing the
    mitigation replay to mutate the fixture.
    """

    def setUp(self):
        self.test_zone = Zone(
            zone_type=ZoneType.DEMAND,
            price_top=100.0,
            price_bottom=95.0,
            origin_start=pd.Timestamp("2024-01-03"),
            origin_end=pd.Timestamp("2024-01-03"),
            formed_from_break_at=pd.Timestamp("2024-01-04"),
            mitigated=False,
            mitigated_at=None,
        )

        self.patcher = patch.multiple(
            "strategy.backtest",

            find_swings=lambda *a, **kw: [],

            analyze_structure=lambda *a, **kw: (
                [],
                None,
                None,
            ),

            find_zones=lambda *a, **kw: [],

            detect_liquidity_levels=lambda *a, **kw: [],

            update_liquidity_sweeps=lambda *a, **kw: [],

            generate_flip_zone_signals=self._signal_dispatch,

            _current_structure_breaks=lambda *a, **kw: [True],

            _visible_liquidity_levels=lambda *a, **kw: [True],

            # The lifecycle tests provide one real Zone directly to the
            # pending-signal resolver.
            _visible_zones=lambda *a, **kw: [self.test_zone],
        )

        self.patcher.start()
        self.addCleanup(self.patcher.stop)

        # Maps replay bar index -> list[TradeSignal].
        self._signals_by_bar = {}

    def _signal_dispatch(self, **kwargs):
        """
        Stand-in for generate_flip_zone_signals.

        run_backtest supplies the current replay position through
        upto_index=i.

        The signal is linked to the real test zone using the SAME
        _stable_cache_key() function that production backtest logic
        actually calls -- not a hand-written reproduction of it. This
        guarantees the test can never drift out of sync with how
        run_backtest() computes zone identity.
        """
        current_i = int(kwargs["upto_index"])

        signals = self._signals_by_bar.get(current_i, [])

        for signal in signals:
            signal.zone_stable_key = _stable_cache_key(self.test_zone)

        return signals


class TestBacktestLongWinLifecycle(BacktestLifecycleTestCase):

    def test_long_win_lifecycle(self):
        rows = [
            ("2024-01-01", 200, 201, 199, 200),      # 0
            ("2024-01-02", 200, 201, 199, 200),      # 1
            ("2024-01-03", 200, 201, 199, 200),      # 2
            ("2024-01-04", 200, 201, 199, 200),      # 3 sweep
            ("2024-01-05", 200, 201, 199, 200),      # 4 signal born
            ("2024-01-06", 105, 107, 103, 104),      # 5 pending
            ("2024-01-07", 100, 101, 99, 100.5),     # 6 retest
            ("2024-01-08", 104, 112, 101, 110),     # 7 TP
        ]

        df = _make_ohlc(rows)

        signal = _make_signal(
            SignalType.LONG,
            entry_price=100.0,
            stop_loss=95.0,
            take_profit=110.0,
            setup_bar_index=3,
            setup_timestamp=df.index[3],
            bar_index=4,
        )

        self._signals_by_bar[4] = [signal]

        result = run_backtest(
            df,
            max_bars_to_retest=20,
        )

        self.assertEqual(result.total_signals_generated, 1)
        self.assertEqual(result.total_trades_triggered, 1)
        self.assertEqual(len(result.trades), 1)

        trade = result.trades[0]

        self.assertEqual(trade.result_status, "WIN")
        self.assertAlmostEqual(trade.fill_price, 100.0)
        self.assertAlmostEqual(trade.exit_price, 110.0)
        self.assertAlmostEqual(trade.initial_risk, 5.0)
        self.assertAlmostEqual(trade.r_multiple, 2.0)


class TestBacktestLongLossLifecycle(BacktestLifecycleTestCase):

    def test_long_loss_lifecycle(self):
        rows = [
            ("2024-01-01", 200, 201, 199, 200),      # 0
            ("2024-01-02", 200, 201, 199, 200),      # 1
            ("2024-01-03", 200, 201, 199, 200),      # 2
            ("2024-01-04", 200, 201, 199, 200),      # 3 sweep
            ("2024-01-05", 200, 201, 199, 200),      # 4 signal born
            ("2024-01-06", 105, 107, 103, 104),      # 5 pending
            ("2024-01-07", 100, 101, 99, 100.5),     # 6 retest
            ("2024-01-08", 97, 98, 90, 92),         # 7 SL
        ]

        df = _make_ohlc(rows)

        signal = _make_signal(
            SignalType.LONG,
            entry_price=100.0,
            stop_loss=95.0,
            take_profit=110.0,
            setup_bar_index=3,
            setup_timestamp=df.index[3],
            bar_index=4,
        )

        self._signals_by_bar[4] = [signal]

        result = run_backtest(
            df,
            max_bars_to_retest=20,
        )

        self.assertEqual(len(result.trades), 1)

        trade = result.trades[0]

        self.assertEqual(trade.result_status, "LOSS")
        self.assertAlmostEqual(trade.exit_price, 95.0)
        self.assertAlmostEqual(trade.r_multiple, -1.0)


class TestBacktestInvalidation(BacktestLifecycleTestCase):

    def test_invalidation_before_entry_takes_precedence(self):
        rows = [
            ("2024-01-01", 200, 201, 199, 200),      # 0
            ("2024-01-02", 200, 201, 199, 200),      # 1
            ("2024-01-03", 200, 201, 199, 200),      # 2
            ("2024-01-04", 200, 201, 199, 200),      # 3 sweep
            ("2024-01-05", 200, 201, 199, 200),      # 4 signal born
            ("2024-01-06", 100, 101, 90, 95),        # 5 stop breach
        ]

        df = _make_ohlc(rows)

        signal = _make_signal(
            SignalType.LONG,
            entry_price=100.0,
            stop_loss=95.0,
            take_profit=110.0,
            setup_bar_index=3,
            setup_timestamp=df.index[3],
            bar_index=4,
        )

        self._signals_by_bar[4] = [signal]

        result = run_backtest(
            df,
            max_bars_to_retest=20,
        )

        self.assertEqual(result.total_invalidated, 1)
        self.assertEqual(result.total_trades_triggered, 0)
        self.assertEqual(result.trades, [])


class TestBacktestExpiration(BacktestLifecycleTestCase):

    def test_signal_expires_without_retest(self):
        rows = [
            ("2024-01-01", 200, 201, 199, 200),      # 0
            ("2024-01-02", 200, 201, 199, 200),      # 1
            ("2024-01-03", 200, 201, 199, 200),      # 2
            ("2024-01-04", 200, 201, 199, 200),      # 3 sweep
            ("2024-01-05", 200, 201, 199, 200),      # 4 signal born
            ("2024-01-06", 150, 151, 149, 150),      # 5 elapsed=1
            ("2024-01-07", 150, 151, 149, 150),      # 6 elapsed=2
            ("2024-01-08", 150, 151, 149, 150),      # 7 elapsed=3
        ]

        df = _make_ohlc(rows)

        signal = _make_signal(
            SignalType.LONG,
            entry_price=100.0,
            stop_loss=95.0,
            take_profit=110.0,
            setup_bar_index=3,
            setup_timestamp=df.index[3],
            bar_index=4,
        )

        self._signals_by_bar[4] = [signal]

        result = run_backtest(
            df,
            max_bars_to_retest=2,
        )

        self.assertEqual(result.total_expired, 1)
        self.assertEqual(result.total_trades_triggered, 0)
        self.assertEqual(result.trades, [])


class TestBacktestDuplicateSignalProtection(BacktestLifecycleTestCase):

    def test_repeated_signal_registered_only_once(self):
        """
        The same setup can appear repeatedly as the signal engine is
        called on later replay bars.

        The backtest must register the setup once and must not create
        duplicate pending signals or duplicate trades.
        """
        rows = [
            ("2024-01-01", 200, 201, 199, 200),      # 0
            ("2024-01-02", 200, 201, 199, 200),      # 1
            ("2024-01-03", 200, 201, 199, 200),      # 2
            ("2024-01-04", 200, 201, 199, 200),      # 3 sweep
            ("2024-01-05", 200, 201, 199, 200),      # 4 signal born
            ("2024-01-06", 105, 107, 103, 104),      # 5 pending
            ("2024-01-07", 100, 101, 99, 100.5),     # 6 retest
            ("2024-01-08", 104, 112, 101, 110),     # 7 TP
        ]

        df = _make_ohlc(rows)

        signal = _make_signal(
            SignalType.LONG,
            entry_price=100.0,
            stop_loss=95.0,
            take_profit=110.0,
            setup_bar_index=3,
            setup_timestamp=df.index[3],
            bar_index=4,
        )

        self._signals_by_bar[4] = [signal]
        self._signals_by_bar[5] = [signal]
        self._signals_by_bar[6] = [signal]
        self._signals_by_bar[7] = [signal]

        result = run_backtest(
            df,
            max_bars_to_retest=20,
        )

        self.assertEqual(result.total_signals_generated, 1)
        self.assertEqual(result.total_trades_triggered, 1)
        self.assertEqual(len(result.trades), 1)


if __name__ == "__main__":
    unittest.main()