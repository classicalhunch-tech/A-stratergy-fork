"""
phase_03_paper/tests/test_performance.py

PerformanceEngine — Phase 3 trading statistics tests.

PURPOSE
-------
Verify win/loss counting (with breakeven treated as a loss), R-based
statistics (total, average, expectancy, drawdown, streaks), trade
duration averaging, and session/daily/monthly bucketing — all
computed purely from JournalEntry history, without touching
PaperTradeEngine or PositionManager.

A lightweight FakeJournal is used, built from real (frozen)
JournalEntry objects, so PerformanceEngine is tested independently of
how those entries came to exist — consistent with how test_monitoring
uses fakes rather than a fully wired runtime.
"""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from phase_03_paper.journal.journal import JournalEntry
from phase_03_paper.performance.performance import (
    BucketStats,
    PerformanceEngine,
    PerformanceSnapshot,
)
from phase_03_paper.positions.manager import PositionState


# ====================================================================
# TEST DOUBLE
# ====================================================================


class FakeJournal:
    """Minimal stand-in for Journal exposing only the read queries
    PerformanceEngine relies on."""

    def __init__(self, entries: List[JournalEntry]) -> None:
        self._entries = entries

    def closed_entries(self) -> List[JournalEntry]:
        return [
            e for e in self._entries if e.position_state == PositionState.CLOSED
        ]

    def entries_for_trade(self, trade_id: str) -> List[JournalEntry]:
        return [e for e in self._entries if e.trade_id == trade_id]


# ====================================================================
# HELPERS
# ====================================================================


def _open_entry(
    trade_id: str,
    *,
    session: str = "LONDON",
    recorded_at: Optional[datetime] = None,
    direction: str = "LONG",
) -> JournalEntry:
    return JournalEntry(
        trade_id=trade_id,
        session=session,
        direction=direction,
        position_state=PositionState.OPEN,
        recorded_at=recorded_at,
        entry=1.1000,
        stop=1.0950,
        target=1.1100,
    )


def _closed_entry(
    trade_id: str,
    *,
    session: str = "LONDON",
    recorded_at: Optional[datetime] = None,
    direction: str = "LONG",
    result_r: Optional[float] = None,
) -> JournalEntry:
    return JournalEntry(
        trade_id=trade_id,
        session=session,
        direction=direction,
        position_state=PositionState.CLOSED,
        recorded_at=recorded_at,
        entry=1.1000,
        stop=1.0950,
        target=1.1100,
        exit_price=1.1050,
        exit_reason=None,
        result_r=result_r,
    )


# ====================================================================
# TESTS
# ====================================================================


class TestPerformanceEngine(unittest.TestCase):
    """Test suite for PerformanceEngine.calculate()."""

    def setUp(self) -> None:
        self.t0 = datetime(2026, 1, 5, 9, 0, tzinfo=timezone.utc)

    def _times(self, *offsets_minutes: int) -> List[datetime]:
        return [self.t0 + timedelta(minutes=m) for m in offsets_minutes]

    # ----------------------------------------------------------
    # Empty state
    # ----------------------------------------------------------

    def test_no_closed_trades_returns_empty_snapshot(self) -> None:
        """With no closed trades, all stats are zero/None, not errors."""

        journal = FakeJournal([])
        engine = PerformanceEngine(journal)

        snapshot = engine.calculate(now=self.t0)

        self.assertIsInstance(snapshot, PerformanceSnapshot)
        self.assertEqual(snapshot.total_trades, 0)
        self.assertIsNone(snapshot.win_rate)
        self.assertIsNone(snapshot.average_r)
        self.assertIsNone(snapshot.expectancy)
        self.assertEqual(snapshot.max_drawdown_r, 0.0)
        self.assertEqual(snapshot.current_streak, 0)
        self.assertEqual(snapshot.session_performance, {})

    # ----------------------------------------------------------
    # Win/loss counting
    # ----------------------------------------------------------

    def test_win_rate_and_totals_computed_correctly(self) -> None:
        """Two wins and one loss produce correct counts, win rate, and total R."""

        t1, t2, t3 = self._times(0, 5, 10)
        entries = [
            _closed_entry("t1", recorded_at=t1, result_r=2.0),
            _closed_entry("t2", recorded_at=t2, result_r=1.5),
            _closed_entry("t3", recorded_at=t3, result_r=-1.0),
        ]
        engine = PerformanceEngine(FakeJournal(entries))

        snapshot = engine.calculate(now=self.t0)

        self.assertEqual(snapshot.total_trades, 3)
        self.assertEqual(snapshot.wins, 2)
        self.assertEqual(snapshot.losses, 1)
        self.assertAlmostEqual(snapshot.win_rate, 2 / 3)
        self.assertAlmostEqual(snapshot.total_r, 2.5)
        self.assertAlmostEqual(snapshot.average_r, 2.5 / 3)

    def test_breakeven_trade_counted_as_loss(self) -> None:
        """A trade with result_r == 0 counts toward losses, not wins."""

        t1, t2 = self._times(0, 5)
        entries = [
            _closed_entry("t1", recorded_at=t1, result_r=2.0),
            _closed_entry("t2", recorded_at=t2, result_r=0.0),
        ]
        engine = PerformanceEngine(FakeJournal(entries))

        snapshot = engine.calculate(now=self.t0)

        self.assertEqual(snapshot.wins, 1)
        self.assertEqual(snapshot.losses, 1)

    def test_closed_entry_missing_result_r_excluded(self) -> None:
        """A CLOSED entry with result_r=None is excluded from statistics."""

        t1, t2 = self._times(0, 5)
        entries = [
            _closed_entry("t1", recorded_at=t1, result_r=1.0),
            _closed_entry("t2", recorded_at=t2, result_r=None),
        ]
        engine = PerformanceEngine(FakeJournal(entries))

        snapshot = engine.calculate(now=self.t0)

        self.assertEqual(snapshot.total_trades, 1)
        self.assertAlmostEqual(snapshot.total_r, 1.0)

    # ----------------------------------------------------------
    # Expectancy
    # ----------------------------------------------------------

    def test_expectancy_computed_correctly(self) -> None:
        """Expectancy matches (win_rate*avg_win) + (loss_rate*avg_loss)."""

        t1, t2, t3, t4 = self._times(0, 5, 10, 15)
        entries = [
            _closed_entry("t1", recorded_at=t1, result_r=2.0),
            _closed_entry("t2", recorded_at=t2, result_r=2.0),
            _closed_entry("t3", recorded_at=t3, result_r=-1.0),
            _closed_entry("t4", recorded_at=t4, result_r=-1.0),
        ]
        engine = PerformanceEngine(FakeJournal(entries))

        snapshot = engine.calculate(now=self.t0)

        # win_rate=0.5, avg_win=2.0, loss_rate=0.5, avg_loss=-1.0
        expected = (0.5 * 2.0) + (0.5 * -1.0)
        self.assertAlmostEqual(snapshot.expectancy, expected)

    # ----------------------------------------------------------
    # Drawdown
    # ----------------------------------------------------------

    def test_max_drawdown_computed_correctly(self) -> None:
        """Drawdown is the largest peak-to-trough drop in cumulative R."""

        times = self._times(0, 5, 10, 15, 20)
        # Cumulative R path: +3, +5 (peak), +2, -1, +1
        # -> drawdown from peak 5 down to -1 => 6
        results = [3.0, 2.0, -3.0, -3.0, 2.0]
        entries = [
            _closed_entry(f"t{i}", recorded_at=times[i], result_r=results[i])
            for i in range(5)
        ]
        engine = PerformanceEngine(FakeJournal(entries))

        snapshot = engine.calculate(now=self.t0)

        self.assertAlmostEqual(snapshot.max_drawdown_r, 6.0)

    # ----------------------------------------------------------
    # Streaks
    # ----------------------------------------------------------

    def test_current_streak_positive_on_win_streak(self) -> None:
        """Two consecutive wins at the end produce current_streak == 2."""

        times = self._times(0, 5, 10)
        results = [-1.0, 1.0, 1.0]
        entries = [
            _closed_entry(f"t{i}", recorded_at=times[i], result_r=results[i])
            for i in range(3)
        ]
        engine = PerformanceEngine(FakeJournal(entries))

        snapshot = engine.calculate(now=self.t0)

        self.assertEqual(snapshot.current_streak, 2)

    def test_current_streak_negative_on_loss_streak(self) -> None:
        """Two consecutive losses at the end produce current_streak == -2."""

        times = self._times(0, 5, 10)
        results = [1.0, -1.0, -0.5]
        entries = [
            _closed_entry(f"t{i}", recorded_at=times[i], result_r=results[i])
            for i in range(3)
        ]
        engine = PerformanceEngine(FakeJournal(entries))

        snapshot = engine.calculate(now=self.t0)

        self.assertEqual(snapshot.current_streak, -2)

    def test_best_and_worst_streaks_tracked_across_full_history(self) -> None:
        """best_winning_streak and worst_losing_streak reflect the full history,
        not just the tail."""

        times = self._times(0, 5, 10, 15, 20, 25, 30)
        # W W W L L W L  -> best win streak = 3, worst loss streak = 2
        results = [1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0]
        entries = [
            _closed_entry(f"t{i}", recorded_at=times[i], result_r=results[i])
            for i in range(len(results))
        ]
        engine = PerformanceEngine(FakeJournal(entries))

        snapshot = engine.calculate(now=self.t0)

        self.assertEqual(snapshot.best_winning_streak, 3)
        self.assertEqual(snapshot.worst_losing_streak, 2)
        self.assertEqual(snapshot.current_streak, -1)

    # ----------------------------------------------------------
    # Duration
    # ----------------------------------------------------------

    def test_average_trade_duration_computed(self) -> None:
        """Average duration is computed from each trade's OPEN -> CLOSED gap."""

        open_t1, close_t1 = self._times(0, 10)  # 10 min
        open_t2, close_t2 = self._times(20, 50)  # 30 min

        entries = [
            _open_entry("t1", recorded_at=open_t1),
            _closed_entry("t1", recorded_at=close_t1, result_r=1.0),
            _open_entry("t2", recorded_at=open_t2),
            _closed_entry("t2", recorded_at=close_t2, result_r=-1.0),
        ]
        engine = PerformanceEngine(FakeJournal(entries))

        snapshot = engine.calculate(now=self.t0)

        expected_avg_seconds = ((10 * 60) + (30 * 60)) / 2
        self.assertAlmostEqual(
            snapshot.avg_trade_duration_seconds,
            expected_avg_seconds,
        )

    def test_trade_missing_open_entry_excluded_from_duration(self) -> None:
        """A closed trade with no matching OPEN entry doesn't break duration calc."""

        close_t1 = self.t0 + timedelta(minutes=10)

        entries = [
            _closed_entry("orphan", recorded_at=close_t1, result_r=1.0),
        ]
        engine = PerformanceEngine(FakeJournal(entries))

        snapshot = engine.calculate(now=self.t0)

        self.assertIsNone(snapshot.avg_trade_duration_seconds)

    # ----------------------------------------------------------
    # Bucketing
    # ----------------------------------------------------------

    def test_session_performance_bucketed_correctly(self) -> None:
        """Trades are grouped into correct per-session BucketStats."""

        t1, t2, t3 = self._times(0, 5, 10)
        entries = [
            _closed_entry("t1", session="LONDON", recorded_at=t1, result_r=1.0),
            _closed_entry("t2", session="LONDON", recorded_at=t2, result_r=-1.0),
            _closed_entry("t3", session="NY", recorded_at=t3, result_r=2.0),
        ]
        engine = PerformanceEngine(FakeJournal(entries))

        snapshot = engine.calculate(now=self.t0)

        self.assertIn("LONDON", snapshot.session_performance)
        self.assertIn("NY", snapshot.session_performance)

        london = snapshot.session_performance["LONDON"]
        self.assertIsInstance(london, BucketStats)
        self.assertEqual(london.trades, 2)
        self.assertEqual(london.wins, 1)
        self.assertEqual(london.losses, 1)
        self.assertAlmostEqual(london.total_r, 0.0)

        ny = snapshot.session_performance["NY"]
        self.assertEqual(ny.trades, 1)
        self.assertEqual(ny.wins, 1)

    def test_daily_performance_bucketed_by_close_date(self) -> None:
        """Trades are grouped by the ISO date of their CLOSED entry."""

        day1 = datetime(2026, 1, 5, 9, 0, tzinfo=timezone.utc)
        day1_later = datetime(2026, 1, 5, 15, 0, tzinfo=timezone.utc)
        day2 = datetime(2026, 1, 6, 9, 0, tzinfo=timezone.utc)

        entries = [
            _closed_entry("t1", recorded_at=day1, result_r=1.0),
            _closed_entry("t2", recorded_at=day1_later, result_r=1.0),
            _closed_entry("t3", recorded_at=day2, result_r=-1.0),
        ]
        engine = PerformanceEngine(FakeJournal(entries))

        snapshot = engine.calculate(now=self.t0)

        self.assertIn("2026-01-05", snapshot.daily_performance)
        self.assertIn("2026-01-06", snapshot.daily_performance)
        self.assertEqual(snapshot.daily_performance["2026-01-05"].trades, 2)
        self.assertEqual(snapshot.daily_performance["2026-01-06"].trades, 1)

    def test_monthly_performance_bucketed_by_close_month(self) -> None:
        """Trades are grouped by the YYYY-MM of their CLOSED entry."""

        jan = datetime(2026, 1, 20, 9, 0, tzinfo=timezone.utc)
        feb = datetime(2026, 2, 3, 9, 0, tzinfo=timezone.utc)

        entries = [
            _closed_entry("t1", recorded_at=jan, result_r=1.0),
            _closed_entry("t2", recorded_at=feb, result_r=-1.0),
        ]
        engine = PerformanceEngine(FakeJournal(entries))

        snapshot = engine.calculate(now=self.t0)

        self.assertIn("2026-01", snapshot.monthly_performance)
        self.assertIn("2026-02", snapshot.monthly_performance)
        self.assertEqual(snapshot.monthly_performance["2026-01"].trades, 1)
        self.assertEqual(snapshot.monthly_performance["2026-02"].trades, 1)

    # ----------------------------------------------------------
    # Immutability
    # ----------------------------------------------------------

    def test_snapshot_is_frozen(self) -> None:
        """PerformanceSnapshot is immutable — mutating it must fail."""

        entries = [
            _closed_entry("t1", recorded_at=self.t0, result_r=1.0),
        ]
        engine = PerformanceEngine(FakeJournal(entries))

        snapshot = engine.calculate(now=self.t0)

        with self.assertRaises(Exception):
            snapshot.total_trades = 99  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()