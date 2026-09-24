"""
phase_03_paper/tests/test_replay_runner.py

Historical Replay — orchestration tests.

PURPOSE
-------
Verify run_historical_replay() drives every CSV candle through the
coordinator in file order, passes each candle's OWN timestamp as
`now` (never wall-clock time), and correctly wires Journal/
MonitoringEngine/PerformanceEngine when supplied.

A FakeReplayCoordinator is used instead of a fully wired
RuntimeCoordinator (which would require SessionEngine,
NotificationEngine, and StrategyAdapter) — this keeps replay
orchestration independently testable, per project rule #7. The fake
satisfies both the Stage2Coordinator protocol (tick/triggered_events)
and MonitoringEngine's expected status() method, mirroring exactly
what the real RuntimeCoordinator provides.
"""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from phase_03_paper.config import Phase3Config
from phase_03_paper.events.audit import AuditLog
from phase_03_paper.journal.journal import Journal
from phase_03_paper.market.engine import Candle, MarketDataEngine
from phase_03_paper.monitoring.monitoring import MonitoringEngine
from phase_03_paper.performance.performance import PerformanceEngine
from phase_03_paper.replay.replay_runner import ReplaySummary, run_historical_replay
from phase_03_paper.runtime.coordinator import RuntimeStatus
from phase_03_paper.trading.paper_engine import PaperTradeEngine


# ====================================================================
# TEST DOUBLES
# ====================================================================


class FakeReplayCoordinator:
    """
    Minimal Stage2Coordinator that also exposes status(), so it can
    additionally back a MonitoringEngine — exactly like the real
    RuntimeCoordinator does, but without requiring SessionEngine,
    NotificationEngine, or StrategyAdapter.
    """

    def __init__(self, engine: MarketDataEngine) -> None:
        self._engine = engine
        self._tick_count = 0
        self._last_status: Optional[RuntimeStatus] = None
        self.seen_now_values: List[datetime] = []

    def tick(self, now: Optional[datetime] = None) -> RuntimeStatus:
        self.seen_now_values.append(now)

        candle = self._engine.next_candle()
        self._tick_count += 1

        status = RuntimeStatus(
            tick_count=self._tick_count,
            last_tick_at=now,
            last_error=None,
            latest_candle=candle,
        )
        self._last_status = status
        return status

    def triggered_events(self) -> list:
        return []

    def status(self) -> RuntimeStatus:
        assert self._last_status is not None
        return self._last_status


class FakeJournal:
    """Minimal Journal stand-in that counts record() calls."""

    def __init__(self) -> None:
        self.record_calls = 0

    def record(self, paper_engine, current_candle_timestamp=None):
        self.record_calls += 1
        return []


# ====================================================================
# HELPERS
# ====================================================================


def _write_csv(tmpdir: str, rows: List[str]) -> str:
    path = Path(tmpdir) / "data.csv"
    header = "timestamp,open,high,low,close,volume\n"
    path.write_text(header + "\n".join(rows) + "\n", encoding="utf-8")
    return str(path)


def _build_paper_engine() -> PaperTradeEngine:
    return PaperTradeEngine(config=Phase3Config(), audit_log=AuditLog())


class TestReplayRunner(unittest.TestCase):
    """Test suite for run_historical_replay()."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)

        self.csv_rows = [
            "2026-01-05T09:00:00+00:00,1.1000,1.1050,1.0950,1.1010,100",
            "2026-01-05T09:05:00+00:00,1.1010,1.1060,1.0990,1.1020,120",
            "2026-01-05T09:10:00+00:00,1.1020,1.1070,1.1000,1.1030,90",
        ]
        self.csv_path = _write_csv(self._tmpdir.name, self.csv_rows)

    # ----------------------------------------------------------
    # Core orchestration
    # ----------------------------------------------------------

    def test_replay_processes_all_candles_in_order(self) -> None:
        """Every CSV row produces exactly one tick, in file order."""

        paper_engine = _build_paper_engine()

        summary = run_historical_replay(
            self.csv_path,
            coordinator_factory=FakeReplayCoordinator,
            paper_engine=paper_engine,
        )

        self.assertIsInstance(summary, ReplaySummary)
        self.assertEqual(summary.total_candles, 3)
        self.assertEqual(summary.ticks_run, 3)
        self.assertEqual(
            summary.first_timestamp,
            datetime.fromisoformat("2026-01-05T09:00:00+00:00"),
        )
        self.assertEqual(
            summary.last_timestamp,
            datetime.fromisoformat("2026-01-05T09:10:00+00:00"),
        )

    def test_replay_passes_candle_timestamp_as_now_not_wallclock(self) -> None:
        """coordinator.tick() receives each candle's OWN timestamp as `now`,
        never real wall-clock time — critical for correct session evaluation."""

        paper_engine = _build_paper_engine()
        captured_coordinator: List[FakeReplayCoordinator] = []

        def factory(engine: MarketDataEngine) -> FakeReplayCoordinator:
            coordinator = FakeReplayCoordinator(engine)
            captured_coordinator.append(coordinator)
            return coordinator

        run_historical_replay(
            self.csv_path,
            coordinator_factory=factory,
            paper_engine=paper_engine,
        )

        coordinator = captured_coordinator[0]
        expected = [
            datetime.fromisoformat("2026-01-05T09:00:00+00:00"),
            datetime.fromisoformat("2026-01-05T09:05:00+00:00"),
            datetime.fromisoformat("2026-01-05T09:10:00+00:00"),
        ]
        self.assertEqual(coordinator.seen_now_values, expected)

    def test_replay_with_no_optional_components_still_works(self) -> None:
        """journal/monitoring_factory/performance are all optional."""

        paper_engine = _build_paper_engine()

        summary = run_historical_replay(
            self.csv_path,
            coordinator_factory=FakeReplayCoordinator,
            paper_engine=paper_engine,
        )

        self.assertIsNone(summary.final_health)
        self.assertIsNone(summary.final_performance)
        self.assertEqual(summary.unhealthy_checks, [])

    # ----------------------------------------------------------
    # Journal integration
    # ----------------------------------------------------------

    def test_replay_calls_journal_record_each_tick(self) -> None:
        """journal.record() is called exactly once per candle processed."""

        paper_engine = _build_paper_engine()
        journal = FakeJournal()

        run_historical_replay(
            self.csv_path,
            coordinator_factory=FakeReplayCoordinator,
            paper_engine=paper_engine,
            journal=journal,
        )

        self.assertEqual(journal.record_calls, 3)

    # ----------------------------------------------------------
    # Monitoring integration
    # ----------------------------------------------------------

    def test_replay_integrates_monitoring_via_factory(self) -> None:
        """monitoring_factory receives the real coordinator instance, and
        final_health reflects the last simulated-time check."""

        paper_engine = _build_paper_engine()

        summary = run_historical_replay(
            self.csv_path,
            coordinator_factory=FakeReplayCoordinator,
            paper_engine=paper_engine,
            monitoring_factory=lambda coordinator: MonitoringEngine(coordinator),
        )

        self.assertIsNotNone(summary.final_health)
        self.assertTrue(summary.final_health.healthy)
        self.assertEqual(
            summary.final_health.last_tick_at,
            datetime.fromisoformat("2026-01-05T09:10:00+00:00"),
        )

    # ----------------------------------------------------------
    # Performance integration
    # ----------------------------------------------------------

    def test_replay_integrates_performance(self) -> None:
        """A pre-built PerformanceEngine (backed by a real Journal) produces
        a final snapshot computed after all candles are processed."""

        paper_engine = _build_paper_engine()
        journal = Journal()
        performance = PerformanceEngine(journal)

        summary = run_historical_replay(
            self.csv_path,
            coordinator_factory=FakeReplayCoordinator,
            paper_engine=paper_engine,
            journal=journal,
            performance=performance,
        )

        self.assertIsNotNone(summary.final_performance)
        # No signals are ever triggered by FakeReplayCoordinator, so no
        # trades open or close — an empty but valid snapshot.
        self.assertEqual(summary.final_performance.total_trades, 0)


if __name__ == "__main__":
    unittest.main()