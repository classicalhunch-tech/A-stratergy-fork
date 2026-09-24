"""
phase_03_paper/tests/test_monitoring.py

MonitoringEngine — Phase 3 health-check tests.

PURPOSE
-------
Verify that MonitoringEngine correctly reports runtime liveness,
runtime errors, new adapter errors (once, not repeatedly), pending
signal counts, new journal activity, and persistence health — while
never mutating the components it observes.

Lightweight fakes are used instead of a fully wired RuntimeCoordinator
(which would require SessionEngine, NotificationEngine, and a real
MarketDataEngine). This keeps MonitoringEngine independently testable,
per project rule #7, and matches how the coordinator/adapter/journal
expose their state through simple properties and a status() snapshot.
"""

from __future__ import annotations

import unittest
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from phase_03_paper.market.engine import Candle
from phase_03_paper.monitoring.monitoring import HealthSnapshot, MonitoringEngine
from phase_03_paper.runtime.coordinator import RuntimeStatus


# ====================================================================
# TEST DOUBLES
# ====================================================================


def _make_candle(timestamp: datetime) -> Candle:
    """Create a minimal deterministic candle for status snapshots."""

    return Candle(
        timestamp=timestamp,
        open=1.1000,
        high=1.1050,
        low=1.0950,
        close=1.1010,
    )


class FakeCoordinator:
    """Minimal stand-in for RuntimeCoordinator exposing only status()."""

    def __init__(self, status: RuntimeStatus) -> None:
        self._status = status

    def status(self) -> RuntimeStatus:
        return self._status


class FakeAdapter:
    """Minimal stand-in for StrategyAdapter's public diagnostics."""

    def __init__(
        self,
        errors: Optional[List[str]] = None,
        pending_count: int = 0,
        candle_count: int = 0,
    ) -> None:
        self.errors = errors if errors is not None else []
        self.pending_count = pending_count
        self.candle_count = candle_count


class FakeJournal:
    """Minimal stand-in for Journal exposing only all_entries()."""

    def __init__(self, entry_count: int = 0) -> None:
        self._entry_count = entry_count

    def all_entries(self) -> list:
        # MonitoringEngine only ever calls len(...) on this, so the
        # contents don't matter — only the count.
        return [object()] * self._entry_count

    def set_entry_count(self, count: int) -> None:
        self._entry_count = count


# ====================================================================
# TESTS
# ====================================================================


class TestMonitoringEngine(unittest.TestCase):
    """Test suite for MonitoringEngine.check_health()."""

    def setUp(self) -> None:
        self.t0 = datetime(2026, 1, 5, 9, 0, tzinfo=timezone.utc)

    def _status(
        self,
        *,
        tick_count: int = 1,
        last_tick_at: Optional[datetime] = None,
        last_error: Optional[str] = None,
        latest_candle: Optional[Candle] = None,
    ) -> RuntimeStatus:
        return RuntimeStatus(
            tick_count=tick_count,
            last_tick_at=last_tick_at,
            last_error=last_error,
            latest_candle=latest_candle,
        )

    # ----------------------------------------------------------
    # Runtime liveness
    # ----------------------------------------------------------

    def test_healthy_when_recent_tick_no_errors(self) -> None:
        """A recent tick with no runtime error produces a healthy snapshot."""

        status = self._status(
            last_tick_at=self.t0,
            latest_candle=_make_candle(self.t0),
        )
        coordinator = FakeCoordinator(status)
        monitoring = MonitoringEngine(coordinator)

        snapshot = monitoring.check_health(
            now=self.t0 + timedelta(seconds=5)
        )

        self.assertTrue(snapshot.runtime_alive)
        self.assertTrue(snapshot.healthy)
        self.assertEqual(snapshot.issues, ())

    def test_unhealthy_when_never_ticked(self) -> None:
        """last_tick_at=None is reported as an issue, not assumed healthy."""

        status = self._status(tick_count=0, last_tick_at=None)
        coordinator = FakeCoordinator(status)
        monitoring = MonitoringEngine(coordinator)

        snapshot = monitoring.check_health(now=self.t0)

        self.assertFalse(snapshot.runtime_alive)
        self.assertFalse(snapshot.healthy)
        self.assertIn("runtime has not ticked yet", snapshot.issues)

    def test_unhealthy_when_tick_stale(self) -> None:
        """A tick older than max_tick_gap_seconds is reported as stale."""

        status = self._status(last_tick_at=self.t0)
        coordinator = FakeCoordinator(status)
        monitoring = MonitoringEngine(
            coordinator,
            max_tick_gap_seconds=30,
        )

        snapshot = monitoring.check_health(
            now=self.t0 + timedelta(seconds=90)
        )

        self.assertFalse(snapshot.runtime_alive)
        self.assertFalse(snapshot.healthy)
        self.assertTrue(
            any("stale" in issue for issue in snapshot.issues)
        )

    def test_runtime_error_reported(self) -> None:
        """A non-None last_error on the coordinator status is surfaced."""

        status = self._status(
            last_tick_at=self.t0,
            last_error="RuntimeError('boom')",
        )
        coordinator = FakeCoordinator(status)
        monitoring = MonitoringEngine(coordinator)

        snapshot = monitoring.check_health(
            now=self.t0 + timedelta(seconds=1)
        )

        self.assertFalse(snapshot.healthy)
        self.assertTrue(
            any("RuntimeError" in issue for issue in snapshot.issues)
        )

    def test_last_candle_at_reflects_latest_candle_timestamp(self) -> None:
        """last_candle_at mirrors the coordinator's latest candle timestamp."""

        candle = _make_candle(self.t0)
        status = self._status(last_tick_at=self.t0, latest_candle=candle)
        coordinator = FakeCoordinator(status)
        monitoring = MonitoringEngine(coordinator)

        snapshot = monitoring.check_health(
            now=self.t0 + timedelta(seconds=1)
        )

        self.assertEqual(snapshot.last_candle_at, self.t0)

    # ----------------------------------------------------------
    # Strategy adapter
    # ----------------------------------------------------------

    def test_new_adapter_errors_detected_once(self) -> None:
        """
        New adapter errors are reported the first time they're seen,
        and not reported again on a subsequent check with no change.
        """

        status = self._status(last_tick_at=self.t0)
        coordinator = FakeCoordinator(status)
        adapter = FakeAdapter(errors=["0: signal discovery error: X"])
        monitoring = MonitoringEngine(coordinator, strategy_adapter=adapter)

        first = monitoring.check_health(now=self.t0 + timedelta(seconds=1))
        self.assertEqual(first.new_adapter_errors, 1)
        self.assertFalse(first.healthy)

        second = monitoring.check_health(now=self.t0 + timedelta(seconds=2))
        self.assertEqual(second.new_adapter_errors, 0)
        self.assertTrue(second.healthy)
        self.assertEqual(second.adapter_error_count, 1)

    def test_additional_adapter_errors_counted_incrementally(self) -> None:
        """A second, later error is reported as exactly one new error."""

        status = self._status(last_tick_at=self.t0)
        coordinator = FakeCoordinator(status)
        adapter = FakeAdapter(errors=["err1"])
        monitoring = MonitoringEngine(coordinator, strategy_adapter=adapter)

        monitoring.check_health(now=self.t0 + timedelta(seconds=1))

        adapter.errors.append("err2")
        snapshot = monitoring.check_health(now=self.t0 + timedelta(seconds=2))

        self.assertEqual(snapshot.new_adapter_errors, 1)
        self.assertEqual(snapshot.adapter_error_count, 2)

    def test_pending_signal_count_reflected(self) -> None:
        """pending_signal_count mirrors the adapter's pending_count."""

        status = self._status(last_tick_at=self.t0)
        coordinator = FakeCoordinator(status)
        adapter = FakeAdapter(pending_count=3, candle_count=42)
        monitoring = MonitoringEngine(coordinator, strategy_adapter=adapter)

        snapshot = monitoring.check_health(now=self.t0 + timedelta(seconds=1))

        self.assertEqual(snapshot.pending_signal_count, 3)
        self.assertEqual(snapshot.candle_count, 42)

    def test_no_adapter_supplied_defaults_to_zero(self) -> None:
        """Without a strategy_adapter, adapter fields default safely to zero."""

        status = self._status(last_tick_at=self.t0)
        coordinator = FakeCoordinator(status)
        monitoring = MonitoringEngine(coordinator)

        snapshot = monitoring.check_health(now=self.t0 + timedelta(seconds=1))

        self.assertEqual(snapshot.adapter_error_count, 0)
        self.assertEqual(snapshot.new_adapter_errors, 0)
        self.assertEqual(snapshot.pending_signal_count, 0)
        self.assertEqual(snapshot.candle_count, 0)

    # ----------------------------------------------------------
    # Journal
    # ----------------------------------------------------------

    def test_new_journal_entries_detected_incrementally(self) -> None:
        """new_journal_entries reflects only entries added since last check."""

        status = self._status(last_tick_at=self.t0)
        coordinator = FakeCoordinator(status)
        journal = FakeJournal(entry_count=2)
        monitoring = MonitoringEngine(coordinator, journal=journal)

        first = monitoring.check_health(now=self.t0 + timedelta(seconds=1))
        self.assertEqual(first.new_journal_entries, 2)

        second = monitoring.check_health(now=self.t0 + timedelta(seconds=2))
        self.assertEqual(second.new_journal_entries, 0)

        journal.set_entry_count(5)
        third = monitoring.check_health(now=self.t0 + timedelta(seconds=3))
        self.assertEqual(third.new_journal_entries, 3)

    # ----------------------------------------------------------
    # Persistence
    # ----------------------------------------------------------

    def test_persistence_check_success(self) -> None:
        """A persistence_check returning True is reported as healthy."""

        status = self._status(last_tick_at=self.t0)
        coordinator = FakeCoordinator(status)
        monitoring = MonitoringEngine(
            coordinator,
            persistence_check=lambda: True,
        )

        snapshot = monitoring.check_health(now=self.t0 + timedelta(seconds=1))

        self.assertTrue(snapshot.persistence_ok)
        self.assertTrue(snapshot.healthy)

    def test_persistence_check_failure_reports_issue(self) -> None:
        """A persistence_check returning False is reported as an issue."""

        status = self._status(last_tick_at=self.t0)
        coordinator = FakeCoordinator(status)
        monitoring = MonitoringEngine(
            coordinator,
            persistence_check=lambda: False,
        )

        snapshot = monitoring.check_health(now=self.t0 + timedelta(seconds=1))

        self.assertFalse(snapshot.persistence_ok)
        self.assertFalse(snapshot.healthy)
        self.assertIn("persistence check failed", snapshot.issues)

    def test_persistence_check_exception_reports_issue(self) -> None:
        """A persistence_check that raises is treated as unhealthy, not crashed."""

        def _boom() -> bool:
            raise ConnectionError("db locked")

        status = self._status(last_tick_at=self.t0)
        coordinator = FakeCoordinator(status)
        monitoring = MonitoringEngine(
            coordinator,
            persistence_check=_boom,
        )

        snapshot = monitoring.check_health(now=self.t0 + timedelta(seconds=1))

        self.assertFalse(snapshot.persistence_ok)
        self.assertFalse(snapshot.healthy)
        self.assertTrue(
            any("persistence check raised" in issue for issue in snapshot.issues)
        )

    def test_no_persistence_check_supplied_is_none(self) -> None:
        """Without a persistence_check, persistence_ok stays None (not False)."""

        status = self._status(last_tick_at=self.t0)
        coordinator = FakeCoordinator(status)
        monitoring = MonitoringEngine(coordinator)

        snapshot = monitoring.check_health(now=self.t0 + timedelta(seconds=1))

        self.assertIsNone(snapshot.persistence_ok)

    # ----------------------------------------------------------
    # History / convenience accessors
    # ----------------------------------------------------------

    def test_history_accumulates_snapshots(self) -> None:
        """Each check_health() call appends exactly one snapshot to history."""

        status = self._status(last_tick_at=self.t0)
        coordinator = FakeCoordinator(status)
        monitoring = MonitoringEngine(coordinator)

        monitoring.check_health(now=self.t0 + timedelta(seconds=1))
        monitoring.check_health(now=self.t0 + timedelta(seconds=2))
        monitoring.check_health(now=self.t0 + timedelta(seconds=3))

        self.assertEqual(len(monitoring.history()), 3)
        self.assertIsInstance(monitoring.history()[0], HealthSnapshot)

    def test_latest_returns_none_before_any_check(self) -> None:
        """latest() returns None until check_health() has been called."""

        status = self._status(last_tick_at=self.t0)
        coordinator = FakeCoordinator(status)
        monitoring = MonitoringEngine(coordinator)

        self.assertIsNone(monitoring.latest())

    def test_is_healthy_reflects_latest_snapshot(self) -> None:
        """is_healthy() tracks the health of only the most recent check."""

        status = self._status(last_tick_at=self.t0)
        coordinator = FakeCoordinator(status)
        adapter = FakeAdapter(errors=["boom"])
        monitoring = MonitoringEngine(coordinator, strategy_adapter=adapter)

        self.assertFalse(monitoring.is_healthy())

        monitoring.check_health(now=self.t0 + timedelta(seconds=1))
        self.assertFalse(monitoring.is_healthy())  # new error this check

        monitoring.check_health(now=self.t0 + timedelta(seconds=2))
        self.assertTrue(monitoring.is_healthy())  # no new error this check

    def test_is_healthy_false_before_first_check(self) -> None:
        """An unchecked system is not assumed healthy."""

        status = self._status(last_tick_at=self.t0)
        coordinator = FakeCoordinator(status)
        monitoring = MonitoringEngine(coordinator)

        self.assertFalse(monitoring.is_healthy())

    # ----------------------------------------------------------
    # Construction validation
    # ----------------------------------------------------------

    def test_max_tick_gap_seconds_must_be_positive(self) -> None:
        """Constructing with a non-positive max_tick_gap_seconds raises."""

        status = self._status(last_tick_at=self.t0)
        coordinator = FakeCoordinator(status)

        with self.assertRaises(ValueError):
            MonitoringEngine(coordinator, max_tick_gap_seconds=0)

        with self.assertRaises(ValueError):
            MonitoringEngine(coordinator, max_tick_gap_seconds=-5)

    def test_snapshot_is_frozen(self) -> None:
        """HealthSnapshot is immutable — mutating it must fail."""

        status = self._status(last_tick_at=self.t0)
        coordinator = FakeCoordinator(status)
        monitoring = MonitoringEngine(coordinator)

        snapshot = monitoring.check_health(now=self.t0 + timedelta(seconds=1))

        with self.assertRaises(Exception):
            snapshot.runtime_alive = False  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()