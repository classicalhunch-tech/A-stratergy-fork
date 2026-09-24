"""
phase_03_paper/monitoring/monitoring.py

MonitoringEngine — Phase 3 system health observer.

PURPOSE
-------
MonitoringEngine answers one question:

    "Is the paper-trading system healthy right now?"

It does this by PULLING state from components that already exist —
RuntimeCoordinator, StrategyAdapter, Journal, and (optionally) a
persistence health-check callable — the same way Journal.record()
pulls from PositionManager and PaperTradeEngine.

MonitoringEngine does NOT:
    - generate trading signals
    - modify strategy logic
    - execute trades
    - calculate performance statistics (that is Performance, later)
    - replace Journal (Journal records what happened)
    - replace Persistence (Persistence saves what happened)
    - contain dashboard/UI logic

DESIGN
------
Like RuntimeCoordinator tracking `_adapter_errors_seen`, and Journal
tracking `_last_recorded` per trade_id, MonitoringEngine tracks its
own cursors (`_adapter_errors_seen`, `_journal_entries_seen`) so that
repeated calls to check_health() report only NEW problems since the
previous check, not the same historical error over and over.

MonitoringEngine is pull-based and synchronous. It does not run its
own loop or thread. Whatever already drives the runtime (a Stage 2
runner, a replay loop, or the dashboard) is expected to call
check_health() once per cycle — the same place it would call
journal.record().

PERSISTENCE
-----------
persistence.py's concrete interface is intentionally NOT assumed
here. Instead, callers pass a zero-argument `persistence_check`
callable that returns True/False (or raises on failure). This keeps
MonitoringEngine decoupled from persistence internals and avoids
guessing at an API that may change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Final, List, Optional, Tuple

from phase_03_paper.journal.journal import Journal
from phase_03_paper.runtime.coordinator import RuntimeCoordinator


_DEFAULT_MAX_TICK_GAP_SECONDS: Final[float] = 60.0


@dataclass(frozen=True)
class HealthSnapshot:
    """Immutable snapshot of system health at one point in time."""

    checked_at: datetime

    # Runtime / market data
    runtime_alive: bool
    last_tick_at: Optional[datetime]
    tick_count: int
    last_runtime_error: Optional[str]

    last_candle_at: Optional[datetime]

    # Strategy adapter
    candle_count: int
    adapter_error_count: int
    new_adapter_errors: int
    pending_signal_count: int

    # Journal activity
    new_journal_entries: int

    # Persistence
    persistence_ok: Optional[bool]

    # Human-readable problems found during THIS check.
    issues: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def healthy(self) -> bool:
        """Return True when no issues were found on this check."""
        return len(self.issues) == 0


class MonitoringEngine:
    """
    Observe the health of the Phase 3 paper-trading system.

    MonitoringEngine is a read-only observer. It never mutates the
    coordinator, adapter, or journal it is given.
    """

    def __init__(
        self,
        coordinator: RuntimeCoordinator,
        strategy_adapter: object | None = None,
        journal: Optional[Journal] = None,
        persistence_check: Optional[Callable[[], bool]] = None,
        max_tick_gap_seconds: float = _DEFAULT_MAX_TICK_GAP_SECONDS,
    ) -> None:
        if max_tick_gap_seconds <= 0:
            raise ValueError("max_tick_gap_seconds must be > 0")

        self._coordinator = coordinator
        self._strategy_adapter = strategy_adapter
        self._journal = journal
        self._persistence_check = persistence_check
        self._max_tick_gap_seconds = max_tick_gap_seconds

        # Cursors — mirror the "seen so far" pattern used by
        # RuntimeCoordinator._adapter_errors_seen and
        # Journal._last_recorded.
        self._adapter_errors_seen: int = 0
        self._journal_entries_seen: int = 0

        self._history: List[HealthSnapshot] = []

    # ============================================================
    # PUBLIC API
    # ============================================================

    def check_health(self, now: datetime | None = None) -> HealthSnapshot:
        """
        Perform one health check and return a HealthSnapshot.

        Safe to call repeatedly (e.g. once per runtime tick). Only
        NEW adapter errors and NEW journal entries since the previous
        call are reported as issues.
        """

        now_utc = self._normalize_now(now)
        issues: List[str] = []

        status = self._coordinator.status()

        # --------------------------------------------------------
        # Runtime liveness
        # --------------------------------------------------------

        runtime_alive = self._is_runtime_alive(
            status.last_tick_at,
            now_utc,
        )

        if status.last_tick_at is None:
            issues.append("runtime has not ticked yet")
        elif not runtime_alive:
            issues.append(
                "runtime stale: no tick within "
                f"{self._max_tick_gap_seconds:.0f}s"
            )

        if status.last_error is not None:
            issues.append(f"runtime error: {status.last_error}")

        latest_candle = status.latest_candle
        last_candle_at = (
            latest_candle.timestamp if latest_candle is not None else None
        )

        # --------------------------------------------------------
        # Strategy adapter
        # --------------------------------------------------------

        candle_count = 0
        adapter_error_count = 0
        new_adapter_errors = 0
        pending_signal_count = 0

        if self._strategy_adapter is not None:
            adapter_errors = list(self._strategy_adapter.errors)
            adapter_error_count = len(adapter_errors)
            new_adapter_errors = max(
                0,
                adapter_error_count - self._adapter_errors_seen,
            )
            self._adapter_errors_seen = adapter_error_count

            pending_signal_count = self._strategy_adapter.pending_count
            candle_count = self._strategy_adapter.candle_count

            if new_adapter_errors > 0:
                issues.append(
                    f"{new_adapter_errors} new adapter error(s)"
                )

        # --------------------------------------------------------
        # Journal activity (observational only — Monitoring does not
        # derive lifecycle state itself; Journal already did that).
        # --------------------------------------------------------

        new_journal_entries = 0

        if self._journal is not None:
            total_entries = len(self._journal.all_entries())
            new_journal_entries = max(
                0,
                total_entries - self._journal_entries_seen,
            )
            self._journal_entries_seen = total_entries

        # --------------------------------------------------------
        # Persistence (optional, caller-supplied health check)
        # --------------------------------------------------------

        persistence_ok: Optional[bool] = None

        if self._persistence_check is not None:
            try:
                persistence_ok = bool(self._persistence_check())
            except Exception as exc:  # noqa: BLE001
                persistence_ok = False
                issues.append(f"persistence check raised: {exc!r}")

            if persistence_ok is False and not issues:
                issues.append("persistence check failed")
            elif persistence_ok is False:
                # Only add the generic message if the exception path
                # above did not already add a more specific one.
                if not any("persistence" in item for item in issues):
                    issues.append("persistence check failed")

        snapshot = HealthSnapshot(
            checked_at=now_utc,
            runtime_alive=runtime_alive,
            last_tick_at=status.last_tick_at,
            tick_count=status.tick_count,
            last_runtime_error=status.last_error,
            last_candle_at=last_candle_at,
            candle_count=candle_count,
            adapter_error_count=adapter_error_count,
            new_adapter_errors=new_adapter_errors,
            pending_signal_count=pending_signal_count,
            new_journal_entries=new_journal_entries,
            persistence_ok=persistence_ok,
            issues=tuple(issues),
        )

        self._history.append(snapshot)
        return snapshot

    def history(self) -> List[HealthSnapshot]:
        """Return every HealthSnapshot recorded so far, in order."""

        return list(self._history)

    def latest(self) -> Optional[HealthSnapshot]:
        """Return the most recent HealthSnapshot, or None if never checked."""

        return self._history[-1] if self._history else None

    def is_healthy(self) -> bool:
        """
        Return whether the system was healthy as of the last check.

        Returns False if check_health() has never been called — an
        unchecked system is not assumed healthy.
        """

        latest = self.latest()
        return latest.healthy if latest is not None else False

    # ============================================================
    # INTERNAL HELPERS
    # ============================================================

    def _is_runtime_alive(
        self,
        last_tick_at: Optional[datetime],
        now_utc: datetime,
    ) -> bool:
        """Return True when the last tick is within the allowed gap."""

        if last_tick_at is None:
            return False

        gap_seconds = (now_utc - last_tick_at).total_seconds()
        return gap_seconds <= self._max_tick_gap_seconds

    @staticmethod
    def _normalize_now(value: datetime | None) -> datetime:
        """Normalize timestamps to timezone-aware UTC datetimes."""

        if value is None:
            return datetime.now(timezone.utc)

        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)

        return value.astimezone(timezone.utc)


__all__ = [
    "HealthSnapshot",
    "MonitoringEngine",
]