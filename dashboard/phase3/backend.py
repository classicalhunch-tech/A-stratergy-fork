"""
dashboard/phase3/backend.py

Phase3Service â€” the ONLY thing the Phase 3 dashboard pages talk to.

PURPOSE
-------
This module wires together components that already exist and are
already independently tested:

    Phase3Config (existing)
    AuditLog (existing)
    SessionEngine (existing)
    NotificationEngine (existing)
    StrategyAdapter (existing, one per replay run)
    RuntimeCoordinator (existing, one per replay run)
    PaperTradeEngine (existing)
    PositionManager (existing, stateless)
    Journal (existing)
    Persistence (existing, optional)
    MonitoringEngine (existing, one per replay run)
    PerformanceEngine (existing)
    ReportEngine (existing)
    run_historical_replay() from phase_03_paper/replay/replay_runner.py

Phase3Service does NOT:
    - implement strategy logic
    - implement session lifecycle rules (SessionEngine owns that)
    - implement execution/fill logic (PaperTradeEngine owns that)
    - calculate performance statistics (PerformanceEngine owns that)
    - decide whether a trade exists (PaperTradeEngine/Journal own that)
    - contain any Streamlit/UI code

Everything this class returns is a real object (or list of real
objects) produced by the actual Phase 3 engine â€” never a fabricated
or estimated value.

LIFETIME
--------
One Phase3Service instance is meant to live for the lifetime of the
Streamlit process (via st.cache_resource in the dashboard page). Its
long-lived components â€” SessionEngine, NotificationEngine,
PaperTradeEngine, Journal, PerformanceEngine, ReportEngine, Persistence
â€” persist and accumulate across multiple replay runs and across
Streamlit reruns.

Its short-lived components â€” MarketDataEngine, StrategyAdapter,
RuntimeCoordinator, MonitoringEngine â€” are rebuilt fresh on every
run_replay() call, because a StrategyAdapter carries causal state
(swings, zones, liquidity, pending signals) that only makes sense for
ONE dataset processed from its own beginning. Re-running the same
dataset through a fresh adapter is safe: PaperTradeEngine already
protects against duplicate signal processing via its own
_processed_signal_keys, so trades from an identical prior run are
never double-counted.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from phase_03_paper.config import DEFAULT_CONFIG, Phase3Config
from phase_03_paper.events.audit import AuditLog
from phase_03_paper.journal.journal import Journal, JournalEntry
from phase_03_paper.market.engine import MarketDataEngine
from phase_03_paper.models import SessionDecision, SessionState, SessionStatus
from phase_03_paper.monitoring.monitoring import HealthSnapshot, MonitoringEngine
from phase_03_paper.notifications.engine import NotificationEngine
from phase_03_paper.performance.performance import PerformanceEngine, PerformanceSnapshot
from phase_03_paper.persistence.persistence import Persistence
from phase_03_paper.positions.manager import PositionManager, PositionView
from phase_03_paper.replay.replay_runner import ReplaySummary, run_historical_replay
from phase_03_paper.runtime.coordinator import RuntimeCoordinator
from phase_03_paper.sessions.engine import SessionEngine
from phase_03_paper.signals.adapter import StrategyAdapter
from phase_03_paper.trading.paper_engine import PaperTradeEngine
from phase_03_paper.ai.report_generator import Report, ReportEngine


def _normalize_now(value: Optional[datetime]) -> datetime:
    """Normalize timestamps to timezone-aware UTC, same rule every
    Phase 3 engine module already uses."""
    if value is None:
        return datetime.now(timezone.utc)
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


# ============================================================
# AUTO-APPROVING COORDINATOR PROXY
# ============================================================


class _AutoApprovingCoordinator:
    """
    Thin proxy around a real RuntimeCoordinator.

    Before delegating to the real coordinator's tick(), it checks
    every configured session for an occurrence that is still
    UNDECIDED and has not yet reached its scheduled start (UPCOMING
    or WARNING), and calls SessionEngine.record_decision(name, TRADE)
    on it â€” the exact same public method a human operator clicks
    "ALLOW TRADING" to call from the Session Control page.

    This proxy does NOT modify SessionEngine's lifecycle rules. It
    only automates the same decision a person would otherwise have
    to make manually for every single session occurrence in a
    multi-day replay. Used only when the dashboard's "auto-approve
    all sessions for this replay run" checkbox is enabled.

    Everything other than tick() is forwarded untouched to the real
    coordinator via __getattr__, so this proxy satisfies whatever
    interface RuntimeCoordinator itself satisfies (status(),
    triggered_events(), audit_log, etc.) without needing to know
    that interface in advance.
    """

    def __init__(
        self,
        coordinator: RuntimeCoordinator,
        session_engine: SessionEngine,
        config: Phase3Config,
    ) -> None:
        self._coordinator = coordinator
        self._session_engine = session_engine
        self._session_names = [
            window.name
            for window in config.sessions.sessions
            if window.enabled
        ]

    def tick(self, now: Optional[datetime] = None):
        now_utc = _normalize_now(now)

        self._session_engine.evaluate(now_utc)

        for session_name in self._session_names:
            try:
                state = self._session_engine.get_state(
                    session_name,
                    now=now_utc,
                )
            except KeyError:
                continue

            if (
                state.decision == SessionDecision.UNDECIDED
                and state.status in (
                    SessionStatus.UPCOMING,
                    SessionStatus.WARNING,
                )
            ):
                self._session_engine.record_decision(
                    session_name,
                    SessionDecision.TRADE,
                    note="auto-approved for replay",
                    now=now_utc,
                )

        return self._coordinator.tick(now_utc)

    def __getattr__(self, name):
        return getattr(self._coordinator, name)


# ============================================================
# PHASE 3 SERVICE
# ============================================================


@dataclass
class Phase3RunResult:
    """Everything one replay run produced, for callers that want more
    than just the ReplaySummary."""

    summary: ReplaySummary
    health: Optional[HealthSnapshot]
    performance: PerformanceSnapshot


class Phase3Service:
    """
    Single application-service boundary between the dashboard and the
    real Phase 3 engine. See module docstring for the full contract.
    """

    def __init__(
        self,
        config: Phase3Config = DEFAULT_CONFIG,
        db_path: Optional[str] = None,
    ) -> None:
        self._config = config

        # --------------------------------------------------------
        # Long-lived components â€” persist for the service's lifetime
        # --------------------------------------------------------

        self._audit_log = AuditLog()
        self._session_engine = SessionEngine(
            config=config,
            audit_log=self._audit_log,
        )
        self._notification_engine = NotificationEngine(
            config=config,
            audit_log=self._audit_log,
        )
        self._paper_engine = PaperTradeEngine(
            config=config,
            audit_log=self._audit_log,
        )
        self._position_manager = PositionManager()
        self._journal = Journal(self._position_manager)
        self._performance_engine = PerformanceEngine(self._journal)
        self._report_engine = ReportEngine(self._journal)

        self._persistence: Optional[Persistence] = None
        if db_path:
            self._persistence = Persistence(db_path)

        # Cursor so save_journal_entries() (append-only) is never
        # called twice with the same entries across multiple runs.
        self._journal_entries_persisted = 0

        # --------------------------------------------------------
        # Short-lived components â€” rebuilt every run_replay() call
        # --------------------------------------------------------

        self._last_strategy_adapter: Optional[StrategyAdapter] = None
        self._last_coordinator: Optional[RuntimeCoordinator] = None
        self._last_monitoring: Optional[MonitoringEngine] = None
        self._last_summary: Optional[ReplaySummary] = None

    # ============================================================
    # COMMANDS â€” REPLAY
    # ============================================================

    def run_replay(
        self,
        csv_path: str,
        auto_approve_sessions: bool = True,
    ) -> ReplaySummary:
        """
        Drive a full historical CSV dataset through the real Phase 3
        pipeline: MarketDataEngine -> RuntimeCoordinator ->
        SessionEngine -> StrategyAdapter -> existing strategy/ ->
        PaperTradeEngine -> PositionManager -> Journal -> Monitoring
        -> Performance.

        Raises FileNotFoundError if csv_path does not exist (surfaced
        by CsvMarketDataSource) â€” callers should catch and display it,
        not swallow it here.
        """

        if not Path(csv_path).exists():
            raise FileNotFoundError(f"Dataset not found: {csv_path}")

        strategy_settings = self._config.strategy

        strategy_adapter = StrategyAdapter(
            max_bars_to_retest=strategy_settings.max_bars_to_retest,
            reward_multiple=strategy_settings.reward_multiple,
            stop_buffer=strategy_settings.stop_buffer,
            entry_mode=strategy_settings.entry_mode,
        )

        def coordinator_factory(engine: MarketDataEngine):
            coordinator = RuntimeCoordinator(
                session_engine=self._session_engine,
                notification_engine=self._notification_engine,
                market_data_engine=engine,
                audit_log=self._audit_log,
                strategy_adapter=strategy_adapter,
            )

            self._last_strategy_adapter = strategy_adapter
            self._last_coordinator = coordinator

            if auto_approve_sessions:
                return _AutoApprovingCoordinator(
                    coordinator,
                    self._session_engine,
                    self._config,
                )

            return coordinator

        def monitoring_factory(coordinator):
            monitoring = MonitoringEngine(
                coordinator=coordinator,
                strategy_adapter=strategy_adapter,
                journal=self._journal,
                persistence_check=(
                    self._check_persistence_health
                    if self._persistence is not None
                    else None
                ),
            )
            self._last_monitoring = monitoring
            return monitoring

        summary = run_historical_replay(
            csv_path,
            coordinator_factory=coordinator_factory,
            paper_engine=self._paper_engine,
            journal=self._journal,
            monitoring_factory=monitoring_factory,
            performance=self._performance_engine,
        )

        self._last_summary = summary

        self._persist_after_run()

        return summary

    def _persist_after_run(self) -> None:
        """Save trades (upsert-safe) and only the journal entries not
        already persisted (journal storage is append-only)."""

        if self._persistence is None:
            return

        self._persistence.save_trades(self._paper_engine.all_trades())

        all_entries = self._journal.all_entries()
        new_entries = all_entries[self._journal_entries_persisted:]

        if new_entries:
            self._persistence.save_journal_entries(new_entries)

        self._journal_entries_persisted = len(all_entries)

    def _check_persistence_health(self) -> bool:
        if self._persistence is None:
            return False
        try:
            self._persistence.load_trades()
            return True
        except Exception:  # noqa: BLE001
            return False

    # ============================================================
    # COMMANDS â€” SESSION CONTROL
    # ============================================================

    def allow_trading(
        self,
        session_name: str,
        note: Optional[str] = None,
    ) -> SessionState:
        """
        Record a TRADE decision for session_name's current
        decision-eligible occurrence (the "ALLOW TRADING" button).

        Raises ValueError if no eligible occurrence exists or the
        relevant occurrence is already terminal (CLOSED/SKIPPED) â€”
        the caller should surface this, not swallow it.
        """
        return self._session_engine.record_decision(
            session_name,
            SessionDecision.TRADE,
            note=note,
        )

    def skip_session(
        self,
        session_name: str,
        note: Optional[str] = None,
    ) -> SessionState:
        """
        Record a SKIP decision for session_name's current
        decision-eligible occurrence (the "SKIP SESSION" button, after
        the UI's own confirmation step).
        """
        return self._session_engine.record_decision(
            session_name,
            SessionDecision.SKIP,
            note=note,
        )

    # ============================================================
    # QUERIES â€” SESSIONS
    # ============================================================

    def get_sessions(self) -> List[SessionState]:
        """All materialized session occurrences, chronological order."""
        return self._session_engine.all_states()

    def get_current_session(self) -> Optional[SessionState]:
        """The currently ACTIVE session occurrence, if any."""
        return self._session_engine.current_session()

    def get_session_state(
        self,
        session_name_or_occurrence_id: str,
    ) -> SessionState:
        """One session's current state, by name or occurrence_id."""
        return self._session_engine.get_state(session_name_or_occurrence_id)

    def get_configured_session_names(self) -> List[str]:
        """Names of every enabled configured session window."""
        return [
            window.name
            for window in self._config.sessions.sessions
            if window.enabled
        ]

    def get_session_audit_events(self, occurrence_id: str):
        """Every audit event tied to one session occurrence (decision
        history, status changes, signal-gate changes) â€” the "Permission
        History" list on the Session Control page."""
        return self._session_engine.audit_log.events_for(occurrence_id)

    # ============================================================
    # QUERIES â€” TRADING
    # ============================================================

    def get_open_positions(self) -> List[PositionView]:
        return self._position_manager.views(self._paper_engine)

    def get_all_positions(self) -> List[PositionView]:
        """Open AND closed positions, as PositionView objects."""
        return self._position_manager.all_views(self._paper_engine)

    def get_closed_trades(self) -> List[JournalEntry]:
        return self._journal.closed_entries()

    # ============================================================
    # QUERIES â€” ANALYTICS
    # ============================================================

    def get_performance(self) -> PerformanceSnapshot:
        return self._performance_engine.calculate()

    def get_latest_report(self) -> Report:
        return self._report_engine.generate()

    # ============================================================
    # QUERIES â€” SYSTEM
    # ============================================================

    def get_system_health(self) -> HealthSnapshot:
        """
        The most recent HealthSnapshot from the last replay run.

        Phase 3 is currently replay-driven, not continuously live, so
        there is no wall-clock "current" health independent of a run
        â€” this reports the last real check_health() result, or an
        honest "no run yet" snapshot if none exists.
        """
        if self._last_monitoring is not None:
            latest = self._last_monitoring.latest()
            if latest is not None:
                return latest

        return HealthSnapshot(
            checked_at=datetime.now(timezone.utc),
            runtime_alive=False,
            last_tick_at=None,
            tick_count=0,
            last_runtime_error=None,
            last_candle_at=None,
            candle_count=0,
            adapter_error_count=0,
            new_adapter_errors=0,
            pending_signal_count=0,
            new_journal_entries=0,
            persistence_ok=(
                self._check_persistence_health()
                if self._persistence is not None
                else None
            ),
            issues=("No replay has been run yet.",),
        )

    def get_health_history(self) -> List[HealthSnapshot]:
        if self._last_monitoring is None:
            return []
        return self._last_monitoring.history()

    def get_last_summary(self) -> Optional[ReplaySummary]:
        return self._last_summary

    def get_audit_events(self):
        """Every audit event recorded so far, oldest first â€” the
        System / Events page."""
        return self._audit_log.all_events()

    def get_config(self) -> Phase3Config:
        return self._config


__all__ = [
    "Phase3Service",
    "Phase3RunResult",
]