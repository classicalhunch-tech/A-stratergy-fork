"""
phase_03_paper/replay/event_collector.py

Replay Result / Event Collection.

PURPOSE
-------
Captures the REAL events already produced by the existing Phase 3
paper-trading pipeline during historical replay.

This module is OBSERVATIONAL ONLY.

It does NOT:
- generate signals
- modify strategy behavior
- open trades
- close trades
- decide trade outcomes
- calculate trade results
- duplicate journal logic
- duplicate persistence logic
- own monitoring logic
- own dashboard/UI logic

It observes the outputs already produced by the existing runtime and
stores them in a structured, ordered event log for later consumers
such as:
    Replay Results
        ->
    Chart Data
        ->
    Visualization
        ->
    Dashboard
        ->
    AI Reporting
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Callable, List, Optional

from phase_03_paper.execution.stage2_runner import (
    Stage2Coordinator,
    run_tick_with_paper_execution,
)
from phase_03_paper.journal.journal import Journal, JournalEntry
from phase_03_paper.market.engine import MarketDataEngine
from phase_03_paper.monitoring.monitoring import (
    HealthSnapshot,
    MonitoringEngine,
)
from phase_03_paper.performance.performance import (
    PerformanceEngine,
    PerformanceSnapshot,
)
from phase_03_paper.replay.csv_source import CsvMarketDataSource
from phase_03_paper.trading.paper_engine import (
    PaperTrade,
    PaperTradeEngine,
)


# ============================================================
# EVENT MODEL
# ============================================================


class ReplayEventType(str, Enum):
    """
    Types of real events observed during historical replay.

    The collector only records events that already exist in the
    underlying Phase 3 pipeline.
    """

    TRADE_OPENED = "TRADE_OPENED"
    TRADE_CLOSED = "TRADE_CLOSED"
    JOURNAL_ENTRY = "JOURNAL_ENTRY"


@dataclass(frozen=True)
class ReplayEvent:
    """
    One immutable fact observed during replay.

    The payload is the ORIGINAL object produced by the component
    responsible for that event.
    """

    event_type: ReplayEventType
    timestamp: Optional[datetime]
    payload: object


@dataclass
class ReplayEventLog:
    """
    Ordered collection of ReplayEvents from one replay run.

    Events are appended in causal processing order.
    """

    events: List[ReplayEvent] = field(default_factory=list)

    def add(self, event: ReplayEvent) -> None:
        """Append one observed event to the log."""
        self.events.append(event)

    # --------------------------------------------------------
    # Basic & Type-Safe Queries
    # --------------------------------------------------------

    def all_events(self) -> List[ReplayEvent]:
        """Return a copy of the complete event list."""
        return list(self.events)

    def events_of_type(self, event_type: ReplayEventType) -> List[ReplayEvent]:
        """Return all events matching a specific ReplayEventType."""
        return [
            event for event in self.events 
            if event.event_type == event_type
        ]

    def trade_opened_events(self) -> List[ReplayEvent]:
        """Return all trade-open events."""
        return self.events_of_type(ReplayEventType.TRADE_OPENED)

    def trade_closed_events(self) -> List[ReplayEvent]:
        """Return all trade-close events."""
        return self.events_of_type(ReplayEventType.TRADE_CLOSED)

    def journal_events(self) -> List[ReplayEvent]:
        """Return all journal events."""
        return self.events_of_type(ReplayEventType.JOURNAL_ENTRY)

    # --------------------------------------------------------
    # Trade-Specific Query
    # --------------------------------------------------------

    def events_for_trade(self, trade_id: str) -> List[ReplayEvent]:
        """
        Return every event associated with one trade.
        Checks both 'trade_id' and 'id' attributes on payloads.
        """
        matching: List[ReplayEvent] = []

        for event in self.events:
            payload = event.payload
            p_trade_id = (
                getattr(payload, "trade_id", None) 
                or getattr(payload, "id", None)
            )

            if p_trade_id == trade_id:
                matching.append(event)

        return matching

    # --------------------------------------------------------
    # Convenience Properties & Metrics
    # --------------------------------------------------------

    @property
    def event_count(self) -> int:
        """Total number of collected events."""
        return len(self.events)

    @property
    def is_empty(self) -> bool:
        """Check if the event log has recorded any events."""
        return len(self.events) == 0

    @property
    def opened_trade_count(self) -> int:
        """Number of observed trade-open events."""
        return len(self.trade_opened_events())

    @property
    def closed_trade_count(self) -> int:
        """Number of observed trade-close events."""
        return len(self.trade_closed_events())

    @property
    def journal_entry_count(self) -> int:
        """Number of observed journal events."""
        return len(self.journal_events())


# ============================================================
# REPLAY SUMMARY
# ============================================================


@dataclass
class ReplayWithEventsSummary:
    """
    Historical replay summary plus the complete structured
    event log.
    """

    total_candles: int
    ticks_run: int

    opened_trades: int
    closed_trades: int

    first_timestamp: Optional[datetime]
    last_timestamp: Optional[datetime]

    final_health: Optional[HealthSnapshot] = None
    final_performance: Optional[PerformanceSnapshot] = None

    unhealthy_checks: List[datetime] = field(default_factory=list)

    event_log: ReplayEventLog = field(
        default_factory=ReplayEventLog
    )


# ============================================================
# COLLECTOR-AWARE REPLAY RUNNER
# ============================================================


def run_historical_replay_with_events(
    csv_path: str,
    coordinator_factory: Callable[
        [MarketDataEngine],
        Stage2Coordinator,
    ],
    paper_engine: PaperTradeEngine,
    journal: Optional[Journal] = None,
    monitoring_factory: Optional[
        Callable[
            [Stage2Coordinator],
            MonitoringEngine,
        ]
    ] = None,
    performance: Optional[PerformanceEngine] = None,
    *,
    timestamp_col: str = "timestamp",
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    volume_col: str = "volume",
) -> ReplayWithEventsSummary:
    """
    Run historical replay while collecting real Phase 3 events.
    """

    # 1. Create the CSV source
    source = CsvMarketDataSource(
        csv_path,
        timestamp_col=timestamp_col,
        open_col=open_col,
        high_col=high_col,
        low_col=low_col,
        close_col=close_col,
        volume_col=volume_col,
    )

    total_candles = len(source)

    # 2. Capture expected timestamps
    expected_timestamps: List[datetime] = [
        MarketDataEngine._normalize(raw).timestamp
        for raw in source.stream()
    ]

    engine = MarketDataEngine(source=source)
    coordinator = coordinator_factory(engine)

    # 3. Optional monitoring
    monitoring: Optional[MonitoringEngine] = None
    if monitoring_factory is not None:
        monitoring = monitoring_factory(coordinator)

    # 4. Initialize collection state
    event_log = ReplayEventLog()

    opened_trades = 0
    closed_trades = 0
    ticks_run = 0

    unhealthy_checks: List[datetime] = []

    # 5. Run the existing Stage 2 causal execution path
    for candle_timestamp in expected_timestamps:

        result = run_tick_with_paper_execution(
            coordinator,
            paper_engine,
            now=candle_timestamp,
        )

        ticks_run += 1

        # 5A. Collect trade-open events
        for trade in result.opened_trades:
            event_log.add(
                ReplayEvent(
                    event_type=ReplayEventType.TRADE_OPENED,
                    timestamp=candle_timestamp,
                    payload=trade,
                )
            )
        opened_trades += len(result.opened_trades)

        # 5B. Collect trade-close events
        for trade in result.closed_trades:
            event_log.add(
                ReplayEvent(
                    event_type=ReplayEventType.TRADE_CLOSED,
                    timestamp=candle_timestamp,
                    payload=trade,
                )
            )
        closed_trades += len(result.closed_trades)

        # 5C. Collect journal entries
        if journal is not None:
            new_entries: List[JournalEntry] = journal.record(
                paper_engine,
                current_candle_timestamp=candle_timestamp,
            )
            for entry in new_entries:
                event_log.add(
                    ReplayEvent(
                        event_type=ReplayEventType.JOURNAL_ENTRY,
                        timestamp=candle_timestamp,
                        payload=entry,
                    )
                )

        # 5D. Observe monitoring state
        if monitoring is not None:
            snapshot = monitoring.check_health(now=candle_timestamp)
            if not snapshot.healthy:
                unhealthy_checks.append(candle_timestamp)

    # 6. Final monitoring state
    final_health = monitoring.latest() if monitoring is not None else None

    # 7. Final performance snapshot
    final_performance: Optional[PerformanceSnapshot] = None
    if performance is not None:
        last_timestamp = expected_timestamps[-1] if expected_timestamps else None
        final_performance = performance.calculate(now=last_timestamp)

    # 8. Return complete replay result summary
    return ReplayWithEventsSummary(
        total_candles=total_candles,
        ticks_run=ticks_run,
        opened_trades=opened_trades,
        closed_trades=closed_trades,
        first_timestamp=expected_timestamps[0] if expected_timestamps else None,
        last_timestamp=expected_timestamps[-1] if expected_timestamps else None,
        final_health=final_health,
        final_performance=final_performance,
        unhealthy_checks=unhealthy_checks,
        event_log=event_log,
    )


__all__ = [
    "ReplayEventType",
    "ReplayEvent",
    "ReplayEventLog",
    "ReplayWithEventsSummary",
    "run_historical_replay_with_events",
]