"""
phase_03_paper/replay/replay_runner.py

Historical Replay — feed historical CSV data through the SAME
Phase 3 pipeline used for live/paper operation.

PURPOSE
-------
Answers: "Does the real Phase 3 architecture behave correctly when
driven end-to-end across a full historical dataset, not just a short
smoke test?"

This module does NOT create a separate fake trading system. It wires
together components that already exist and are already independently
tested:

    CsvMarketDataSource   (new — implements MarketDataSource)
    MarketDataEngine      (existing — validates/normalizes candles)
    RuntimeCoordinator    (existing — built by the caller)
    stage2_runner         (existing — causal Candle -> Signal -> Trade order)
    Journal               (existing — lifecycle recording)
    MonitoringEngine      (existing — health observation)
    PerformanceEngine     (existing — statistics)

ReplayRunner's only new responsibility is orchestration: loop once
per historical candle, in order, calling each existing component the
same way a live run would.

WHY `now` IS PRE-COMPUTED PER CANDLE
-------------------------------------
RuntimeCoordinator.tick(now) uses `now` to evaluate SessionEngine
*before* it knows which candle next_candle() will return. In live
operation that's correct — `now` is wall-clock time. In replay, `now`
must instead be the historical candle's own timestamp, or session
evaluation (and therefore trading permission) would be based on
today's real date instead of the simulated moment — a direct
violation of the project's no-look-ahead-bias / causal-processing
rules.

To get each candle's timestamp before tick() consumes it, ReplayRunner
reuses MarketDataEngine._normalize() (the exact same normalization
logic the engine itself will apply) to pre-compute the ordered list of
timestamps from the same CSV rows the CsvMarketDataSource will yield.
This does not duplicate validation logic — it calls the existing
static method directly.

CALLER RESPONSIBILITY
----------------------
ReplayRunner does not construct SessionEngine, NotificationEngine, or
StrategyAdapter — those require configuration this module has no
visibility into. Instead the caller supplies a `coordinator_factory`
that receives the MarketDataEngine ReplayRunner builds from the CSV,
and returns a fully wired RuntimeCoordinator (or any compatible
Stage2Coordinator).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, List, Optional

from phase_03_paper.execution.stage2_runner import (
    Stage2Coordinator,
    run_tick_with_paper_execution,
)
from phase_03_paper.journal.journal import Journal
from phase_03_paper.market.engine import MarketDataEngine
from phase_03_paper.monitoring.monitoring import HealthSnapshot, MonitoringEngine
from phase_03_paper.performance.performance import (
    PerformanceEngine,
    PerformanceSnapshot,
)
from phase_03_paper.replay.csv_source import CsvMarketDataSource
from phase_03_paper.trading.paper_engine import PaperTradeEngine


@dataclass
class ReplaySummary:
    """Result of one complete historical replay run."""

    total_candles: int
    ticks_run: int

    opened_trades: int
    closed_trades: int

    first_timestamp: Optional[datetime]
    last_timestamp: Optional[datetime]

    final_health: Optional[HealthSnapshot] = None
    final_performance: Optional[PerformanceSnapshot] = None

    # Timestamps of any monitoring checks that came back unhealthy,
    # for post-run inspection without needing full history retained.
    unhealthy_checks: List[datetime] = field(default_factory=list)


def run_historical_replay(
    csv_path: str,
    coordinator_factory: Callable[[MarketDataEngine], Stage2Coordinator],
    paper_engine: PaperTradeEngine,
    journal: Optional[Journal] = None,
    monitoring_factory: Optional[
        Callable[[Stage2Coordinator], MonitoringEngine]
    ] = None,
    performance: Optional[PerformanceEngine] = None,
    *,
    timestamp_col: str = "timestamp",
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    volume_col: str = "volume",
) -> ReplaySummary:
    """
    Drive a full historical CSV dataset through the real Phase 3
    pipeline, tick by tick, in causal order.

    Parameters
    ----------
    csv_path:
        Path to a historical OHLC(V) CSV file.

    coordinator_factory:
        Called once with the MarketDataEngine ReplayRunner builds from
        the CSV. Must return a fully wired RuntimeCoordinator (or any
        object satisfying the Stage2Coordinator protocol) — with
        SessionEngine, NotificationEngine, and StrategyAdapter already
        configured by the caller.

    paper_engine:
        PaperTradeEngine used for simulated trade execution, passed
        through unchanged to stage2_runner.run_tick_with_paper_execution.

    journal:
        Optional Journal. If provided, journal.record() is called
        once per tick, exactly as a live loop would.

    monitoring_factory:
        Optional callable receiving the constructed coordinator and
        returning a MonitoringEngine. A factory is required here
        rather than a pre-built MonitoringEngine because
        MonitoringEngine needs a coordinator instance to construct,
        and that coordinator does not exist until coordinator_factory
        has run inside this function. If provided, check_health() is
        called once per tick using that candle's own timestamp as
        `now`, so staleness detection reflects simulated time, not
        wall-clock time.

    performance:
        Optional PerformanceEngine, pre-built by the caller (it only
        depends on Journal, which already exists before this call).
        If provided, calculate() is called once at the end of the
        replay, using the final candle's timestamp as `now`.

    timestamp_col / open_col / high_col / low_col / close_col / volume_col:
        Column-name overrides passed through to CsvMarketDataSource.

    Returns
    -------
    ReplaySummary
        Aggregate counts plus the final MonitoringEngine and
        PerformanceEngine snapshots, if those were supplied.
    """

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

    # Pre-compute each candle's normalized timestamp, in order, by
    # reusing MarketDataEngine's own normalization logic. This gives
    # ReplayRunner the correct `now` to pass to coordinator.tick()
    # BEFORE that tick pulls the candle itself.
    expected_timestamps: List[datetime] = [
        MarketDataEngine._normalize(raw).timestamp for raw in source.stream()
    ]

    engine = MarketDataEngine(source=source)
    coordinator = coordinator_factory(engine)

    monitoring: Optional[MonitoringEngine] = (
        monitoring_factory(coordinator) if monitoring_factory is not None else None
    )

    opened_trades = 0
    closed_trades = 0
    ticks_run = 0
    unhealthy_checks: List[datetime] = []

    for candle_timestamp in expected_timestamps:
        result = run_tick_with_paper_execution(
            coordinator,
            paper_engine,
            now=candle_timestamp,
        )

        ticks_run += 1
        opened_trades += len(result.opened_trades)
        closed_trades += len(result.closed_trades)

        if journal is not None:
            journal.record(
                paper_engine,
                current_candle_timestamp=candle_timestamp,
            )

        if monitoring is not None:
            snapshot = monitoring.check_health(now=candle_timestamp)
            if not snapshot.healthy:
                unhealthy_checks.append(candle_timestamp)

    final_health = monitoring.latest() if monitoring is not None else None

    final_performance: Optional[PerformanceSnapshot] = None
    if performance is not None:
        last_ts = expected_timestamps[-1] if expected_timestamps else None
        final_performance = performance.calculate(now=last_ts)

    return ReplaySummary(
        total_candles=total_candles,
        ticks_run=ticks_run,
        opened_trades=opened_trades,
        closed_trades=closed_trades,
        first_timestamp=expected_timestamps[0] if expected_timestamps else None,
        last_timestamp=expected_timestamps[-1] if expected_timestamps else None,
        final_health=final_health,
        final_performance=final_performance,
        unhealthy_checks=unhealthy_checks,
    )


__all__ = [
    "ReplaySummary",
    "run_historical_replay",
]