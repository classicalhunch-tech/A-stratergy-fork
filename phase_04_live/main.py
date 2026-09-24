"""
phase_04_live/main.py

Live runner entry point.

Wires together:

    LiveMarketSource (Phase 4)
        ->
    MarketDataEngine (Phase 3, reused as-is)
        ->
    RuntimeCoordinator (Phase 3, reused as-is)
        ->
    LiveEventEngine (Phase 4)

Phase 3 remains responsible for its existing runtime behavior:
    SessionEngine
    NotificationEngine
    StrategyAdapter
    AuditLog

Phase 4 adds:
    ClockIntegrityChecker
    LiveEventEngine
    LiveMarketSource

IMPORTANT -- SESSION DECISIONS

SessionEngine sessions default to UNDECIDED. RuntimeCoordinator
suppresses triggered strategy signals until a session decision has
been recorded for the relevant occurrence.

This runner does NOT automatically approve session decisions.

Therefore:

    market polling        -> runs
    candle validation     -> runs
    strategy processing   -> runs
    clock checks          -> run
    session gating        -> remains active
    live trade approval   -> NOT automatic

This is intentional. Phase 4 must not bypass the existing Phase 3
session safety gate merely because it is running live.

IMPORTANT -- QUALITY ISSUE ROUTING

LiveMarketSource must be constructed before LiveEventEngine exists
(engine needs coordinator, coordinator needs market_data_engine,
market_data_engine needs live_source -- live_source is necessarily
built first). QualityIssueRelay below is the dependency that solves
this ordering problem: it is handed to LiveMarketSource as its
on_quality_issue callback at construction time, then bound to the
real engine.on_quality_issue once the engine exists. This is why
every QualityIssue both logs AND lands in the engine's ordered
LiveEvent stream -- per event_engine.py's own docstring, that stream
is supposed to capture every QualityIssue raised during a tick, not
just print it. No globals, no reaching into private attributes after
construction.

IMPORTANT -- MT5 CONNECTION TIMING

ClockIntegrityChecker.check_drift() calls mt5.symbol_info_tick()
directly and runs BEFORE coordinator.tick() in every cycle (by
design -- see event_engine.py's ordering: clock issues must surface
before anything else, since a bad clock can invalidate the whole
cycle's data). LiveMarketSource.connect() is otherwise lazy and only
fires on the first iteration of stream(), which only happens once
coordinator.tick() reaches market_data_engine.next_candle() -- i.e.
AFTER the clock check already ran. Left alone, this means the very
first cycle's clock check hits MT5 before it's ever been initialized
and raises "No IPC connection". The fix is NOT to reorder the clock
check (that would defeat its purpose); it's to connect MT5 explicitly
at startup, before either subsystem needs it -- see main() below.

USAGE:

    Smoke test:
        python -m phase_04_live.main --once

    Continuous loop:
        python -m phase_04_live.main
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Callable, Optional

from phase_03_paper.events.audit import AuditLog
from phase_03_paper.market.engine import MarketDataEngine
from phase_03_paper.notifications.engine import NotificationEngine
from phase_03_paper.runtime.coordinator import RuntimeCoordinator
from phase_03_paper.sessions.engine import SessionEngine
from phase_03_paper.signals.adapter import StrategyAdapter

from phase_04_live.clock.integrity import ClockIntegrityChecker
from phase_04_live.clock.models import ClockIssue
from phase_04_live.events.event_engine import LiveEventEngine
from phase_04_live.market.live_source import LiveMarketSource
from phase_04_live.market.models import QualityIssue


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SYMBOL = "XAUUSD"
TIMEFRAME = "M15"

POLL_INTERVAL_SECONDS = 5.0

CLOCK_CHECK_INTERVAL_SECONDS = 30.0
MAX_DRIFT_SECONDS = 2.0
STALE_AFTER_MISSED_CANDLES = 2


# ---------------------------------------------------------------------------
# Diagnostics Callbacks
# ---------------------------------------------------------------------------

def handle_quality_issue(issue: QualityIssue) -> None:
    """
    Console diagnostic for market-quality issues.
    """
    logger.warning(
        "[QUALITY] %s | %s %s | %s",
        issue.issue_type.value,
        issue.symbol,
        issue.timeframe,
        issue.detail,
    )


def handle_clock_issue(issue: ClockIssue) -> None:
    """
    Console diagnostic for clock/feed-integrity issues.
    """
    logger.warning(
        "[CLOCK] %s | measured=%.2fs threshold=%.2fs | %s",
        issue.issue_type.value,
        issue.measured_seconds,
        issue.threshold_seconds,
        issue.detail,
    )


class QualityIssueRelay:
    """
    Explicit forwarding dependency -- solves the construction-order
    problem described in the module docstring without a global or a
    reach-into-private-attribute hack.

    Passed to LiveMarketSource as on_quality_issue at construction
    time (when the engine doesn't exist yet), then bound to
    engine.on_quality_issue once the engine is built. Every call
    logs AND forwards, so QualityIssue events reliably reach the
    engine's ordered LiveEvent stream, not just the console.
    """

    def __init__(self) -> None:
        self._target: Optional[Callable[[QualityIssue], None]] = None

    def bind(self, target: Callable[[QualityIssue], None]) -> None:
        self._target = target

    def __call__(self, issue: QualityIssue) -> None:
        handle_quality_issue(issue)
        if self._target is not None:
            self._target(issue)


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------

def build_event_engine() -> tuple[LiveEventEngine, LiveMarketSource]:
    """
    Build the complete Phase 4 live event engine using explicit
    dependency injection -- no global state, no closure workarounds,
    no post-construction reach into private attributes.

    Returns (engine, live_source). live_source is returned alongside
    the engine so the caller can explicitly connect() it before the
    first cycle and disconnect() it on shutdown -- see the module
    docstring's IMPORTANT -- MT5 CONNECTION TIMING note for why this
    can't be left to LiveMarketSource's lazy internal connect().
    """

    # 1. Relay must exist before LiveMarketSource so it can be passed
    #    in as a real constructor argument, not patched in later.
    quality_relay = QualityIssueRelay()

    # 2. Instantiate LiveMarketSource with the relay as its callback.
    live_source = LiveMarketSource(
        symbol=SYMBOL,
        timeframe=TIMEFRAME,
        poll_interval_seconds=POLL_INTERVAL_SECONDS,
        on_quality_issue=quality_relay,
    )

    # 3. Wrap via Phase 3 MarketDataEngine
    market_data_engine = MarketDataEngine(live_source)

    # 4. Build Phase 3 Runtime Stack (exact construction pattern from tests)
    audit_log = AuditLog()
    session_engine = SessionEngine(audit_log=audit_log)
    notification_engine = NotificationEngine(audit_log=audit_log)
    strategy_adapter = StrategyAdapter()

    coordinator = RuntimeCoordinator(
        session_engine=session_engine,
        notification_engine=notification_engine,
        market_data_engine=market_data_engine,
        audit_log=audit_log,
        strategy_adapter=strategy_adapter,
    )

    # 5. Build Clock Checker
    clock_checker = ClockIntegrityChecker(
        symbol=SYMBOL,
        max_drift_seconds=MAX_DRIFT_SECONDS,
        stale_after_missed_candles=STALE_AFTER_MISSED_CANDLES,
        on_issue=handle_clock_issue,
    )

    # 6. Assemble LiveEventEngine
    engine = LiveEventEngine(
        coordinator=coordinator,
        clock_checker=clock_checker,
        timeframe=TIMEFRAME,
        poll_interval_seconds=POLL_INTERVAL_SECONDS,
        clock_check_interval_seconds=CLOCK_CHECK_INTERVAL_SECONDS,
    )

    # 7. Now that the engine exists, bind the relay to it so every
    #    QualityIssue raised inside live_source's stream() reaches
    #    the engine's ordered event stream, not just the log line.
    quality_relay.bind(engine.on_quality_issue)

    return engine, live_source


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    """
    CLI entry point.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Phase 4 live runner",
    )

    parser.add_argument(
        "--once",
        action="store_true",
        help="Run exactly one live cycle and exit.",
    )

    args = parser.parse_args()

    engine, live_source = build_event_engine()

    # Connect MT5 explicitly, BEFORE the first cycle runs. Do not
    # rely on LiveMarketSource's lazy internal connect() here -- see
    # the module docstring's "IMPORTANT -- MT5 CONNECTION TIMING"
    # note for why the clock check would otherwise hit MT5 cold on
    # cycle one and raise "No IPC connection".
    print(f"Connecting to MT5 for {SYMBOL}...")
    live_source.connect()

    try:
        if args.once:
            print(f"Running ONE live cycle for {SYMBOL} {TIMEFRAME}...")
            events = engine.run_once()
            print(f"Cycle complete. {len(events)} event(s) produced:")
            for event in events:
                print(f"  {event.event_type.value} @ {event.occurred_at} -> {event.payload}")
            return 0

        print(f"Running live loop for {SYMBOL} {TIMEFRAME} (poll every {POLL_INTERVAL_SECONDS}s).")
        print("Session decisions remain under the existing Phase 3 session gate; this runner does not auto-approve them.")

        try:
            engine.run_forever()
        except KeyboardInterrupt:
            print("\nStopped by user.")

        return 0
    finally:
        live_source.disconnect()


if __name__ == "__main__":
    sys.exit(main())