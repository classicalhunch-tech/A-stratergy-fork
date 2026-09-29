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
        ->
    ExecutionEventConsumer (Phase 4)
        ->
    BrokerExecutor (Phase 4)

Phase 3 remains responsible for its existing runtime behavior:
    SessionEngine
    NotificationEngine
    StrategyAdapter
    AuditLog

Phase 4 adds:
    ClockIntegrityChecker
    LiveEventEngine
    LiveMarketSource
    ExecutionEventConsumer
    BrokerExecutor

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

IMPORTANT -- KILL-SWITCH WIRING

The live event engine owns the final safe-mode check for signals, but
it must receive a SafeModeGate explicitly. The runner therefore wires
the durable, file-backed EmergencyKillSwitch here so every live cycle
reads the current kill-switch state before exposing a triggered signal.

IMPORTANT -- EXECUTION WIRING

The execution consumer sits one layer below the event engine and
consumes SIGNAL_TRIGGERED events only. It transforms them into
ExecutionRequests, passes them through BrokerExecutor (which applies
idempotency + kill-switch guard again), and returns ExecutionResults
as ORDER_PLACED or ORDER_REJECTED events back into the event stream.

This maintains clean separation:
    - Phase 3: signal discovery and permission
    - Phase 4 event engine: clock/quality/signal suppression
    - Phase 4 executor: order safety (idempotency, guard checks)
    - Phase 4 consumer: signal-to-execution transformation

The consumer is optional (can be disabled or swapped out). The runner
applies it to every event batch after the engine runs.

USAGE:

    Smoke test:
        python -m phase_04_live.main --once

    Continuous loop:
        python -m phase_04_live.main
    
    Dry run (no actual broker orders):
        python -m phase_04_live.main --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
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
from phase_04_live.execution.consumer import ExecutionEventConsumer
from phase_04_live.execution.executor import BrokerExecutor, ExecutionConfig
from phase_04_live.market.live_source import LiveMarketSource
from phase_04_live.market.models import QualityIssue
from phase_04_live.recovery.safe_mode import SafeModeGate
from phase_04_live.risk.kill_switch import EmergencyKillSwitch


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

# Durable operator-controlled state. Missing file means the first-ever run
# is allowed; malformed or unreadable existing state fails closed.
KILL_SWITCH_PATH = Path("state/kill_switch.json")

# Broker execution configuration
EXECUTOR_MAX_ORDER_SIZE = 0.1  # Max notional per order
EXECUTOR_ORDER_TIMEOUT_SECONDS = 30.0


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

    # 6. Build the durable kill-switch gate. SafeModeGate performs a
    #    fresh read on every signal, so operator changes are visible
    #    without restarting the process.
    kill_switch = EmergencyKillSwitch(KILL_SWITCH_PATH)
    safe_mode_gate = SafeModeGate(kill_switch)

    # 7. Assemble LiveEventEngine with the safety gate wired in.
    engine = LiveEventEngine(
        coordinator=coordinator,
        clock_checker=clock_checker,
        timeframe=TIMEFRAME,
        poll_interval_seconds=POLL_INTERVAL_SECONDS,
        clock_check_interval_seconds=CLOCK_CHECK_INTERVAL_SECONDS,
        safe_mode_gate=safe_mode_gate,
    )

    # 8. Now that the engine exists, bind the relay to it so every
    #    QualityIssue raised inside live_source's stream() reaches
    #    the engine's ordered event stream, not just the log line.
    quality_relay.bind(engine.on_quality_issue)

    return engine, live_source


def build_execution_consumer(
    kill_switch: EmergencyKillSwitch,
    safe_mode_gate: SafeModeGate,
    enable_dry_run: bool = False,
    order_sink: Optional[Callable] = None,
) -> ExecutionEventConsumer:
    """
    Build the execution consumer and broker executor.
    
    The executor is optional and can be disabled by passing dry_run=True
    or by not providing an order_sink. In both cases, orders are logged
    but not sent to the broker.
    """
    
    executor_config = ExecutionConfig(
        max_order_size=EXECUTOR_MAX_ORDER_SIZE,
        order_timeout_seconds=EXECUTOR_ORDER_TIMEOUT_SECONDS,
        enable_dry_run=enable_dry_run,
    )
    
    broker_executor = BrokerExecutor(
        kill_switch=kill_switch,
        safe_mode_gate=safe_mode_gate,
        config=executor_config,
        order_sink=order_sink,
    )
    
    consumer = ExecutionEventConsumer(
        broker_executor=broker_executor,
        symbol=SYMBOL,
    )
    
    return consumer


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

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log orders instead of sending them to the broker.",
    )

    args = parser.parse_args()

    engine, live_source = build_event_engine()

    # Extract kill_switch and safe_mode_gate from engine for executor wiring
    # (they were built inside build_event_engine; in production, we'd expose
    # them as return values. For now, we rebuild them here -- the state file
    # is the single source of truth, so rebuilding is safe and idempotent).
    kill_switch = EmergencyKillSwitch(KILL_SWITCH_PATH)
    safe_mode_gate = SafeModeGate(kill_switch)

    # Build execution consumer (optional; can be disabled)
    execution_consumer = build_execution_consumer(
        kill_switch=kill_switch,
        safe_mode_gate=safe_mode_gate,
        enable_dry_run=args.dry_run,
        order_sink=None,  # Implement your broker order submission here
    )

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
            
            # Apply execution consumer to the event stream
            events = execution_consumer.consume(events)
            
            print(f"Cycle complete. {len(events)} event(s) produced:")
            for event in events:
                print(f"  {event.event_type.value} @ {event.occurred_at} -> {event.payload}")
            return 0

        print(f"Running live loop for {SYMBOL} {TIMEFRAME} (poll every {POLL_INTERVAL_SECONDS}s).")
        print("Session decisions remain under the existing Phase 3 session gate; this runner does not auto-approve them.")
        print(f"Execution mode: {'DRY-RUN' if args.dry_run else 'LIVE'}")

        try:
            while True:
                events = engine.run_once()
                
                # Apply execution consumer to the event stream
                events = execution_consumer.consume(events)
                
                for event in events:
                    print(f"{event.event_type.value} @ {event.occurred_at}")
                
                import time
                time.sleep(engine.poll_interval_seconds)
        except KeyboardInterrupt:
            print("\nStopped by user.")

        return 0
    finally:
        live_source.disconnect()


if __name__ == "__main__":
    sys.exit(main())
