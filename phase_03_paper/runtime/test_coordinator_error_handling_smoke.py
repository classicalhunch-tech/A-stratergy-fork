"""
phase_03_paper/runtime/test_coordinator_error_handling_smoke.py

Smoke test for RuntimeCoordinator error handling:

    component raises exception
        |
        v
    RuntimeCoordinator catches it
        |
        v
    tick() does not crash
        |
        v
    last_error is populated
        |
        v
    AuditLog receives an ERROR event

This test does NOT construct a fake/mock component to force the
failure. It uses a real MarketDataEngine wired to a minimal
MarketDataSource that yields one malformed raw record (missing the
required "close" field). MarketDataEngine's own _normalize() already
raises KeyError on a malformed record (see market/engine.py) and
re-raises after recording its own internal ERROR status -- this test
simply confirms RuntimeCoordinator catches that real exception rather
than letting it propagate and crash the runtime.

It also confirms the coordinator recovers on the next tick(): once
the failing source's single record has been consumed and its
generator is exhausted, next_candle() returns None (not an error),
so a subsequent tick() should evaluate cleanly and clear last_error.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterator

from phase_03_paper.config import DEFAULT_CONFIG
from phase_03_paper.events.audit import AuditLog
from phase_03_paper.market.engine import (
    MarketDataEngine,
    MarketDataSource,
    MarketDataStatus,
)
from phase_03_paper.models import NotificationSeverity
from phase_03_paper.notifications.engine import NotificationEngine
from phase_03_paper.runtime.coordinator import RuntimeCoordinator
from phase_03_paper.sessions.engine import SessionEngine


class _FailingSource(MarketDataSource):
    """
    A minimal, real MarketDataSource that yields one malformed raw
    record (missing the required "close" field), to trigger a real
    normalization failure inside MarketDataEngine -- not a simulated
    or mocked error.
    """

    def stream(self) -> Iterator[dict]:
        yield {
            "timestamp": "2026-01-01T07:00:00+00:00",
            "open": 1.1000,
            "high": 1.1010,
            "low": 1.0990,
            # "close" intentionally omitted.
        }


def run_smoke_test() -> None:
    audit_log = AuditLog()

    session_engine = SessionEngine(config=DEFAULT_CONFIG, audit_log=audit_log)
    notification_engine = NotificationEngine(
        config=DEFAULT_CONFIG, audit_log=audit_log
    )
    market_data_engine = MarketDataEngine(source=_FailingSource())

    coordinator = RuntimeCoordinator(
        session_engine=session_engine,
        notification_engine=notification_engine,
        market_data_engine=market_data_engine,
        audit_log=audit_log,
    )

    now = datetime(2026, 1, 1, 6, 0, tzinfo=timezone.utc)

    # ------------------------------------------------------------
    # TICK 1: the malformed candle triggers a real KeyError inside
    # MarketDataEngine. The coordinator must catch it, not crash.
    # ------------------------------------------------------------

    status = coordinator.tick(now=now)

    assert status.tick_count == 1
    assert status.healthy is False
    assert status.last_error is not None
    assert "close" in status.last_error

    assert market_data_engine.status().status == MarketDataStatus.ERROR

    error_events = audit_log.events_of_type("runtime_tick_error")
    assert len(error_events) == 1
    assert error_events[0].severity == NotificationSeverity.ERROR

    # ------------------------------------------------------------
    # TICK 2: the failing source's single record has been consumed
    # and its generator is now exhausted, so next_candle() returns
    # None (not an error). The coordinator should recover cleanly.
    # ------------------------------------------------------------

    status = coordinator.tick(now=now)

    assert status.tick_count == 2
    assert status.healthy is True
    assert status.last_error is None

    # No additional error event should have been recorded on the
    # recovered tick.
    assert len(audit_log.events_of_type("runtime_tick_error")) == 1


if __name__ == "__main__":
    run_smoke_test()