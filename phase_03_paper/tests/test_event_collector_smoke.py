"""
phase_03_paper/tests/test_event_collector_smoke.py

Smoke test for event_collector.run_historical_replay_with_events().

PURPOSE
-------
Confirms the event collector, run against the SAME real 500-candle
dataset and SAME wiring as test_full_replay_integration.py, produces
an event log that is CONSISTENT with what that already-passing test
proved about this dataset:

    Opened trades   : 0
    Closed trades   : 0
    Pending signals : 1

If the collector's event counts do not match those already-confirmed
numbers, that means the collector itself has a bug -- it must NEVER
report a different reality than the real pipeline already produced.

WIRING SOURCE
-------------
This wiring is copied verbatim from the reconstructed
test_full_replay_integration.py's _coordinator_factory, which itself
was built from confirmed real constructors in test_stage1_integration.py
and test_stage2_integration.py. No session decisions are auto-approved
here, matching the same deliberate choice made there.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from phase_03_paper.config import Phase3Config
from phase_03_paper.events.audit import AuditLog
from phase_03_paper.journal.journal import Journal
from phase_03_paper.market.engine import MarketDataEngine
from phase_03_paper.notifications.engine import NotificationEngine
from phase_03_paper.replay.event_collector import (
    ReplayEventType,
    run_historical_replay_with_events,
)
from phase_03_paper.runtime.coordinator import RuntimeCoordinator
from phase_03_paper.sessions.engine import SessionEngine
from phase_03_paper.signals.adapter import StrategyAdapter
from phase_03_paper.trading.paper_engine import PaperTradeEngine


def _coordinator_factory(market_data_engine: MarketDataEngine) -> RuntimeCoordinator:
    adapter = StrategyAdapter()
    session_engine = SessionEngine()
    notification_engine = NotificationEngine()

    return RuntimeCoordinator(
        session_engine=session_engine,
        notification_engine=notification_engine,
        market_data_engine=market_data_engine,
        strategy_adapter=adapter,
    )


def _build_wiring():
    """Returns (coordinator_factory, paper_engine, journal)."""
    paper_engine = PaperTradeEngine(
        config=Phase3Config(),
        audit_log=AuditLog(),
    )
    journal = Journal()
    return _coordinator_factory, paper_engine, journal


class TestEventCollectorSmoke(unittest.TestCase):
    def test_event_log_matches_known_replay_result(self) -> None:
        csv_path = str(
            Path(__file__).resolve().parents[2] / "your_data_file_500.csv"
        )

        coordinator_factory, paper_engine, journal = _build_wiring()

        summary = run_historical_replay_with_events(
            csv_path,
            coordinator_factory=coordinator_factory,
            paper_engine=paper_engine,
            journal=journal,
        )

        print("=" * 70)
        print("EVENT COLLECTOR SMOKE TEST")
        print("=" * 70)
        print(f"Total candles      : {summary.total_candles}")
        print(f"Ticks run          : {summary.ticks_run}")
        print(f"Opened trades      : {summary.opened_trades}")
        print(f"Closed trades      : {summary.closed_trades}")
        print(f"Event log size     : {summary.event_log.event_count}")
        print(f"  TRADE_OPENED     : {summary.event_log.opened_trade_count}")
        print(f"  TRADE_CLOSED     : {summary.event_log.closed_trade_count}")
        print(f"  JOURNAL_ENTRY    : {summary.event_log.journal_entry_count}")
        print("=" * 70)

        self.assertEqual(summary.total_candles, 500)
        self.assertEqual(summary.ticks_run, 500)
        self.assertEqual(summary.opened_trades, 0)
        self.assertEqual(summary.closed_trades, 0)
        self.assertEqual(
            summary.event_log.opened_trade_count, summary.opened_trades
        )
        self.assertEqual(
            summary.event_log.closed_trade_count, summary.closed_trades
        )
        self.assertEqual(summary.event_log.journal_entry_count, 0)

        for event in summary.event_log.all_events():
            self.assertIn(
                event.event_type,
                (
                    ReplayEventType.TRADE_OPENED,
                    ReplayEventType.TRADE_CLOSED,
                    ReplayEventType.JOURNAL_ENTRY,
                ),
            )
            self.assertIsNotNone(event.timestamp)


if __name__ == "__main__":
    unittest.main()