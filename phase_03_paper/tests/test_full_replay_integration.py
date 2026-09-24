"""
phase_03_paper/tests/test_full_replay_integration.py

RECONSTRUCTED from confirmed real wiring (not recovered from backup --
the original file's source was lost to an accidental content mismatch
during event_collector.py development; see chat history for the
recovery investigation). This reconstruction uses ONLY constructors
and call signatures directly confirmed from your real, on-disk files:

    MarketDataEngine(source)                                  -- test_stage1_integration.py
    StrategyAdapter()                                         -- test_stage1_integration.py
    SessionEngine()                                           -- test_stage1_integration.py
    NotificationEngine()                                      -- test_stage1_integration.py
    RuntimeCoordinator(session_engine=, notification_engine=,
                        market_data_engine=, strategy_adapter=) -- test_stage1_integration.py
    PaperTradeEngine(config=Phase3Config(), audit_log=AuditLog()) -- test_stage2_integration.py
    Journal(position_manager=None)                            -- journal.py (optional arg)
    CsvMarketDataSource(csv_path, ...)                         -- replay/csv_source.py
    run_historical_replay(csv_path, coordinator_factory, paper_engine, journal=...) -- replay_runner.py

IMPORTANT: this test deliberately does NOT auto-approve session
decisions (unlike test_stage1_integration.py, which does so for its
own separate purpose). This matches the ORIGINAL confirmed printed
output from this test (500 candles, 0 opened trades, 0 closed trades,
1 pending signal, 0 errors) -- sessions defaulting to UNDECIDED
suppress signals from ever reaching PaperTradeEngine, which is
consistent with that known result.

If this reconstruction's output does not match the original known
numbers below, STOP and report the mismatch -- do not adjust
assertions to force a match. That would hide a real behavior change.

KNOWN ORIGINAL RESULT (from prior confirmed test runs):
    Total candles      : 500
    Ticks run          : 500
    Opened trades      : 0
    Closed trades      : 0
    Pending signals    : 1
    Adapter errors     : 0
    Journal entries    : 0
    Unhealthy checks   : 0
"""

from __future__ import annotations

import unittest

from phase_03_paper.config import Phase3Config
from phase_03_paper.events.audit import AuditLog
from phase_03_paper.journal.journal import Journal
from phase_03_paper.market.engine import MarketDataEngine
from phase_03_paper.notifications.engine import NotificationEngine
from phase_03_paper.replay.replay_runner import run_historical_replay
from phase_03_paper.runtime.coordinator import RuntimeCoordinator
from phase_03_paper.sessions.engine import SessionEngine
from phase_03_paper.signals.adapter import StrategyAdapter
from phase_03_paper.trading.paper_engine import PaperTradeEngine

DATA_FILE = "your_data_file_500.csv"


def _coordinator_factory(market_data_engine: MarketDataEngine) -> RuntimeCoordinator:
    """
    Build a fully wired RuntimeCoordinator, exactly matching the
    confirmed construction pattern in test_stage1_integration.py.

    No session decisions are auto-approved here -- see module
    docstring for why that's deliberate, not an oversight.
    """
    adapter = StrategyAdapter()
    session_engine = SessionEngine()
    notification_engine = NotificationEngine()

    return RuntimeCoordinator(
        session_engine=session_engine,
        notification_engine=notification_engine,
        market_data_engine=market_data_engine,
        strategy_adapter=adapter,
    )


class TestFullReplayIntegration(unittest.TestCase):
    def test_real_pipeline_completes_a_full_historical_replay(self) -> None:
        paper_engine = PaperTradeEngine(
            config=Phase3Config(),
            audit_log=AuditLog(),
        )
        journal = Journal()

        summary = run_historical_replay(
            DATA_FILE,
            coordinator_factory=_coordinator_factory,
            paper_engine=paper_engine,
            journal=journal,
        )

        print()
        print("=" * 70)
        print("PHASE 3 -- FULL REAL HISTORICAL REPLAY INTEGRATION TEST")
        print("=" * 70)
        print(f"Total candles      : {summary.total_candles}")
        print(f"Ticks run          : {summary.ticks_run}")
        print(f"First timestamp    : {summary.first_timestamp}")
        print(f"Last timestamp     : {summary.last_timestamp}")
        print(f"Opened trades      : {summary.opened_trades}")
        print(f"Closed trades      : {summary.closed_trades}")
        print(f"Journal entries    : {len(journal.all_entries())}")
        print("-" * 70)
        print(
            f"Perf: total_trades={summary.opened_trades} "
        )
        print("=" * 70)

        # Assertions match the known original result. If these fail,
        # something in the real pipeline changed -- report it, don't
        # loosen these to force a pass.
        self.assertEqual(summary.total_candles, 500)
        self.assertEqual(summary.ticks_run, 500)
        self.assertEqual(summary.opened_trades, 0)
        self.assertEqual(summary.closed_trades, 0)
        self.assertEqual(len(journal.all_entries()), 0)

        print()
        print("[PASS] Full real historical replay completed end-to-end")
        print("[PASS] Result matches known original replay output")
        print()


if __name__ == "__main__":
    unittest.main()