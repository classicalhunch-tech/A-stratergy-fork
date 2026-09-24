"""
phase_03_paper/tests/test_report_generator_smoke.py

Smoke test for ReportEngine (phase_03_paper/ai/report_generator.py).
"""

import unittest
from datetime import datetime, timezone

from phase_03_paper.ai.report_generator import ReportEngine
from phase_03_paper.journal.journal import Journal, JournalEntry
from phase_03_paper.positions.manager import PositionState


def _closed_entry(trade_id, session, direction, result_r, recorded_at):
    return JournalEntry(
        trade_id=trade_id,
        session=session,
        direction=direction,
        position_state=PositionState.CLOSED,
        recorded_at=recorded_at,
        entry=1.0,
        stop=0.9,
        target=1.2,
        exit_price=1.1,
        exit_reason=None,
        result_r=result_r,
    )


class TestReportGeneratorSmoke(unittest.TestCase):
    def test_empty_journal_reports_no_trades(self):
        journal = Journal()
        engine = ReportEngine(journal)

        report = engine.generate(now=datetime(2026, 1, 1, tzinfo=timezone.utc))

        self.assertEqual(report.snapshot.total_trades, 0)
        self.assertIn("No closed trades yet", report.narrative)
        self.assertIsNone(report.best_trade)
        self.assertIsNone(report.worst_trade)

    def test_generates_narrative_with_best_and_worst_trade(self):
        journal = Journal()
        journal._entries.extend([
            _closed_entry(
                "t1", "London", "LONG", 2.5,
                datetime(2026, 1, 1, tzinfo=timezone.utc),
            ),
            _closed_entry(
                "t2", "NewYork", "SHORT", -1.0,
                datetime(2026, 1, 2, tzinfo=timezone.utc),
            ),
            _closed_entry(
                "t3", "London", "LONG", 1.5,
                datetime(2026, 1, 3, tzinfo=timezone.utc),
            ),
        ])
        journal._last_recorded.update({
            "t1": PositionState.CLOSED,
            "t2": PositionState.CLOSED,
            "t3": PositionState.CLOSED,
        })

        engine = ReportEngine(journal)
        report = engine.generate(now=datetime(2026, 1, 4, tzinfo=timezone.utc))

        self.assertEqual(report.snapshot.total_trades, 3)
        self.assertEqual(report.snapshot.wins, 2)
        self.assertEqual(report.snapshot.losses, 1)

        self.assertIsNotNone(report.best_trade)
        self.assertEqual(report.best_trade.trade_id, "t1")
        self.assertEqual(report.best_trade.result_r, 2.5)

        self.assertIsNotNone(report.worst_trade)
        self.assertEqual(report.worst_trade.trade_id, "t2")
        self.assertEqual(report.worst_trade.result_r, -1.0)

        self.assertIn("3 closed trades", report.narrative)
        self.assertIn("Best trade", report.narrative)
        self.assertIn("Worst trade", report.narrative)
        self.assertIn("London", report.narrative)

    def test_narrative_is_deterministic(self):
        journal = Journal()
        journal._entries.append(
            _closed_entry(
                "t1", "London", "LONG", 1.0,
                datetime(2026, 1, 1, tzinfo=timezone.utc),
            )
        )
        journal._last_recorded["t1"] = PositionState.CLOSED

        engine = ReportEngine(journal)
        now = datetime(2026, 1, 5, tzinfo=timezone.utc)

        report_a = engine.generate(now=now)
        report_b = engine.generate(now=now)

        self.assertEqual(report_a.narrative, report_b.narrative)


if __name__ == "__main__":
    unittest.main()