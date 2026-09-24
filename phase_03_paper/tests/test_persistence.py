"""
phase_03_paper/tests/test_persistence.py

Tests for Persistence (SQLite storage for trades + journal entries).
"""

import os
import tempfile
import unittest
from datetime import datetime, timezone

from phase_03_paper.journal.journal import JournalEntry
from phase_03_paper.persistence.persistence import Persistence
from phase_03_paper.positions.manager import PositionState
from phase_03_paper.trading.paper_engine import (
    ExitReason,
    PaperTrade,
    PaperTradeStatus,
)
from strategy.signals import SignalType


def _make_trade(trade_id="t1", status=PaperTradeStatus.OPEN):
    return PaperTrade(
        trade_id=trade_id,
        direction=SignalType.LONG,
        session="London",
        requested_entry=100.0,
        stop=95.0,
        target=110.0,
        entry=100.5,
        opened_at=datetime(2024, 1, 1, 9, 0, tzinfo=timezone.utc),
        status=status,
        signal_key=("zone", "abc123", "LONG"),
    )


def _make_journal_entry(trade_id="t1", state=PositionState.OPEN):
    return JournalEntry(
        trade_id=trade_id,
        session="London",
        direction="LONG",
        position_state=state,
        recorded_at=datetime(2024, 1, 1, 9, 0, tzinfo=timezone.utc),
        entry=100.5,
        stop=95.0,
        target=110.0,
    )


class TestPersistenceTrades(unittest.TestCase):

    def setUp(self):
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self.db_path = path
        self.persistence = Persistence(self.db_path)

    def tearDown(self):
        os.remove(self.db_path)

    def test_save_and_load_single_trade(self):
        trade = _make_trade()
        self.persistence.save_trade(trade)

        loaded = self.persistence.load_trades()
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].trade_id, "t1")
        self.assertEqual(loaded[0].direction, SignalType.LONG)
        self.assertEqual(loaded[0].status, PaperTradeStatus.OPEN)
        self.assertEqual(loaded[0].signal_key, ("zone", "abc123", "LONG"))
        self.assertEqual(loaded[0].opened_at, trade.opened_at)

    def test_upsert_overwrites_existing_trade(self):
        trade = _make_trade(status=PaperTradeStatus.OPEN)
        self.persistence.save_trade(trade)

        trade.status = PaperTradeStatus.CLOSED
        trade.closed_at = datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)
        trade.exit_price = 110.0
        trade.exit_reason = ExitReason.TARGET_HIT
        trade.result_r = 2.0
        self.persistence.save_trade(trade)

        loaded = self.persistence.load_trades()
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].status, PaperTradeStatus.CLOSED)
        self.assertEqual(loaded[0].exit_reason, ExitReason.TARGET_HIT)
        self.assertEqual(loaded[0].result_r, 2.0)

    def test_save_multiple_trades(self):
        self.persistence.save_trades([_make_trade("t1"), _make_trade("t2")])
        loaded = self.persistence.load_trades()
        self.assertEqual({t.trade_id for t in loaded}, {"t1", "t2"})

    def test_trade_without_signal_key_round_trips_as_none(self):
        trade = _make_trade()
        trade.signal_key = None
        self.persistence.save_trade(trade)
        loaded = self.persistence.load_trades()
        self.assertIsNone(loaded[0].signal_key)


class TestPersistenceJournal(unittest.TestCase):

    def setUp(self):
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self.db_path = path
        self.persistence = Persistence(self.db_path)

    def tearDown(self):
        os.remove(self.db_path)

    def test_save_and_load_single_entry(self):
        entry = _make_journal_entry()
        self.persistence.save_journal_entry(entry)

        loaded = self.persistence.load_journal_entries()
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].trade_id, "t1")
        self.assertEqual(loaded[0].position_state, PositionState.OPEN)

    def test_append_only_preserves_lifecycle_history(self):
        self.persistence.save_journal_entry(
            _make_journal_entry(state=PositionState.OPEN)
        )
        self.persistence.save_journal_entry(
            _make_journal_entry(state=PositionState.MANAGED)
        )
        self.persistence.save_journal_entry(
            _make_journal_entry(state=PositionState.CLOSED)
        )

        loaded = self.persistence.load_journal_entries()
        self.assertEqual(len(loaded), 3)
        self.assertEqual(
            [e.position_state for e in loaded],
            [PositionState.OPEN, PositionState.MANAGED, PositionState.CLOSED],
        )

    def test_entries_for_trade_filters_correctly(self):
        self.persistence.save_journal_entries(
            [_make_journal_entry("t1"), _make_journal_entry("t2")]
        )
        loaded = self.persistence.journal_entries_for_trade("t1")
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].trade_id, "t1")


class TestPersistenceRestart(unittest.TestCase):
    """Reopening the same db_path must see previously saved data."""

    def setUp(self):
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self.db_path = path

    def tearDown(self):
        os.remove(self.db_path)

    def test_reopening_persistence_sees_prior_data(self):
        p1 = Persistence(self.db_path)
        p1.save_trade(_make_trade())
        p1.save_journal_entry(_make_journal_entry())

        # Simulate a restart: new Persistence instance, same file path.
        p2 = Persistence(self.db_path)
        self.assertEqual(len(p2.load_trades()), 1)
        self.assertEqual(len(p2.load_journal_entries()), 1)


if __name__ == "__main__":
    unittest.main()