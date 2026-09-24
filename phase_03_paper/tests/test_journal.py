"""
phase_03_paper/tests/test_journal.py

Journal — Phase 3 lifecycle-transition journal tests.

PURPOSE
-------
Verify that Journal correctly records one entry per forward
lifecycle transition (OPEN -> MANAGED -> CLOSED), never duplicates a
transition already recorded, and exposes the expected read queries.
"""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from phase_03_paper.config import Phase3Config
from phase_03_paper.events.audit import AuditLog
from phase_03_paper.journal.journal import Journal, JournalEntry
from phase_03_paper.market.engine import Candle
from phase_03_paper.positions.manager import PositionState
from phase_03_paper.trading.paper_engine import (
    ExitReason,
    PaperTradeEngine,
    PaperTradeStatus,
)
from strategy.signals import SignalStatus, SignalType, TradeSignal


def _make_candle(
    timestamp: datetime,
    *,
    open_price: float = 1.1000,
    high: float = 1.1100,
    low: float = 1.0900,
    close: float = 1.1050,
) -> Candle:
    """Create a deterministic test candle."""

    return Candle(
        timestamp=timestamp,
        open=open_price,
        high=high,
        low=low,
        close=close,
    )


def _make_long_signal(
    timestamp: datetime,
    *,
    entry: float = 1.1000,
    stop: float = 1.0950,
    target: float = 1.1100,
    session: str = "TEST",
    bar_index: int = 0,
) -> TradeSignal:
    """Create a deterministic triggered LONG signal."""

    return TradeSignal(
        signal_type=SignalType.LONG,
        setup_timestamp=timestamp,
        triggered_at_time=timestamp,
        entry_price=entry,
        stop_loss=stop,
        take_profit=target,
        status=SignalStatus.TRIGGERED,
        session=session,
        bar_index=bar_index,
    )


def _build_engine() -> PaperTradeEngine:
    """Build a clean paper engine for each test."""

    return PaperTradeEngine(
        config=Phase3Config(),
        audit_log=AuditLog(),
    )


class TestJournal(unittest.TestCase):
    """Test suite for Journal lifecycle-transition recording."""

    def setUp(self) -> None:
        self.journal = Journal()

        self.t0 = datetime(
            2026,
            1,
            5,
            9,
            0,
            tzinfo=timezone.utc,
        )

        self.t1 = self.t0 + timedelta(minutes=5)
        self.t2 = self.t0 + timedelta(minutes=10)

    def open_long_trade(
        self,
        engine: PaperTradeEngine,
        *,
        session: str = "TEST",
        bar_index: int = 0,
        target: float = 1.1100,
    ):
        """Open one deterministic LONG paper trade."""

        signal = _make_long_signal(
            self.t0,
            session=session,
            bar_index=bar_index,
            target=target,
        )

        trade = engine.on_signal(
            signal,
            now=self.t0,
        )

        self.assertIsNotNone(trade)
        self.assertEqual(
            trade.status,
            PaperTradeStatus.OPEN,
        )

        return trade

    def test_record_on_open_creates_one_open_entry(self) -> None:
        """Recording immediately after opening produces one OPEN entry."""

        engine = _build_engine()
        trade = self.open_long_trade(engine)

        new_entries = self.journal.record(
            engine,
            current_candle_timestamp=self.t0,
        )

        self.assertEqual(len(new_entries), 1)
        self.assertEqual(new_entries[0].trade_id, trade.trade_id)
        self.assertEqual(new_entries[0].position_state, PositionState.OPEN)
        self.assertEqual(len(self.journal.all_entries()), 1)

    def test_repeated_record_same_state_does_not_duplicate(self) -> None:
        """Calling record() again with no state change adds nothing."""

        engine = _build_engine()
        self.open_long_trade(engine)

        self.journal.record(engine, current_candle_timestamp=self.t0)
        second_call = self.journal.record(engine, current_candle_timestamp=self.t0)

        self.assertEqual(len(second_call), 0)
        self.assertEqual(len(self.journal.all_entries()), 1)

    def test_record_transitions_to_managed_on_later_candle(self) -> None:
        """A still-open trade produces a MANAGED entry on a later candle."""

        engine = _build_engine()
        trade = self.open_long_trade(engine)

        self.journal.record(engine, current_candle_timestamp=self.t0)
        new_entries = self.journal.record(engine, current_candle_timestamp=self.t1)

        self.assertEqual(len(new_entries), 1)
        self.assertEqual(new_entries[0].position_state, PositionState.MANAGED)

        history = self.journal.entries_for_trade(trade.trade_id)
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0].position_state, PositionState.OPEN)
        self.assertEqual(history[1].position_state, PositionState.MANAGED)

    def test_record_transitions_to_closed(self) -> None:
        """A closed trade produces exactly one CLOSED entry with result_r."""

        engine = _build_engine()
        trade = self.open_long_trade(engine)

        self.journal.record(engine, current_candle_timestamp=self.t0)

        engine.on_candle(
            _make_candle(
                self.t1,
                high=1.1200,
            )
        )

        new_entries = self.journal.record(engine, current_candle_timestamp=self.t1)

        self.assertEqual(len(new_entries), 1)
        closed_entry = new_entries[0]

        self.assertEqual(closed_entry.position_state, PositionState.CLOSED)
        self.assertEqual(closed_entry.trade_id, trade.trade_id)
        self.assertIsNotNone(closed_entry.exit_price)
        self.assertIsNotNone(closed_entry.exit_reason)
        self.assertIsNotNone(closed_entry.result_r)

        history = self.journal.entries_for_trade(trade.trade_id)
        self.assertEqual(len(history), 2)
        self.assertEqual(history[-1].position_state, PositionState.CLOSED)

    def test_closed_entries_returns_only_closed(self) -> None:
        """closed_entries() must exclude OPEN/MANAGED entries."""

        engine = _build_engine()
        open_trade = self.open_long_trade(
            engine,
            target=1.1300,
        )
        closed_trade = self.open_long_trade(
            engine,
            session="SECOND",
            bar_index=1,
        )

        self.journal.record(engine, current_candle_timestamp=self.t0)

        engine.on_candle(
            _make_candle(
                self.t1,
                high=1.1200,
                low=1.0975,
            )
        )

        self.journal.record(engine, current_candle_timestamp=self.t1)

        closed = self.journal.closed_entries()

        self.assertEqual(len(closed), 1)
        self.assertEqual(closed[0].trade_id, closed_trade.trade_id)
        self.assertNotIn(
            open_trade.trade_id,
            [entry.trade_id for entry in closed],
        )

    def test_entries_for_session_filters_correctly(self) -> None:
        """entries_for_session() must only return matching-session entries."""

        engine = _build_engine()
        self.open_long_trade(engine, session="ALPHA")
        self.open_long_trade(engine, session="BETA", bar_index=1)

        self.journal.record(engine, current_candle_timestamp=self.t0)

        alpha_entries = self.journal.entries_for_session("ALPHA")

        self.assertEqual(len(alpha_entries), 1)
        self.assertEqual(alpha_entries[0].session, "ALPHA")

    def test_last_state_of_tracks_progress(self) -> None:
        """last_state_of() reflects the most recently recorded transition."""

        engine = _build_engine()
        trade = self.open_long_trade(engine)

        self.assertIsNone(self.journal.last_state_of(trade.trade_id))

        self.journal.record(engine, current_candle_timestamp=self.t0)
        self.assertEqual(
            self.journal.last_state_of(trade.trade_id),
            PositionState.OPEN,
        )

        self.journal.record(engine, current_candle_timestamp=self.t1)
        self.assertEqual(
            self.journal.last_state_of(trade.trade_id),
            PositionState.MANAGED,
        )

    def test_entry_snapshots_are_frozen(self) -> None:
        """JournalEntry is immutable — attempting to mutate it must fail."""

        engine = _build_engine()
        self.open_long_trade(engine)

        entries = self.journal.record(engine, current_candle_timestamp=self.t0)
        entry = entries[0]

        self.assertIsInstance(entry, JournalEntry)

        with self.assertRaises(Exception):
            entry.position_state = PositionState.CLOSED  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()