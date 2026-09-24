"""
phase_03_paper/tests/test_position_manager.py

PositionManager â€” Phase 3 lifecycle tests.

PURPOSE
-------
Verify that PositionManager correctly derives position lifecycle
states from PaperTradeEngine records.

Tests cover:

    - OPEN state on setup candle
    - MANAGED state on later candles
    - CLOSED state on closed trades
    - Missing timestamp fallbacks
    - Active views vs all-trade views
    - Session-based filtering (open and all)
    - Directional checks (LONG vs SHORT)
    - Exposure count tracking
"""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from phase_03_paper.config import Phase3Config
from phase_03_paper.events.audit import AuditLog
from phase_03_paper.market.engine import Candle
from phase_03_paper.positions.manager import (
    PositionManager,
    PositionState,
)
from phase_03_paper.trading.paper_engine import (
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
    """
    Create a deterministic triggered LONG signal.

    bar_index is included in TradeSignal's duplicate-signal identity
    (see PaperTradeEngine._signal_key's fallback branch). Tests that
    open more than one trade in the same test method must pass a
    distinct bar_index per call, or the engine will correctly treat
    the second signal as a duplicate of the first and return None.
    """

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


def _make_short_signal(
    timestamp: datetime,
    *,
    entry: float = 1.1000,
    stop: float = 1.1050,
    target: float = 1.0900,
    session: str = "TEST",
    bar_index: int = 0,
) -> TradeSignal:
    """Create a deterministic triggered SHORT signal. See _make_long_signal
    for why bar_index matters when opening more than one trade per test."""

    return TradeSignal(
        signal_type=SignalType.SHORT,
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


class TestPositionManager(unittest.TestCase):
    """Test suite for PositionManager lifecycle behavior."""

    def setUp(self) -> None:
        self.manager = PositionManager()

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

    def open_short_trade(
        self,
        engine: PaperTradeEngine,
        *,
        session: str = "TEST",
        bar_index: int = 0,
    ):
        """Open one deterministic SHORT paper trade."""

        signal = _make_short_signal(
            self.t0,
            session=session,
            bar_index=bar_index,
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

    def test_open_same_candle_is_open(self) -> None:
        """A trade remains OPEN on the candle where it was opened."""

        engine = _build_engine()
        trade = self.open_long_trade(engine)

        views = self.manager.views(
            engine,
            current_candle_timestamp=self.t0,
        )

        self.assertEqual(len(views), 1)
        self.assertIs(views[0].trade, trade)
        self.assertEqual(
            views[0].position_state,
            PositionState.OPEN,
        )

    def test_open_later_candle_becomes_managed(self) -> None:
        """A still-open trade becomes MANAGED on a later candle."""

        engine = _build_engine()
        trade = self.open_long_trade(engine)

        views = self.manager.views(
            engine,
            current_candle_timestamp=self.t1,
        )

        self.assertEqual(len(views), 1)
        self.assertIs(views[0].trade, trade)
        self.assertEqual(
            views[0].position_state,
            PositionState.MANAGED,
        )

    def test_same_timestamp_remains_open(self) -> None:
        """Equal timestamps must preserve the same-candle OPEN state."""

        engine = _build_engine()
        trade = self.open_long_trade(engine)

        state = self.manager.position_state_of(
            trade,
            current_candle_timestamp=trade.opened_at,
        )

        self.assertEqual(
            state,
            PositionState.OPEN,
        )

    def test_missing_current_timestamp_is_open(self) -> None:
        """Without a current candle timestamp, an open trade is OPEN."""

        engine = _build_engine()
        trade = self.open_long_trade(engine)

        state = self.manager.position_state_of(
            trade,
            current_candle_timestamp=None,
        )

        self.assertEqual(
            state,
            PositionState.OPEN,
        )

    def test_closed_trade_is_closed(self) -> None:
        """A closed PaperTrade must produce CLOSED."""

        engine = _build_engine()
        trade = self.open_long_trade(engine)

        closed_trades = engine.on_candle(
            _make_candle(
                self.t1,
                high=1.1200,
            )
        )

        self.assertEqual(len(closed_trades), 1)
        self.assertEqual(
            trade.status,
            PaperTradeStatus.CLOSED,
        )

        state = self.manager.position_state_of(
            trade,
            current_candle_timestamp=self.t1,
        )

        self.assertEqual(
            state,
            PositionState.CLOSED,
        )

    def test_views_returns_only_open_positions(self) -> None:
        """views() must exclude closed trades."""

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

        engine.on_candle(
            _make_candle(
                self.t1,
                high=1.1200,
                low=1.0975,
            )
        )

        open_trades = engine.open_trades()

        self.assertEqual(len(open_trades), 1)
        self.assertIs(open_trades[0], open_trade)
        self.assertNotIn(closed_trade, open_trades)

        views = self.manager.views(
            engine,
            current_candle_timestamp=self.t1,
        )

        self.assertEqual(len(views), 1)
        self.assertIs(views[0].trade, open_trade)

    def test_all_views_includes_closed_trades(self) -> None:
        """all_views() must include both open and closed trades."""

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

        engine.on_candle(
            _make_candle(
                self.t1,
                high=1.1200,
                low=1.0975,
            )
        )

        views = self.manager.all_views(
            engine,
            current_candle_timestamp=self.t1,
        )

        self.assertEqual(len(views), 2)

        trade_map = {
            view.trade.trade_id: view.position_state
            for view in views
        }

        self.assertEqual(
            trade_map[open_trade.trade_id],
            PositionState.MANAGED,
        )
        self.assertEqual(
            trade_map[closed_trade.trade_id],
            PositionState.CLOSED,
        )

    def test_positions_by_session_filters_open_positions(self) -> None:
        """positions_by_session() must filter current open positions."""

        engine = _build_engine()
        alpha_trade = self.open_long_trade(
            engine,
            session="ALPHA",
        )
        beta_trade = self.open_long_trade(
            engine,
            session="BETA",
            bar_index=1,
        )

        alpha_views = self.manager.positions_by_session(
            engine,
            session="ALPHA",
            current_candle_timestamp=self.t1,
        )

        self.assertEqual(len(alpha_views), 1)
        self.assertIs(
            alpha_views[0].trade,
            alpha_trade,
        )
        self.assertNotIn(
            beta_trade,
            [view.trade for view in alpha_views],
        )

    def test_all_views_by_session_includes_closed_trade(self) -> None:
        """all_views_by_session() must include closed lifecycle records."""

        engine = _build_engine()
        alpha_trade = self.open_long_trade(
            engine,
            session="ALPHA",
        )

        engine.on_candle(
            _make_candle(
                self.t1,
                high=1.1200,
            )
        )

        views = self.manager.all_views_by_session(
            engine,
            session="ALPHA",
            current_candle_timestamp=self.t1,
        )

        self.assertEqual(len(views), 1)
        self.assertIs(
            views[0].trade,
            alpha_trade,
        )
        self.assertEqual(
            views[0].position_state,
            PositionState.CLOSED,
        )

    def test_has_open_long_position(self) -> None:
        """has_open_position() detects an open LONG."""

        engine = _build_engine()
        self.open_long_trade(engine)

        self.assertTrue(
            self.manager.has_open_position(
                engine,
                SignalType.LONG,
            )
        )

    def test_has_open_short_position(self) -> None:
        """has_open_position() detects an open SHORT."""

        engine = _build_engine()
        self.open_short_trade(engine)

        self.assertTrue(
            self.manager.has_open_position(
                engine,
                SignalType.SHORT,
            )
        )

    def test_direction_check_does_not_confuse_long_and_short(self) -> None:
        """LONG and SHORT exposure checks must remain independent."""

        engine = _build_engine()
        self.open_long_trade(engine)

        self.assertTrue(
            self.manager.has_open_position(
                engine,
                SignalType.LONG,
            )
        )
        self.assertFalse(
            self.manager.has_open_position(
                engine,
                SignalType.SHORT,
            )
        )

    def test_open_exposure_count(self) -> None:
        """open_exposure_count() returns the number of open trades."""

        engine = _build_engine()

        self.assertEqual(
            self.manager.open_exposure_count(engine),
            0,
        )

        self.open_long_trade(
            engine,
            session="ONE",
        )
        self.open_long_trade(
            engine,
            session="TWO",
            bar_index=1,
        )

        self.assertEqual(
            self.manager.open_exposure_count(engine),
            2,
        )

        engine.on_candle(
            _make_candle(
                self.t1,
                high=1.1200,
            )
        )

        self.assertEqual(
            self.manager.open_exposure_count(engine),
            0,
        )


if __name__ == "__main__":
    unittest.main()