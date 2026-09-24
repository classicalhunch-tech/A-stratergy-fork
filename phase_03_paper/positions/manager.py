"""
phase_03_paper/positions/manager.py

PositionManager — Phase 3 position lifecycle layer.

PURPOSE
-------
PositionManager provides a read-only view of the position lifecycle
owned by PaperTradeEngine.

Lifecycle:

    NO_POSITION -> OPEN -> MANAGED -> CLOSED

PaperTrade remains the source of truth for:

    - direction
    - entry price
    - stop loss
    - take profit
    - timestamps
    - session
    - exit reason
    - result R
    - trade status

PositionManager does NOT duplicate those fields.
It only derives the current lifecycle state.


STATE MODEL
-----------
OPEN
    The trade exists and has not yet reached a later candle than the
    candle on which it opened.

MANAGED
    The trade is still open and the current candle timestamp is
    strictly later than the trade's opened_at timestamp.

CLOSED
    PaperTradeEngine has already closed the trade.

NO_POSITION
    Used when a supplied trade is not currently an active position.


CAUSAL RULE
-----------
A trade opened on candle N is OPEN during candle N.
It becomes eligible for MANAGED state beginning with a later candle timestamp.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import List, Optional

from phase_03_paper.trading.paper_engine import (
    PaperTrade,
    PaperTradeEngine,
    PaperTradeStatus,
)
from strategy.signals import SignalType


class PositionState(Enum):
    """Derived lifecycle state for a paper position."""

    NO_POSITION = "NO_POSITION"
    OPEN = "OPEN"
    MANAGED = "MANAGED"
    CLOSED = "CLOSED"


@dataclass(frozen=True)
class PositionView:
    """Read-only view of one PaperTrade and its derived lifecycle state."""

    trade: PaperTrade
    position_state: PositionState


class PositionManager:
    """
    Stateless position lifecycle classifier.

    PositionManager never stores or mutates trades.
    It reads PaperTrade records from PaperTradeEngine and derives
    PositionState from the trade's current status and timestamps.
    """

    @staticmethod
    def position_state_of(
        trade: PaperTrade,
        current_candle_timestamp: Optional[datetime] = None,
    ) -> PositionState:
        """
        Classify one trade's current lifecycle state.
        """
        if trade.status == PaperTradeStatus.CLOSED:
            return PositionState.CLOSED

        if trade.status != PaperTradeStatus.OPEN:
            return PositionState.NO_POSITION

        if (
            trade.opened_at is None
            or current_candle_timestamp is None
        ):
            return PositionState.OPEN

        if current_candle_timestamp > trade.opened_at:
            return PositionState.MANAGED

        return PositionState.OPEN

    def views(
        self,
        paper_engine: PaperTradeEngine,
        current_candle_timestamp: Optional[datetime] = None,
    ) -> List[PositionView]:
        """
        Return views of currently open paper positions.
        CLOSED trades are intentionally excluded.
        """
        return [
            PositionView(
                trade=trade,
                position_state=self.position_state_of(
                    trade=trade,
                    current_candle_timestamp=current_candle_timestamp,
                ),
            )
            for trade in paper_engine.open_trades()
        ]

    def all_views(
        self,
        paper_engine: PaperTradeEngine,
        current_candle_timestamp: Optional[datetime] = None,
    ) -> List[PositionView]:
        """
        Return lifecycle views for all recorded trades (including CLOSED).
        """
        return [
            PositionView(
                trade=trade,
                position_state=self.position_state_of(
                    trade=trade,
                    current_candle_timestamp=current_candle_timestamp,
                ),
            )
            for trade in paper_engine.all_trades()
        ]

    def positions_by_session(
        self,
        paper_engine: PaperTradeEngine,
        session: str,
        current_candle_timestamp: Optional[datetime] = None,
    ) -> List[PositionView]:
        """Return currently open positions belonging to one session."""
        return [
            view
            for view in self.views(
                paper_engine=paper_engine,
                current_candle_timestamp=current_candle_timestamp,
            )
            if view.trade.session == session
        ]

    def all_views_by_session(
        self,
        paper_engine: PaperTradeEngine,
        session: str,
        current_candle_timestamp: Optional[datetime] = None,
    ) -> List[PositionView]:
        """Return all lifecycle views belonging to one session (open & closed)."""
        return [
            view
            for view in self.all_views(
                paper_engine=paper_engine,
                current_candle_timestamp=current_candle_timestamp,
            )
            if view.trade.session == session
        ]

    @staticmethod
    def has_open_position(
        paper_engine: PaperTradeEngine,
        direction: SignalType,
    ) -> bool:
        """Return True if an open position exists in the requested direction."""
        return any(
            trade.direction == direction
            for trade in paper_engine.open_trades()
        )

    @staticmethod
    def open_exposure_count(
        paper_engine: PaperTradeEngine,
    ) -> int:
        """Return the number of currently open paper positions."""
        return len(paper_engine.open_trades())


__all__ = [
    "PositionState",
    "PositionView",
    "PositionManager",
]