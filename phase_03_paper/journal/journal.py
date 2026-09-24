"""
phase_03_paper/journal/journal.py

Journal â€” Phase 3 trade journal.

PURPOSE
-------
Journal maintains an in-memory, append-only log of every meaningful
lifecycle transition a PaperTrade goes through, as classified by
PositionManager:

    OPEN -> MANAGED -> CLOSED

Journal does NOT decide position state itself â€” it defers entirely
to PositionManager.position_state_of() so there is only one
implementation of the lifecycle rules in the codebase.

Journal DOES:
    - record one entry the first time a trade is observed OPEN
    - record one entry the first time a trade transitions to MANAGED
    - record one entry the first time a trade is observed CLOSED
    - never record the same transition twice for the same trade
    - never move a trade "backwards" through the lifecycle
    - provide simple read queries over the recorded history

Journal explicitly does NOT:
    - persist anything to disk (that is Persistence, later in the
      roadmap)
    - mutate PaperTrade or PositionManager
    - compute performance statistics (that is a later phase)


SNAPSHOTTING
------------
Each JournalEntry copies the relevant PaperTrade fields *at the
moment the transition was recorded*. PaperTrade is mutable and keeps
changing after OPEN (entry stays fixed, but exit_price/exit_reason/
result_r/closed_at only become meaningful at CLOSED) â€” a JournalEntry
freezes what was true at that instant, so earlier entries in the
history remain accurate even after the underlying trade closes.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

from phase_03_paper.positions.manager import PositionManager, PositionState
from phase_03_paper.trading.paper_engine import (
    ExitReason,
    PaperTrade,
    PaperTradeEngine,
)


# ============================================================
# JOURNAL ENTRY MODEL
# ============================================================


@dataclass(frozen=True)
class JournalEntry:
    """One immutable record of a single trade lifecycle transition."""

    trade_id: str
    session: str
    direction: str  # SignalType.value snapshot, e.g. "LONG" / "SHORT"

    position_state: PositionState
    recorded_at: Optional[datetime]

    entry: Optional[float]
    stop: float
    target: float

    exit_price: Optional[float] = None
    exit_reason: Optional[ExitReason] = None
    result_r: Optional[float] = None


# ============================================================
# JOURNAL
# ============================================================


class Journal:
    """
    Append-only, in-memory trade journal.

    Tracks the last PositionState recorded per trade_id so repeated
    calls to record() across many candles produce at most one entry
    per lifecycle transition â€” never one entry per candle.
    """

    # Lifecycle states worth journaling, in the order they occur.
    # NO_POSITION is intentionally excluded: it means "not an active
    # position" and has nothing to journal.
    _TRACKED_STATES = (
        PositionState.OPEN,
        PositionState.MANAGED,
        PositionState.CLOSED,
    )

    def __init__(self, position_manager: Optional[PositionManager] = None) -> None:
        self._position_manager = position_manager or PositionManager()
        self._entries: List[JournalEntry] = []

        # trade_id -> last PositionState recorded for that trade.
        self._last_recorded: Dict[str, PositionState] = {}

    # --------------------------------------------------------------
    # Recording
    # --------------------------------------------------------------

    def record(
        self,
        paper_engine: PaperTradeEngine,
        current_candle_timestamp: Optional[datetime] = None,
    ) -> List[JournalEntry]:
        """
        Inspect every trade PaperTradeEngine currently knows about
        (open and closed) via PositionManager.all_views(), and append
        a JournalEntry for any trade whose PositionState has advanced
        since the last call.

        Returns only the entries newly appended by this call (an
        empty list if nothing changed).
        """

        new_entries: List[JournalEntry] = []

        views = self._position_manager.all_views(
            paper_engine=paper_engine,
            current_candle_timestamp=current_candle_timestamp,
        )

        for view in views:
            entry = self._maybe_record(
                trade=view.trade,
                position_state=view.position_state,
                current_candle_timestamp=current_candle_timestamp,
            )
            if entry is not None:
                new_entries.append(entry)

        return new_entries

    def _maybe_record(
        self,
        trade: PaperTrade,
        position_state: PositionState,
        current_candle_timestamp: Optional[datetime],
    ) -> Optional[JournalEntry]:
        """
        Record one JournalEntry if, and only if, position_state is a
        genuinely new, forward lifecycle transition for this trade.
        """

        if position_state not in self._TRACKED_STATES:
            return None

        last_state = self._last_recorded.get(trade.trade_id)

        if last_state == position_state:
            return None

        if last_state is not None:
            last_index = self._TRACKED_STATES.index(last_state)
            new_index = self._TRACKED_STATES.index(position_state)
            if new_index <= last_index:
                # Never move backwards (or sideways) through the
                # lifecycle, even if a caller passes states out of
                # order.
                return None

        recorded_at = (
            current_candle_timestamp
            if current_candle_timestamp is not None
            else (trade.closed_at or trade.opened_at)
        )

        entry = JournalEntry(
            trade_id=trade.trade_id,
            session=trade.session,
            direction=trade.direction.value,
            position_state=position_state,
            recorded_at=recorded_at,
            entry=trade.entry,
            stop=trade.stop,
            target=trade.target,
            exit_price=trade.exit_price,
            exit_reason=trade.exit_reason,
            result_r=(
                trade.result_r if position_state == PositionState.CLOSED else None
            ),
        )

        self._entries.append(entry)
        self._last_recorded[trade.trade_id] = position_state

        return entry

    # --------------------------------------------------------------
    # Queries
    # --------------------------------------------------------------

    def all_entries(self) -> List[JournalEntry]:
        """Return the full journal history, in the order recorded."""
        return list(self._entries)

    def entries_for_trade(self, trade_id: str) -> List[JournalEntry]:
        """Return the recorded lifecycle history of one trade."""
        return [e for e in self._entries if e.trade_id == trade_id]

    def entries_for_session(self, session: str) -> List[JournalEntry]:
        """Return every recorded entry belonging to one session."""
        return [e for e in self._entries if e.session == session]

    def closed_entries(self) -> List[JournalEntry]:
        """Return only CLOSED entries (final outcomes)."""
        return [e for e in self._entries if e.position_state == PositionState.CLOSED]

    def last_state_of(self, trade_id: str) -> Optional[PositionState]:
        """Return the last PositionState recorded for a trade, if any."""
        return self._last_recorded.get(trade_id)

    # --------------------------------------------------------------
    # Reconstruction (from persisted entries)
    # --------------------------------------------------------------

    def load_entries(self, entries: List[JournalEntry]) -> None:
        # Rebuild this Journal's in-memory state from a list of
        # already-persisted JournalEntry records (e.g. loaded via
        # Persistence.load_journal_entries()).
        #
        # For reconstruction only -- never called by the live
        # record() path. Entries are assumed to already be in
        # recorded order and to represent a valid lifecycle history
        # (no validation against PositionManager is performed here,
        # since these are transitions that already happened, not
        # new ones being observed).
        #
        # Replaces this Journal's existing in-memory entries and
        # _last_recorded index entirely -- intended to be called
        # once, right after construction, on a fresh Journal.
        self._entries = list(entries)
        self._last_recorded = {}
        for entry in self._entries:
            self._last_recorded[entry.trade_id] = entry.position_state

__all__ = [
    "JournalEntry",
    "Journal",
]
