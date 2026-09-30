"""
phase_03_paper/persistence/persistence.py

Persistence -- Phase 3 SQLite-backed storage.

PURPOSE
-------
Make PaperTradeEngine's trades and Journal's lifecycle entries
survive a process restart.

Persistence is the ONLY component in Phase 3 responsible for writing
to disk. It does not calculate anything (that's Performance) and it
does not decide what "should" be recorded (that's Journal / the
PaperTradeEngine). It only saves what it is given, and can reload it
back into equivalent Python objects.

SCOPE (current)
----------------
    - trades           (from PaperTradeEngine: PENDING/OPEN/CLOSED/REJECTED)
    - journal_entries   (from Journal: OPEN/MANAGED/CLOSED snapshots)

NOT in scope here (future persistence work, per the roadmap):
    - audit events
    - daily performance summaries
    - arbitrary system/session state

PositionManager is intentionally NOT persisted: it is a stateless,
read-only view computed from PaperTradeEngine's trades, so it can
always be recomputed after trades are reloaded. Persisting it
separately would just be a second, riskier copy of the same data.

WRITE SEMANTICS
----------------
    - trades:          upsert by trade_id (a trade's row is
                        overwritten as its status evolves, e.g.
                        OPEN -> CLOSED)
    - journal_entries:  append-only insert (JournalEntry is
                        immutable, and Journal already guarantees at
                        most one entry per lifecycle transition per
                        trade, so there is never a conflict to
                        resolve here)
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Union

from phase_03_paper.journal.journal import JournalEntry
from phase_03_paper.positions.manager import PositionState
from phase_03_paper.trading.paper_engine import (
    ExitReason,
    PaperTrade,
    PaperTradeStatus,
)
from strategy.signals import SignalType


# Repo root, derived from this file's location -- invariant to cwd.
# phase_03_paper/persistence/persistence.py -> parents[2] == repo root.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


_SCHEMA = """
CREATE TABLE IF NOT EXISTS trades (
    trade_id        TEXT PRIMARY KEY,
    direction       TEXT NOT NULL,
    session         TEXT NOT NULL,
    requested_entry REAL NOT NULL,
    stop            REAL NOT NULL,
    target          REAL NOT NULL,
    entry           REAL,
    opened_at       TEXT,
    closed_at       TEXT,
    exit_price      REAL,
    exit_reason     TEXT,
    result_r        REAL NOT NULL,
    status          TEXT NOT NULL,
    signal_key      TEXT
);

CREATE TABLE IF NOT EXISTS journal_entries (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_id        TEXT NOT NULL,
    session         TEXT NOT NULL,
    direction       TEXT NOT NULL,
    position_state  TEXT NOT NULL,
    recorded_at     TEXT,
    entry           REAL,
    stop            REAL NOT NULL,
    target          REAL NOT NULL,
    exit_price      REAL,
    exit_reason     TEXT,
    result_r        REAL
);

CREATE INDEX IF NOT EXISTS idx_journal_trade_id
    ON journal_entries (trade_id);
"""


def _dt_to_str(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    return dt.isoformat()


def _str_to_dt(s: Optional[str]) -> Optional[datetime]:
    if s is None:
        return None
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


class Persistence:
    """
    SQLite-backed storage for PaperTrade and JournalEntry records.

    Usage:

        persistence = Persistence("phase_03_paper.db")
        persistence.save_trades(paper_engine.all_trades())
        persistence.save_journal_entries(journal.all_entries())

        # ... process stops, then restarts ...

        trades = persistence.load_trades()
        entries = persistence.load_journal_entries()

    Each public method opens and closes its own connection. Phase 3
    is not high-frequency (candle-driven, not tick-driven), so this
    keeps the class simple and avoids holding a long-lived connection
    open across restarts/crashes. If profiling ever shows this is a
    bottleneck, that's an optimization to revisit later -- not now.
    """

    def __init__(self, db_path: Union[str, Path]) -> None:
        # Anchor relative db_path to PROJECT_ROOT (repo root), not cwd.
        # A cwd-relative path silently creates/opens a different DB
        # depending on where the process was launched from -- which can
        # disagree with the read-only reader in frontend_dashboard/server.py.
        path = Path(db_path)
        if not path.is_absolute():
            path = _PROJECT_ROOT / path
        self._db_path = str(path)

        with closing(self._connect()) as conn:
            conn.executescript(_SCHEMA)
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    # --------------------------------------------------------------
    # Trades
    # --------------------------------------------------------------

    def save_trade(self, trade: PaperTrade) -> None:
        self.save_trades([trade])

    def save_trades(self, trades: List[PaperTrade]) -> None:
        """
        Upsert each trade by trade_id. Safe to call repeatedly with
        the same trade as its fields evolve (PENDING -> OPEN ->
        CLOSED) -- the latest call always wins.
        """
        if not trades:
            return

        rows = []
        for trade in trades:
            signal_key = (
                json.dumps(list(trade.signal_key))
                if trade.signal_key is not None
                else None
            )
            rows.append(
                (
                    trade.trade_id,
                    trade.direction.value,
                    trade.session,
                    trade.requested_entry,
                    trade.stop,
                    trade.target,
                    trade.entry,
                    _dt_to_str(trade.opened_at),
                    _dt_to_str(trade.closed_at),
                    trade.exit_price,
                    trade.exit_reason.value if trade.exit_reason else None,
                    trade.result_r,
                    trade.status.value,
                    signal_key,
                )
            )

        with closing(self._connect()) as conn:
            conn.executemany(
                """
                INSERT INTO trades (
                    trade_id, direction, session, requested_entry,
                    stop, target, entry, opened_at, closed_at,
                    exit_price, exit_reason, result_r, status,
                    signal_key
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(trade_id) DO UPDATE SET
                    direction=excluded.direction,
                    session=excluded.session,
                    requested_entry=excluded.requested_entry,
                    stop=excluded.stop,
                    target=excluded.target,
                    entry=excluded.entry,
                    opened_at=excluded.opened_at,
                    closed_at=excluded.closed_at,
                    exit_price=excluded.exit_price,
                    exit_reason=excluded.exit_reason,
                    result_r=excluded.result_r,
                    status=excluded.status,
                    signal_key=excluded.signal_key
                """,
                rows,
            )
            conn.commit()

    def load_trades(self) -> List[PaperTrade]:
        with closing(self._connect()) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM trades")
            return [self._row_to_trade(row) for row in cursor.fetchall()]

    @staticmethod
    def _row_to_trade(row: sqlite3.Row) -> PaperTrade:
        signal_key = (
            tuple(json.loads(row["signal_key"]))
            if row["signal_key"] is not None
            else None
        )
        return PaperTrade(
            trade_id=row["trade_id"],
            direction=SignalType(row["direction"]),
            session=row["session"],
            requested_entry=row["requested_entry"],
            stop=row["stop"],
            target=row["target"],
            entry=row["entry"],
            opened_at=_str_to_dt(row["opened_at"]),
            closed_at=_str_to_dt(row["closed_at"]),
            exit_price=row["exit_price"],
            exit_reason=(
                ExitReason(row["exit_reason"]) if row["exit_reason"] else None
            ),
            result_r=row["result_r"],
            status=PaperTradeStatus(row["status"]),
            signal_key=signal_key,
        )

    # --------------------------------------------------------------
    # Journal entries
    # --------------------------------------------------------------

    def save_journal_entry(self, entry: JournalEntry) -> None:
        self.save_journal_entries([entry])

    def save_journal_entries(self, entries: List[JournalEntry]) -> None:
        """
        Append-only insert. JournalEntry is immutable/frozen, and
        Journal already guarantees at most one entry per lifecycle
        transition per trade -- so there's no conflict case here,
        every call just appends new rows.
        """
        if not entries:
            return

        rows = [
            (
                entry.trade_id,
                entry.session,
                entry.direction,
                entry.position_state.value,
                _dt_to_str(entry.recorded_at),
                entry.entry,
                entry.stop,
                entry.target,
                entry.exit_price,
                entry.exit_reason.value if entry.exit_reason else None,
                entry.result_r,
            )
            for entry in entries
        ]

        with closing(self._connect()) as conn:
            conn.executemany(
                """
                INSERT INTO journal_entries (
                    trade_id, session, direction, position_state,
                    recorded_at, entry, stop, target, exit_price,
                    exit_reason, result_r
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            conn.commit()

    def load_journal_entries(self) -> List[JournalEntry]:
        with closing(self._connect()) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM journal_entries ORDER BY id ASC"
            )
            return [self._row_to_journal_entry(row) for row in cursor.fetchall()]

    def journal_entries_for_trade(self, trade_id: str) -> List[JournalEntry]:
        with closing(self._connect()) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM journal_entries WHERE trade_id = ? ORDER BY id ASC",
                (trade_id,),
            )
            return [self._row_to_journal_entry(row) for row in cursor.fetchall()]

    @staticmethod
    def _row_to_journal_entry(row: sqlite3.Row) -> JournalEntry:
        return JournalEntry(
            trade_id=row["trade_id"],
            session=row["session"],
            direction=row["direction"],
            position_state=PositionState(row["position_state"]),
            recorded_at=_str_to_dt(row["recorded_at"]),
            entry=row["entry"],
            stop=row["stop"],
            target=row["target"],
            exit_price=row["exit_price"],
            exit_reason=(
                ExitReason(row["exit_reason"]) if row["exit_reason"] else None
            ),
            result_r=row["result_r"],
        )


__all__ = ["Persistence"]
