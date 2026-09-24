"""
phase_04_live/persistence/store.py

Durable SQLite-backed state for Phase 4 -- restart-safe idempotency,
expected-position tracking, order-attempt recovery, and initial
trade-risk persistence.

Follows the EXACT same persistence pattern as
phase_03_paper/persistence/persistence.py:
per-call connections, db_path as a constructor arg, and schema
auto-created via executescript(CREATE TABLE IF NOT EXISTS...).

This is a separate SQLite FILE from Phase 3's phase_03_paper.db
(per project decision), but deliberately uses the SAME persistence
style -- not a second, different pattern.

SCOPE
-----
    - execution_identities
        Durable backing for the existing IdempotencyStore Protocol.

    - expected_positions
        Durable backing for ExpectedPosition tracking.

    - trade_risk
        Durable record of the initial dollar risk assigned to a
        successfully accepted trade. This is the denominator used
        later to convert realized dollar P&L into realized R.

    - order_attempts
        Durable pre-send record used by restart/recovery correlation
        when an order outcome is unknown after a crash.

This module does NOT:
    - make recovery decisions
    - reconcile anything
    - query MT5
    - calculate position size
    - decide when an expected position should be added/removed
    - decide when a trade-risk record should be created
      (the caller records it after an accepted order)
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Union

from phase_04_live.execution.idempotency import (
    encode_execution_identity,
    ExecutionIdentity,
)
from phase_04_live.orders.order_manager import OrderRequest
from phase_04_live.reconciliation.discrepancies import ExpectedPosition


_SCHEMA = """
CREATE TABLE IF NOT EXISTS execution_identities (
    identity_key    TEXT PRIMARY KEY,
    identity_repr   TEXT NOT NULL,
    recorded_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS expected_positions (
    ticket          INTEGER PRIMARY KEY,
    symbol          TEXT NOT NULL,
    direction       TEXT NOT NULL,
    volume          REAL NOT NULL,
    stop_loss       REAL,
    take_profit     REAL,
    recorded_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS trade_risk (
    ticket          INTEGER PRIMARY KEY,
    dollar_risk     REAL NOT NULL,
    recorded_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS attempt_risk (
    identity_key    TEXT PRIMARY KEY,
    dollar_risk     REAL NOT NULL,
    recorded_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS order_attempts (
    identity_key    TEXT PRIMARY KEY,
    symbol          TEXT NOT NULL,
    order_type      INTEGER NOT NULL,
    volume          REAL NOT NULL,
    price           REAL NOT NULL,
    sl              REAL NOT NULL,
    tp              REAL NOT NULL,
    deviation       INTEGER NOT NULL,
    magic           INTEGER NOT NULL,
    comment         TEXT NOT NULL,
    filling_mode    INTEGER NOT NULL,
    attempted_at    TEXT NOT NULL,
    resolved_at     TEXT
);
"""


def _utc_now_str() -> str:
    return datetime.now(timezone.utc).isoformat()


def _encode_identity(identity: ExecutionIdentity) -> str:
    """
    Thin wrapper kept for internal call-site stability. Delegates to
    phase_04_live.execution.idempotency.encode_execution_identity(),
    the single source of truth for this encoding -- execution_identities,
    order_attempts, and guard_and_place_order() must all produce
    identical keys for the same identity.
    """
    return encode_execution_identity(identity)


@dataclass(frozen=True)
class OrderAttempt:
    """
    One durably-recorded order attempt: the exact OrderRequest that
    was about to be sent to the broker, plus when it was attempted.

    Recorded by Phase4Persistence.record_order_attempt() immediately
    before order_manager.place_order_for_signal() calls mt5.order_send().

    resolved_at is None while the outcome is still unknown.
    A caller (eventually Recovery/reconciliation) marks it resolved
    once the outcome is confirmed.
    """

    identity_key: str
    request: OrderRequest
    attempted_at: datetime
    resolved_at: Optional[datetime]


class Phase4Persistence:
    """
    SQLite-backed durable state for Phase 4 live execution.

    Implements the has()/record() shape of the
    phase_04_live.execution.idempotency.IdempotencyStore Protocol
    directly (structural typing).

    Each public method opens and closes its own connection, matching
    Phase 3 persistence behavior and keeping the persistence layer
    simple across process restarts/crashes.
    """

    def __init__(self, db_path: Union[str, Path]) -> None:
        self._db_path = str(db_path)

        with closing(self._connect()) as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.executescript(_SCHEMA)
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    # ================================================================
    # IdempotencyStore Protocol (has / record)
    # ================================================================

    def has(self, identity: ExecutionIdentity) -> bool:
        key = _encode_identity(identity)

        with closing(self._connect()) as conn:
            cursor = conn.execute(
                "SELECT 1 FROM execution_identities WHERE identity_key = ?",
                (key,),
            )
            return cursor.fetchone() is not None

    def record(self, identity: ExecutionIdentity) -> None:
        key = _encode_identity(identity)

        with closing(self._connect()) as conn:
            conn.execute(
                """
                INSERT INTO execution_identities (
                    identity_key, identity_repr, recorded_at
                ) VALUES (?, ?, ?)
                ON CONFLICT(identity_key) DO NOTHING
                """,
                (
                    key,
                    repr(identity),
                    _utc_now_str(),
                ),
            )
            conn.commit()

    # ================================================================
    # Expected positions
    # ================================================================

    def save_expected_position(
        self,
        position: ExpectedPosition,
    ) -> None:
        """
        Upsert by ticket.

        Caller decides WHEN a position should be tracked as expected.
        This method only stores it durably.
        """
        with closing(self._connect()) as conn:
            conn.execute(
                """
                INSERT INTO expected_positions (
                    ticket,
                    symbol,
                    direction,
                    volume,
                    stop_loss,
                    take_profit,
                    recorded_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(ticket) DO UPDATE SET
                    symbol=excluded.symbol,
                    direction=excluded.direction,
                    volume=excluded.volume,
                    stop_loss=excluded.stop_loss,
                    take_profit=excluded.take_profit,
                    recorded_at=excluded.recorded_at
                """,
                (
                    position.ticket,
                    position.symbol,
                    position.direction,
                    position.volume,
                    position.stop_loss,
                    position.take_profit,
                    _utc_now_str(),
                ),
            )
            conn.commit()

    def remove_expected_position(self, ticket: int) -> None:
        """
        Remove one ticket from durable expected-position tracking.

        Safe to call for a ticket that is not present.
        """
        with closing(self._connect()) as conn:
            conn.execute(
                "DELETE FROM expected_positions WHERE ticket = ?",
                (ticket,),
            )
            conn.commit()

    def load_expected_positions(self) -> List[ExpectedPosition]:
        with closing(self._connect()) as conn:
            conn.row_factory = sqlite3.Row

            cursor = conn.execute(
                "SELECT * FROM expected_positions"
            )

            return [
                self._row_to_expected_position(row)
                for row in cursor.fetchall()
            ]

    @staticmethod
    def _row_to_expected_position(
        row: sqlite3.Row,
    ) -> ExpectedPosition:
        return ExpectedPosition(
            ticket=row["ticket"],
            symbol=row["symbol"],
            direction=row["direction"],
            volume=row["volume"],
            stop_loss=row["stop_loss"],
            take_profit=row["take_profit"],
        )

    # ================================================================
    # Trade risk
    # ================================================================

    def record_trade_risk(
        self,
        ticket: int,
        dollar_risk: float,
    ) -> None:
        """
        Durably record the initial dollar risk assigned to a trade.

        This value is the R denominator for later realized-R
        calculations:

            realized_R = realized_profit / dollar_risk

        The record is keyed by ticket and is idempotent.
        An existing ticket is never overwritten.
        """
        with closing(self._connect()) as conn:
            conn.execute(
                """
                INSERT INTO trade_risk (
                    ticket,
                    dollar_risk,
                    recorded_at
                )
                VALUES (?, ?, ?)
                ON CONFLICT(ticket) DO NOTHING
                """,
                (
                    ticket,
                    dollar_risk,
                    _utc_now_str(),
                ),
            )
            conn.commit()

    def get_trade_risk(
        self,
        ticket: int,
    ) -> Optional[float]:
        """
        Return the recorded initial dollar risk for a ticket.

        Returns None when no risk record exists.
        """
        with closing(self._connect()) as conn:
            cursor = conn.execute(
                """
                SELECT dollar_risk
                FROM trade_risk
                WHERE ticket = ?
                """,
                (ticket,),
            )

            row = cursor.fetchone()

            return (
                float(row[0])
                if row is not None
                else None
            )

    # ================================================================
    # Order attempts (durable pre-send record)
    # ================================================================

    def record_attempt_risk(
        self,
        identity_key: str,
        dollar_risk: float,
    ) -> None:
        """
        Durably record the initial dollar risk for an execution
        identity BEFORE its order is sent.

        Restart recovery uses it to attach the R denominator to the
        broker position adopted for that identity.

        Idempotent: an existing identity_key is never overwritten.
        """
        with closing(self._connect()) as conn:
            conn.execute(
                """
                INSERT INTO attempt_risk (
                    identity_key,
                    dollar_risk,
                    recorded_at
                )
                VALUES (?, ?, ?)
                ON CONFLICT(identity_key) DO NOTHING
                """,
                (
                    identity_key,
                    dollar_risk,
                    _utc_now_str(),
                ),
            )
            conn.commit()

    def get_attempt_risk(
        self,
        identity_key: str,
    ) -> "Optional[float]":
        """Return the recorded dollar risk for an identity, or None."""
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT dollar_risk FROM attempt_risk "
                "WHERE identity_key = ?",
                (identity_key,),
            ).fetchone()

        return float(row[0]) if row else None

    def record_order_attempt(
        self,
        identity_key: str,
        request: OrderRequest,
        attempted_at: datetime,
    ) -> None:
        """
        Durably record ONE order attempt.

        The identity_key uses the same encoding as
        execution_identities.

        Idempotent: a second call for the same identity_key is a no-op.
        """
        with closing(self._connect()) as conn:
            conn.execute(
                """
                INSERT INTO order_attempts (
                    identity_key,
                    symbol,
                    order_type,
                    volume,
                    price,
                    sl,
                    tp,
                    deviation,
                    magic,
                    comment,
                    filling_mode,
                    attempted_at,
                    resolved_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
                ON CONFLICT(identity_key) DO NOTHING
                """,
                (
                    identity_key,
                    request.symbol,
                    request.order_type,
                    request.volume,
                    request.price,
                    request.sl,
                    request.tp,
                    request.deviation,
                    request.magic,
                    request.comment,
                    request.filling_mode,
                    attempted_at.isoformat(),
                ),
            )
            conn.commit()

    def resolve_order_attempt(
        self,
        identity_key: str,
    ) -> None:
        """
        Mark one order attempt resolved.

        Safe to call for a missing or already-resolved identity_key.
        Does not decide the outcome; it only removes the attempt from
        load_pending_order_attempts().
        """
        with closing(self._connect()) as conn:
            conn.execute(
                """
                UPDATE order_attempts
                SET resolved_at = ?
                WHERE identity_key = ?
                  AND resolved_at IS NULL
                """,
                (
                    _utc_now_str(),
                    identity_key,
                ),
            )
            conn.commit()

    def load_pending_order_attempts(self) -> List[OrderAttempt]:
        """
        Load every unresolved order attempt.

        These are candidates for restart/recovery correlation against
        broker-side executions whose local outcome is unknown.
        """
        with closing(self._connect()) as conn:
            conn.row_factory = sqlite3.Row

            cursor = conn.execute(
                """
                SELECT *
                FROM order_attempts
                WHERE resolved_at IS NULL
                """
            )

            return [
                self._row_to_order_attempt(row)
                for row in cursor.fetchall()
            ]

    @staticmethod
    def _row_to_order_attempt(
        row: sqlite3.Row,
    ) -> "OrderAttempt":
        return OrderAttempt(
            identity_key=row["identity_key"],
            request=OrderRequest(
                symbol=row["symbol"],
                order_type=row["order_type"],
                volume=row["volume"],
                price=row["price"],
                sl=row["sl"],
                tp=row["tp"],
                deviation=row["deviation"],
                magic=row["magic"],
                comment=row["comment"],
                filling_mode=row["filling_mode"],
            ),
            attempted_at=datetime.fromisoformat(
                row["attempted_at"]
            ),
            resolved_at=(
                datetime.fromisoformat(row["resolved_at"])
                if row["resolved_at"] is not None
                else None
            ),
        )


__all__ = [
    "Phase4Persistence",
    "OrderAttempt",
]