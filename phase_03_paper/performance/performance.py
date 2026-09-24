"""
phase_03_paper/performance/performance.py

PerformanceEngine — Phase 3 trading statistics.

PURPOSE
-------
PerformanceEngine answers one question:

    "How is the trading system performing?"

It CONSUMES data already recorded by Journal. It does not execute
trades, does not touch PaperTradeEngine or PositionManager directly,
and does not duplicate any lifecycle-classification logic.

PerformanceEngine does NOT:
    - modify strategy logic
    - execute trades
    - replace Journal (Journal records what happened)
    - replace Persistence (Persistence saves what happened)
    - replace Monitoring (Monitoring checks system health)

DESIGN
------
Recomputes fully from the complete closed-trade history on every
call rather than tracking an incremental cursor. Streaks, drawdown,
and windowed (daily/monthly) stats are whole-history calculations —
do not optimize this prematurely, only after profiling proves it is
a real bottleneck.

AGGREGATE STATS VS. ORDER-DEPENDENT STATS
------------------------------------------
Two different validity requirements are used deliberately:

    - total_trades, wins, losses, total_r, average_r, win_rate,
      expectancy only require a closed entry with a non-None
      result_r. They do not depend on timestamps or ordering.

    - current_streak, best/worst streak, max_drawdown_r, average
      duration, and daily/monthly bucketing all depend on knowing
      each trade's position in chronological order (or its close
      date), so they additionally require a non-None recorded_at.

A closed trade with a valid result_r but a missing recorded_at (an
edge case Journal should not normally produce, but which is not
strategy/execution logic PerformanceEngine should assume away) still
counts toward win rate and total R. It is simply excluded from the
order-dependent statistics, since it cannot be placed in sequence.

BREAKEVEN HANDLING
-------------------
A closed trade with result_r == 0 is classified as a LOSS for
win-rate and streak purposes (conservative default).

BUCKETING
---------
Daily and monthly buckets are keyed by the CLOSE date/month of a
trade (JournalEntry.recorded_at of its CLOSED entry) — "performance
on this day" means trades that closed on this day. Session bucketing
does not require a timestamp.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional

from phase_03_paper.journal.journal import Journal, JournalEntry
from phase_03_paper.positions.manager import PositionState


# ============================================================
# BUCKET STATS
# ============================================================


@dataclass(frozen=True)
class BucketStats:
    """Aggregate performance statistics for a specific time or session bucket."""

    trades: int
    wins: int
    losses: int
    win_rate: Optional[float]
    total_r: float
    average_r: Optional[float]


# ============================================================
# PERFORMANCE SNAPSHOT
# ============================================================


@dataclass(frozen=True)
class PerformanceSnapshot:
    """Immutable snapshot of trading performance at one point in time."""

    computed_at: datetime

    total_trades: int
    wins: int
    losses: int
    win_rate: Optional[float]

    total_r: float
    average_r: Optional[float]
    expectancy: Optional[float]

    max_drawdown_r: float

    current_streak: int
    best_winning_streak: int
    worst_losing_streak: int

    avg_trade_duration_seconds: Optional[float]

    session_performance: Dict[str, BucketStats] = field(default_factory=dict)
    daily_performance: Dict[str, BucketStats] = field(default_factory=dict)
    monthly_performance: Dict[str, BucketStats] = field(default_factory=dict)


# ============================================================
# PERFORMANCE ENGINE
# ============================================================


class PerformanceEngine:
    """Calculates immutable trading performance metrics from Journal history."""

    def __init__(self, journal: Journal) -> None:
        self._journal = journal

    def calculate(self, now: datetime | None = None) -> PerformanceSnapshot:
        """Recompute a complete performance snapshot from scratch."""

        computed_at = self._normalize_now(now)

        # Aggregate stats only need result_r — no timestamp required.
        all_entries = self._valid_closed_entries()
        total_trades = len(all_entries)

        if total_trades == 0:
            return self._empty_snapshot(computed_at)

        results = [entry.result_r for entry in all_entries]

        wins = sum(1 for r in results if self._is_win(r))
        losses = total_trades - wins

        total_r = sum(results)
        average_r = total_r / total_trades
        win_rate = wins / total_trades
        expectancy = self._expectancy(results, win_rate)

        # Order-dependent stats additionally require recorded_at, and
        # are computed only over the subset that has it, sorted
        # chronologically.
        ordered_entries = self._chronological(all_entries)
        ordered_results = [entry.result_r for entry in ordered_entries]

        max_drawdown_r = self._max_drawdown(ordered_results)
        current_streak, best_win, worst_loss = self._streaks(ordered_results)
        avg_duration = self._average_duration_seconds(ordered_entries)

        session_perf = self._bucket_by(
            all_entries,
            key_fn=lambda e: e.session,
        )
        daily_perf = self._bucket_by(
            ordered_entries,
            key_fn=lambda e: self._date_key(e.recorded_at),
        )
        monthly_perf = self._bucket_by(
            ordered_entries,
            key_fn=lambda e: self._month_key(e.recorded_at),
        )

        return PerformanceSnapshot(
            computed_at=computed_at,
            total_trades=total_trades,
            wins=wins,
            losses=losses,
            win_rate=win_rate,
            total_r=total_r,
            average_r=average_r,
            expectancy=expectancy,
            max_drawdown_r=max_drawdown_r,
            current_streak=current_streak,
            best_winning_streak=best_win,
            worst_losing_streak=worst_loss,
            avg_trade_duration_seconds=avg_duration,
            session_performance=session_perf,
            daily_performance=daily_perf,
            monthly_performance=monthly_perf,
        )

    @staticmethod
    def _empty_snapshot(computed_at: datetime) -> PerformanceSnapshot:
        return PerformanceSnapshot(
            computed_at=computed_at,
            total_trades=0,
            wins=0,
            losses=0,
            win_rate=None,
            total_r=0.0,
            average_r=None,
            expectancy=None,
            max_drawdown_r=0.0,
            current_streak=0,
            best_winning_streak=0,
            worst_losing_streak=0,
            avg_trade_duration_seconds=None,
        )

    # ============================================================
    # DATA GATHERING
    # ============================================================

    def _valid_closed_entries(self) -> List[JournalEntry]:
        """
        CLOSED entries with a usable result_r.

        No ordering or recorded_at guarantee here — see _chronological
        for the subset used by order-dependent statistics.
        """

        return [
            e for e in self._journal.closed_entries() if e.result_r is not None
        ]

    def _chronological(
        self,
        entries: List[JournalEntry],
    ) -> List[JournalEntry]:
        """
        Subset of entries that have a recorded_at, sorted chronologically.

        Entries missing a timestamp cannot be placed in sequence, so
        they are excluded here — but they were already counted in the
        aggregate totals computed from the unordered set.
        """

        timed = [e for e in entries if e.recorded_at is not None]
        return sorted(timed, key=lambda e: self._to_utc(e.recorded_at))

    @staticmethod
    def _to_utc(timestamp: datetime) -> datetime:
        if timestamp.tzinfo is None:
            return timestamp.replace(tzinfo=timezone.utc)
        return timestamp.astimezone(timezone.utc)

    # ============================================================
    # CLASSIFICATION
    # ============================================================

    @staticmethod
    def _is_win(result_r: float) -> bool:
        """A trade is a WIN only when result_r is strictly positive."""

        return result_r > 0.0

    # ============================================================
    # STATISTICS
    # ============================================================

    def _expectancy(
        self,
        results: List[float],
        win_rate: float,
    ) -> Optional[float]:
        """
        Classic per-trade expectancy:

            (win_rate * avg_win_r) + (loss_rate * avg_loss_r)
        """

        if not results:
            return None

        wins = [r for r in results if self._is_win(r)]
        losses = [r for r in results if not self._is_win(r)]

        avg_win = sum(wins) / len(wins) if wins else 0.0
        avg_loss = sum(losses) / len(losses) if losses else 0.0
        loss_rate = 1.0 - win_rate

        return (win_rate * avg_win) + (loss_rate * avg_loss)

    @staticmethod
    def _max_drawdown(results: List[float]) -> float:
        """Peak-to-trough drawdown of the cumulative R curve, as a
        positive number of R. Requires results in chronological order."""

        cumulative_r = 0.0
        peak_r = 0.0
        max_dd = 0.0

        for r in results:
            cumulative_r += r
            peak_r = max(peak_r, cumulative_r)
            max_dd = max(max_dd, peak_r - cumulative_r)

        return max_dd

    def _streaks(self, results: List[float]) -> tuple[int, int, int]:
        """Return (current_streak, best_winning_streak, worst_losing_streak).
        Requires results in chronological order."""

        if not results:
            return 0, 0, 0

        best_win = 0
        worst_loss = 0
        run_win = 0
        run_loss = 0

        for r in results:
            if self._is_win(r):
                run_win += 1
                run_loss = 0
                best_win = max(best_win, run_win)
            else:
                run_loss += 1
                run_win = 0
                worst_loss = max(worst_loss, run_loss)

        current = run_win if self._is_win(results[-1]) else -run_loss
        return current, best_win, worst_loss

    def _average_duration_seconds(
        self,
        ordered_entries: List[JournalEntry],
    ) -> Optional[float]:
        """
        Average trade duration in seconds, from each trade's OPEN
        entry to its CLOSED entry. Trades whose OPEN entry cannot be
        found are skipped rather than treated as zero duration.
        """

        durations: List[float] = []

        for closed in ordered_entries:
            open_entry = self._find_open_entry(closed.trade_id)
            if open_entry is None or open_entry.recorded_at is None:
                continue

            delta = (
                self._to_utc(closed.recorded_at)
                - self._to_utc(open_entry.recorded_at)
            ).total_seconds()

            if delta >= 0:
                durations.append(delta)

        if not durations:
            return None

        return sum(durations) / len(durations)

    def _find_open_entry(self, trade_id: str) -> Optional[JournalEntry]:
        """Return the OPEN JournalEntry for a trade, if one was recorded."""

        open_entries = [
            e
            for e in self._journal.entries_for_trade(trade_id)
            if e.position_state == PositionState.OPEN
        ]
        if not open_entries:
            return None

        return min(
            open_entries,
            key=lambda e: (
                self._to_utc(e.recorded_at)
                if e.recorded_at is not None
                else datetime.max.replace(tzinfo=timezone.utc)
            ),
        )

    # ============================================================
    # BUCKETING
    # ============================================================

    def _bucket_by(
        self,
        entries: List[JournalEntry],
        *,
        key_fn: Callable[[JournalEntry], Optional[str]],
    ) -> Dict[str, BucketStats]:
        """Group entries by key_fn and compute BucketStats per group."""

        grouped: Dict[str, List[float]] = defaultdict(list)

        for entry in entries:
            key = key_fn(entry)
            if key is not None:
                grouped[key].append(entry.result_r)

        buckets: Dict[str, BucketStats] = {}
        for key, results in grouped.items():
            trades = len(results)
            wins = sum(1 for r in results if self._is_win(r))
            losses = trades - wins
            total_r = sum(results)

            buckets[key] = BucketStats(
                trades=trades,
                wins=wins,
                losses=losses,
                win_rate=wins / trades if trades else None,
                total_r=total_r,
                average_r=total_r / trades if trades else None,
            )

        return buckets

    @staticmethod
    def _date_key(timestamp: Optional[datetime]) -> Optional[str]:
        """ISO date key (YYYY-MM-DD) for daily bucketing."""

        return timestamp.date().isoformat() if timestamp else None

    @staticmethod
    def _month_key(timestamp: Optional[datetime]) -> Optional[str]:
        """YYYY-MM key for monthly bucketing."""

        return timestamp.strftime("%Y-%m") if timestamp else None

    @staticmethod
    def _normalize_now(value: datetime | None) -> datetime:
        if value is None:
            return datetime.now(timezone.utc)
        return PerformanceEngine._to_utc(value)


__all__ = [
    "BucketStats",
    "PerformanceSnapshot",
    "PerformanceEngine",
]