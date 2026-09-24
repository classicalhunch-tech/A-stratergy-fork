"""
phase_03_paper/ai/report_generator.py

ReportEngine -- Phase 3 AI Reports (deterministic narrative, v1).

PURPOSE
-------
Turns a PerformanceSnapshot (already computed by PerformanceEngine)
into a human-readable natural-language report.

ReportEngine CONSUMES PerformanceEngine and Journal. It does not
recompute any statistic PerformanceEngine already provides, and does
not touch PaperTradeEngine, PositionManager, or strategy logic.

ReportEngine does NOT:
    - compute performance statistics itself (PerformanceEngine owns
      that)
    - execute trades or modify strategy state
    - persist anything (Persistence, later)
    - call any external model / API

DESIGN
------
v1 is fully deterministic: the same PerformanceSnapshot always
produces exactly the same narrative text, using plain string
templates. No network calls, no external model, nothing
non-reproducible -- correctness and testability first.

NARRATIVE GENERATION SEAM
--------------------------
All prose generation is isolated behind `_narrative_from_snapshot()`
and its small helpers below it. A future LLM-backed narrative layer
(e.g. reusing the existing Ollama wiring in
phase2_dashboard/ai_assistant.py) should REPLACE only that method and
its helpers -- everything else in this module (data gathering,
notable-trade selection, the Report dataclass, the public
ReportEngine.generate() contract) stays unchanged. This keeps a clean
"one component, one responsibility" boundary: ReportEngine assembles
facts, the narrative function turns facts into prose.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional, Tuple

from phase_03_paper.journal.journal import Journal, JournalEntry
from phase_03_paper.performance.performance import (
    PerformanceEngine,
    PerformanceSnapshot,
)


# ============================================================
# NOTABLE TRADE
# ============================================================


@dataclass(frozen=True)
class NotableTrade:
    """A single closed trade highlighted in a report."""

    trade_id: str
    session: str
    direction: str
    result_r: float
    recorded_at: Optional[datetime]


# ============================================================
# REPORT
# ============================================================


@dataclass(frozen=True)
class Report:
    """Immutable AI Report: computed facts plus generated narrative."""

    generated_at: datetime
    snapshot: PerformanceSnapshot
    best_trade: Optional[NotableTrade]
    worst_trade: Optional[NotableTrade]
    narrative: str


# ============================================================
# REPORT ENGINE
# ============================================================


class ReportEngine:
    """Generates AI Reports from Journal history via PerformanceEngine."""

    def __init__(self, journal: Journal) -> None:
        self._journal = journal
        self._performance_engine = PerformanceEngine(journal)

    def generate(self, now: Optional[datetime] = None) -> Report:
        """Generate one Report from the complete current Journal history."""

        generated_at = self._normalize_now(now)
        snapshot = self._performance_engine.calculate(now=generated_at)

        best_trade, worst_trade = self._notable_trades()

        narrative = self._narrative_from_snapshot(
            snapshot=snapshot,
            best_trade=best_trade,
            worst_trade=worst_trade,
        )

        return Report(
            generated_at=generated_at,
            snapshot=snapshot,
            best_trade=best_trade,
            worst_trade=worst_trade,
            narrative=narrative,
        )

    # ------------------------------------------------------------
    # NOTABLE TRADES
    # ------------------------------------------------------------

    def _notable_trades(
        self,
    ) -> Tuple[Optional[NotableTrade], Optional[NotableTrade]]:
        """
        Return (best, worst) closed trades by result_r.

        Reads directly from Journal.closed_entries() -- this is a
        selection, not a statistic, so it deliberately does not go
        through PerformanceEngine.
        """

        entries = [
            e for e in self._journal.closed_entries() if e.result_r is not None
        ]

        if not entries:
            return None, None

        best_entry = max(entries, key=lambda e: e.result_r)
        worst_entry = min(entries, key=lambda e: e.result_r)

        return (
            self._to_notable(best_entry),
            self._to_notable(worst_entry),
        )

    @staticmethod
    def _to_notable(entry: JournalEntry) -> NotableTrade:
        return NotableTrade(
            trade_id=entry.trade_id,
            session=entry.session,
            direction=entry.direction,
            result_r=entry.result_r,
            recorded_at=entry.recorded_at,
        )

    # ------------------------------------------------------------
    # NARRATIVE GENERATION -- deterministic v1
    #
    # Replace ONLY this method (and the helpers below it) to add an
    # LLM-backed narrative layer later.
    # ------------------------------------------------------------

    def _narrative_from_snapshot(
        self,
        snapshot: PerformanceSnapshot,
        best_trade: Optional[NotableTrade],
        worst_trade: Optional[NotableTrade],
    ) -> str:
        if snapshot.total_trades == 0:
            return "No closed trades yet -- nothing to report."

        lines: List[str] = []

        lines.append(self._headline(snapshot))
        lines.append(self._streak_line(snapshot))

        drawdown_line = self._drawdown_line(snapshot)
        if drawdown_line:
            lines.append(drawdown_line)

        if best_trade is not None:
            lines.append(self._notable_trade_line("Best trade", best_trade))
        if worst_trade is not None:
            lines.append(self._notable_trade_line("Worst trade", worst_trade))

        session_line = self._top_session_line(snapshot)
        if session_line:
            lines.append(session_line)

        return " ".join(lines)

    @staticmethod
    def _headline(snapshot: PerformanceSnapshot) -> str:
        win_rate_pct = (
            f"{snapshot.win_rate * 100:.1f}%"
            if snapshot.win_rate is not None
            else "n/a"
        )
        return (
            f"{snapshot.total_trades} closed trade"
            f"{'s' if snapshot.total_trades != 1 else ''} "
            f"({snapshot.wins}W / {snapshot.losses}L, {win_rate_pct} win rate), "
            f"totaling {snapshot.total_r:+.2f}R "
            f"(avg {snapshot.average_r:+.2f}R/trade)."
        )

    @staticmethod
    def _streak_line(snapshot: PerformanceSnapshot) -> str:
        if snapshot.current_streak > 0:
            streak_desc = (
                f"currently on a {snapshot.current_streak}-trade winning streak"
            )
        elif snapshot.current_streak < 0:
            streak_desc = (
                f"currently on a {abs(snapshot.current_streak)}-trade "
                f"losing streak"
            )
        else:
            streak_desc = "no active streak"

        return (
            f"{streak_desc.capitalize()}; "
            f"best winning streak {snapshot.best_winning_streak}, "
            f"worst losing streak {snapshot.worst_losing_streak}."
        )

    @staticmethod
    def _drawdown_line(snapshot: PerformanceSnapshot) -> Optional[str]:
        if snapshot.max_drawdown_r <= 0:
            return None
        return f"Max drawdown so far: {snapshot.max_drawdown_r:.2f}R."

    @staticmethod
    def _notable_trade_line(label: str, trade: NotableTrade) -> str:
        when = (
            trade.recorded_at.date().isoformat()
            if trade.recorded_at is not None
            else "unknown date"
        )
        return (
            f"{label}: {trade.direction} in {trade.session} session, "
            f"{trade.result_r:+.2f}R ({when})."
        )

    @staticmethod
    def _top_session_line(snapshot: PerformanceSnapshot) -> Optional[str]:
        if not snapshot.session_performance:
            return None

        best_session_name, best_session_stats = max(
            snapshot.session_performance.items(),
            key=lambda item: item[1].total_r,
        )

        return (
            f"Best-performing session so far: {best_session_name} "
            f"({best_session_stats.trades} trades, "
            f"{best_session_stats.total_r:+.2f}R)."
        )

    @staticmethod
    def _normalize_now(value: Optional[datetime]) -> datetime:
        if value is None:
            return datetime.now(timezone.utc)
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)


__all__ = [
    "NotableTrade",
    "Report",
    "ReportEngine",
]