"""
dashboard/phase3/ai_helper.py

Local AI integration for the Machet Mechanics Phase 3 (paper trading /
forward validation) dashboard tab. Reuses the Ollama connection layer
from phase2_dashboard.claude_helper -- only the system prompt and the
context formatting are Phase-3-specific.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from phase2_dashboard.claude_helper import (
    _send_chat_request,
    _truncate,
    _clean_conversation,
)

PHASE3_SYSTEM_PROMPT = """
You are the Machet Mechanics Forward-Testing Assistant.

You are embedded inside a paper-trading / forward-validation dashboard
(Phase 3). This is NOT the same as Phase 2 backtest diagnostics -- this
data comes from a real, session-gated replay/paper execution engine.

Rules:

1. Base conclusions only on the replay summary, performance stats,
   trade journal, and health data supplied in context.

2. Never invent trades, R-multiples, drawdown figures, or session
   outcomes not present in the supplied data.

3. Compare forward/paper results to what Phase 2 backtesting implied
   ONLY if backtest figures are explicitly given to you -- otherwise
   say you don't have that comparison available.

4. Look for:
   - divergence between expected and actual win rate / expectancy
   - drawdown behavior vs. what the strategy's risk rules assume
   - session-specific weaknesses (e.g. one session dragging performance)
   - health/monitoring issues (adapter errors, stale ticks, pending
     signals) that could be distorting results
   - small-sample caveats when trade count is low

5. You are a research interpretation assistant, not an execution
   engine. Do not suggest placing trades or changing live config.

6. Keep answers concise and concrete; suggest specific next checks.
""".strip()


def _format_phase3_context(
    summary: Any,
    performance: Any,
    closed_trades: Sequence[Any],
    health: Any,
    report_narrative: str | None,
) -> str:
    """Build a readable context block from real Phase3Service objects."""
    lines: list[str] = []

    if summary is not None:
        lines.append(
            f"REPLAY SUMMARY: {summary.total_candles} candles, "
            f"{summary.ticks_run} ticks, {summary.opened_trades} opened, "
            f"{summary.closed_trades} closed."
        )

    if performance is not None and performance.total_trades:
        lines.append(
            f"PERFORMANCE: {performance.total_trades} trades, "
            f"win_rate={performance.win_rate:.3f}, "
            f"total_r={performance.total_r:+.2f}, "
            f"avg_r={performance.average_r:+.2f}, "
            f"max_drawdown_r={performance.max_drawdown_r:.2f}, "
            f"current_streak={performance.current_streak}"
        )
        if performance.session_performance:
            for name, stats in performance.session_performance.items():
                lines.append(
                    f"  SESSION {name}: trades={stats.trades}, "
                    f"win_rate={stats.win_rate:.3f}, total_r={stats.total_r:+.2f}"
                )

    if closed_trades:
        lines.append(f"CLOSED TRADES (last {min(len(closed_trades), 20)}):")
        for entry in closed_trades[-20:]:
            lines.append(
                f"  {entry.trade_id} {entry.direction} {entry.session} "
                f"result_r={entry.result_r} exit={entry.exit_reason}"
            )

    if health is not None:
        lines.append(
            f"HEALTH: runtime_alive={health.runtime_alive}, "
            f"adapter_errors={health.adapter_error_count}, "
            f"pending_signals={health.pending_signal_count}, "
            f"issues={health.issues}"
        )

    if report_narrative:
        lines.append("DETERMINISTIC REPORT NARRATIVE:")
        lines.append(report_narrative)

    return _truncate("\n".join(lines))


def chat_with_phase3_ai(
    summary: Any,
    performance: Any,
    closed_trades: Sequence[Any],
    health: Any,
    report_narrative: str | None,
    conversation: Sequence[Mapping[str, Any]] | None = None,
) -> str:
    """Answer a user question grounded in real Phase 3 replay/paper data."""
    context_text = _format_phase3_context(
        summary, performance, closed_trades, health, report_narrative
    )

    contextual_system_prompt = (
        PHASE3_SYSTEM_PROMPT
        + "\n\nCURRENT PHASE 3 CONTEXT\n========================\n"
        + context_text
        + "\n\nUse this context when answering the user's questions."
    )

    messages: list[dict[str, str]] = [
        {"role": "system", "content": contextual_system_prompt}
    ]
    messages.extend(_clean_conversation(conversation))

    if len(messages) == 1:
        return "Ask me a question about the current Phase 3 replay/paper results."

    return _send_chat_request(messages)
