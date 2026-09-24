"""
frontend_dashboard/runtime_copilot.py

Read-only Phase 3 / Phase 4 runtime supervisor AI, using the same
local Ollama model already used for Phase 2 research.

STRUCTURAL SAFETY BOUNDARY (not just prompt text):
This module imports ONLY read functions from frontend_dashboard.server
(phase3_trades, phase3_performance, phase4_expected_positions,
phase4_order_attempts) -- it never imports Phase4Persistence,
guard_and_place_order, order_manager, broker connection code, or
anything else capable of writing to a database or placing an order.
There is no write path physically reachable from this module,
regardless of what the model is asked to do.

Context is gathered by calling server.py's own functions directly
(same process, no HTTP round-trip needed) -- this can never diverge
from what /api/phase3/... and /api/phase4/... themselves return.
"""

from __future__ import annotations

import json
import os
from typing import Any, Mapping, Sequence

from frontend_dashboard import ollama_transport
from frontend_dashboard.server import (
    phase3_performance,
    phase3_trades,
    phase4_expected_positions,
    phase4_order_attempts,
)

MODEL_ID = os.environ.get("OLLAMA_MODEL", "qwen2.5-coder:3b")

MAX_CONTEXT_CHARS = 8000


SYSTEM_PROMPT = """
You are the Machet Mechanics Runtime Copilot.

You are embedded inside a read-only FastAPI bridge over the live
Phase 3 (paper trading) and Phase 4 (live execution) databases.

Your role, per the operator's own written duty assignment:

1. You are STRICTLY READ-ONLY. You have no ability to write to any
   database, modify any config, place or cancel any order, or
   interfere with the live runtime loop in any way -- this is a
   structural fact about your environment, not merely a rule you
   are asked to follow.

2. Base every conclusion only on the runtime context actually
   supplied to you in this conversation. Never invent trades,
   tickets, prices, R-multiples, win rates, or any other number.

3. Clearly distinguish what the data shows from your interpretation
   of it.

4. When context shows zero trades, zero positions, or zero pending
   attempts, say so plainly -- do not imply activity that has not
   happened.

5. Help with: inspecting paper trading history and performance,
   monitoring live expected positions and order attempts, debugging
   why a trade did or didn't happen, and explaining live runtime
   state on demand.

6. You are an analyst, not an execution engine. You cannot and will
   not place, modify, or cancel trades -- if asked to, explain that
   this capability does not exist for you and point to the actual
   runtime process (phase_04_live.runtime.loop) as the only thing
   that submits orders.

7. If the available context is insufficient to answer, say so
   directly rather than guessing.

8. Keep answers clear and reasonably concise unless asked for more
   depth.
""".strip()


def _truncate(text: str, limit: int = MAX_CONTEXT_CHARS) -> str:
    if len(text) <= limit:
        return text
    omitted = len(text) - limit
    return text[:limit] + f"\n\n[... {omitted} characters truncated ...]"


def gather_runtime_context() -> dict[str, Any]:
    """
    Pull the current real state from every read-only source this
    copilot is allowed to see. Called fresh on every request -- no
    caching -- so the AI is never grounded in stale runtime state.
    """
    return {
        "phase3_trades": phase3_trades(),
        "phase3_performance": phase3_performance(),
        "phase4_expected_positions": phase4_expected_positions(),
        "phase4_order_attempts": phase4_order_attempts(),
    }


def _format_context(context: dict[str, Any]) -> str:
    return _truncate(json.dumps(context, indent=2, default=str))


def chat_with_runtime_copilot(
    conversation: Sequence[Mapping[str, Any]] | None = None,
) -> str:
    """
    Answer a user's question using freshly-gathered Phase 3/4 runtime
    context plus recent conversation history.
    """
    context = gather_runtime_context()
    context_text = _format_context(context)

    contextual_system_prompt = (
        SYSTEM_PROMPT
        + "\n\n"
        + "CURRENT RUNTIME CONTEXT (live, just fetched)\n"
        + "=============================================\n"
        + context_text
        + "\n\n"
        + "Use this context when answering the user's questions."
    )

    messages: list[dict[str, str]] = [
        {"role": "system", "content": contextual_system_prompt}
    ]
    messages.extend(ollama_transport.clean_conversation(conversation))

    if len(messages) == 1:
        return "Ask me about your paper trading history or live runtime state."

    return ollama_transport.send_chat_request(MODEL_ID, messages)


__all__ = ["gather_runtime_context", "chat_with_runtime_copilot"]
