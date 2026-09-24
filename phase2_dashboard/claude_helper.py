"""
phase2_dashboard/claude_helper.py

Local AI integration for the Machet Mechanics Phase 2 Dashboard.

Uses Ollama instead of a paid cloud API.

The AI layer is completely separate from the trading strategy and
diagnostic engines. If Ollama fails, the research dashboard continues
to work normally.
"""

from __future__ import annotations

import json
import os
from typing import Any, Mapping, Sequence

import requests
import streamlit as st


# ============================================================================
# LOCAL AI CONFIGURATION
# ============================================================================

OLLAMA_HOST = os.environ.get(
    "OLLAMA_HOST",
    "http://localhost:11434",
).rstrip("/")

OLLAMA_CHAT_URL = f"{OLLAMA_HOST}/api/chat"
OLLAMA_TAGS_URL = f"{OLLAMA_HOST}/api/tags"

MODEL_ID = os.environ.get(
    "OLLAMA_MODEL",
    "qwen2.5-coder:3b",
)

MAX_RESULT_CHARS = 8000
MAX_CHAT_MESSAGES = 12

REQUEST_TIMEOUT_SECONDS = 120
TAGS_TIMEOUT_SECONDS = 3


# ============================================================================
# SYSTEM PROMPT
# ============================================================================

SYSTEM_PROMPT = """
You are the Machet Mechanics Research Assistant.

You are embedded inside a quantitative robustness and research dashboard.

Your job is to help interpret backtest and Phase 2 diagnostic results
objectively.

Rules:

1. Base numerical conclusions only on information supplied in the
   diagnostic context.

2. Never invent statistics, trades, returns, drawdowns, p-values,
   probabilities, sample sizes, or other numbers.

3. Clearly distinguish observed evidence from possible explanations.

4. Look for:
   - statistical weaknesses
   - instability
   - sensitivity to assumptions
   - overfitting risk
   - selection bias
   - transaction-cost sensitivity
   - small-sample problems
   - inconsistent results
   - suspiciously strong results
   - areas requiring additional validation

5. If the available evidence is insufficient to answer a question,
   say so directly.

6. Do not give generic motivational commentary.

7. Do not claim that a strategy has a durable market edge merely because
   a backtest or diagnostic looks good.

8. When suggesting next research steps, make them concrete and
   testable.

9. You are a research interpretation assistant, not an execution engine.
   Do not place trades or control the trading system.

10. Keep answers clear and reasonably concise unless the user asks
    for deeper analysis.
""".strip()


# ============================================================================
# HELPERS
# ============================================================================

def _truncate(
    text: str,
    limit: int = MAX_RESULT_CHARS,
) -> str:
    """Keep large diagnostic payloads manageable for local inference."""

    if len(text) <= limit:
        return text

    omitted = len(text) - limit

    return (
        text[:limit]
        + f"\n\n[... {omitted} characters truncated ...]"
    )


def _format_diagnostic_result(
    diagnostic_result: Any,
) -> str:
    """Convert common Python result objects into readable text."""

    try:
        if isinstance(
            diagnostic_result,
            (dict, list, tuple),
        ):
            return json.dumps(
                diagnostic_result,
                indent=2,
                default=str,
            )
    except Exception:
        pass

    return str(diagnostic_result)


def _clean_conversation(
    conversation: Sequence[Mapping[str, Any]] | None,
) -> list[dict[str, str]]:
    """
    Keep only valid user/assistant messages.

    This prevents Streamlit session-state objects or malformed entries
    from being sent to Ollama.
    """

    if not conversation:
        return []

    cleaned: list[dict[str, str]] = []

    for message in conversation[-MAX_CHAT_MESSAGES:]:
        if not isinstance(message, Mapping):
            continue

        role = str(
            message.get("role", "")
        ).strip().lower()

        content = str(
            message.get("content", "")
        ).strip()

        if role not in {"user", "assistant"}:
            continue

        if not content:
            continue

        cleaned.append(
            {
                "role": role,
                "content": content,
            }
        )

    return cleaned


# ============================================================================
# OLLAMA CONNECTION
# ============================================================================

@st.cache_data(
    ttl=30,
    show_spinner=False,
)
def _list_local_models() -> list[str] | None:
    """
    Return installed Ollama model names.

    Returns None when the Ollama server cannot be reached.
    """

    try:
        response = requests.get(
            OLLAMA_TAGS_URL,
            timeout=TAGS_TIMEOUT_SECONDS,
        )

        response.raise_for_status()

    except requests.exceptions.RequestException:
        return None

    try:
        payload = response.json()

    except ValueError:
        return None

    models = payload.get(
        "models",
        [],
    )

    if not isinstance(models, list):
        return []

    names: list[str] = []

    for model in models:

        if not isinstance(model, Mapping):
            continue

        name = str(
            model.get("name", "")
        ).strip()

        if name:
            names.append(name)

    return names


def _model_is_pulled(
    model_id: str,
    installed: list[str],
) -> bool:
    """
    Match both exact model names and tags.

    Example:
        llama3
        llama3:8b
    """

    return any(
        name == model_id
        or name.startswith(f"{model_id}:")
        for name in installed
    )


# ============================================================================
# OLLAMA REQUEST
# ============================================================================

def _send_chat_request(
    messages: list[dict[str, str]],
) -> str:
    """
    Send a conversation to Ollama.

    Always returns a readable message rather than raising an exception
    into the Streamlit dashboard.
    """

    installed_models = _list_local_models()

    if installed_models is None:
        return (
            "Local AI unavailable.\n\n"
            "Ollama is not running on this computer.\n\n"
            "Start Ollama and try again."
        )

    if not _model_is_pulled(
        MODEL_ID,
        installed_models,
    ):
        return (
            f"Local AI model '{MODEL_ID}' is not installed.\n\n"
            f"Run `ollama pull {MODEL_ID}` in a terminal, "
            "then try again."
        )

    payload = {
        "model": MODEL_ID,
        "stream": False,
        "messages": messages,
        "options": {
            "temperature": 0.2,
        },
    }

    try:

        response = requests.post(
            OLLAMA_CHAT_URL,
            json=payload,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )

        response.raise_for_status()

        data = response.json()

    except requests.exceptions.Timeout:

        return (
            "Local AI request timed out.\n\n"
            "The model may be too large for the current hardware, "
            "or the diagnostic context may be taking too long to process."
        )

    except requests.exceptions.ConnectionError:

        return (
            "Could not connect to the local Ollama server.\n\n"
            "Make sure Ollama is running and try again."
        )

    except requests.exceptions.RequestException as exc:

        return (
            "Local AI request failed:\n\n"
            f"{exc}"
        )

    except ValueError:

        return (
            "Ollama returned an invalid response."
        )

    message = data.get(
        "message",
        {},
    )

    if not isinstance(message, Mapping):
        return (
            "Local AI returned an unexpected response."
        )

    text = str(
        message.get(
            "content",
            "",
        )
    ).strip()

    if not text:
        return (
            "Local AI returned no text content."
        )

    return text


# ============================================================================
# ONE-SHOT DIAGNOSTIC REVIEW
# ============================================================================

def analyze_diagnostic_with_local_ai(
    diagnostic_title: str,
    diagnostic_result: Any,
) -> str:
    """
    Produce a one-shot research critique for a completed diagnostic.

    This remains available for future dashboard features.
    """

    result_text = _truncate(
        _format_diagnostic_result(
            diagnostic_result
        )
    )

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": (
                f"Analyze this Phase 2 diagnostic:\n\n"
                f"Diagnostic: {diagnostic_title}\n\n"
                f"Output:\n{result_text}\n\n"
                "Provide:\n"
                "1. Key quantitative findings\n"
                "2. Robustness concerns\n"
                "3. Overfitting or selection risk\n"
                "4. Instability or sensitivity\n"
                "5. Important red flags\n"
                "6. What should be investigated next"
            ),
        },
    ]

    return _send_chat_request(messages)


# ============================================================================
# MULTI-TURN RESEARCH CHAT
# ============================================================================

def chat_with_local_ai(
    diagnostic_title: str,
    diagnostic_result: Any,
    conversation: Sequence[Mapping[str, Any]] | None = None,
) -> str:
    """
    Answer a user's research question using:

        - the current diagnostic result
        - the current diagnostic title
        - recent conversation history

    The diagnostic context is supplied on every request so the assistant
    remains grounded in the currently displayed research result.
    """

    result_text = _truncate(
        _format_diagnostic_result(
            diagnostic_result
        )
    )

    contextual_system_prompt = (
        SYSTEM_PROMPT
        + "\n\n"
        + "CURRENT RESEARCH CONTEXT\n"
        + "========================\n"
        + f"Diagnostic: {diagnostic_title}\n\n"
        + "Diagnostic output:\n"
        + result_text
        + "\n\n"
        + "Use this context when answering the user's questions."
    )

    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": contextual_system_prompt,
        }
    ]

    messages.extend(
        _clean_conversation(
            conversation
        )
    )

    if len(messages) == 1:
        return (
            "Ask me a question about the current research result."
        )

    return _send_chat_request(messages)


# ============================================================================
# BACKWARD COMPATIBILITY
# ============================================================================

# Existing code may still import the old function name.
# Keep it working while the dashboard transitions to local AI.

analyze_diagnostic_with_claude = (
    analyze_diagnostic_with_local_ai
)