"""
frontend_dashboard/ollama_transport.py

Shared, framework-agnostic Ollama chat transport.

Extracted from phase2_dashboard/claude_helper.py's connection logic
(_send_chat_request, _list_local_models, _clean_conversation) so
frontend_dashboard/runtime_copilot.py can reuse it without depending
on Streamlit or duplicating the request/error-handling logic.

phase2_dashboard/claude_helper.py itself is left completely
untouched -- it is a working, tested Phase 2 component; refactoring
it to import from here would risk breaking it for no benefit here.
Some duplication between the two is the safer tradeoff.

This module does NOT:
    - know anything about trading, Phase 3, Phase 4, or diagnostics
    - cache anything (no st.cache_data -- callers decide caching)
    - import streamlit
"""

from __future__ import annotations

import os
from typing import Mapping, Sequence

import requests


OLLAMA_HOST = os.environ.get(
    "OLLAMA_HOST",
    "http://localhost:11434",
).rstrip("/")

OLLAMA_CHAT_URL = f"{OLLAMA_HOST}/api/chat"
OLLAMA_TAGS_URL = f"{OLLAMA_HOST}/api/tags"

MAX_CHAT_MESSAGES = 12

REQUEST_TIMEOUT_SECONDS = 120
TAGS_TIMEOUT_SECONDS = 3


def list_local_models() -> list[str] | None:
    """Return installed Ollama model names, or None if unreachable."""
    try:
        response = requests.get(OLLAMA_TAGS_URL, timeout=TAGS_TIMEOUT_SECONDS)
        response.raise_for_status()
    except requests.exceptions.RequestException:
        return None

    try:
        payload = response.json()
    except ValueError:
        return None

    models = payload.get("models", [])
    if not isinstance(models, list):
        return []

    names: list[str] = []
    for model in models:
        if not isinstance(model, Mapping):
            continue
        name = str(model.get("name", "")).strip()
        if name:
            names.append(name)

    return names


def model_is_pulled(model_id: str, installed: list[str]) -> bool:
    return any(
        name == model_id or name.startswith(f"{model_id}:")
        for name in installed
    )


def clean_conversation(
    conversation: Sequence[Mapping[str, str]] | None,
    max_messages: int = MAX_CHAT_MESSAGES,
) -> list[dict[str, str]]:
    """Keep only valid user/assistant messages, most recent N."""
    if not conversation:
        return []

    cleaned: list[dict[str, str]] = []
    for message in conversation[-max_messages:]:
        if not isinstance(message, Mapping):
            continue
        role = str(message.get("role", "")).strip().lower()
        content = str(message.get("content", "")).strip()
        if role not in {"user", "assistant"}:
            continue
        if not content:
            continue
        cleaned.append({"role": role, "content": content})

    return cleaned


def send_chat_request(model_id: str, messages: list[dict[str, str]]) -> str:
    """
    Send a conversation to Ollama. Always returns a readable message
    rather than raising -- callers (Streamlit or FastAPI) can display
    the result directly either way.
    """
    installed_models = list_local_models()

    if installed_models is None:
        return (
            "Local AI unavailable.\n\n"
            "Ollama is not running on this computer.\n\n"
            "Start Ollama and try again."
        )

    if not model_is_pulled(model_id, installed_models):
        return (
            f"Local AI model '{model_id}' is not installed.\n\n"
            f"Run `ollama pull {model_id}` in a terminal, then try again."
        )

    payload = {
        "model": model_id,
        "stream": False,
        "messages": messages,
        "options": {"temperature": 0.2},
    }

    try:
        response = requests.post(
            OLLAMA_CHAT_URL, json=payload, timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.Timeout:
        return (
            "Local AI request timed out.\n\n"
            "The model may be too large for the current hardware, "
            "or the context may be taking too long to process."
        )
    except requests.exceptions.ConnectionError:
        return (
            "Could not connect to the local Ollama server.\n\n"
            "Make sure Ollama is running and try again."
        )
    except requests.exceptions.RequestException as exc:
        return f"Local AI request failed:\n\n{exc}"
    except ValueError:
        return "Ollama returned an invalid response."

    message = data.get("message", {})
    if not isinstance(message, Mapping):
        return "Local AI returned an unexpected response."

    text = str(message.get("content", "")).strip()
    if not text:
        return "Local AI returned no text content."

    return text


__all__ = [
    "list_local_models",
    "model_is_pulled",
    "clean_conversation",
    "send_chat_request",
]
