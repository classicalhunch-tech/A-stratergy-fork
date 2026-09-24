"""
phase2_dashboard/ai_assistant.py

Machet Mechanics — Local AI Research Assistant.

Provides:
    - A chat interface backed by the local Ollama model.
    - Notifications and alerts from persisted Phase 2 diagnostics.
    - Real signal-activity notifications, read from results.csv.
    - An equity snapshot section (placeholder until live trading starts).
    - A session-start greeting highlighting important issues.

This module is READ-ONLY.

It does NOT:
    - execute trades
    - place orders
    - modify strategy logic
    - modify diagnostic results

The assistant is intended for research, diagnostics, and explanation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import streamlit as st

from phase2_dashboard.claude_helper import (
    MODEL_ID,
    chat_with_local_ai,
)


# ============================================================================
# OPTIONAL BACKEND IMPORTS
# ============================================================================

try:
    from phase2_dashboard.runner import DIAGNOSTICS
    from phase2_dashboard.results import get_latest

    BACKEND_OK = True

except Exception:
    DIAGNOSTICS: dict[str, Any] = {}
    get_latest = None
    BACKEND_OK = False


# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# results.csv is real per-bar signal output (base_signal / final_signal),
# written by the MTF pipeline. This is what the "🎯 Session Signals"
# notification is built from — no invented numbers.
RESULTS_CSV_PATH = PROJECT_ROOT / "results.csv"

# Optional file the live/paper trading system can write once you're
# actually trading. Expected shape, e.g.:
#
# {
#     "balance": 10432.18,
#     "starting_balance": 10000.0,
#     "open_positions": 1,
#     "as_of": "2026-09-08 14:05:00",
#     "return_pct": 4.32           <- optional, only shown if present
# }
#
# Until this file exists, the equity panel shows a placeholder instead
# of inventing a number.
EQUITY_LOG_PATH = PROJECT_ROOT / "logs" / "equity_snapshot.json"

ASSISTANT_NAME = "Machet AI"

ASSISTANT_AVATAR = "🤖"
USER_AVATAR = "🧑‍💻"

# Prevent an unlimited session history from being sent to Ollama.
MAX_CHAT_MESSAGES = 40

# Tokens recognized as actual directional signals vs. "nothing happened".
LONG_SIGNAL_TOKENS = {"LONG", "BUY"}
SHORT_SIGNAL_TOKENS = {"SHORT", "SELL"}
NO_SIGNAL_TOKENS = {"NONE", "NO_SIGNAL", "NAN", "NULL", ""}


# ============================================================================
# SAFE HELPERS
# ============================================================================

def _safe_number(
    value: Any,
    default: float = 0.0,
) -> float:
    """Convert a value to float without crashing the dashboard."""

    try:
        return float(value)

    except (TypeError, ValueError):
        return default


def _safe_text(
    value: Any,
    default: str = "",
) -> str:
    """Convert a value to clean display text."""

    if value is None:
        return default

    return str(value).strip()


def _normalize_signal_token(value: Any) -> str:
    """Uppercase/trim a raw final_signal value for comparison."""

    return _safe_text(value).upper()


# ============================================================================
# DIAGNOSTIC DATA
# ============================================================================

def _get_latest_diagnostic(
    key: str,
) -> Any:
    """Return the latest persisted result for one diagnostic."""

    if not BACKEND_OK:
        return None

    if get_latest is None:
        return None

    try:
        return get_latest(key)

    except Exception:
        return None


def _diagnostic_alerts() -> list[dict[str, str]]:
    """
    Identify Phase 2 diagnostics whose latest recorded run failed.

    No diagnostic is treated as failed merely because it has never run.
    """

    alerts: list[dict[str, str]] = []

    if not BACKEND_OK:
        return alerts

    if get_latest is None:
        return alerts

    if not isinstance(DIAGNOSTICS, Mapping):
        return alerts

    for key in DIAGNOSTICS:

        item = _get_latest_diagnostic(key)

        if item is None:
            continue

        if isinstance(item, Mapping):

            status = item.get("status")
            succeeded = item.get("succeeded")

        else:

            status = getattr(
                item,
                "status",
                None,
            )

            succeeded = getattr(
                item,
                "succeeded",
                None,
            )

        status_text = (
            _safe_text(status)
            .upper()
        )

        failed = (
            status_text
            in {
                "FAIL",
                "FAILED",
                "ERROR",
                "BROKEN",
                "NEEDS REVIEW",
            }
            or succeeded is False
        )

        if not failed:
            continue

        title = (
            key
            .replace("_", " ")
            .title()
        )

        alerts.append(
            {
                "level": "critical",
                "title": (
                    f"🚨 {title} needs review"
                ),
                "detail": (
                    "The latest recorded run "
                    "for this diagnostic did not pass."
                ),
            }
        )

    return alerts


# ============================================================================
# SESSION SIGNAL SUMMARY (real data, from results.csv)
# ============================================================================

def _load_signal_summary() -> dict[str, Any] | None:
    """
    Read results.csv and summarize the most recent session's signals.

    results.csv holds per-bar signal generation output
    (base_signal / final_signal) — not trade outcomes. There is no
    win/loss or P&L in this file, so none is reported here.

    Robust to a missing "timestamp" column, malformed timestamps,
    timezone-aware timestamps, and an empty or missing file. Returns
    None whenever there is nothing usable to report — it never raises.
    """

    if not RESULTS_CSV_PATH.exists():
        return None

    try:
        import pandas as pd

        df = pd.read_csv(RESULTS_CSV_PATH)

    except Exception:
        return None

    if df.empty:
        return None

    if "timestamp" not in df.columns or "final_signal" not in df.columns:
        return None

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
    )

    df = df.dropna(subset=["timestamp"])

    if df.empty:
        return None

    try:
        if df["timestamp"].dt.tz is not None:
            df["timestamp"] = df["timestamp"].dt.tz_localize(None)
    except Exception:
        pass

    latest_date = df["timestamp"].dt.date.max()

    session_rows = df[
        df["timestamp"].dt.date == latest_date
    ]

    if session_rows.empty:
        return None

    normalized_signals = session_rows["final_signal"].apply(
        _normalize_signal_token
    )

    raw_counts = normalized_signals.value_counts().to_dict()

    long_count = sum(
        count
        for token, count in raw_counts.items()
        if token in LONG_SIGNAL_TOKENS
    )

    short_count = sum(
        count
        for token, count in raw_counts.items()
        if token in SHORT_SIGNAL_TOKENS
    )

    last_row = session_rows.iloc[-1]
    last_signal_token = _normalize_signal_token(last_row["final_signal"])

    last_signal = (
        "No signal"
        if last_signal_token in NO_SIGNAL_TOKENS
        else last_signal_token
    )

    return {
        "session_date": str(latest_date),
        "raw_counts": raw_counts,
        "long_count": long_count,
        "short_count": short_count,
        "last_signal": last_signal,
        "last_time": _safe_text(last_row["timestamp"]),
    }


def _build_signal_notification() -> dict[str, str]:
    """Build a concise notification from today's real signal activity."""

    summary = _load_signal_summary()

    if summary is None:
        return {
            "level": "info",
            "title": "🎯 Signal data unavailable",
            "detail": (
                f"Couldn't read usable signal data from "
                f"{RESULTS_CSV_PATH.name} — run the pipeline to "
                "generate signals."
            ),
        }

    return {
        "level": "info",
        "title": (
            f"🎯 Session {summary['session_date']} · "
            f"LONG/BUY: {summary['long_count']} · "
            f"SHORT/SELL: {summary['short_count']}"
        ),
        "detail": (
            f"Latest signal: {summary['last_signal']} "
            f"at {summary['last_time']}."
        ),
    }


def _build_signal_context() -> dict[str, Any]:
    """
    Full signal picture for the AI's context — not just the notification.

    Includes the complete raw signal-value counts (whatever labels
    actually exist in the dataset) so Machet AI can reason about the
    real data rather than only the summarized notification text.
    """

    summary = _load_signal_summary()

    if summary is None:
        return {"available": False}

    return {
        "available": True,
        "session_date": summary["session_date"],
        "raw_signal_counts": summary["raw_counts"],
        "long_count": summary["long_count"],
        "short_count": summary["short_count"],
        "last_signal": summary["last_signal"],
        "last_signal_time": summary["last_time"],
    }


# ============================================================================
# EQUITY SNAPSHOT (placeholder until live/paper trading writes one)
# ============================================================================

def _load_equity_snapshot() -> dict[str, Any] | None:
    """
    Load an optional equity snapshot written by the live/paper system.

    Returns None when the file does not exist or is invalid — no
    equity number or return figure is ever invented.
    """

    if not EQUITY_LOG_PATH.exists():
        return None

    try:

        with EQUITY_LOG_PATH.open(
            "r",
            encoding="utf-8",
        ) as file:

            data = json.load(file)

    except (
        OSError,
        json.JSONDecodeError,
    ):

        return None

    if not isinstance(data, dict):
        return None

    return data


def _build_equity_notification() -> dict[str, str]:
    """
    Build a notification for the equity panel.

    Only ever labeled as a balance/equity change unless the snapshot
    itself supplies an explicit "return_pct" field — a change in
    balance is not the same claim as a percentage return, so the two
    are never conflated here.
    """

    snapshot = _load_equity_snapshot()

    if snapshot is None:
        return {
            "level": "info",
            "title": "💰 Equity: not connected yet",
            "detail": (
                "Once you're live or paper trading, write balance "
                f"snapshots to {EQUITY_LOG_PATH.relative_to(PROJECT_ROOT)} "
                "and a real equity snapshot will show up here."
            ),
        }

    balance = _safe_number(snapshot.get("balance", 0.0))

    starting_balance = _safe_number(
        snapshot.get("starting_balance", balance)
    )

    open_positions = snapshot.get("open_positions", 0)

    change = balance - starting_balance
    level = "good" if change >= 0 else "warning"

    detail_parts = [f"Balance change: {change:+,.2f}"]

    if "return_pct" in snapshot:

        return_pct = _safe_number(snapshot.get("return_pct"))

        detail_parts.append(
            f"Reported return: {return_pct:+.2f}%"
        )

    detail_parts.append(f"Open positions: {open_positions}")

    detail_parts.append(
        f"As of {_safe_text(snapshot.get('as_of'), 'unknown time')}"
    )

    return {
        "level": level,
        "title": (
            f"💰 Equity: {balance:,.2f} "
            f"(balance change: {change:+,.2f})"
        ),
        "detail": " · ".join(detail_parts),
    }


# ============================================================================
# NOTIFICATIONS
# ============================================================================

def _build_notifications() -> list[dict[str, str]]:
    """Build all available research notifications."""

    notifications = _diagnostic_alerts()

    notifications.append(_build_signal_notification())
    notifications.append(_build_equity_notification())

    return notifications


# ============================================================================
# RESEARCH CONTEXT
# ============================================================================

def _build_research_context() -> dict[str, Any]:
    """
    Collect persisted Phase 2 diagnostic results.

    This is what allows the local AI to reason about actual
    research results instead of only seeing notification text.
    Only diagnostics that actually have a persisted result are
    included.
    """

    context: dict[str, Any] = {}

    if not BACKEND_OK:
        return context

    if not isinstance(DIAGNOSTICS, Mapping):
        return context

    for key in DIAGNOSTICS:

        result = _get_latest_diagnostic(key)

        if result is None:
            continue

        context[key] = result

    return context


# ============================================================================
# CHAT HISTORY
# ============================================================================

def _get_chat_messages() -> list[dict[str, str]]:
    """Return the assistant's current conversation history."""

    if "ai_assistant_messages" not in st.session_state:

        st.session_state.ai_assistant_messages = []

    messages = st.session_state.ai_assistant_messages

    if not isinstance(messages, list):

        st.session_state.ai_assistant_messages = []

        return st.session_state.ai_assistant_messages

    return messages


def _append_message(
    role: str,
    content: str,
) -> None:
    """Append a chat message and enforce the history limit."""

    messages = _get_chat_messages()

    messages.append(
        {
            "role": role,
            "content": content,
        }
    )

    if len(messages) > MAX_CHAT_MESSAGES:

        del messages[
            : len(messages)
            - MAX_CHAT_MESSAGES
        ]


# ============================================================================
# NOTIFICATION RENDERING (native Streamlit — no HTML)
# ============================================================================

def _render_notifications() -> None:
    """Render notifications using native Streamlit components."""

    st.subheader(
        "🔔 Notifications & Alerts"
    )

    notifications = _build_notifications()

    for note in notifications:

        level = note["level"]

        title = note["title"]
        detail = note["detail"]

        if level == "critical":

            st.error(
                f"**{title}**\n\n"
                f"{detail}"
            )

        elif level == "warning":

            st.warning(
                f"**{title}**\n\n"
                f"{detail}"
            )

        elif level == "good":

            st.success(
                f"**{title}**\n\n"
                f"{detail}"
            )

        else:

            st.info(
                f"**{title}**\n\n"
                f"{detail}"
            )

    if st.button(
        "🔄 Refresh alerts",
        use_container_width=True,
        key="refresh_ai_alerts",
    ):

        # Only re-run the page to pull fresh notification data.
        # Do NOT reset ai_assistant_greeted here — that would make
        # the session-start greeting reappear on every refresh.
        st.rerun()


# ============================================================================
# CHAT RENDERING
# ============================================================================

def _render_chat() -> None:
    """Render the local Ollama research assistant chat."""

    st.subheader(
        f"{ASSISTANT_AVATAR} "
        f"Chat with {ASSISTANT_NAME}"
    )

    st.caption(
        f"Running locally on `{MODEL_ID}` "
        "· research data stays on this machine. 🔒"
    )

    messages = _get_chat_messages()

    for message in messages:

        role = message.get(
            "role",
            "assistant",
        )

        content = message.get(
            "content",
            "",
        )

        avatar = (
            ASSISTANT_AVATAR
            if role == "assistant"
            else USER_AVATAR
        )

        with st.chat_message(
            role,
            avatar=avatar,
        ):

            st.markdown(content)

    prompt = st.chat_input(
        f"Ask {ASSISTANT_NAME} "
        "about your research..."
    )

    if not prompt:
        return

    prompt = prompt.strip()

    if not prompt:
        return

    _append_message(
        "user",
        prompt,
    )

    with st.chat_message(
        "user",
        avatar=USER_AVATAR,
    ):

        st.markdown(prompt)

    diagnostic_context = {
        "diagnostics": _build_research_context(),
        "notifications": _build_notifications(),
        "signals": _build_signal_context(),
    }

    with st.chat_message(
        "assistant",
        avatar=ASSISTANT_AVATAR,
    ):

        with st.spinner(
            f"{ASSISTANT_NAME} is analyzing... 🧠"
        ):

            try:

                reply = chat_with_local_ai(
                    diagnostic_title=(
                        "Machet Mechanics "
                        "Phase 2 Research"
                    ),
                    diagnostic_result=(
                        diagnostic_context
                    ),
                    conversation=(
                        _get_chat_messages()
                    ),
                )

            except Exception as exc:

                reply = (
                    "I could not reach the local "
                    "AI model right now. The rest of the "
                    "dashboard is still usable.\n\n"
                    f"Error: `{exc}`"
                )

        st.markdown(reply)

    _append_message(
        "assistant",
        reply,
    )


# ============================================================================
# SESSION GREETING
# ============================================================================

def _send_session_start_greeting() -> None:
    """
    Add one deterministic greeting per Streamlit session.

    The greeting is deliberately generated locally without asking
    Ollama to produce it. This only fires once per session — the
    Refresh button does not reset it.
    """

    if st.session_state.get(
        "ai_assistant_greeted",
        False,
    ):

        return

    st.session_state.ai_assistant_greeted = True

    critical = [
        note
        for note in _build_notifications()
        if note["level"] == "critical"
    ]

    if critical:

        lines = "\n".join(
            f"- {note['title']}"
            for note in critical
        )

        greeting = (
            "Hey, welcome back! 👋 "
            "A few research items need "
            "your attention:\n\n"
            f"{lines}"
        )

    else:

        greeting = (
            "Hey, welcome back! 👋 "
            "Nothing urgent is flagged "
            "right now — smooth sailing. 🎉"
        )

    _append_message(
        "assistant",
        greeting,
    )


# ============================================================================
# PUBLIC ENTRY POINT
# ============================================================================

def render_ai_assistant_page() -> None:
    """
    Render the complete local AI assistant page.

    Call this function from the Phase 2 dashboard navigation.
    """

    _send_session_start_greeting()

    st.title(
        f"{ASSISTANT_AVATAR} "
        f"{ASSISTANT_NAME}"
    )

    st.caption(
        "Local research assistant for "
        "Machet Mechanics Phase 2. "
        "Read-only: it does not execute trades "
        "or modify your strategy."
    )

    st.divider()

    left, right = st.columns(
        [2, 1]
    )

    with left:

        _render_chat()

    with right:

        _render_notifications()