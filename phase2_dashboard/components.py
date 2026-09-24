"""
phase2_dashboard/components.py

Phase 2 native Streamlit UI components.

This module is intentionally UI-only.

It provides:
- navigation buttons
- diagnostic cards
- metric cards
- locked phase cards
- history entries
- status labels
- native progress indicators
- terminal-style header
- terminal status dots
- terminal KPI tiles
- terminal engine-status rows
- terminal stylesheet injection

It does NOT:
- run diagnostics
- write results
- modify backend state
- execute trades
- perform persistence
- calculate research statistics
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Mapping

import streamlit as st


# ============================================================================
# STATUS
# ============================================================================

def _normalize_status(status: Any) -> str:
    """Normalize status text."""

    return str(status or "NOT RUN").strip().upper()


def status_label(status: str) -> str:
    """
    Return a normalized diagnostic status label.

    This is intentionally a small presentation-level normalization.
    Detailed terminal status treatment is handled separately by
    _TERMINAL_STATUS_MAP.
    """

    normalized = _normalize_status(status)

    if normalized in {
        "PASS",
        "PASSED",
        "SUCCESS",
        "SUCCEEDED",
        "HEALTHY",
        "ONLINE",
        "READY",
    }:
        return "PASS"

    if normalized in {
        "FAIL",
        "FAILED",
        "ERROR",
        "CRITICAL",
        "OFFLINE",
    }:
        return "FAIL"

    if normalized in {
        "WARNING",
        "WARN",
        "DEGRADED",
        "REVIEW",
    }:
        return "WARNING"

    if normalized in {
        "RUNNING",
        "IN PROGRESS",
        "PENDING",
        "SYNCING",
    }:
        return "RUNNING"

    if normalized in {
        "TIMED OUT",
        "TIMEOUT",
    }:
        return "TIMED OUT"

    if normalized == "INTERRUPTED":
        return "INTERRUPTED"

    return "NOT RUN"


def status_badge_html(status: str) -> str:
    """
    Backward-compatible status helper.

    The terminal frontend now uses terminal_status_dot().
    This function remains so existing imports do not break.

    Returns:
        Plain normalized status text.
    """

    return status_label(status)


# ============================================================================
# TERMINAL STATUS
# ============================================================================

# Single source of truth for terminal status presentation.
#
# CSS classes are defined by terminal.css.
_TERMINAL_STATUS_MAP = {
    # Healthy / successful
    "PASS": ("pass", "PASS"),
    "PASSED": ("pass", "PASS"),
    "SUCCESS": ("pass", "PASS"),
    "SUCCEEDED": ("pass", "PASS"),
    "ONLINE": ("pass", "ONLINE"),
    "READY": ("pass", "READY"),
    "HEALTHY": ("pass", "HEALTHY"),

    # Warning / active
    "WARNING": ("warn", "WARNING"),
    "WARN": ("warn", "WARNING"),
    "DEGRADED": ("warn", "DEGRADED"),
    "RUNNING": ("warn", "RUNNING"),
    "REVIEW": ("warn", "REVIEW"),
    "PENDING": ("warn", "PENDING"),
    "SYNCING": ("warn", "SYNCING"),

    # Failure
    "FAIL": ("fail", "FAILED"),
    "FAILED": ("fail", "FAILED"),
    "ERROR": ("fail", "ERROR"),
    "CRITICAL": ("fail", "CRITICAL"),
    "OFFLINE": ("fail", "OFFLINE"),
    "TIMED OUT": ("fail", "TIMED OUT"),
    "TIMEOUT": ("fail", "TIMED OUT"),
    "INTERRUPTED": ("fail", "INTERRUPTED"),

    # Neutral
    "NOT RUN": ("muted", "NOT RUN"),
    "LOCKED": ("muted", "LOCKED"),
    "UNAVAILABLE": ("muted", "UNAVAILABLE"),

    # Accent / informational
    "INFO": ("accent", "INFO"),
    "ACTIVE": ("accent", "ACTIVE"),
}


def terminal_status_dot(
    status: str,
    *,
    label: str | None = None,
) -> str:
    """
    Return an HTML fragment containing a terminal-style status dot
    and label.

    The actual visual styling is defined by terminal.css.

    Example:

        st.markdown(
            terminal_status_dot("PASS"),
            unsafe_allow_html=True,
        )
    """

    normalized = _normalize_status(status)

    css_suffix, default_label = _TERMINAL_STATUS_MAP.get(
        normalized,
        ("muted", normalized or "UNKNOWN"),
    )

    display_label = (
        label
        if label is not None
        else default_label
    )

    return (
        '<span class="mm-status">'
        f'<span class="mm-dot mm-dot-{css_suffix}"></span>'
        f'<span class="mm-label-{css_suffix}">'
        f"{display_label}"
        "</span>"
        "</span>"
    )


def render_status_line(
    status: str,
    *,
    label: str | None = None,
) -> None:
    """Render one terminal-style status line."""

    st.markdown(
        terminal_status_dot(
            status,
            label=label,
        ),
        unsafe_allow_html=True,
    )


# ============================================================================
# TERMINAL HEADER
# ============================================================================

def render_terminal_header(
    title: str,
    subtitle: str,
    *,
    system_status: str = "ONLINE",
    dataset: str | None = None,
) -> None:
    """
    Render the persistent terminal header.

    Values are supplied by the caller. This component does not
    determine backend health itself.
    """

    now = datetime.now().strftime(
        "%d %b %Y %H:%M"
    )

    status_html = terminal_status_dot(
        system_status
    )

    dataset_line = ""

    if dataset:
        dataset_line = f" · {dataset}"

    st.markdown(
        f"""
        <div class="mm-header">

            <div class="mm-header-left">

                <div class="mm-header-title">
                    {title}
                </div>

                <div class="mm-header-subtitle">
                    {subtitle}{dataset_line}
                </div>

            </div>

            <div class="mm-header-right">

                {status_html}

                <div class="mm-header-clock">
                    {now}
                </div>

            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================================
# TERMINAL KPI TILE
# ============================================================================

def render_kpi_tile(
    value: Any,
    label: str,
    *,
    status: str | None = None,
) -> None:
    """
    Render one compact terminal KPI tile.

    The value is display-only.

    Optional status adds a visual value class for terminal.css.
    """

    display_value = (
        "—"
        if value is None
        else str(value)
    )

    value_class = "mm-tile-value"

    if status:
        normalized = _normalize_status(status)

        if normalized in {
            "PASS",
            "PASSED",
            "SUCCESS",
            "SUCCEEDED",
            "HEALTHY",
        }:
            value_class += " mm-tile-value-pass"

        elif normalized in {
            "WARNING",
            "WARN",
            "DEGRADED",
            "REVIEW",
        }:
            value_class += " mm-tile-value-warn"

        elif normalized in {
            "FAIL",
            "FAILED",
            "ERROR",
            "CRITICAL",
        }:
            value_class += " mm-tile-value-fail"

        elif normalized in {
            "ACTIVE",
            "INFO",
        }:
            value_class += " mm-tile-value-accent"

    st.markdown(
        f"""
        <div class="mm-tile">

            <div class="{value_class}">
                {display_value}
            </div>

            <div class="mm-tile-label">
                {label}
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================================
# TERMINAL SECTION LABEL
# ============================================================================

def render_section_label(label: str) -> None:
    """Render a compact terminal section label."""

    st.markdown(
        f"""
        <div class="mm-section-label">
            {label}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================================
# TERMINAL ENGINE STATUS ROW
# ============================================================================

def render_engine_status_row(
    name: str,
    status: str,
    meta: str,
) -> None:
    """
    Render one research-engine status row.

    Example:

        MONTE CARLO       PASS       12.42s
    """

    status_html = terminal_status_dot(
        status
    )

    st.markdown(
        f"""
        <div class="mm-engine-row">

            <span class="mm-engine-name">
                {name}
            </span>

            {status_html}

            <span class="mm-engine-meta">
                {meta}
            </span>

        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================================
# TERMINAL THEME
# ============================================================================

def inject_terminal_theme(
    css_path: str | Path,
) -> None:
    """
    Load and inject terminal.css.

    If the stylesheet does not exist or cannot be read,
    the application continues with native Streamlit styling.
    """

    try:
        css_file = Path(css_path)

        with css_file.open(
            "r",
            encoding="utf-8",
        ) as file:
            css = file.read()

    except OSError:
        return

    if not css.strip():
        return

    st.markdown(
        f"<style>{css}</style>",
        unsafe_allow_html=True,
    )


# ============================================================================
# NAVIGATION
# ============================================================================

def render_nav_buttons(
    pages: list[str],
    active_page: str,
    *,
    key_prefix: str = "nav",
) -> str:
    """
    Render native Streamlit navigation buttons.

    Returns:
        Selected page name.

        If no button was clicked, active_page is returned.
    """

    selected_page = active_page

    if not pages:
        return selected_page

    columns = st.columns(
        len(pages)
    )

    for column, page in zip(
        columns,
        pages,
    ):

        is_active = (
            page == active_page
        )

        with column:

            button_type = (
                "primary"
                if is_active
                else "secondary"
            )

            if st.button(
                page,
                key=f"{key_prefix}_{page}",
                use_container_width=True,
                type=button_type,
            ):
                selected_page = page

    return selected_page


def render_locked_nav_item(
    label: str,
    *,
    key_prefix: str = "nav_locked",
) -> None:
    """Render a disabled future-phase navigation item."""

    st.button(
        f"🔒 {label}",
        key=f"{key_prefix}_{label}",
        use_container_width=True,
        disabled=True,
    )


# ============================================================================
# DIAGNOSTIC CARD
# ============================================================================

def render_diagnostic_card(
    *,
    key: str,
    title: str,
    description: str,
    status: str,
    run_label: str = "Run Diagnostic",
    category: str | None = None,
    progress: float | None = None,
    subtitle: str | None = None,
    button_key: str | None = None,
) -> bool:
    """
    Render a diagnostic card.

    This function is UI-only.

    Returns:
        True  -> run button clicked
        False -> no click
    """

    normalized_status = status_label(
        status
    )

    action_key = (
        button_key
        if button_key
        else f"run_{key}"
    )

    with st.container(
        border=True
    ):

        if category:
            st.caption(category)

        st.subheader(title)

        if subtitle:
            st.caption(subtitle)

        render_status_line(
            normalized_status
        )

        st.write(description)

        if progress is not None:

            try:
                progress_ratio = float(
                    progress
                )

                if progress_ratio > 1.0:
                    progress_ratio /= 100.0

                progress_ratio = max(
                    0.0,
                    min(
                        1.0,
                        progress_ratio,
                    ),
                )

                st.caption(
                    "Diagnostic progress · "
                    f"{progress_ratio * 100:.0f}%"
                )

                st.progress(
                    progress_ratio
                )

            except (
                TypeError,
                ValueError,
            ):
                pass

        st.caption(
            f"TEST · {key}"
        )

        return st.button(
            run_label,
            key=action_key,
            use_container_width=True,
            type="primary",
        )


# ============================================================================
# METRIC CARD
# ============================================================================

def render_metric_card(
    label: str,
    value: Any,
    *,
    help_text: str | None = None,
    delta: Any | None = None,
    delta_label: str | None = None,
    delta_color: Literal[
        "normal",
        "inverse",
        "off",
    ] = "normal",
    icon: str | None = None,
) -> None:
    """
    Render a native Streamlit metric.

    Values are display-only.
    """

    display_label = str(
        label
    )

    if icon:
        display_label = (
            f"{icon} {display_label}"
        )

    value_text = (
        "—"
        if value is None
        else str(value)
    )

    delta_text: str | None = None

    if delta is not None:

        delta_text = str(
            delta
        )

        if delta_label:
            delta_text = (
                f"{delta_text} "
                f"{delta_label}"
            )

    st.metric(
        label=display_label,
        value=value_text,
        delta=delta_text,
        delta_color=delta_color,
        help=help_text,
    )


# ============================================================================
# LOCKED PHASE CARD
# ============================================================================

def render_locked_phase_card(
    phase: str,
    title: str,
    description: str,
    *,
    requirement: str = (
        "Complete Phase 2 validation first"
    ),
) -> None:
    """Render a locked future-phase card."""

    with st.container(
        border=True
    ):

        st.caption(phase)

        st.subheader(
            f"🔒 {title}"
        )

        st.write(
            description
        )

        st.divider()

        st.info(
            f"Requirement: {requirement}"
        )

        st.button(
            "🔒 Locked",
            disabled=True,
            use_container_width=True,
            key=(
                f"locked_"
                f"{phase}_"
                f"{title}"
            ),
        )


# ============================================================================
# HISTORY
# ============================================================================

def render_history_entry(
    record: Mapping[str, Any] | Any,
) -> None:
    """
    Render one historical diagnostic run.

    Supports:
        - dictionaries
        - DiagnosticRecord-style objects

    Status resolution:

        returncode 0   -> PASS
        returncode -1  -> TIMED OUT
        returncode -2  -> INTERRUPTED
        other code     -> FAIL

    If returncode is unavailable, the legacy succeeded field
    is used for compatibility.
    """

    def get_value(
        name: str,
        default: Any = None,
    ) -> Any:

        if isinstance(
            record,
            Mapping,
        ):
            return record.get(
                name,
                default,
            )

        return getattr(
            record,
            name,
            default,
        )

    key = get_value(
        "key",
        "Unknown",
    )

    # ------------------------------------------------------------------------
    # Canonical return-code semantics
    # ------------------------------------------------------------------------

    returncode = get_value(
        "returncode",
        None,
    )

    if returncode == 0:
        status = "PASS"

    elif returncode == -1:
        status = "TIMED OUT"

    elif returncode == -2:
        status = "INTERRUPTED"

    elif returncode is not None:
        status = "FAIL"

    else:
        # Legacy compatibility.
        succeeded = bool(
            get_value(
                "succeeded",
                False,
            )
        )

        status = (
            "PASS"
            if succeeded
            else "FAIL"
        )

    # ------------------------------------------------------------------------
    # Timestamp
    # ------------------------------------------------------------------------

    started_at_display = get_value(
        "started_at_display",
        None,
    )

    started_at = get_value(
        "started_at",
        None,
    )

    if started_at_display:
        timestamp = str(
            started_at_display
        )

    elif started_at is None:
        timestamp = "Unknown time"

    else:
        timestamp = str(
            started_at
        )

    # ------------------------------------------------------------------------
    # Duration
    # ------------------------------------------------------------------------

    duration_seconds = get_value(
        "duration_seconds",
        None,
    )

    if duration_seconds is None:
        duration = "—"

    else:

        try:
            duration = (
                f"{float(duration_seconds):.2f}s"
            )

        except (
            TypeError,
            ValueError,
        ):
            duration = str(
                duration_seconds
            )

    # ------------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------------

    with st.container(
        border=True
    ):

        columns = st.columns(
            [3, 1]
        )

        with columns[0]:

            st.write(
                f"**{key}**"
            )

            st.caption(
                f"◷ {timestamp} · {duration}"
            )

        with columns[1]:

            render_status_line(
                status
            )


# ============================================================================
# SECTION DIVIDER
# ============================================================================

def render_section_divider(
    label: str | None = None,
) -> None:
    """Render a native Streamlit section divider."""

    if label:
        st.subheader(label)

    st.divider()


# ============================================================================
# INFO BANNER
# ============================================================================

def render_info_banner(
    title: str,
    message: str,
    *,
    icon: str = "◆",
) -> None:
    """Render a compact native informational banner."""

    st.info(
        f"{icon} **{title}**\n\n{message}"
    )


# ============================================================================
# EMPTY STATE
# ============================================================================

def render_empty_state(
    title: str,
    message: str,
    *,
    icon: str = "◇",
) -> None:
    """Render a native empty-state panel."""

    st.info(
        f"{icon} **{title}**\n\n{message}"
    )