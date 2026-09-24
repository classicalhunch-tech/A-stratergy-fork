"""
dashboard/app.py

Machet Mechanics Research Engine â€” main application.

Run from the project root:

    streamlit run dashboard/app.py

Phase 1:
    dashboard/pages/1_ðŸ“Š_Research.py

Phase 2:
    dashboard/pages/2_â—†_Phase2_Lab.py

Machet AI:
    dashboard/pages/3_ðŸ¤–_Machet_AI.py

Frontend rules:
    - Native Streamlit components only.
    - No custom HTML.
    - No unsafe_allow_html.
    - No custom CSS.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Machet Mechanics â€” Research Engine",
    page_icon="â—†",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# BOOT SEQUENCE ANIMATION
#
# Native animation: a single st.empty() placeholder rewritten
# frame-by-frame in a short loop. No CSS/HTML/JS involved â€” this
# is plain Streamlit rerender-in-place. Runs once per session.
# ============================================================

def _render_boot_sequence() -> None:
    if st.session_state.get("boot_sequence_shown", False):
        return

    st.session_state.boot_sequence_shown = True

    frames = [
        ":violet[â—† INITIALIZING MACHET MECHANICS...]",
        ":violet[â—† INITIALIZING MACHET MECHANICS...]  :blue[loading strategy engine]",
        ":violet[â—† INITIALIZING MACHET MECHANICS...]  :blue[loading strategy engine]  :orange[linking research modules]",
        ":green[â—† RESEARCH ENGINE ONLINE]",
    ]

    placeholder = st.empty()

    for frame in frames:
        placeholder.markdown(f"##### {frame}")
        time.sleep(0.25)

    time.sleep(0.4)
    placeholder.empty()


_render_boot_sequence()


# ============================================================
# PHASE DATA
#
# Single source of truth. Nav bar, status cards, sidebar summary,
# and progress bar are all derived from this list â€” update a
# phase here and every surface in the app stays in sync.
# ============================================================

@dataclass(frozen=True)
class Phase:
    number: str
    name: str
    subtitle: str
    description: str
    active: bool
    items: list[str]
    page_path: str | None = None  # None => locked, no navigation target


PHASES: list[Phase] = [
    Phase(
        number="01",
        name="Historical Research",
        subtitle="Backtesting & validation",
        description=(
            "Validate the strategy against historical market data "
            "before advancing to robustness research."
        ),
        active=True,
        items=["Backtesting", "Equity analysis", "Drawdown analysis", "Trade ledger"],
        page_path="pages/1_ðŸ“Š_Research.py",
    ),
    Phase(
        number="02",
        name="Optimization",
        subtitle="Robustness & parameter discovery",
        description=(
            "Test parameter stability, walk-forward behavior, "
            "Monte Carlo robustness, transaction costs, and overfit risk."
        ),
        active=True,
        items=["Parameter sweeps", "Walk-forward", "Monte Carlo", "Overfit detection"],
        page_path="pages/2_â—†_Phase2_Lab.py",
    ),
    Phase(
        number="03",
        name="Paper Trading",
        subtitle="Forward testing",
        description=(
            "Evaluate execution behavior against a live market feed "
            "without using real capital."
        ),
        active=True,
        items=["Market feed", "Paper positions", "Slippage", "Latency"],
        page_path="pages/4_ðŸ“ˆ_Paper_Trading.py",
    ),
    Phase(
        number="04",
        name="Live Execution",
        subtitle="Production",
        description=(
            "Future production stage with execution controls, "
            "monitoring, audit logging, and safety systems."
        ),
        active=False,
        items=["MT5 execution", "Risk engine", "Kill switch", "Audit logs"],
    ),
]

ACTIVE_PHASES = [p for p in PHASES if p.active]
LOCKED_PHASES = [p for p in PHASES if not p.active]


# ============================================================
# NAVIGATION HELPER
# ============================================================
def switch_page_safely(page_path: str) -> None:
    """Navigate to a dashboard page."""
    st.switch_page(page_path)

# ============================================================
# RENDER HELPERS
# ============================================================

def render_nav_button(phase: Phase, *, key_suffix: str) -> None:
    """Render a single top-nav button â€” active phases navigate,
    locked phases render disabled with a lock icon."""
    if phase.active and phase.page_path:
        if st.button(
            f"PHASE {phase.number} â€” {phase.name.upper()}",
            key=f"nav_{key_suffix}",
            use_container_width=True,
            type="primary",
        ):
            switch_page_safely(phase.page_path)
    else:
        st.button(
            f"ðŸ”’ PHASE {phase.number} â€” {phase.name.upper()}",
            key=f"nav_{key_suffix}",
            use_container_width=True,
            disabled=True,
        )


def render_phase_card(phase: Phase) -> None:
    """Render a phase card using native Streamlit components only."""
    with st.container(border=True):
        header_left, header_right = st.columns([3, 1])

        with header_left:
            st.caption(f"PHASE {phase.number}")

        with header_right:
            if phase.active:
                st.markdown(":green[**ACTIVE**]")
            else:
                st.markdown(":orange[**LOCKED**]")

        st.subheader(phase.name)
        st.caption(phase.subtitle)
        st.write(phase.description)

        st.divider()

        marker = "âœ“" if phase.active else "â—‹"
        for item in phase.items:
            st.write(f"{marker} {item}")

        if phase.active and phase.page_path:
            st.button(
                f"Open Phase {phase.number}",
                key=f"open_{phase.number}",
                use_container_width=True,
                on_click=switch_page_safely,
                args=(phase.page_path,),
            )
        elif not phase.active:
            st.info("ðŸ”’ Locked â€” requires completion of prior phases.")


# ============================================================
# HERO
# ============================================================

st.caption("SYSTEM DEVELOPMENT ROADMAP")
st.markdown("# :violet[â—†] Machet Mechanics :blue[Research Engine]")
st.write("Strategy development Â· validation Â· execution")

status_col, progress_col = st.columns([1, 3])

with status_col:
    st.markdown("### :green[â—] :green[RESEARCH ENGINE ONLINE]")

with progress_col:
    progress_pct = len(ACTIVE_PHASES) / len(PHASES)
    st.progress(progress_pct, text=f"{len(ACTIVE_PHASES)} of {len(PHASES)} phases active")

st.divider()


# ============================================================
# PHASE NAVIGATION
# ============================================================

st.markdown("### :blue[System Phases]")

nav_cols = st.columns(len(PHASES))
for col, phase in zip(nav_cols, PHASES):
    with col:
        render_nav_button(phase, key_suffix=phase.number)


# ============================================================
# CURRENT SYSTEM STATUS
# ============================================================

st.markdown("### :orange[Current Development Status]")

if ACTIVE_PHASES:
    active_cols = st.columns(len(ACTIVE_PHASES), gap="large")
    for col, phase in zip(active_cols, ACTIVE_PHASES):
        with col:
            render_phase_card(phase)
else:
    st.info("No phases are currently active.")


# ============================================================
# FUTURE PHASES
# ============================================================

if LOCKED_PHASES:
    st.divider()
    st.markdown("### :violet[Future Execution Stages]")

    locked_cols = st.columns(len(LOCKED_PHASES), gap="large")
    for col, phase in zip(locked_cols, LOCKED_PHASES):
        with col:
            render_phase_card(phase)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("## :violet[â—†] Machet Mechanics")
    st.caption("Research Engine")

    st.divider()
    st.markdown("#### :blue[System Status]")

    for phase in PHASES:
        label = f"Phase {phase.number} â€” {phase.name}"
        if phase.active:
            st.markdown(f":green[âœ“] {label}")
        else:
            st.caption(f"ðŸ”’ {label}")

    st.divider()
    st.markdown("#### :orange[Architecture]")

    pipeline_steps = [
        "Historical Data",
        "Strategy Engine",
        "Backtest",
        "Robustness Research",
        "Future Paper Trading",
        "Future Execution",
    ]
    for i, step in enumerate(pipeline_steps):
        st.caption(step)
        if i < len(pipeline_steps) - 1:
            st.caption("â†“")

    st.divider()
    st.markdown("#### :violet[Research Assistant]")

    st.page_link(
        "pages/3_ðŸ¤–_Machet_AI.py",
        label="ðŸ¤– Machet AI â€” Chat",
        help="Local Ollama-powered research assistant (read-only).",
    )

    st.divider()
    st.caption("Phases unlock sequentially.")


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "ðŸ§ª Research before execution. "
    "Each phase must produce sufficient evidence before "
    "the system progresses to the next stage."
)
st.caption("Machet Mechanics Research Engine Â· Phase-controlled development")
