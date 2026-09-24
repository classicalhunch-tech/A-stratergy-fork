"""
dashboard/pages/4_📈_Paper_Trading.py

Phase 3 — Paper Trading dashboard page.

This page is DISPLAY ONLY.

It contains no:
    - trading logic
    - strategy logic
    - session logic
    - execution logic
    - persistence logic
    - performance calculations

It only reads state produced by the real Phase 3 engine through
the dashboard-facing Phase3Service application layer.

Architecture
------------

    Streamlit Page
          |
          v
    dashboard.phase3.backend.Phase3Service
          |
          v
    phase_03_paper/*
          |
          v
    Real Phase 3 components

Phase 1 and Phase 2 remain separate dashboard areas.
This page only provides navigation links to them.

The dashboard never creates duplicate Phase 3 engine components
just for display.

============================================================
VISUAL DESIGN
============================================================
Institutional / bank-statement styling using ONLY native Streamlit
components — no custom HTML, no unsafe_allow_html, no custom CSS.
Color/theme polish comes from .streamlit/config.toml (see the
companion snippet), not from injected styling here. Polish in this
file comes from: a KPI summary strip, colored status badges, a
session countdown banner, an equity curve, formatted tables via
column_config, and clearly iconed tabs. No visual element here
invents or estimates a number — everything rendered comes directly
from a real backend object.

============================================================
WIRING STATUS: LIVE
============================================================
All Phase 3 backend access goes through Phase3Service. This page
does not fabricate values, infer success from stdout, duplicate
backend calculations, or implement trading/strategy behavior.
The AI Assistant tab is the one exception to "no third party" —
it calls a local Ollama model (see dashboard/phase3/ai_helper.py)
purely to answer questions about data already shown on this page;
it never computes new statistics itself.

APPLICATION STATE: get_phase3_service() uses st.cache_resource so
one Phase3Service instance (and its managed Journal / Monitoring /
Performance state) survives every Streamlit rerun.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Machet Mechanics — Paper Trading",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# DASHBOARD APPLICATION SERVICE
# ============================================================

BACKEND_AVAILABLE = True
BACKEND_ERROR: str | None = None

try:
    from dashboard.phase3.backend import Phase3Service

except Exception as exc:  # noqa: BLE001
    BACKEND_AVAILABLE = False
    BACKEND_ERROR = str(exc)
    Phase3Service = None  # type: ignore[assignment]


@st.cache_resource
def get_phase3_service() -> "Phase3Service":
    """
    Return the single cached Phase3Service instance.

    Streamlit reruns pages frequently. cache_resource ensures the
    application service and its managed state survive those reruns.
    """
    if Phase3Service is None:
        raise RuntimeError("Phase3Service backend module could not be imported.")
    return Phase3Service()


# ============================================================
# DISPLAY ADAPTERS
# ============================================================

def get_system_health():
    return get_phase3_service().get_system_health()


def get_open_positions():
    return get_phase3_service().get_open_positions()


def get_closed_trades():
    return get_phase3_service().get_closed_trades()


def get_performance():
    return get_phase3_service().get_performance()


def get_latest_report():
    return get_phase3_service().get_latest_report()


def get_sessions():
    return get_phase3_service().get_sessions()


def run_historical_replay_call(csv_path: str, auto_approve_sessions: bool):
    return get_phase3_service().run_replay(
        csv_path,
        auto_approve_sessions=auto_approve_sessions,
    )


# ============================================================
# STATUS BADGE HELPER
# ============================================================

def status_badge(label: str, kind: str) -> None:
    """
    Render a colored status line using only native Streamlit
    components. kind: "good" | "warn" | "bad" | "info".
    """
    normalized = (label or "").upper()
    if kind == "good":
        st.success(normalized)
    elif kind == "warn":
        st.warning(normalized)
    elif kind == "bad":
        st.error(normalized)
    else:
        st.info(normalized)


def _fmt_pct(value) -> str:
    return f"{value * 100:.1f}%" if value is not None else "—"


def _fmt_r(value) -> str:
    return f"{value:+.2f}R" if value is not None else "—"


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("## :violet[◆] Machet Mechanics")
    st.caption("Phase 03 — Forward Testing")
    st.divider()

    st.page_link("app.py", label="← Back to Overview")
    st.page_link("pages/1_📊_Research.py", label="📊 Phase 1 — Research")
    st.page_link("pages/2_◆_Phase2_Lab.py", label="◆ Phase 2 — Lab")
    st.page_link("pages/3_🤖_Machet_AI.py", label="🤖 Machet AI — Chat")

    st.divider()
    st.caption("Display-only workspace. All execution logic resides in the backend service.")

    if not BACKEND_AVAILABLE:
        st.error("Phase 3 dashboard backend unavailable.")
        with st.expander("Backend error details"):
            st.code(BACKEND_ERROR or "Unknown error", language="text")


# ============================================================
# HERO SECTION
# ============================================================

st.caption("PHASE 03 · FORWARD TESTING ENGINE")
st.markdown("# :teal[📈] Institutional Paper Trading")
st.write(
    "Session-gated automation · simulated fill execution · "
    "audit journal · real-time monitoring."
)

st.divider()

if not BACKEND_AVAILABLE:
    st.error("Phase 3 dashboard backend failed to load. Review sidebar logs for diagnostic traces.")
    st.stop()


# ============================================================
# ACCOUNT SUMMARY STRIP (bank-statement style KPI row)
# ============================================================

service = get_phase3_service()
_perf_preview = None
try:
    _perf_preview = get_performance()
except Exception:  # noqa: BLE001
    _perf_preview = None

_open_positions_preview = []
try:
    _open_positions_preview = get_open_positions()
except Exception:  # noqa: BLE001
    _open_positions_preview = []

with st.container(border=True):
    kpi_cols = st.columns(4)

    total_r_value = _perf_preview.total_r if _perf_preview else 0.0
    kpi_cols[0].metric(
        "TOTAL R",
        _fmt_r(total_r_value) if _perf_preview else "—",
        delta=None,
    )

    win_rate_value = _perf_preview.win_rate if _perf_preview else None
    kpi_cols[1].metric(
        "WIN RATE",
        _fmt_pct(win_rate_value),
        delta=(f"{(win_rate_value - 0.5) * 100:+.1f} pts vs 50%" if win_rate_value is not None else None),
    )

    kpi_cols[2].metric(
        "OPEN POSITIONS",
        len(_open_positions_preview),
    )

    system_state = "ONLINE" if BACKEND_AVAILABLE else "OFFLINE"
    with kpi_cols[3]:
        st.caption("SYSTEM STATUS")
        status_badge(system_state, "good" if BACKEND_AVAILABLE else "bad")

st.divider()


# ============================================================
# CONTROLS PANEL
# ============================================================

with st.container(border=True):
    st.markdown("### :blue[Execution Controls]")

    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([2, 2, 3])

    with ctrl_col1:
        run_replay = st.button(
            "▶ Run Historical Replay",
            type="primary",
            width="stretch",
        )

    with ctrl_col2:
        refresh_health = st.button(
            "↻ Refresh System Health",
            width="stretch",
        )

    with ctrl_col3:
        dataset = st.selectbox(
            "Select Backtest Dataset",
            [
                "your_data_file.csv (5,000 candles — full)",
                "your_data_file_4000.csv",
                "your_data_file_3000.csv",
                "your_data_file_2000.csv",
                "your_data_file_1000.csv",
                "your_data_file_500.csv",
            ],
            label_visibility="collapsed",
        )

    auto_approve = st.checkbox(
        "Auto-approve all sessions for this replay run",
        value=True,
        help=(
            "When enabled, Phase3Service pre-approves session "
            "occurrences via SessionEngine's public API. "
            "When disabled, normal decision rules apply."
        ),
    )

if run_replay:
    with st.spinner("Executing replay through live Phase 3 pipeline..."):
        try:
            csv_name = dataset.split(" ", 1)[0].strip()
            csv_path = str(PROJECT_ROOT / csv_name)

            summary = run_historical_replay_call(csv_path, auto_approve)

            st.success(
                f"Replay complete — {summary.ticks_run} ticks processed, "
                f"{summary.opened_trades} trades opened, "
                f"{summary.closed_trades} trades closed."
            )

        except Exception as exc:  # noqa: BLE001
            st.error(f"Replay execution failed: {exc}")

if refresh_health:
    st.session_state["health_refreshed_at"] = datetime.now(timezone.utc)

st.divider()


# ============================================================
# SESSION BANNER (bank "next statement" style strip)
# ============================================================

try:
    sessions = get_sessions()
except Exception:  # noqa: BLE001
    sessions = []

if sessions:
    now_utc = datetime.now(timezone.utc)
    active = [s for s in sessions if s.status.value == "ACTIVE"]
    upcoming = [s for s in sessions if s.scheduled_start > now_utc]
    upcoming.sort(key=lambda s: s.scheduled_start)

    with st.container(border=True):
        session_cols = st.columns([2, 2, 2])

        with session_cols[0]:
            st.caption("CURRENT SESSION")
            if active:
                st.markdown(f"**{active[0].session_name}**")
                status_badge("ACTIVE", "good")
            else:
                st.markdown("**No active session**")
                status_badge("CLOSED", "info")

        with session_cols[1]:
            st.caption("NEXT SESSION")
            if upcoming:
                next_state = upcoming[0]
                minutes_away = (next_state.scheduled_start - now_utc).total_seconds() / 60.0
                st.markdown(f"**{next_state.session_name}**")
                st.caption(f"in {minutes_away:,.0f} min · {next_state.status.value}")
            else:
                st.markdown("—")

        with session_cols[2]:
            st.caption("TOTAL OCCURRENCES")
            st.markdown(f"**{len(sessions)}**")
            st.caption("materialized this run")

    st.divider()


# ============================================================
# SYSTEM STATUS TELEMETRY
# ============================================================

st.markdown("### :blue[System Health & Telemetry]")

try:
    health = get_system_health()
except Exception as exc:  # noqa: BLE001
    st.error(f"Failed to fetch system health: {exc}")
    health = None

if health is not None:
    with st.container(border=True):
        status_cols = st.columns(5)

        with status_cols[0]:
            st.caption("RUNTIME")
            status_badge("ALIVE" if health.runtime_alive else "STOPPED", "good" if health.runtime_alive else "warn")

        status_cols[1].metric("Candles Processed", health.candle_count)
        status_cols[2].metric("Adapter Errors", health.adapter_error_count)
        status_cols[3].metric("Pending Signals", health.pending_signal_count)

        with status_cols[4]:
            st.caption("PERSISTENCE")
            if health.persistence_ok is None:
                status_badge("N/A", "info")
            elif health.persistence_ok:
                status_badge("OK", "good")
            else:
                status_badge("ISSUE", "bad")

    if health.issues:
        with st.container(border=True):
            st.markdown(":orange[**Active Health Warnings**]")
            for issue in health.issues:
                st.caption(f"• {issue}")

st.divider()


# ============================================================
# MAIN TABS VIEW
# ============================================================

(
    tab_overview,
    tab_positions,
    tab_trades,
    tab_performance,
    tab_report,
    tab_ai_chat,
) = st.tabs(
    [
        "🧭 Overview",
        "📋 Positions",
        "📒 Trades",
        "📊 Performance",
        "🧾 AI Report",
        "💬 AI Assistant",
    ]
)


# ============================================================
# OVERVIEW TAB
# ============================================================

with tab_overview:
    st.subheader("Engine Pipeline Flow")

    with st.container(border=True):
        pipeline = [
            "Market Data", "Runtime Coordinator", "Session Engine", "Strategy Adapter",
            "Signal Event", "Paper Trade Engine", "Position Manager", "Journal",
            "SQLite Persistence", "Monitoring", "Performance", "Report Engine",
        ]
        step_cols = st.columns(len(pipeline))
        for col, step in zip(step_cols, pipeline):
            with col:
                st.markdown(":green[●]")
                st.caption(step)

    summary = service.get_last_summary()

    if summary is None:
        st.info("No replay session executed yet. Trigger a run from the controls above.")
    else:
        st.subheader("Last Replay Metrics Summary")
        with st.container(border=True):
            sum_cols = st.columns(4)
            sum_cols[0].metric("Total Candles", summary.total_candles)
            sum_cols[1].metric("Ticks Run", summary.ticks_run)
            sum_cols[2].metric("Opened Trades", summary.opened_trades)
            sum_cols[3].metric("Closed Trades", summary.closed_trades)

            if summary.first_timestamp and summary.last_timestamp:
                st.caption(f"Time Range: {summary.first_timestamp} → {summary.last_timestamp}")

        if summary.unhealthy_checks:
            st.warning(f"{len(summary.unhealthy_checks)} health check warning(s) flagged during run.")


# ============================================================
# OPEN POSITIONS TAB
# ============================================================

with tab_positions:
    st.subheader("Active Positions Monitor")

    try:
        positions = get_open_positions()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Error fetching open positions: {exc}")
        positions = []

    if not positions:
        st.caption("No open positions currently active.")
    else:
        rows = []
        for view in positions:
            trade = view.trade
            rows.append(
                {
                    "Trade ID": trade.trade_id,
                    "Direction": trade.direction.value,
                    "Session": trade.session,
                    "Entry": trade.entry,
                    "Stop": trade.stop,
                    "Target": trade.target,
                    "Opened At": trade.opened_at,
                    "State": view.position_state.value,
                }
            )
        df = pd.DataFrame(rows)
        st.dataframe(
            df,
            width="stretch",
            column_config={
                "Entry": st.column_config.NumberColumn(format="%.5f"),
                "Stop": st.column_config.NumberColumn(format="%.5f"),
                "Target": st.column_config.NumberColumn(format="%.5f"),
            },
            hide_index=True,
        )


# ============================================================
# CLOSED TRADES TAB
# ============================================================

with tab_trades:
    st.subheader("Trade Audit Journal")

    try:
        trades = get_closed_trades()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Error fetching closed trades: {exc}")
        trades = []

    if not trades:
        st.caption("No closed trade records logged yet.")
    else:
        rows = []
        for entry in trades:
            rows.append(
                {
                    "Trade ID": entry.trade_id,
                    "Direction": entry.direction,
                    "Session": entry.session,
                    "Entry": entry.entry,
                    "Stop": entry.stop,
                    "Target": entry.target,
                    "Exit Price": entry.exit_price,
                    "Exit Reason": entry.exit_reason.value if entry.exit_reason else None,
                    "Result (R)": entry.result_r,
                    "Closed At": entry.recorded_at,
                }
            )

        df = pd.DataFrame(rows)
        st.dataframe(
            df,
            width="stretch",
            hide_index=True,
            column_config={
                "Entry": st.column_config.NumberColumn(format="%.5f"),
                "Stop": st.column_config.NumberColumn(format="%.5f"),
                "Target": st.column_config.NumberColumn(format="%.5f"),
                "Exit Price": st.column_config.NumberColumn(format="%.5f"),
                "Result (R)": st.column_config.NumberColumn(format="%+.2f"),
            },
        )

        st.download_button(
            "Download Journal as CSV",
            df.to_csv(index=False),
            file_name="paper_trades_audit.csv",
            mime="text/csv",
        )


# ============================================================
# PERFORMANCE TAB
# ============================================================

with tab_performance:
    st.subheader("Performance Analytics")

    try:
        perf = get_performance()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Error fetching performance metrics: {exc}")
        perf = None

    if perf is None or perf.total_trades == 0:
        st.caption("No performance analytics available. Complete a replay cycle first.")
    else:
        with st.container(border=True):
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Trades", perf.total_trades)
            col2.metric("Win Rate", _fmt_pct(perf.win_rate))
            col3.metric("Total R", _fmt_r(perf.total_r))
            col4.metric("Avg R / Trade", _fmt_r(perf.average_r))

            col1, col2, col3 = st.columns(3)
            col1.metric("Current Streak", perf.current_streak)
            col2.metric("Best Winning Streak", perf.best_winning_streak)
            col3.metric("Max Drawdown (R)", f"{perf.max_drawdown_r:.2f}")

        # ------------------------------------------------------
        # Equity curve — cumulative R across closed trades in
        # chronological order, computed here purely for display
        # from real, already-persisted result_r values (no new
        # statistic is invented; this is a running sum of numbers
        # PerformanceEngine already validated).
        # ------------------------------------------------------
        try:
            closed = get_closed_trades()
        except Exception:  # noqa: BLE001
            closed = []

        timed_closed = [e for e in closed if e.recorded_at is not None and e.result_r is not None]
        timed_closed.sort(key=lambda e: e.recorded_at)

        if timed_closed:
            st.caption("Cumulative R Curve")
            cumulative = []
            running = 0.0
            for entry in timed_closed:
                running += entry.result_r
                cumulative.append(running)
            curve_df = pd.DataFrame(
                {"Cumulative R": cumulative},
                index=[e.recorded_at for e in timed_closed],
            )
            st.line_chart(curve_df, width="stretch")

        if perf.session_performance:
            st.caption("Session Performance Breakdown")
            session_rows = [
                {
                    "Session": name,
                    "Trades": stats.trades,
                    "Win Rate": _fmt_pct(stats.win_rate),
                    "Total R": _fmt_r(stats.total_r),
                }
                for name, stats in perf.session_performance.items()
            ]
            st.dataframe(pd.DataFrame(session_rows), width="stretch", hide_index=True)


# ============================================================
# AI REPORT TAB (deterministic, no LLM)
# ============================================================

with tab_report:
    st.subheader("Deterministic Audit Report")
    st.caption("Generated directly by ReportEngine — deterministic insights with zero third-party LLM latency.")

    try:
        report = get_latest_report()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Error fetching report narrative: {exc}")
        report = None

    if report is None:
        st.caption("No report snapshot found. Execute a replay session to generate output.")
    else:
        with st.container(border=True):
            st.write(report.narrative)

        st.download_button(
            "Download Report as Text",
            report.narrative,
            file_name="phase3_report.txt",
            mime="text/plain",
        )


# ============================================================
# AI ASSISTANT TAB (Ollama, grounded in real Phase 3 data)
# ============================================================

with tab_ai_chat:
    st.subheader("Ask the Local AI About These Results")
    st.caption(
        "Grounded in the replay summary, performance, trades, and health "
        "shown above. Runs entirely on your local Ollama instance."
    )

    from dashboard.phase3.ai_helper import chat_with_phase3_ai

    if "phase3_ai_messages" not in st.session_state:
        st.session_state["phase3_ai_messages"] = []

    chat_container = st.container(border=True, height=420)
    with chat_container:
        if not st.session_state["phase3_ai_messages"]:
            st.caption("No messages yet — ask a question below to get started.")
        for msg in st.session_state["phase3_ai_messages"]:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

    user_question = st.chat_input(
        "Ask about win rate, drawdown, session performance, health issues..."
    )

    if user_question:
        st.session_state["phase3_ai_messages"].append(
            {"role": "user", "content": user_question}
        )

        with st.spinner("Thinking..."):
            reply = chat_with_phase3_ai(
                summary=service.get_last_summary(),
                performance=perf if "perf" in dir() else None,
                closed_trades=closed if "closed" in dir() else [],
                health=health,
                report_narrative=report.narrative if "report" in dir() and report is not None else None,
                conversation=st.session_state["phase3_ai_messages"],
            )

        st.session_state["phase3_ai_messages"].append(
            {"role": "assistant", "content": reply}
        )
        st.rerun()

    clear_col, _ = st.columns([1, 5])
    with clear_col:
        if st.button("Clear conversation", width="stretch"):
            st.session_state["phase3_ai_messages"] = []
            st.rerun()


# ============================================================
# FOOTER
# ============================================================

st.divider()
st.caption(
    "Machet Mechanics · Phase 03 — Paper Trading Engine · "
    "Strictly Display-Only Architecture (AI Assistant tab excepted)"
)