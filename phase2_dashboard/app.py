"""
phase2_dashboard/app.py

MACHET MECHANICS — Research Terminal

Architecture:

    Native Streamlit UI (this file)
        |
        +--> Phase 1 UI
        |       |
        |       +--> existing strategy/backtest layer
        |
        +--> Phase2Service
        |       |
        |       +--> phase2_dashboard.runner
        |       +--> phase2_dashboard.results
        |
        +--> Phase3Service
                |
                +--> phase_03_paper/* (real Phase 3 engine)

This module:

- renders the terminal UI using ONLY native Streamlit widgets
  (no custom HTML/CSS injection, no `components.py` theme layer)
- routes between dashboard sections
- collects user input
- displays backend results

This module does NOT:

- implement strategy logic
- implement research diagnostics
- execute live trades
- implement persistence directly
- calculate new research statistics
- fabricate missing result values

Run with:

    streamlit run phase2_dashboard/app.py
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import streamlit as st


# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Machet Mechanics — Research Terminal",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================================
# PROJECT PATH
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# PERSISTED DATASET CHOICE
#
# st.session_state resets whenever the app process restarts, which is
# why re-uploading the same CSV every time felt necessary. The actual
# dataset files already live on disk, so instead of requiring a fresh
# browser upload each run, we persist the chosen path to a tiny local
# file and reload it on startup.
# ============================================================================

_LAST_DATASET_FILE = PROJECT_ROOT / "phase2_dashboard" / ".last_dataset.txt"


def _load_last_dataset_path() -> str | None:
    """Return the last-used dataset path, if it was saved and still exists."""
    try:
        if _LAST_DATASET_FILE.exists():
            saved = _LAST_DATASET_FILE.read_text(encoding="utf-8").strip()
            if saved and Path(saved).exists():
                return saved
    except Exception:
        pass
    return None


def _save_last_dataset_path(path: str) -> None:
    """Persist the chosen dataset path so future app restarts remember it."""
    try:
        _LAST_DATASET_FILE.parent.mkdir(parents=True, exist_ok=True)
        _LAST_DATASET_FILE.write_text(path, encoding="utf-8")
    except Exception:
        pass


def _discover_project_csvs() -> list[Path]:
    """
    List candidate dataset CSVs already sitting in the project —
    the root-level staged datasets plus anything previously uploaded
    through Phase 01, so they're pickable without a new upload.
    """
    candidates: list[Path] = []

    candidates.extend(sorted(PROJECT_ROOT.glob("your_data_file*.csv")))

    uploaded_dir = PROJECT_ROOT / "phase2_dashboard" / "uploaded_datasets"
    if uploaded_dir.exists():
        candidates.extend(sorted(uploaded_dir.glob("*.csv")))

    # de-dupe while preserving order
    seen = set()
    unique = []
    for path in candidates:
        resolved = str(path.resolve())
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


# ============================================================================
# BACKEND — PHASE 2
# ============================================================================

BACKEND_AVAILABLE = True
BACKEND_ERROR: str | None = None

try:
    from phase2_dashboard.backend import Phase2Service, formatter

    service = Phase2Service()

except Exception as exc:
    BACKEND_AVAILABLE = False
    BACKEND_ERROR = str(exc)
    service = None
    formatter = None


# ============================================================================
# BACKEND — PHASE 3
# ============================================================================

PHASE3_BACKEND_AVAILABLE = True
PHASE3_BACKEND_ERROR: str | None = None

try:
    from dashboard.phase3.backend import Phase3Service

except Exception as exc:
    PHASE3_BACKEND_AVAILABLE = False
    PHASE3_BACKEND_ERROR = str(exc)
    Phase3Service = None  # type: ignore[assignment]


@st.cache_resource
def get_phase3_service() -> "Phase3Service":
    """
    Return the single cached Phase3Service instance.

    Streamlit reruns pages frequently. cache_resource ensures the
    application service and its managed state (Journal, Monitoring,
    Performance, etc.) survive those reruns.
    """
    if Phase3Service is None:
        raise RuntimeError("Phase3Service backend module could not be imported.")
    return Phase3Service()


# ============================================================================
# AI ASSISTANT
# ============================================================================

AI_ASSISTANT_AVAILABLE = True
AI_ASSISTANT_ERROR: str | None = None

try:
    from phase2_dashboard.ai_assistant import (
        render_ai_assistant_page,
    )

except Exception as exc:
    AI_ASSISTANT_AVAILABLE = False
    AI_ASSISTANT_ERROR = str(exc)
    render_ai_assistant_page = None


# ============================================================================
# DIAGNOSTIC REGISTRY
# ============================================================================

DIAGNOSTIC_META = {
    "monte_carlo": {
        "number": "01",
        "title": "Monte Carlo",
        "group": "Robustness",
        "description": (
            "Tests whether observed performance remains stable "
            "when trade ordering and sampling are randomized."
        ),
    },
    "warmup_probe": {
        "number": "02",
        "title": "Warmup Sensitivity",
        "group": "Robustness",
        "description": (
            "Checks whether results depend heavily on the amount "
            "of historical data used before evaluation."
        ),
    },
    "walk_forward": {
        "number": "03",
        "title": "Walk-Forward",
        "group": "Robustness",
        "description": (
            "Tests performance across sequential historical "
            "evaluation windows rather than one full-sample result."
        ),
    },
    "parameter_sensitivity": {
        "number": "04",
        "title": "Parameter Stability",
        "group": "Stability",
        "description": (
            "Tests whether nearby parameter settings produce "
            "similar results instead of one isolated optimum."
        ),
    },
    "stop_buffer_diagnostic": {
        "number": "05",
        "title": "Stop Buffer",
        "group": "Costs",
        "description": (
            "Measures sensitivity to stop-placement assumptions "
            "and small execution changes."
        ),
    },
    "spread_sensitivity": {
        "number": "06",
        "title": "Transaction Costs",
        "group": "Costs",
        "description": (
            "Measures how increasing trading friction changes "
            "expected performance."
        ),
    },
    "trial_registry": {
        "number": "07",
        "title": "Trial Registry",
        "group": "Overfitting",
        "description": (
            "Records the configurations tested during the "
            "Phase 2 research process."
        ),
    },
    "overfit_detection": {
        "number": "08",
        "title": "DSR + PBO",
        "group": "Overfitting",
        "description": (
            "Evaluates whether apparent performance may be "
            "explained by research selection effects and "
            "backtest luck."
        ),
    },
    "pbo_no_confounds": {
        "number": "09",
        "title": "PBO — Clean Trial",
        "group": "Overfitting",
        "description": (
            "Repeats the overfitting analysis after removing "
            "the known confounded trial."
        ),
    },
    "pbo_blocks_8": {
        "number": "10",
        "title": "PBO — 8 Blocks",
        "group": "Overfitting",
        "description": (
            "Checks whether the PBO conclusion changes when "
            "the CSCV block configuration changes."
        ),
    },
}


PHASE02_GROUPS = {
    "ROBUSTNESS": [
        "monte_carlo",
        "warmup_probe",
        "walk_forward",
    ],
    "STABILITY": [
        "parameter_sensitivity",
    ],
    "COSTS": [
        "stop_buffer_diagnostic",
        "spread_sensitivity",
    ],
    "OVERFITTING": [
        "trial_registry",
        "overfit_detection",
        "pbo_no_confounds",
        "pbo_blocks_8",
    ],
}


PHASE03_SUBPAGES = [
    "OVERVIEW",
    "POSITIONS",
    "TRADES",
    "PERFORMANCE",
    "CHART",
    "AI ASSISTANT",
    "AI REPORT",
]


PHASE03_DATASETS = [
    "your_data_file.csv",
    "your_data_file_4000.csv",
    "your_data_file_3000.csv",
    "your_data_file_2000.csv",
    "your_data_file_1000.csv",
    "your_data_file_500.csv",
]


# ============================================================================
# SESSION STATE
# ============================================================================

DEFAULT_STATE = {
    "mm_section": "OVERVIEW",
    "mm_phase01_page": "DATA UPLOAD",
    "mm_phase02_page": "ROBUSTNESS",
    "mm_phase03_page": "OVERVIEW",
    "running": False,
    "confirm_clear_history": False,
    "uploaded_dataset_path": None,
    "backtest_result": None,
    "phase3_last_dataset": None,
}


for key, value in DEFAULT_STATE.items():

    if key not in st.session_state:
        st.session_state[key] = value

if st.session_state.uploaded_dataset_path is None:
    _restored = _load_last_dataset_path()
    if _restored:
        st.session_state.uploaded_dataset_path = _restored


# ============================================================================
# HELPERS
# ============================================================================

def _safe_elapsed(value: Any) -> str:
    """Format a duration without inventing a value."""

    try:
        return f"{float(value):.2f}s"

    except (
        TypeError,
        ValueError,
    ):
        return "—"


def diagnostic_is_available(
    key: str,
) -> bool:
    """Return whether the backend currently exposes a diagnostic."""

    if not BACKEND_AVAILABLE:
        return False

    try:
        return (
            key
            in service.get_available_diagnostics()
        )

    except Exception:
        return False


def get_status_value(
    key: str,
) -> str:
    """
    Return the canonical backend DiagnosticStatus value.

    Falls back to NOT_RUN if the backend cannot be queried.
    """

    if not BACKEND_AVAILABLE:
        return "NOT_RUN"

    try:
        return service.get_status(
            key
        ).value

    except Exception:
        return "NOT_RUN"


def run_one(
    key: str,
) -> None:
    """
    Run one diagnostic through Phase2Service.

    Phase2Service owns execution and persistence.
    """

    if (
        not BACKEND_AVAILABLE
        or key not in DIAGNOSTIC_META
        or not diagnostic_is_available(key)
    ):
        st.error(
            "This diagnostic is not available."
        )
        return

    meta = DIAGNOSTIC_META[key]

    st.session_state.running = True

    try:

        with st.spinner(
            f"Running {meta['title']}..."
        ):
            service.run_diagnostic(
                key
            )

    except Exception as exc:

        st.error(
            f"{meta['title']} failed to run: {exc}"
        )

    finally:
        st.session_state.running = False


def system_status_value() -> str:
    """Return terminal-level backend status."""

    if not BACKEND_AVAILABLE:
        return "OFFLINE"

    return "ONLINE"


def status_badge(status: str) -> None:
    """Render a status value as a native Streamlit badge/message."""

    normalized = (status or "").upper()

    if normalized in {"SUCCESS", "ONLINE", "PASS", "VALID"}:
        st.success(normalized)

    elif normalized in {"FAILED", "TIMED_OUT", "INTERRUPTED", "OFFLINE", "FAIL"}:
        st.error(normalized)

    elif normalized in {"RUNNING"}:
        st.info(normalized)

    else:
        st.warning(normalized or "NOT RUN")


def phase3_badge(label: str, kind: str) -> None:
    """
    Render a colored status line for Phase 3 surfaces.

    kind: "good" | "warn" | "bad" | "info". Kept separate from
    status_badge() above because Phase 3 states (ALIVE/STOPPED,
    OK/ISSUE, ACTIVE/CLOSED) don't map onto the Phase 2 diagnostic
    vocabulary (SUCCESS/FAILED/RUNNING/NOT_RUN).
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


def _fmt_pct(value: Any) -> str:
    return f"{value * 100:.1f}%" if value is not None else "—"


def _fmt_r(value: Any) -> str:
    return f"{value:+.2f}R" if value is not None else "—"


_TIMESTAMP_CANDIDATES = (
    "timestamp",
    "date",
    "datetime",
    "time",
)


def load_ohlc_csv(path: str) -> "pd.DataFrame":
    """
    Load a dataset CSV and return it indexed by a DatetimeIndex.

    strategy/backtest.py requires the dataframe's index to be a
    DatetimeIndex (raises "Backtest dataframe must be indexed by
    timestamp (DatetimeIndex)." otherwise). CSVs store the timestamp
    as a plain column, so it must be parsed and set as the index
    here rather than left as a default RangeIndex.

    Raises ValueError if no recognizable timestamp column is found,
    rather than silently fabricating one.
    """

    import pandas as pd

    df = pd.read_csv(path)

    lower_cols = {
        str(column).lower(): column
        for column in df.columns
    }

    date_col = None

    for candidate in _TIMESTAMP_CANDIDATES:

        if candidate in lower_cols:
            date_col = lower_cols[candidate]
            break

    if date_col is None:

        raise ValueError(
            "No timestamp/date/datetime/time column found in the "
            "dataset — cannot build the DatetimeIndex the backtest "
            "engine requires."
        )

    df[date_col] = pd.to_datetime(
        df[date_col],
        errors="coerce",
    )

    if df[date_col].isna().any():

        raise ValueError(
            f"Column '{date_col}' contains values that could not be "
            "parsed as timestamps."
        )

    df = df.set_index(date_col).sort_index()

    df.index.name = "timestamp"

    return df


def _get(
    obj: Any,
    *names: str,
    default: Any = "Not available",
) -> Any:
    """
    Read a real field from a result object.

    No fallback value is calculated or invented.
    """

    for name in names:

        if isinstance(
            obj,
            dict,
        ) and name in obj:
            return obj[name]

        if hasattr(
            obj,
            name,
        ):
            return getattr(
                obj,
                name,
            )

    return default


# ============================================================================
# SYSTEM ERROR PANEL
# ============================================================================

def render_system_error(
    component: str,
    message: str,
) -> None:
    """Render a controlled application error panel."""

    with st.container(
        border=True
    ):

        st.subheader(
            "System Error"
        )

        st.markdown(
            f"**Component:** {component}"
        )

        st.error(
            "FAILED"
        )

        st.write(
            f"Message: {message}"
        )

        with st.expander(
            "Technical details"
        ):
            st.code(
                message,
                language="text",
            )


# ============================================================================
# HEADER
# ============================================================================

_SECTION_SUBTITLES = {
    "OVERVIEW": "SYSTEM OVERVIEW",
    "PHASE01": "PHASE 01 · DATA ENGINE",
    "PHASE02": "PHASE 02 · ROBUSTNESS LAB",
    "PHASE03": "PHASE 03 · PAPER TRADING ENGINE",
    "CHART_ASSISTANT": "CHART ASSISTANT",
    "AI_ANALYST": "AI RESEARCH ANALYST",
    "REPORTS": "REPORT CENTER",
}


dataset_name = None

if st.session_state.uploaded_dataset_path:

    dataset_name = Path(
        st.session_state.uploaded_dataset_path
    ).name


header_left, header_right = st.columns([3, 1])

with header_left:

    st.title(
        "Machet Mechanics — Research Terminal"
    )

    st.caption(
        _SECTION_SUBTITLES.get(
            st.session_state.mm_section,
            "",
        )
    )

with header_right:

    status_badge(system_status_value())

    if dataset_name:
        st.caption(f"Dataset: {dataset_name}")

    st.caption(datetime.now().strftime("%d %b %Y %H:%M"))


if not BACKEND_AVAILABLE:

    render_system_error(
        "Phase 2 Backend",
        BACKEND_ERROR
        or "Unknown error",
    )


# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================

with st.sidebar:

    st.markdown(
        "**MACHET MECHANICS**"
    )

    st.caption(
        "RESEARCH TERMINAL"
    )

    st.divider()

    top_sections = [
        ("OVERVIEW", "OVERVIEW"),
        ("PHASE01", "PHASE 01"),
        ("PHASE02", "PHASE 02"),
        ("PHASE03", "PHASE 03"),
        (
            "CHART_ASSISTANT",
            "CHART ASSISTANT",
        ),
        (
            "AI_ANALYST",
            "AI RESEARCH ANALYST",
        ),
        (
            "REPORTS",
            "REPORTS",
        ),
    ]

    for section_key, section_label in top_sections:

        is_active = (
            st.session_state.mm_section
            == section_key
        )

        if st.button(
            section_label,
            key=f"nav_{section_key}",
            width="stretch",
            type=(
                "primary"
                if is_active
                else "secondary"
            ),
        ):

            st.session_state.mm_section = (
                section_key
            )

            st.rerun()

        # ------------------------------------------------------------
        # Phase 1 sub-navigation
        # ------------------------------------------------------------

        if (
            section_key == "PHASE01"
            and is_active
        ):

            for sub in [
                "DATA UPLOAD",
                "STRATEGY / BACKTEST",
            ]:

                if st.button(
                    f"   {sub}",
                    key=f"nav_p1_{sub}",
                    width="stretch",
                    type=(
                        "primary"
                        if (
                            st.session_state
                            .mm_phase01_page
                            == sub
                        )
                        else "secondary"
                    ),
                ):

                    st.session_state.mm_phase01_page = (
                        sub
                    )

                    st.rerun()

        # ------------------------------------------------------------
        # Phase 2 sub-navigation
        # ------------------------------------------------------------

        if (
            section_key == "PHASE02"
            and is_active
        ):

            for sub in PHASE02_GROUPS:

                if st.button(
                    f"   {sub}",
                    key=f"nav_p2_{sub}",
                    width="stretch",
                    type=(
                        "primary"
                        if (
                            st.session_state
                            .mm_phase02_page
                            == sub
                        )
                        else "secondary"
                    ),
                ):

                    st.session_state.mm_phase02_page = (
                        sub
                    )

                    st.rerun()

        # ------------------------------------------------------------
        # Phase 3 sub-navigation
        # ------------------------------------------------------------

        if (
            section_key == "PHASE03"
            and is_active
        ):

            for sub in PHASE03_SUBPAGES:

                if st.button(
                    f"   {sub}",
                    key=f"nav_p3_{sub}",
                    width="stretch",
                    type=(
                        "primary"
                        if (
                            st.session_state
                            .mm_phase03_page
                            == sub
                        )
                        else "secondary"
                    ),
                ):

                    st.session_state.mm_phase03_page = (
                        sub
                    )

                    st.rerun()

    st.divider()

    st.caption(
        "ROADMAP"
    )

    st.button(
        "PHASE 04 🔒 LIVE TRADING",
        disabled=True,
        width="stretch",
    )

    st.divider()

    st.caption(
        "SYSTEM"
    )

    status_badge(system_status_value())

    if not PHASE3_BACKEND_AVAILABLE:
        st.warning(
            "Phase 3 engine unavailable."
        )

    if not AI_ASSISTANT_AVAILABLE:
        st.warning(
            "AI Research Analyst unavailable."
        )


# ============================================================================
# OVERVIEW
# ============================================================================

def render_overview() -> None:
    """Render the Phase 2 research terminal overview."""

    total = len(
        DIAGNOSTIC_META
    )

    passed = 0
    failed = 0
    pending = 0

    for key in DIAGNOSTIC_META:

        status = get_status_value(
            key
        )

        if status == "SUCCESS":
            passed += 1

        elif status in {
            "FAILED",
            "TIMED_OUT",
            "INTERRUPTED",
        }:
            failed += 1

        else:
            pending += 1

    completion_pct = (
        f"{passed / total * 100:.0f}%"
        if total
        else "0%"
    )

    st.subheader(
        "System Overview"
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Diagnostic Engines",
        total,
    )

    c2.metric(
        "Passed",
        passed,
    )

    c3.metric(
        "Needs Review",
        failed,
    )

    c4.metric(
        "Complete",
        completion_pct,
    )

    st.write("")

    st.subheader(
        "Research Engine Status"
    )

    for key, meta in DIAGNOSTIC_META.items():

        latest = (
            service.get_latest(key)
            if BACKEND_AVAILABLE
            else None
        )

        duration = (
            _safe_elapsed(
                getattr(
                    latest,
                    "duration_seconds",
                    None,
                )
            )
            if latest
            else "—"
        )

        row_a, row_b, row_c = st.columns([3, 1, 1])

        with row_a:
            st.write(meta["title"])

        with row_b:
            status_badge(get_status_value(key))

        with row_c:
            st.caption(duration)

    st.write("")

    st.subheader(
        "Research Baseline"
    )

    st.caption(
        "Baseline configuration is displayed only when sourced "
        "from the actual strategy/backtest configuration. "
        "No placeholder values are shown here."
    )

    baseline_cols = st.columns(4)

    baseline_items = [
        ("RETEST WINDOW", "Not available"),
        ("REWARD MULTIPLE", "Not available"),
        ("STOP BUFFER", "Not available"),
        ("ENTRY MODE", "Not available"),
    ]

    for col, (
        label,
        value,
    ) in zip(
        baseline_cols,
        baseline_items,
    ):

        with col:
            st.metric(
                label,
                value,
            )

    st.caption(
        "Phase 2 is an analytical validation layer. "
        "Diagnostic results are research evidence, "
        "not guarantees of future performance."
    )


# ============================================================================
# PHASE 01 — DATA UPLOAD
# ============================================================================

def render_phase01_upload() -> None:
    """Render the independent Phase 1 dataset ingestion page."""

    st.subheader(
        "Dataset Ingestion"
    )

    existing_csvs = _discover_project_csvs()

    if existing_csvs:

        with st.container(border=True):

            st.markdown("**Load an existing dataset**")
            st.caption(
                "These CSVs are already on disk — no upload needed."
            )

            labels = [path.name for path in existing_csvs]

            current_path = st.session_state.uploaded_dataset_path
            default_index = 0
            if current_path:
                for i, path in enumerate(existing_csvs):
                    if str(path.resolve()) == str(Path(current_path).resolve()):
                        default_index = i
                        break

            picked_label = st.selectbox(
                "Dataset already in the project",
                labels,
                index=default_index,
                key="p1_existing_dataset_pick",
            )

            if st.button("Load Selected Dataset", type="primary", key="p1_load_existing"):
                picked_path = next(
                    p for p in existing_csvs if p.name == picked_label
                )
                st.session_state.uploaded_dataset_path = str(picked_path)
                _save_last_dataset_path(str(picked_path))
                st.rerun()

        st.divider()
        st.markdown("**Or upload a new dataset**")

    uploaded = st.file_uploader(
        "Drop CSV file here or browse",
        type=["csv"],
    )

    if uploaded is not None:

        try:

            import pandas as pd

            df = pd.read_csv(
                uploaded
            )

            upload_dir = (
                PROJECT_ROOT
                / "phase2_dashboard"
                / "uploaded_datasets"
            )

            upload_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            save_path = (
                upload_dir
                / uploaded.name
            )

            df.to_csv(
                save_path,
                index=False,
            )

            st.session_state.uploaded_dataset_path = (
                str(save_path)
            )

            _save_last_dataset_path(str(save_path))

            date_col = None

            for candidate in (
                "timestamp",
                "date",
                "datetime",
                "time",
            ):

                if candidate in {
                    str(column).lower()
                    for column in df.columns
                }:

                    date_col = next(
                        (
                            column
                            for column in df.columns
                            if str(column).lower()
                            == candidate
                        ),
                        None,
                    )

                    break

            date_range = (
                "Not available"
            )

            if date_col is not None:

                try:

                    parsed = pd.to_datetime(
                        df[date_col],
                        errors="coerce",
                    ).dropna()

                    if not parsed.empty:

                        date_range = (
                            f"{parsed.min()} "
                            f"→ "
                            f"{parsed.max()}"
                        )

                except Exception:
                    pass

            lower_cols = {
                str(column).lower()
                for column in df.columns
            }

            required_ohlcv = {
                "open",
                "high",
                "low",
                "close",
            }

            is_valid = (
                required_ohlcv
                .issubset(lower_cols)
            )

            st.subheader(
                "Dataset Status"
            )

            with st.container(
                border=True
            ):

                st.write(
                    f"**File**  {uploaded.name}"
                )

                st.write(
                    f"**Candles**  {len(df):,}"
                )

                st.write(
                    f"**Date range**  {date_range}"
                )

                st.write(
                    "**Columns**  "
                    + ", ".join(
                        str(column)
                        for column in df.columns
                    )
                )

                if is_valid:

                    st.success(
                        "VALID"
                    )

                else:

                    st.error(
                        "Missing required OHLC columns"
                    )

        except Exception as exc:

            render_system_error(
                "CSV Ingestion",
                str(exc),
            )

    elif st.session_state.uploaded_dataset_path:

        st.caption(
            "Currently loaded: "
            f"{Path(st.session_state.uploaded_dataset_path).name}"
        )

    else:

        st.info(
            "No dataset loaded. "
            "Upload a CSV to begin."
        )


# ============================================================================
# PHASE 01 — STRATEGY / BACKTEST
# ============================================================================

def render_phase01_backtest() -> None:
    """Render the independent Phase 1 strategy/backtest page."""

    if not st.session_state.uploaded_dataset_path:

        st.info(
            "No dataset loaded. "
            "Go to Phase 01 → Data Upload first."
        )

        return

    st.subheader(
        "Strategy Configuration"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        entry_mode = st.selectbox(
            "Entry mode",
            [
                "midpoint",
                "extreme",
            ],
        )

    with col2:

        reward_multiple = st.number_input(
            "Reward multiple",
            value=2.0,
            step=0.1,
        )

    with col3:

        max_bars_to_retest = st.number_input(
            "Max bars to retest",
            value=20,
            step=1,
        )

    with col4:

        stop_buffer = st.number_input(
            "Stop buffer",
            value=0.0,
            step=0.01,
            format="%.2f",
        )

    run_clicked = st.button(
        "Run Backtest",
        type="primary",
    )

    if run_clicked:

        try:

            from strategy.backtest import (
                run_backtest,
            )

            df = load_ohlc_csv(
                st.session_state.uploaded_dataset_path
            )

            with st.spinner(
                "Running backtest..."
            ):

                result = run_backtest(
                    df,
                    max_bars_to_retest=int(
                        max_bars_to_retest
                    ),
                    reward_multiple=float(
                        reward_multiple
                    ),
                    stop_buffer=float(
                        stop_buffer
                    ),
                    entry_mode=entry_mode,
                )

            st.session_state.backtest_result = (
                result
            )

        except Exception as exc:

            render_system_error(
                "strategy.backtest.run_backtest",
                str(exc),
            )

            st.session_state.backtest_result = (
                None
            )

    result = (
        st.session_state.backtest_result
    )

    if result is None:

        st.info(
            "No completed run available. "
            "Configure and run the backtest above."
        )

        return

    st.subheader(
        "Results"
    )

    fields = [
        (
            "SIGNALS GENERATED",
            _get(
                result,
                "total_signals_generated",
            ),
        ),
        (
            "TRADES TRIGGERED",
            _get(
                result,
                "total_trades_triggered",
            ),
        ),
        (
            "INVALIDATED",
            _get(
                result,
                "total_invalidated",
            ),
        ),
        (
            "EXPIRED",
            _get(
                result,
                "total_expired",
            ),
        ),
        (
            "WIN RATE",
            _get(
                result,
                "win_rate",
            ),
        ),
        (
            "EXPECTANCY",
            _get(
                result,
                "expectancy",
            ),
        ),
    ]

    trades_list = _get(result, "trades", default=None)

    errors_list = _get(result, "errors", default=None)

    cols = st.columns(3)

    for index, (
        label,
        value,
    ) in enumerate(fields):

        with cols[index % 3]:

            display_value = value

            if label == "WIN RATE" and isinstance(value, (int, float)):
                display_value = f"{value:.1f}%"

            st.metric(
                label,
                str(display_value),
            )

    if isinstance(trades_list, list):
        st.caption(f"{len(trades_list)} trade record(s) returned.")

    if isinstance(errors_list, list) and errors_list:
        st.warning(f"{len(errors_list)} error(s) reported by the backtest engine.")
        with st.expander("ERRORS"):
            for err in errors_list:
                st.text(str(err))

    with st.expander(
        "RAW RESULT OBJECT"
    ):

        st.code(
            str(result),
            language="text",
        )


# ============================================================================
# PHASE 02 — DIAGNOSTIC GROUP
# ============================================================================

def render_diagnostic_card(
    key: str,
    title: str,
    description: str,
    status: str,
    run_label: str,
    category: str,
    subtitle: str | None,
) -> bool:
    """Render one diagnostic card using only native Streamlit widgets.

    Returns True if the run button was clicked this run.
    """

    clicked = False

    with st.container(border=True):

        st.caption(category)
        st.subheader(title)
        st.write(description)

        if subtitle:
            st.caption(subtitle)

        status_badge(status)

        clicked = st.button(
            run_label,
            key=f"run_{key}",
            width="stretch",
        )

    return clicked


def render_phase02_group(
    group_name: str,
) -> None:
    """Render one Phase 2 diagnostic group."""

    keys = PHASE02_GROUPS[
        group_name
    ]

    st.subheader(
        group_name
    )

    columns = st.columns(
        min(
            len(keys),
            3,
        )
        or 1
    )

    for index, key in enumerate(keys):

        meta = DIAGNOSTIC_META[
            key
        ]

        column = columns[
            index % len(columns)
        ]

        with column:

            if not diagnostic_is_available(
                key
            ):

                with st.container(
                    border=True
                ):

                    st.caption(
                        f"TEST {meta['number']} "
                        f"· {meta['group']}"
                    )

                    st.subheader(
                        meta["title"]
                    )

                    st.write(
                        meta["description"]
                    )

                    st.warning(
                        "Backend registration unavailable."
                    )

                continue

            status = get_status_value(
                key
            )

            latest = (
                service.get_latest(key)
                if BACKEND_AVAILABLE
                else None
            )

            subtitle = None

            if latest is not None:

                started_at = getattr(
                    latest,
                    "started_at_display",
                    None,
                )

                if not started_at:
                    started_at = getattr(
                        latest,
                        "started_at",
                        "Unknown time",
                    )

                subtitle = (
                    f"Last run: {started_at} "
                    f"· Duration: "
                    f"{_safe_elapsed(getattr(latest, 'duration_seconds', None))}"
                )

            clicked = render_diagnostic_card(
                key=key,
                title=meta["title"],
                description=meta["description"],
                status=status,
                run_label=f"Run {meta['title']}",
                category=(
                    f"TEST {meta['number']} "
                    f"· {meta['group']}"
                ),
                subtitle=subtitle,
            )

            if (
                clicked
                and not st.session_state.running
            ):

                run_one(
                    key
                )

                st.rerun()

            if latest is not None:

                with st.expander(
                    "VIEW RESULTS"
                ):

                    output = getattr(
                        latest,
                        "combined_output",
                        None,
                    )

                    if output is None:
                        output = getattr(
                            latest,
                            "output",
                            "",
                        )

                    st.code(
                        output or "(no output)",
                        language="text",
                    )

                with st.expander(
                    "HISTORY"
                ):

                    history = (
                        service
                        .get_history(key)
                        .get(
                            key,
                            [],
                        )
                    )

                    for record in reversed(
                        history
                    ):

                        st.text(
                            str(record)
                        )


# ============================================================================
# PHASE 03 — PAPER TRADING
# ============================================================================

def _phase3_service_or_none():
    """Return the cached Phase3Service, or None if the backend failed to load."""
    if not PHASE3_BACKEND_AVAILABLE:
        return None
    try:
        return get_phase3_service()
    except Exception:
        return None


def render_phase03_controls() -> None:
    """Replay controls shared at the top of the Phase 3 workspace."""

    with st.container(border=True):

        st.markdown("**Execution Controls**")

        ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([2, 2, 3])

        with ctrl_col1:
            run_replay = st.button(
                "▶ Run Historical Replay",
                type="primary",
                width="stretch",
                key="p3_run_replay",
            )

        with ctrl_col2:
            refresh_health = st.button(
                "↻ Refresh System Health",
                width="stretch",
                key="p3_refresh_health",
            )

        with ctrl_col3:
            dataset = st.selectbox(
                "Select Backtest Dataset",
                PHASE03_DATASETS,
                label_visibility="collapsed",
                key="p3_dataset_select",
            )

        auto_approve = st.checkbox(
            "Auto-approve all sessions for this replay run",
            value=True,
            key="p3_auto_approve",
            help=(
                "When enabled, Phase3Service pre-approves session "
                "occurrences via SessionEngine's public API. "
                "When disabled, normal decision rules apply."
            ),
        )

    if run_replay:

        service3 = _phase3_service_or_none()

        if service3 is None:
            render_system_error(
                "Phase 3 Backend",
                PHASE3_BACKEND_ERROR or "Phase3Service unavailable.",
            )
            return

        with st.spinner("Executing replay through live Phase 3 pipeline..."):

            try:

                csv_path = str(PROJECT_ROOT / dataset)

                summary = service3.run_replay(
                    csv_path,
                    auto_approve_sessions=auto_approve,
                )

                st.session_state.phase3_last_dataset = csv_path

                st.success(
                    f"Replay complete — {summary.ticks_run} ticks processed, "
                    f"{summary.opened_trades} trades opened, "
                    f"{summary.closed_trades} trades closed."
                )

            except Exception as exc:

                st.error(f"Replay execution failed: {exc}")

    if refresh_health:
        st.session_state["health_refreshed_at"] = datetime.now(timezone.utc)


def render_phase03_overview() -> None:
    """Phase 3 overview sub-page: KPI strip, session banner, health, pipeline."""

    render_phase03_controls()

    service3 = _phase3_service_or_none()

    if service3 is None:
        render_system_error(
            "Phase 3 Backend",
            PHASE3_BACKEND_ERROR or "Phase3Service unavailable.",
        )
        return

    st.write("")

    perf_preview = None
    open_positions_preview = []

    try:
        perf_preview = service3.get_performance()
    except Exception:
        pass

    try:
        open_positions_preview = service3.get_open_positions()
    except Exception:
        pass

    with st.container(border=True):

        kpi_cols = st.columns(4)

        total_r_value = perf_preview.total_r if perf_preview else None
        kpi_cols[0].metric("TOTAL R", _fmt_r(total_r_value))

        win_rate_value = perf_preview.win_rate if perf_preview else None
        kpi_cols[1].metric("WIN RATE", _fmt_pct(win_rate_value))

        kpi_cols[2].metric("OPEN POSITIONS", len(open_positions_preview))

        with kpi_cols[3]:
            st.caption("SYSTEM STATUS")
            phase3_badge("ONLINE" if PHASE3_BACKEND_AVAILABLE else "OFFLINE",
                         "good" if PHASE3_BACKEND_AVAILABLE else "bad")

    st.write("")

    try:
        sessions = service3.get_sessions()
    except Exception:
        sessions = []

    if sessions:

        now_utc = datetime.now(timezone.utc)
        active = [s for s in sessions if s.status.value == "ACTIVE"]
        upcoming = [s for s in sessions if s.scheduled_start > now_utc]
        upcoming.sort(key=lambda s: s.scheduled_start)

        with st.container(border=True):

            session_cols = st.columns(3)

            with session_cols[0]:
                st.caption("CURRENT SESSION")
                if active:
                    st.markdown(f"**{active[0].session_name}**")
                    phase3_badge("ACTIVE", "good")
                else:
                    st.markdown("**No active session**")
                    phase3_badge("CLOSED", "info")

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

        st.write("")

    st.subheader("System Health & Telemetry")

    try:
        health = service3.get_system_health()
    except Exception as exc:
        st.error(f"Failed to fetch system health: {exc}")
        health = None

    if health is not None:

        with st.container(border=True):

            status_cols = st.columns(5)

            with status_cols[0]:
                st.caption("RUNTIME")
                phase3_badge(
                    "ALIVE" if health.runtime_alive else "STOPPED",
                    "good" if health.runtime_alive else "warn",
                )

            status_cols[1].metric("Candles Processed", health.candle_count)
            status_cols[2].metric("Adapter Errors", health.adapter_error_count)
            status_cols[3].metric("Pending Signals", health.pending_signal_count)

            with status_cols[4]:
                st.caption("PERSISTENCE")
                if health.persistence_ok is None:
                    phase3_badge("N/A", "info")
                elif health.persistence_ok:
                    phase3_badge("OK", "good")
                else:
                    phase3_badge("ISSUE", "bad")

        if health.issues:
            with st.container(border=True):
                st.markdown(":orange[**Active Health Warnings**]")
                for issue in health.issues:
                    st.caption(f"• {issue}")

    st.write("")

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

    summary = service3.get_last_summary()

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


def render_phase03_positions() -> None:
    """Phase 3 open-positions sub-page."""

    import pandas as pd

    st.subheader("Active Positions Monitor")

    service3 = _phase3_service_or_none()

    if service3 is None:
        render_system_error(
            "Phase 3 Backend",
            PHASE3_BACKEND_ERROR or "Phase3Service unavailable.",
        )
        return

    try:
        positions = service3.get_open_positions()
    except Exception as exc:
        st.error(f"Error fetching open positions: {exc}")
        positions = []

    if not positions:
        st.caption("No open positions currently active.")
        return

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


def render_phase03_trades() -> None:
    """Phase 3 closed-trades / audit journal sub-page."""

    import pandas as pd

    st.subheader("Trade Audit Journal")

    service3 = _phase3_service_or_none()

    if service3 is None:
        render_system_error(
            "Phase 3 Backend",
            PHASE3_BACKEND_ERROR or "Phase3Service unavailable.",
        )
        return

    try:
        trades = service3.get_closed_trades()
    except Exception as exc:
        st.error(f"Error fetching closed trades: {exc}")
        trades = []

    if not trades:
        st.caption("No closed trade records logged yet.")
        return

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


def render_phase03_performance() -> None:
    """Phase 3 performance analytics sub-page."""

    import pandas as pd

    st.subheader("Performance Analytics")

    service3 = _phase3_service_or_none()

    if service3 is None:
        render_system_error(
            "Phase 3 Backend",
            PHASE3_BACKEND_ERROR or "Phase3Service unavailable.",
        )
        return

    try:
        perf = service3.get_performance()
    except Exception as exc:
        st.error(f"Error fetching performance metrics: {exc}")
        perf = None

    if perf is None or perf.total_trades == 0:
        st.caption("No performance analytics available. Complete a replay cycle first.")
        return

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

    try:
        closed = service3.get_closed_trades()
    except Exception:
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


def _build_phase3_chart_figure(service3, dataset_path: str):
    """
    Build the Phase 3 candlestick chart (with real trade markers) from
    the last replay's dataset.

    Shared by render_phase03_chart() (what the person sees) and
    render_phase03_ai_assistant()'s vision mode (what the AI sees) so
    the two can never drift apart -- same figure, same data, always.

    Returns (fig, df) on success. Returns (None, df) if plotly isn't
    installed (df still loaded so a fallback line chart can be drawn).
    Returns (None, None) if the dataset itself can't be loaded or is
    missing required columns -- callers should treat that as "nothing
    to show" and say why via the returned error string.
    """

    import pandas as pd

    try:
        df = pd.read_csv(dataset_path)
    except Exception as exc:
        return None, None, f"Chart Data Load error: {exc}"

    lower_cols = {str(column).lower(): column for column in df.columns}
    required = {"open", "high", "low", "close"}

    if not required.issubset(lower_cols.keys()):
        return None, None, "Dataset is missing OHLC columns required for a candlestick chart."

    x_col = None
    for candidate in ("timestamp", "date", "datetime", "time"):
        if candidate in lower_cols:
            x_col = lower_cols[candidate]
            break

    x_values = df[x_col] if x_col else df.index

    try:
        import plotly.graph_objects as go
    except ImportError:
        return None, df, None

    fig = go.Figure(
        data=[
            go.Candlestick(
                x=x_values,
                open=df[lower_cols["open"]],
                high=df[lower_cols["high"]],
                low=df[lower_cols["low"]],
                close=df[lower_cols["close"]],
                name="Price",
            )
        ]
    )

    try:
        closed = service3.get_closed_trades()
    except Exception:
        closed = []

    try:
        open_positions = service3.get_open_positions()
    except Exception:
        open_positions = []

    entry_x, entry_y, entry_color = [], [], []
    exit_x, exit_y, exit_color = [], [], []

    for entry in closed:
        if entry.result_r is not None:
            exit_color.append("#00C2A8" if entry.result_r >= 0 else "#E05C5C")
        else:
            exit_color.append("#888888")
        exit_x.append(entry.recorded_at)
        exit_y.append(entry.exit_price)

    for view in open_positions:
        trade = view.trade
        entry_x.append(trade.opened_at)
        entry_y.append(trade.entry)
        entry_color.append("#3AA0FF" if trade.direction.value == "LONG" else "#F2A93B")

    if entry_x:
        fig.add_trace(
            go.Scatter(
                x=entry_x,
                y=entry_y,
                mode="markers",
                marker=dict(symbol="triangle-up", size=10, color=entry_color),
                name="Trade Entry (open)",
            )
        )

    if exit_x:
        fig.add_trace(
            go.Scatter(
                x=exit_x,
                y=exit_y,
                mode="markers",
                marker=dict(symbol="x", size=9, color=exit_color),
                name="Trade Exit (closed)",
            )
        )

    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        height=560,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h"),
    )

    return fig, df, None


def render_phase03_chart() -> None:
    """
    Phase 3 market-chart visualizer sub-page.

    Plots the candles from the last replay's dataset with trade
    entry/exit markers overlaid from the real Journal/Position data.
    This is a separate sub-page from AI Report — no chat, no
    narrative text here, just the chart.
    """

    st.subheader("Market Chart — Last Replay")

    service3 = _phase3_service_or_none()

    if service3 is None:
        render_system_error(
            "Phase 3 Backend",
            PHASE3_BACKEND_ERROR or "Phase3Service unavailable.",
        )
        return

    dataset_path = st.session_state.get("phase3_last_dataset")

    if not dataset_path:
        st.info(
            "No replay has been run yet. Go to Overview and run a "
            "historical replay first, then return here to view the chart."
        )
        return

    fig, df, error = _build_phase3_chart_figure(service3, dataset_path)

    if error is not None:
        st.warning(error)
        return

    if fig is None:
        st.info("Plotly is unavailable. Showing the close-price line instead.")
        lower_cols = {str(column).lower(): column for column in df.columns}
        st.line_chart(df[lower_cols["close"]])
    else:
        st.plotly_chart(fig, width="stretch")

    st.caption(
        f"Dataset: {Path(dataset_path).name} · {len(df):,} candles. "
        "Markers are drawn only from real Journal/PositionManager data — "
        "nothing here is estimated."
    )


def render_phase03_ai_assistant() -> None:
    """
    Phase 3 AI Assistant sub-page.

    Reuses the same local Ollama model as Phase 2's Chart Assistant
    (phase2_dashboard.claude_helper.chat_with_local_ai), but supplies
    it with real Phase 3 engine facts as context instead of raw
    uploaded-CSV stats: the last replay summary, current performance
    snapshot, recent closed trades, and current session.

    The assistant must not invent signals, trades, performance, or
    strategy events -- so every value handed to it below comes
    directly from Phase3Service, never estimated here.
    """

    st.subheader("AI Assistant")

    service3 = _phase3_service_or_none()

    if service3 is None:
        render_system_error(
            "Phase 3 Backend",
            PHASE3_BACKEND_ERROR or "Phase3Service unavailable.",
        )
        return

    dataset_path = st.session_state.get("phase3_last_dataset")

    try:
        config = service3.get_config()
        symbol = config.market_data.symbol
        timeframe = config.market_data.timeframe
    except Exception:
        symbol = None
        timeframe = None

    dataset_label = Path(dataset_path).name if dataset_path else "No replay run yet"

    phase1_status = (
        "Phase 1 backtest: available"
        if st.session_state.get("backtest_result") is not None
        else "Phase 1 backtest: none run yet"
    )

    st.caption(f"Context: {symbol or '—'} / {timeframe or '—'} / {dataset_label}")
    st.caption(phase1_status)

    if "phase3_chat_history" not in st.session_state:
        st.session_state["phase3_chat_history"] = []

    for turn in st.session_state["phase3_chat_history"]:
        with st.container(border=True):
            st.caption(turn["role"].upper())
            st.write(turn["content"])

    use_vision = st.checkbox(
        "🖼️ Include chart image (vision)",
        value=False,
        key="p3_ai_use_vision",
        help=(
            "Sends a picture of the current market chart to a "
            "vision-capable local model (separate from the text "
            "model) alongside the same real trade data. Requires "
            "a vision model pulled via `ollama pull moondream` "
            "(or similar) and the `kaleido` package -- if either "
            "is missing, the assistant will say so instead of "
            "failing silently."
        ),
    )

    prompt = st.chat_input("Ask about the displayed data...", key="p3_ai_chat_input")

    if prompt:

        try:

            from phase2_dashboard.claude_helper import chat_with_local_ai

            last_summary = service3.get_last_summary()
            perf = None
            try:
                perf = service3.get_performance()
            except Exception:
                pass

            try:
                closed = service3.get_closed_trades()
            except Exception:
                closed = []

            try:
                current_session = service3.get_current_session()
            except Exception:
                current_session = None

            phase1_result = st.session_state.get("backtest_result")
            phase1_dataset_path = st.session_state.get("uploaded_dataset_path")

            phase1_backtest = None
            if phase1_result is not None:
                phase1_backtest = {
                    "dataset": (
                        Path(phase1_dataset_path).name
                        if phase1_dataset_path
                        else None
                    ),
                    "signals_generated": _get(
                        phase1_result, "total_signals_generated"
                    ),
                    "trades_triggered": _get(
                        phase1_result, "total_trades_triggered"
                    ),
                    "invalidated": _get(phase1_result, "total_invalidated"),
                    "expired": _get(phase1_result, "total_expired"),
                    "win_rate": _get(phase1_result, "win_rate"),
                    "expectancy": _get(phase1_result, "expectancy"),
                }

            phase3_context = {
                "symbol": symbol,
                "timeframe": timeframe,
                "dataset": dataset_label,
                "phase1_backtest": phase1_backtest,
                "replay_summary": (
                    {
                        "total_candles": last_summary.total_candles,
                        "ticks_run": last_summary.ticks_run,
                        "opened_trades": last_summary.opened_trades,
                        "closed_trades": last_summary.closed_trades,
                        "first_timestamp": str(last_summary.first_timestamp)
                        if last_summary.first_timestamp
                        else None,
                        "last_timestamp": str(last_summary.last_timestamp)
                        if last_summary.last_timestamp
                        else None,
                    }
                    if last_summary is not None
                    else None
                ),
                "performance": (
                    {
                        "total_trades": perf.total_trades,
                        "win_rate": perf.win_rate,
                        "total_r": perf.total_r,
                        "average_r": perf.average_r,
                        "current_streak": perf.current_streak,
                        "best_winning_streak": perf.best_winning_streak,
                        "worst_losing_streak": perf.worst_losing_streak,
                        "max_drawdown_r": perf.max_drawdown_r,
                    }
                    if perf is not None and perf.total_trades
                    else None
                ),
                "recent_closed_trades": [
                    {
                        "trade_id": entry.trade_id,
                        "direction": entry.direction,
                        "session": entry.session,
                        "result_r": entry.result_r,
                        "closed_at": str(entry.recorded_at)
                        if entry.recorded_at
                        else None,
                    }
                    for entry in closed[-10:]
                ],
                "current_session": (
                    current_session.session_name
                    if current_session is not None
                    else None
                ),
            }

            with st.spinner("Thinking..."):

                if use_vision:

                    from phase2_dashboard.claude_helper import (
                        chat_about_chart_image,
                    )

                    if not dataset_path:
                        reply = (
                            "No replay has been run yet, so there's no "
                            "chart to look at. Run a replay from "
                            "Overview first."
                        )
                    else:
                        fig, _df, chart_error = _build_phase3_chart_figure(
                            service3, dataset_path
                        )

                        if chart_error is not None:
                            reply = f"Couldn't build the chart image: {chart_error}"
                        elif fig is None:
                            reply = (
                                "Plotly isn't installed, so no chart "
                                "image can be generated. Install it "
                                "(`pip install plotly --break-system-packages`) "
                                "and try again."
                            )
                        else:
                            try:
                                image_bytes = fig.to_image(format="png")
                            except Exception as exc:
                                reply = (
                                    "Couldn't export the chart as an "
                                    "image -- the `kaleido` package is "
                                    "likely missing. Install it with "
                                    "`pip install kaleido "
                                    "--break-system-packages` and try "
                                    f"again. ({exc})"
                                )
                            else:
                                reply = chat_about_chart_image(
                                    diagnostic_title=(
                                        "Phase 3 Paper Trading — "
                                        "AI Assistant (chart image)"
                                    ),
                                    diagnostic_result=phase3_context,
                                    image_bytes=image_bytes,
                                    conversation=[
                                        {
                                            "role": "user",
                                            "content": prompt,
                                        }
                                    ],
                                )

                else:

                    reply = chat_with_local_ai(
                        diagnostic_title="Phase 3 Paper Trading — AI Assistant",
                        diagnostic_result=phase3_context,
                        conversation=[
                            {
                                "role": "user",
                                "content": prompt,
                            }
                        ],
                    )

            st.session_state["phase3_chat_history"].append(
                {
                    "role": "user",
                    "content": prompt + (" 🖼️" if use_vision else ""),
                }
            )
            st.session_state["phase3_chat_history"].append(
                {"role": "assistant", "content": reply}
            )

            st.rerun()

        except Exception as exc:

            render_system_error(
                "Phase 3 AI Assistant / local AI",
                str(exc),
            )


def render_phase03_report() -> None:
    """Phase 3 AI Report sub-page — deterministic narrative, no chat."""

    st.subheader("Deterministic Audit Report")
    st.caption(
        "Generated directly by ReportEngine — deterministic insights, "
        "no external LLM, no chat interface."
    )

    service3 = _phase3_service_or_none()

    if service3 is None:
        render_system_error(
            "Phase 3 Backend",
            PHASE3_BACKEND_ERROR or "Phase3Service unavailable.",
        )
        return

    try:
        report = service3.get_latest_report()
    except Exception as exc:
        st.error(f"Error fetching report narrative: {exc}")
        report = None

    if report is None:
        st.caption("No report snapshot found. Execute a replay session to generate output.")
        return

    with st.container(border=True):
        st.write(report.narrative)

    st.download_button(
        "Download Report as Text",
        report.narrative,
        file_name="phase3_report.txt",
        mime="text/plain",
    )


def render_phase03(subpage: str) -> None:
    """Route to the active Phase 3 sub-page."""

    if subpage == "OVERVIEW":
        render_phase03_overview()
    elif subpage == "POSITIONS":
        render_phase03_positions()
    elif subpage == "TRADES":
        render_phase03_trades()
    elif subpage == "PERFORMANCE":
        render_phase03_performance()
    elif subpage == "CHART":
        render_phase03_chart()
    elif subpage == "AI ASSISTANT":
        render_phase03_ai_assistant()
    elif subpage == "AI REPORT":
        render_phase03_report()
    else:
        render_phase03_overview()


# ============================================================================
# CHART ASSISTANT
# ============================================================================

def render_chart_assistant() -> None:
    """
    Render the independent Chart Assistant.

    This page currently provides:

    - dataset loading
    - OHLC validation
    - candlestick visualization
    - basic chart context
    - conversational assistant

    Strategy overlays are deliberately not fabricated.
    """

    st.subheader(
        "Market Chart"
    )

    if not st.session_state.uploaded_dataset_path:

        st.info(
            "No dataset loaded. Load a CSV under "
            "Phase 01 → Data Upload to view a chart."
        )

        return

    try:

        import pandas as pd

        df = pd.read_csv(
            st.session_state.uploaded_dataset_path
        )

        lower_cols = {
            str(column).lower(): column
            for column in df.columns
        }

        required = {
            "open",
            "high",
            "low",
            "close",
        }

        if not required.issubset(
            lower_cols.keys()
        ):

            st.warning(
                "Dataset is missing OHLC "
                "columns required for "
                "a candlestick chart."
            )

            return

        x_col = None

        for candidate in (
            "timestamp",
            "date",
            "datetime",
            "time",
        ):

            if candidate in lower_cols:

                x_col = lower_cols[
                    candidate
                ]

                break

        try:

            import plotly.graph_objects as go

            fig = go.Figure(
                data=[
                    go.Candlestick(
                        x=(
                            df[x_col]
                            if x_col
                            else df.index
                        ),
                        open=df[
                            lower_cols["open"]
                        ],
                        high=df[
                            lower_cols["high"]
                        ],
                        low=df[
                            lower_cols["low"]
                        ],
                        close=df[
                            lower_cols["close"]
                        ],
                    )
                ]
            )

            fig.update_layout(
                margin=dict(
                    l=10,
                    r=10,
                    t=10,
                    b=10,
                ),
                height=480,
                xaxis_rangeslider_visible=False,
            )

            st.plotly_chart(
                fig,
                width="stretch",
            )

        except ImportError:

            st.info(
                "Plotly is unavailable. "
                "Showing the close-price line instead."
            )

            st.line_chart(
                df[
                    lower_cols["close"]
                ]
            )

        st.caption(
            "Structure / Liquidity / Zones / Signals "
            "overlays are not yet wired. They will be connected "
            "only after the existing strategy module output "
            "contracts are confirmed."
        )

        st.subheader(
            "Chart Assistant"
        )

        prompt = st.chat_input(
            "Ask about this chart..."
        )

        if prompt:

            try:

                from phase2_dashboard.claude_helper import (
                    chat_with_local_ai,
                )

                close_series = df[
                    lower_cols["close"]
                ]

                high_series = df[
                    lower_cols["high"]
                ]

                low_series = df[
                    lower_cols["low"]
                ]

                chart_context = {
                    "candles": len(df),
                    "last_close": float(
                        close_series.iloc[-1]
                    ),
                    "period_high": float(
                        high_series.max()
                    ),
                    "period_low": float(
                        low_series.min()
                    ),
                }

                with st.spinner(
                    "Analyzing chart..."
                ):

                    reply = chat_with_local_ai(
                        diagnostic_title=(
                            "Chart Assistant — "
                            "loaded dataset"
                        ),
                        diagnostic_result=(
                            chart_context
                        ),
                        conversation=[
                            {
                                "role": "user",
                                "content": prompt,
                            }
                        ],
                    )

                st.markdown(
                    reply
                )

            except Exception as exc:

                render_system_error(
                    "Chart Assistant / local AI",
                    str(exc),
                )

    except Exception as exc:

        render_system_error(
            "Chart Assistant",
            str(exc),
        )


# ============================================================================
# REPORTS
# ============================================================================

def render_reports() -> None:
    """
    Render the current Phase 2 report view.

    This is currently a dashboard-level presentation of persisted
    diagnostic state. It does not replace the project's dedicated
    report-generation layer.
    """

    st.subheader(
        "Latest Report"
    )

    if not BACKEND_AVAILABLE:

        render_system_error(
            "Phase 2 Backend",
            BACKEND_ERROR
            or "Unknown error",
        )

        return

    lines = [
        "# MACHET MECHANICS — "
        "PHASE 2 RESEARCH REPORT",
        "",
        (
            "Generated: "
            f"{datetime.now().strftime('%d %b %Y %H:%M')}"
        ),
    ]

    if st.session_state.uploaded_dataset_path:

        lines.append(
            "Dataset: "
            f"{Path(st.session_state.uploaded_dataset_path).name}"
        )

    lines.extend(
        [
            "",
            "## Diagnostic Summary",
            "",
            "| Diagnostic | Status | Last Run | Duration |",
            "|---|---|---|---|",
        ]
    )

    for key, meta in DIAGNOSTIC_META.items():

        latest = service.get_latest(
            key
        )

        status = get_status_value(
            key
        )

        if latest is not None:

            last_run = getattr(
                latest,
                "started_at_display",
                None,
            )

            if not last_run:

                last_run = getattr(
                    latest,
                    "started_at",
                    "—",
                )

            duration = _safe_elapsed(
                getattr(
                    latest,
                    "duration_seconds",
                    None,
                )
            )

        else:

            last_run = "—"
            duration = "—"

        lines.append(
            f"| {meta['title']} | "
            f"{status} | "
            f"{last_run} | "
            f"{duration} |"
        )

    report_text = "\n".join(
        lines
    )

    with st.container(
        border=True
    ):

        st.markdown(
            report_text
        )

    st.download_button(
        "Export Report (.md)",
        data=report_text,
        file_name=(
            "machet_mechanics_report_"
            f"{datetime.now().strftime('%Y%m%d_%H%M')}.md"
        ),
        mime="text/markdown",
    )

    if st.button(
        "Refresh"
    ):

        st.rerun()


# ============================================================================
# HISTORY
# ============================================================================

def render_history_page() -> None:
    """Render persisted Phase 2 diagnostic history."""

    st.subheader(
        "Research History"
    )

    if not BACKEND_AVAILABLE:

        render_system_error(
            "Phase 2 Backend",
            BACKEND_ERROR
            or "Unknown error",
        )

        return

    filter_key = st.selectbox(
        "Diagnostic",
        [
            "ALL",
            *DIAGNOSTIC_META.keys(),
        ],
    )

    all_history = service.get_history(
        None
        if filter_key == "ALL"
        else filter_key
    )

    flat: list[Any] = []

    for records in all_history.values():
        flat.extend(
            records
        )

    def history_sort_key(
        record: Any,
    ) -> Any:

        return getattr(
            record,
            "started_at",
            datetime.min,
        )

    flat.sort(
        key=history_sort_key
    )

    if not flat:

        st.info(
            "No research history yet. "
            "Run a diagnostic to populate history."
        )

        return

    for record in reversed(
        flat[-50:]
    ):

        st.text(
            str(record)
        )

    if st.button(
        "Clear History"
    ):

        st.session_state.confirm_clear_history = (
            True
        )

    if st.session_state.confirm_clear_history:

        st.warning(
            "This permanently removes persisted "
            "history. This cannot be undone."
        )

        c1, c2 = st.columns(2)

        with c1:

            if st.button(
                "Confirm Clear",
                type="primary",
            ):

                service.clear_history(
                    None
                    if filter_key == "ALL"
                    else filter_key
                )

                st.session_state.confirm_clear_history = (
                    False
                )

                st.rerun()

        with c2:

            if st.button(
                "Cancel"
            ):

                st.session_state.confirm_clear_history = (
                    False
                )

                st.rerun()


# ============================================================================
# ROUTING
# ============================================================================

section = (
    st.session_state.mm_section
)


if section == "OVERVIEW":

    render_overview()


elif section == "PHASE01":

    if (
        st.session_state.mm_phase01_page
        == "DATA UPLOAD"
    ):

        render_phase01_upload()

    else:

        render_phase01_backtest()


elif section == "PHASE02":

    render_phase02_group(
        st.session_state.mm_phase02_page
    )


elif section == "PHASE03":

    render_phase03(
        st.session_state.mm_phase03_page
    )


elif section == "CHART_ASSISTANT":

    render_chart_assistant()


elif section == "AI_ANALYST":

    if AI_ASSISTANT_AVAILABLE:

        render_ai_assistant_page()

    else:

        render_system_error(
            "AI Research Analyst",
            AI_ASSISTANT_ERROR
            or "Unknown error",
        )


elif section == "REPORTS":

    render_reports()

    st.divider()

    render_history_page()


# ============================================================================
# FOOTER
# ============================================================================

st.divider()

st.caption(
    "MACHET MECHANICS · "
    "QUANTITATIVE RESEARCH TERMINAL"
)