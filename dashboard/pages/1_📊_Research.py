"""
dashboard/pages/1_📊_Research.py

Phase 1 Research Page
---------------------

Responsibilities:
    - Load and clean historical OHLC data
    - Validate chronological market data
    - Run the canonical Phase 1 backtest engine
    - Provide deterministic research metrics
    - Display historical backtest results

This module does NOT contain:
    - Strategy logic
    - Signal-generation logic
    - Optimization
    - Monte Carlo simulation
    - Walk-forward logic
    - Broker/live execution
    - AI interpretation

Design:
    - Native Streamlit components only
    - No custom HTML
    - No unsafe_allow_html
    - No custom CSS
    - Strategy remains inside strategy/
    - Dashboard is display/research UI only
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Phase 1 — Research Dashboard",
    page_icon="📊",
    layout="wide",
)


# ============================================================================
# PROJECT PATH
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from strategy.backtest import run_backtest  # noqa: E402


# ============================================================================
# CONSTANTS
# ============================================================================

TIMESTAMP_CANDIDATES = [
    "timestamp",
    "datetime",
    "date",
    "time",
]

OHLC_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
]


# ============================================================================
# SAFE VALUE HELPERS
# ============================================================================

def _safe_int(value: Any, default: int = 0) -> int:
    """Safely convert a value to int."""
    if value is None:
        return default

    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert a value to float."""
    if value is None:
        return default

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_enum_like(value: Any) -> str:
    """
    Normalize strings and Enum-like objects to uppercase text.
    """
    inner = getattr(value, "value", value)

    if inner is None:
        return ""

    return str(inner).strip().upper()


_normalize_status = _normalize_enum_like


# ============================================================================
# UPLOADED FILE IDENTITY
# ============================================================================

def _uploaded_file_signature(uploaded_file) -> str:
    """
    Return a deterministic identity for the uploaded file.

    Prevents the same dataset from being backtested again on ordinary
    Streamlit reruns.
    """
    return hashlib.sha256(
        uploaded_file.getvalue()
    ).hexdigest()


# ============================================================================
# TIMESTAMP RESOLUTION
# ============================================================================

def _resolve_timestamp(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, str]:
    """
    Resolve the timestamp source.

    Supported:
        - timestamp
        - datetime
        - date + time
        - date
        - time

    Returns:
        (dataframe, timestamp_column)
    """

    if "date" in df.columns and "time" in df.columns:
        df = df.copy()

        df["__ts__"] = (
            df["date"].astype(str)
            + " "
            + df["time"].astype(str)
        )

        df = df.drop(
            columns=["date", "time"]
        )

        return df, "__ts__"

    timestamp_column = next(
        (
            column
            for column in TIMESTAMP_CANDIDATES
            if column in df.columns
        ),
        None,
    )

    if timestamp_column is None:
        raise ValueError(
            "No timestamp-like column found. "
            f"Expected one of: {TIMESTAMP_CANDIDATES}"
        )

    return df, timestamp_column


# ============================================================================
# HISTORICAL OHLC LOADER
# ============================================================================

def load_ohlc_csv(
    path_or_buffer,
) -> tuple[pd.DataFrame, int]:
    """
    Load, clean, validate, and chronologically sort OHLC data.

    Returns:
        dataframe
        number of rows removed during cleaning
    """

    df = pd.read_csv(path_or_buffer)

    if df.empty:
        raise ValueError(
            "The CSV file contains no rows."
        )

    # ------------------------------------------------------------------------
    # Normalize column names
    # ------------------------------------------------------------------------

    df.columns = [
        str(column).strip().lower()
        for column in df.columns
    ]

    # ------------------------------------------------------------------------
    # Resolve timestamp
    # ------------------------------------------------------------------------

    df, timestamp_column = _resolve_timestamp(df)

    # ------------------------------------------------------------------------
    # Validate required OHLC columns
    # ------------------------------------------------------------------------

    missing = set(OHLC_COLUMNS) - set(df.columns)

    if missing:
        raise ValueError(
            "CSV missing required columns: "
            f"{sorted(missing)}"
        )

    dropped_rows = 0

    # ------------------------------------------------------------------------
    # Timestamp cleaning
    # ------------------------------------------------------------------------

    df[timestamp_column] = pd.to_datetime(
        df[timestamp_column],
        errors="coerce",
    )

    bad_timestamp_mask = (
        df[timestamp_column].isna()
    )

    bad_timestamp_count = int(
        bad_timestamp_mask.sum()
    )

    if bad_timestamp_count:
        df = df.loc[
            ~bad_timestamp_mask
        ].copy()

        dropped_rows += bad_timestamp_count

    # ------------------------------------------------------------------------
    # Numeric OHLC cleaning
    # ------------------------------------------------------------------------

    for column in OHLC_COLUMNS:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    bad_numeric_mask = (
        df[OHLC_COLUMNS]
        .isna()
        .any(axis=1)
    )

    bad_numeric_count = int(
        bad_numeric_mask.sum()
    )

    if bad_numeric_count:
        df = df.loc[
            ~bad_numeric_mask
        ].copy()

        dropped_rows += bad_numeric_count

    if df.empty:
        raise ValueError(
            "No usable rows remained after "
            "timestamp and OHLC cleaning."
        )

    # ------------------------------------------------------------------------
    # OHLC structural validation
    # ------------------------------------------------------------------------

    invalid_ohlc_mask = (
        (df["high"] < df["low"])
        | (df["high"] < df["open"])
        | (df["high"] < df["close"])
        | (df["low"] > df["open"])
        | (df["low"] > df["close"])
    )

    invalid_ohlc_count = int(
        invalid_ohlc_mask.sum()
    )

    if invalid_ohlc_count:
        df = df.loc[
            ~invalid_ohlc_mask
        ].copy()

        dropped_rows += invalid_ohlc_count

    # ------------------------------------------------------------------------
    # Positive price validation
    # ------------------------------------------------------------------------

    invalid_price_mask = (
        df[OHLC_COLUMNS] <= 0
    ).any(axis=1)

    invalid_price_count = int(
        invalid_price_mask.sum()
    )

    if invalid_price_count:
        df = df.loc[
            ~invalid_price_mask
        ].copy()

        dropped_rows += invalid_price_count

    if df.empty:
        raise ValueError(
            "No usable rows remained after "
            "OHLC validation."
        )

    # ------------------------------------------------------------------------
    # Chronological index
    # ------------------------------------------------------------------------

    df = (
        df
        .set_index(timestamp_column)
        .sort_index(kind="mergesort")
    )

    # ------------------------------------------------------------------------
    # Duplicate timestamps
    # ------------------------------------------------------------------------

    duplicate_mask = (
        df.index.duplicated(
            keep="first"
        )
    )

    duplicate_count = int(
        duplicate_mask.sum()
    )

    if duplicate_count:
        df = df.loc[
            ~duplicate_mask
        ].copy()

        dropped_rows += duplicate_count

    if df.empty:
        raise ValueError(
            "No usable rows remained after "
            "duplicate timestamp removal."
        )

    # ------------------------------------------------------------------------
    # Final chronology validation
    # ------------------------------------------------------------------------

    if not df.index.is_monotonic_increasing:
        raise ValueError(
            "Historical data is not chronologically ordered."
        )

    return df, dropped_rows


# ============================================================================
# CANONICAL BACKTEST WRAPPER
# ============================================================================

def run(
    df: pd.DataFrame,
    **kwargs,
):
    """
    Thin wrapper around the canonical Phase 1 backtest engine.

    No strategy logic belongs in this dashboard.
    """
    return run_backtest(
        df,
        **kwargs,
    )


# ============================================================================
# DATASET SUMMARY
# ============================================================================

def dataset_summary(
    df: pd.DataFrame,
) -> dict[str, Any]:
    """
    Produce display-only dataset metadata.
    """

    if df.empty:
        return {
            "rows": 0,
            "start": None,
            "end": None,
            "median_interval": None,
        }

    intervals = df.index.to_series().diff().dropna()

    median_interval = (
        intervals.median()
        if not intervals.empty
        else None
    )

    return {
        "rows": len(df),
        "start": df.index.min(),
        "end": df.index.max(),
        "median_interval": median_interval,
    }


# ============================================================================
# TRADE FILTERING
# ============================================================================

def _closed_trades(trades):
    """Return only closed WIN/LOSS trades."""

    return [
        trade
        for trade in trades
        if _normalize_status(
            getattr(
                trade,
                "result_status",
                None,
            )
        ) in {"WIN", "LOSS"}
    ]


# ============================================================================
# EQUITY CURVE
# ============================================================================

def compute_equity_curve(
    closed_trades,
):
    """
    Build cumulative R from chronologically ordered
    closed trades.
    """

    equity = []
    running_r = 0.0

    for trade in closed_trades:
        r_multiple = _safe_float(
            getattr(
                trade,
                "r_multiple",
                0.0,
            )
        )

        running_r += r_multiple
        equity.append(running_r)

    return equity


# ============================================================================
# MAXIMUM DRAWDOWN
# ============================================================================

def compute_max_drawdown(
    equity_curve,
):
    """
    Calculate maximum peak-to-trough drawdown in R.

    Starting equity baseline = 0R.
    """

    if not equity_curve:
        return 0.0

    peak = 0.0
    max_drawdown = 0.0

    for value in equity_curve:
        value = float(value)

        peak = max(
            peak,
            value,
        )

        drawdown = peak - value

        max_drawdown = max(
            max_drawdown,
            drawdown,
        )

    return max_drawdown


# ============================================================================
# STREAK ANALYSIS
# ============================================================================

def compute_streaks(
    closed_trades,
):
    """Calculate longest winning and losing streaks."""

    longest_win = 0
    longest_loss = 0

    current_win = 0
    current_loss = 0

    for trade in closed_trades:
        status = _normalize_status(
            getattr(
                trade,
                "result_status",
                None,
            )
        )

        if status == "WIN":
            current_win += 1
            current_loss = 0

            longest_win = max(
                longest_win,
                current_win,
            )

        elif status == "LOSS":
            current_loss += 1
            current_win = 0

            longest_loss = max(
                longest_loss,
                current_loss,
            )

    return {
        "longest_win": longest_win,
        "longest_loss": longest_loss,
    }


# ============================================================================
# SIDE STATISTICS
# ============================================================================

def side_stats(
    trades,
):
    """Calculate statistics for a trade subset."""

    if not trades:
        return {
            "count": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "expectancy": 0.0,
            "net_r": 0.0,
        }

    wins = sum(
        1
        for trade in trades
        if _normalize_status(
            getattr(
                trade,
                "result_status",
                None,
            )
        ) == "WIN"
    )

    losses = sum(
        1
        for trade in trades
        if _normalize_status(
            getattr(
                trade,
                "result_status",
                None,
            )
        ) == "LOSS"
    )

    total_r = sum(
        _safe_float(
            getattr(
                trade,
                "r_multiple",
                0.0,
            )
        )
        for trade in trades
    )

    count = len(trades)

    return {
        "count": count,
        "wins": wins,
        "losses": losses,
        "win_rate": (
            wins / count * 100.0
            if count
            else 0.0
        ),
        "expectancy": (
            total_r / count
            if count
            else 0.0
        ),
        "net_r": total_r,
    }


# ============================================================================
# PROFIT FACTOR
# ============================================================================

def compute_profit_factor(
    closed_trades,
):
    """
    Gross profit divided by gross loss.

    Returns:
        inf when profit exists without losses
        0.0 when there is no gross profit
    """

    r_multiples = [
        _safe_float(
            getattr(
                trade,
                "r_multiple",
                0.0,
            )
        )
        for trade in closed_trades
    ]

    gross_profit = sum(
        r
        for r in r_multiples
        if r > 0
    )

    gross_loss = abs(
        sum(
            r
            for r in r_multiples
            if r < 0
        )
    )

    if gross_loss == 0:
        return (
            float("inf")
            if gross_profit > 0
            else 0.0
        )

    return gross_profit / gross_loss


# ============================================================================
# PAGE HEADER
# ============================================================================

st.caption(
    "PHASE 01 · HISTORICAL RESEARCH"
)

st.title(
    "📊 Historical Research Dashboard"
)

st.write(
    "Upload historical OHLC data to run the canonical "
    "backtest engine and inspect historical strategy behaviour."
)

st.divider()


# ============================================================================
# DATA UPLOAD
# ============================================================================

with st.container(border=True):

    st.markdown(
        "### 📥 Historical Data"
    )

    uploaded_file = st.file_uploader(
        "Upload OHLC CSV File",
        type=["csv"],
        key="phase1_file_uploader",
    )


# ============================================================================
# DATASET EXECUTION / CACHE
# ============================================================================

if uploaded_file is not None:

    current_signature = (
        _uploaded_file_signature(
            uploaded_file
        )
    )

    previous_signature = (
        st.session_state.get(
            "phase1_file_signature"
        )
    )

    is_new_dataset = (
        current_signature
        != previous_signature
        or "phase1_results"
        not in st.session_state
        or "phase1_df"
        not in st.session_state
    )

    if is_new_dataset:

        try:

            with st.spinner(
                "Cleaning historical data and running backtest..."
            ):

                df, dropped = load_ohlc_csv(
                    uploaded_file
                )

                results = run(
                    df
                )

        except Exception as exc:

            st.error(
                f"Error processing historical data: {exc}"
            )

            st.stop()

        st.session_state[
            "phase1_file_signature"
        ] = current_signature

        st.session_state[
            "phase1_df"
        ] = df

        st.session_state[
            "phase1_results"
        ] = results

        st.session_state[
            "phase1_dropped"
        ] = dropped


# ============================================================================
# RESTORE CACHED RESULTS
# ============================================================================

if (
    "phase1_df"
    not in st.session_state
    or "phase1_results"
    not in st.session_state
):

    st.info(
        "Upload an OHLC CSV file to initiate "
        "the Phase 1 historical research backtest."
    )

    st.stop()


df = st.session_state[
    "phase1_df"
]

results = st.session_state[
    "phase1_results"
]

dropped = st.session_state.get(
    "phase1_dropped",
    0,
)


# ============================================================================
# DATASET SUMMARY
# ============================================================================

summary = dataset_summary(
    df
)

st.success(
    f"Historical data loaded — "
    f"{summary['rows']:,} valid rows."
)

if dropped:
    st.warning(
        f"{dropped:,} invalid or duplicate rows "
        "were removed during data cleaning."
    )

with st.container(border=True):

    st.markdown(
        "### 🗂 Dataset Summary"
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Rows",
        f"{summary['rows']:,}",
    )

    col2.metric(
        "Start",
        (
            summary["start"].strftime(
                "%Y-%m-%d %H:%M"
            )
            if summary["start"] is not None
            else "—"
        ),
    )

    col3.metric(
        "End",
        (
            summary["end"].strftime(
                "%Y-%m-%d %H:%M"
            )
            if summary["end"] is not None
            else "—"
        ),
    )

    col4.metric(
        "Median Interval",
        (
            str(summary["median_interval"])
            if summary["median_interval"]
            is not None
            else "—"
        ),
    )

st.divider()


# ============================================================================
# BACKTEST RESULT EXTRACTION
# ============================================================================

all_trades = list(
    getattr(
        results,
        "trades",
        []
    ) or []
)

closed_trades = _closed_trades(
    all_trades
)

open_trades = [
    trade
    for trade in all_trades
    if _normalize_status(
        getattr(
            trade,
            "result_status",
            None,
        )
    ) == "OPEN"
]

uncategorized_trades = (
    len(all_trades)
    - len(closed_trades)
    - len(open_trades)
)


# ============================================================================
# ENGINE SUMMARY
# ============================================================================

total_signals = _safe_int(
    getattr(
        results,
        "total_signals_generated",
        0,
    )
)

total_triggered = _safe_int(
    getattr(
        results,
        "total_trades_triggered",
        len(all_trades),
    )
)

total_invalidated = _safe_int(
    getattr(
        results,
        "total_invalidated",
        0,
    )
)

total_expired = _safe_int(
    getattr(
        results,
        "total_expired",
        0,
    )
)

total_mtf_rejected = _safe_int(
    getattr(
        results,
        "total_mtf_rejected",
        0,
    )
)


# ============================================================================
# NO CLOSED TRADES
# ============================================================================

if not closed_trades:

    st.warning(
        "Backtest completed, but no closed WIN/LOSS "
        "trades were produced by the engine."
    )

    with st.container(border=True):

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Signals",
            total_signals,
        )

        col2.metric(
            "Triggered",
            total_triggered,
        )

        col3.metric(
            "Invalidated",
            total_invalidated,
        )

        col4.metric(
            "Expired",
            total_expired,
        )

    if total_mtf_rejected:
        st.caption(
            f"MTF-filtered candidates: "
            f"{total_mtf_rejected:,}"
        )

    if open_trades:
        st.info(
            f"{len(open_trades)} trade(s) remain OPEN "
            "at the end of the dataset."
        )

    if uncategorized_trades > 0:
        st.warning(
            f"{uncategorized_trades} trade(s) had "
            "an unrecognized result status."
        )

    st.stop()


# ============================================================================
# CORE PERFORMANCE METRICS
# ============================================================================

stats = side_stats(
    closed_trades
)

equity_curve = compute_equity_curve(
    closed_trades
)

max_drawdown = compute_max_drawdown(
    equity_curve
)

profit_factor = compute_profit_factor(
    closed_trades
)

streaks = compute_streaks(
    closed_trades
)


# ============================================================================
# KPI SUMMARY
# ============================================================================

with st.container(border=True):

    kpi_cols = st.columns(5)

    kpi_cols[0].metric(
        "Closed Trades",
        stats["count"],
    )

    kpi_cols[1].metric(
        "Net R",
        f'{stats["net_r"]:.2f}R',
    )

    kpi_cols[2].metric(
        "Win Rate",
        f'{stats["win_rate"]:.1f}%',
    )

    kpi_cols[3].metric(
        "Expectancy",
        f'{stats["expectancy"]:.3f}R',
    )

    with kpi_cols[4]:

        st.caption(
            "DATA STATUS"
        )

        if dropped:
            st.warning(
                "CLEANED"
            )
        else:
            st.success(
                "CLEAN"
            )


if open_trades or uncategorized_trades > 0:

    notes = []

    if open_trades:
        notes.append(
            f"{len(open_trades)} still OPEN"
        )

    if uncategorized_trades > 0:
        notes.append(
            f"{uncategorized_trades} unrecognized status"
        )

    st.caption(
        f"ℹ️ {len(all_trades)} total trade(s) returned "
        f"by the engine — {', '.join(notes)}. "
        "These are excluded from closed-trade metrics."
    )


st.divider()


# ============================================================================
# MAIN TABS
# ============================================================================

(
    tab_overview,
    tab_equity,
    tab_direction,
    tab_engine,
) = st.tabs(
    [
        "🧭 Overview",
        "📈 Equity Curve",
        "↔️ Direction Analysis",
        "⚙️ Engine Activity",
    ]
)


# ============================================================================
# OVERVIEW
# ============================================================================

with tab_overview:

    st.subheader(
        "Risk & Consistency"
    )

    with st.container(border=True):

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Max Drawdown",
            f"{max_drawdown:.2f}R",
        )

        pf_display = (
            "∞"
            if profit_factor == float("inf")
            else f"{profit_factor:.2f}"
        )

        col2.metric(
            "Profit Factor",
            pf_display,
        )

        col3.metric(
            "Longest Win Streak",
            streaks["longest_win"],
        )

        col4.metric(
            "Longest Loss Streak",
            streaks["longest_loss"],
        )

    st.caption(
        "These statistics describe the supplied historical sample "
        "and are not forecasts of future performance."
    )


# ============================================================================
# EQUITY CURVE
# ============================================================================

with tab_equity:

    st.subheader(
        "Equity Curve — Cumulative R"
    )

    st.caption(
        "Running sum of closed-trade R-multiples "
        "in chronological order."
    )

    equity_df = pd.DataFrame(
        {
            "Cumulative R": equity_curve
        }
    )

    st.line_chart(
        equity_df,
    )


# ============================================================================
# DIRECTION ANALYSIS
# ============================================================================

with tab_direction:

    st.subheader(
        "Long / Short Breakdown"
    )

    long_trades = [
        trade
        for trade in closed_trades
        if _normalize_enum_like(
            getattr(
                trade,
                "direction",
                "",
            )
        ) in {
            "LONG",
            "BUY",
        }
    ]

    short_trades = [
        trade
        for trade in closed_trades
        if _normalize_enum_like(
            getattr(
                trade,
                "direction",
                "",
            )
        ) in {
            "SHORT",
            "SELL",
        }
    ]

    long_stats = side_stats(
        long_trades
    )

    short_stats = side_stats(
        short_trades
    )

    direction_df = pd.DataFrame(
        [
            {
                "Direction": "LONG",
                "Trades": long_stats["count"],
                "Wins": long_stats["wins"],
                "Losses": long_stats["losses"],
                "Win Rate %": round(
                    long_stats["win_rate"],
                    2,
                ),
                "Expectancy R": round(
                    long_stats["expectancy"],
                    3,
                ),
                "Net R": round(
                    long_stats["net_r"],
                    2,
                ),
            },
            {
                "Direction": "SHORT",
                "Trades": short_stats["count"],
                "Wins": short_stats["wins"],
                "Losses": short_stats["losses"],
                "Win Rate %": round(
                    short_stats["win_rate"],
                    2,
                ),
                "Expectancy R": round(
                    short_stats["expectancy"],
                    3,
                ),
                "Net R": round(
                    short_stats["net_r"],
                    2,
                ),
            },
        ]
    )

    with st.container(border=True):

        st.dataframe(
            direction_df,
            hide_index=True,
        )


# ============================================================================
# ENGINE ACTIVITY
# ============================================================================

with tab_engine:

    st.subheader(
        "Signal Pipeline Activity"
    )

    with st.container(border=True):

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Signals Generated",
            total_signals,
        )

        col2.metric(
            "Trades Triggered",
            total_triggered,
        )

        col3.metric(
            "Invalidated",
            total_invalidated,
        )

        col4.metric(
            "Expired",
            total_expired,
        )

    if total_mtf_rejected:
        st.info(
            f"{total_mtf_rejected:,} candidate signal(s) "
            "were rejected by the optional MTF filter."
        )

    st.caption(
        "Engine counters are read directly from the canonical "
        "backtest result. No strategy calculations are performed here."
    )


# ============================================================================
# RESEARCH INTEGRITY
# ============================================================================

st.divider()

st.caption(
    "🧪 Phase 1 Research — historical backtesting baseline. "
    "No broker connection, live execution, optimization, "
    "Monte Carlo simulation, walk-forward testing, or AI "
    "interpretation is active on this page."
)