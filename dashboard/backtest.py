"""
dashboard/backtest.py

Data-loading + metrics layer for the dashboard. This is the ONLY
module that reaches into the strategy engine (strategy/backtest.py) --
app.py and components.py should not import strategy.* directly, so
the engine's interface only needs updating in one place if it changes.
"""

import sys
from pathlib import Path

# strategy/ lives one level above dashboard/, and this file may be run
# from any working directory (e.g. `streamlit run dashboard/app.py`
# from the project root), so make sure the project root is importable
# regardless of cwd.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from strategy.backtest import run_backtest  # noqa: E402

TIMESTAMP_CANDIDATES = ["timestamp", "date", "datetime", "time"]
OHLC_COLUMNS = ["open", "high", "low", "close"]


def load_ohlc_csv(path_or_buffer) -> tuple[pd.DataFrame, int]:
    """
    Load a CSV into the DataFrame contract run_backtest() expects:
    DatetimeIndex + lowercase open/high/low/close columns.

    Mirrors the column normalization strategy/backtest.py already does
    internally, then additionally finds and indexes a timestamp-like
    column, since run_backtest() requires a DatetimeIndex rather than
    a plain 'timestamp' column.

    Returns (df, dropped_rows) where dropped_rows counts rows removed
    for an unparseable timestamp, non-numeric OHLC data, or a duplicate
    timestamp (in that order), so the caller can surface a single
    "N rows dropped" warning to the user.
    """
    df = pd.read_csv(path_or_buffer)
    df.columns = [str(c).strip().lower() for c in df.columns]

    ts_col = next((c for c in TIMESTAMP_CANDIDATES if c in df.columns), None)
    if ts_col is None:
        raise ValueError(
            f"No timestamp-like column found. Looked for: {TIMESTAMP_CANDIDATES}. "
            f"Columns present: {list(df.columns)}"
        )

    required = set(OHLC_COLUMNS)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")

    dropped_rows = 0

    # 1) Parse timestamps; drop rows that don't parse.
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    bad_ts = int(df[ts_col].isna().sum())
    if bad_ts:
        df = df[df[ts_col].notna()]
        dropped_rows += bad_ts

    # 2) Coerce OHLC to numeric; drop rows where any of them fail to
    # parse (stray text, thousands separators, blanks, etc.) instead
    # of letting an object-dtype column fail confusingly downstream.
    for col in OHLC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    bad_ohlc = int(df[OHLC_COLUMNS].isna().any(axis=1).sum())
    if bad_ohlc:
        df = df[df[OHLC_COLUMNS].notna().all(axis=1)]
        dropped_rows += bad_ohlc

    df = df.set_index(ts_col).sort_index()

    # 3) Duplicate timestamps break the "one bar per index entry"
    # assumption the strategy engine relies on -- keep the first
    # occurrence and drop the rest rather than passing an ambiguous
    # index downstream.
    dup_mask = df.index.duplicated(keep="first")
    n_dupes = int(dup_mask.sum())
    if n_dupes:
        df = df[~dup_mask]
        dropped_rows += n_dupes

    if df.empty:
        raise ValueError(
            "No usable rows remained after cleaning -- check the timestamp "
            "format and that open/high/low/close are numeric."
        )

    return df, dropped_rows


def run(df: pd.DataFrame, **kwargs):
    """Thin passthrough to the strategy engine's run_backtest()."""
    return run_backtest(df, **kwargs)


def compute_streaks(closed_trades):
    """Longest winning and losing streaks in chronological order."""
    max_win_streak = max_loss_streak = 0
    cur_win_streak = cur_loss_streak = 0

    for trade in closed_trades:
        if trade.result_status == "WIN":
            cur_win_streak += 1
            cur_loss_streak = 0
        elif trade.result_status == "LOSS":
            cur_loss_streak += 1
            cur_win_streak = 0
        else:
            # Handles BE (Break-Even) or other non-win/loss outcomes safely
            cur_win_streak = 0
            cur_loss_streak = 0

        max_win_streak = max(max_win_streak, cur_win_streak)
        max_loss_streak = max(max_loss_streak, cur_loss_streak)

    return max_win_streak, max_loss_streak


def compute_equity_curve(closed_trades):
    """Cumulative R multiple, in the chronological order trades closed."""
    equity = []
    running = 0.0
    for trade in closed_trades:
        running += trade.r_multiple
        equity.append(running)
    return equity


def compute_max_drawdown(equity_curve):
    """
    Max drawdown in R units, off the running peak of the equity curve.

    The running peak starts at 0.0 (equity before any trade has
    closed), not -inf -- otherwise a losing streak right at the start
    of the curve reports 0 drawdown instead of the real one, since the
    first point would be treated as its own peak.
    """
    peak = 0.0
    max_dd = 0.0
    for value in equity_curve:
        peak = max(peak, value)
        max_dd = max(max_dd, peak - value)
    return max_dd


def split_by_direction(closed_trades, direction):
    return [t for t in closed_trades if t.direction == direction]


def side_stats(trades):
    if not trades:
        return {
            "count": 0,
            "win_rate": 0.0,
            "expectancy": 0.0,
            "net_r": 0.0,
        }

    wins = [t for t in trades if t.result_status == "WIN"]
    total_r = sum(t.r_multiple for t in trades)

    return {
        "count": len(trades),
        "win_rate": len(wins) / len(trades) * 100.0,
        "expectancy": total_r / len(trades),
        "net_r": total_r,
    }