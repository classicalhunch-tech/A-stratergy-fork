"""
frontend_dashboard/server.py

Read-only FastAPI bridge over:
    - phase_04_live.db    (Phase 4: expected_positions, order_attempts)
    - phase3_paper.db      (Phase 3: trades, journal_entries -> performance, report)
    - a staged historical CSV dataset (candles -- see note below)
    - a live MT5 terminal connection (XAUUSD candles + account -- see note below)

Does NOT read:
    - live price/candle data via /api/candles (that endpoint still serves
      a STAGED HISTORICAL CSV, not live data). Live candles ARE available
      read-only via /api/mt5/candles, backed by a single persistent MT5
      terminal connection (see lifespan handler below) -- this does not
      place orders or interfere with phase_04_live.runtime.loop in any way.
    - notifications (Phase4NotificationEngine is in-memory only inside
      the loop.py process -- not persisted, not readable from here)
    - Phase 2 diagnostic results (phase2_dashboard/results.py -- not
      yet wired in, separate JSON-backed store, different pattern)
    - true SessionEngine state (/api/session-check below is a STUB --
      see its docstring)

This module does NOT write to either database, place orders, or
interfere with phase_04_live.runtime.loop in any way -- it opens its
own short-lived read connections (SQLite), plus one persistent
read-only MT5 terminal connection managed by the lifespan handler below.

The /api/paper/* and /api/live/* risk-config + control endpoints below
are UI STAGING STUBS ONLY: in-memory, per-process state that resets on
every restart/reload. They do not persist anywhere, do not place
orders, and do not touch phase_04_live.runtime.loop or any database.
They exist purely so the dashboard's "UI Stub" controls have something
to call instead of 404ing.

EXCEPTION: /api/paper/risk-config's POST handler below now ALSO writes
PAPER_RISK_CONFIG_FILE (state/paper_risk_config.json) to disk on every
update, alongside its in-memory _paper_risk_config state. This is the
one place this module does write something a running loop.py process
actually reads -- see phase_04_live/risk/config_file.py. Everything
else in this "UI staging stubs" section remains in-memory only.
"""

from __future__ import annotations

import csv
import sqlite3
from contextlib import asynccontextmanager, closing
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import ctypes
import json
import os
import signal
import subprocess
import sys
import time

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import MetaTrader5 as mt5

from phase_03_paper.ai.report_generator import ReportEngine
from phase_03_paper.journal.journal import Journal
from phase_03_paper.performance.performance import PerformanceEngine
from phase_03_paper.persistence.persistence import Persistence as Phase3Persistence
from phase_03_paper.config import DEFAULT_CONFIG as PHASE3_DEFAULT_CONFIG
from phase_03_paper.sessions.engine import SessionEngine
from phase_03_paper.models import SessionDecision, SessionStatus

PHASE4_DB_PATH = "phase_04_live.db"
PHASE3_DB_PATH = "phase_03_paper/data/phase3_paper.db"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIR = Path(__file__).resolve().parent
CANDLE_CSV_PATH = PROJECT_ROOT / "real_gold_data_mt5_5000.csv"
LOG_DIR = PROJECT_ROOT / "logs"
STATE_DIR = PROJECT_ROOT / "state"
PAPER_PID_FILE = STATE_DIR / "paper_loop.json"
PAPER_RISK_CONFIG_FILE = STATE_DIR / "paper_risk_config.json"

_session_engine = SessionEngine(config=PHASE3_DEFAULT_CONFIG)

MT5_SYMBOL = "XAUUSD"
MT5_TIMEFRAME_MAP = {
    "5m": mt5.TIMEFRAME_M5,
    "15m": mt5.TIMEFRAME_M15,
    "4h": mt5.TIMEFRAME_H4,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: open one MT5 terminal connection for the life of the process.
    # Per-request initialize()/shutdown() is wasteful and can race --
    # this connection is shared read-only across all /api/mt5/* requests.
    if not mt5.initialize():
        print("MT5 initialize() failed, error code =", mt5.last_error())
    yield
    # Shutdown: close it cleanly.
    mt5.shutdown()


app = FastAPI(title="Trading Dashboard Bridge (read-only)", lifespan=lifespan)


def _connect_readonly(db_path: str) -> sqlite3.Connection:
    uri = f"file:{Path(db_path).resolve()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


@app.get("/api/health")
def health() -> dict:
    """Confirm the bridge can read both live databases."""
    with closing(_connect_readonly(PHASE4_DB_PATH)) as conn:
        conn.execute("SELECT 1")
    with closing(_connect_readonly(PHASE3_DB_PATH)) as conn:
        conn.execute("SELECT 1")
    return {"status": "ok", "phase4_db": PHASE4_DB_PATH, "phase3_db": PHASE3_DB_PATH}


@app.get("/api/phase4/expected_positions")
def phase4_expected_positions() -> list[dict]:
    with closing(_connect_readonly(PHASE4_DB_PATH)) as conn:
        rows = conn.execute("SELECT * FROM expected_positions").fetchall()
        return [dict(row) for row in rows]


@app.get("/api/phase4/order_attempts")
def phase4_order_attempts(pending_only: bool = False) -> list[dict]:
    with closing(_connect_readonly(PHASE4_DB_PATH)) as conn:
        if pending_only:
            rows = conn.execute(
                "SELECT * FROM order_attempts WHERE resolved_at IS NULL"
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM order_attempts ORDER BY attempted_at DESC"
            ).fetchall()
        return [dict(row) for row in rows]


@app.get("/api/phase3/trades")
def phase3_trades() -> list[dict]:
    persistence = Phase3Persistence(PHASE3_DB_PATH)
    trades = persistence.load_trades()
    return [
        {
            "trade_id": t.trade_id,
            "direction": t.direction.value,
            "session": t.session,
            "requested_entry": t.requested_entry,
            "stop": t.stop,
            "target": t.target,
            "entry": t.entry,
            "opened_at": t.opened_at.isoformat() if t.opened_at else None,
            "closed_at": t.closed_at.isoformat() if t.closed_at else None,
            "exit_price": t.exit_price,
            "exit_reason": t.exit_reason.value if t.exit_reason else None,
            "result_r": t.result_r,
            "status": t.status.value,
        }
        for t in trades
    ]


def _load_journal() -> Journal:
    persistence = Phase3Persistence(PHASE3_DB_PATH)
    entries = persistence.load_journal_entries()
    journal = Journal()
    journal.load_entries(entries)
    return journal


@app.get("/api/phase3/performance")
def phase3_performance() -> dict:
    journal = _load_journal()
    engine = PerformanceEngine(journal)
    snapshot = engine.calculate()
    result = asdict(snapshot)
    result["computed_at"] = snapshot.computed_at.isoformat()
    return result


def _notable_to_dict(trade) -> dict | None:
    if trade is None:
        return None
    return {
        "trade_id": trade.trade_id,
        "session": trade.session,
        "direction": trade.direction,
        "result_r": trade.result_r,
        "recorded_at": trade.recorded_at.isoformat() if trade.recorded_at else None,
    }


@app.get("/api/phase3/report")
def phase3_report() -> dict:
    journal = _load_journal()
    engine = ReportEngine(journal)
    report = engine.generate()
    return {
        "generated_at": report.generated_at.isoformat(),
        "narrative": report.narrative,
        "best_trade": _notable_to_dict(report.best_trade),
        "worst_trade": _notable_to_dict(report.worst_trade),
    }


@app.get("/api/candles")
def candles(limit: int = 500) -> list[dict]:
    if not CANDLE_CSV_PATH.exists():
        return []

    rows: list[dict] = []
    with open(CANDLE_CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        lower_map = {name.lower(): name for name in fieldnames}

        ts_col = next(
            (lower_map[c] for c in ("timestamp", "date", "datetime", "time") if c in lower_map),
            None,
        )
        open_col = lower_map.get("open")
        high_col = lower_map.get("high")
        low_col = lower_map.get("low")
        close_col = lower_map.get("close")

        if not all([ts_col, open_col, high_col, low_col, close_col]):
            return []

        for row in reader:
            raw_ts = row.get(ts_col, "")
            try:
                ts = datetime.fromisoformat(raw_ts.replace("Z", "+00:00"))
            except ValueError:
                continue

            try:
                rows.append(
                    {
                        "time": int(ts.timestamp()),
                        "open": float(row[open_col]),
                        "high": float(row[high_col]),
                        "low": float(row[low_col]),
                        "close": float(row[close_col]),
                    }
                )
            except (TypeError, ValueError):
                continue

    rows.sort(key=lambda r: r["time"])
    return rows[-limit:]


@app.get("/api/mt5/candles")
def mt5_candles(timeframe: str = "5m", limit: int = 500) -> list[dict]:
    """Live MT5 OHLC candles for XAUUSD. Read-only: uses copy_rates_from_pos,
    never places orders, never touches phase_04_live.runtime.loop. Requires
    the MT5 terminal to be running and logged in on this machine -- if the
    startup initialize() failed, this returns an empty list rather than
    raising, so the frontend can fall back gracefully.

    timeframe: one of "5m", "15m", "4h" (see MT5_TIMEFRAME_MAP above).
    """
    mt5_timeframe = MT5_TIMEFRAME_MAP.get(timeframe)
    if mt5_timeframe is None:
        return []

    rates = mt5.copy_rates_from_pos(MT5_SYMBOL, mt5_timeframe, 0, limit)
    if rates is None:
        return []

    return [
        {
            "time": int(r["time"]),
            "open": float(r["open"]),
            "high": float(r["high"]),
            "low": float(r["low"]),
            "close": float(r["close"]),
        }
        for r in rates
    ]


@app.get("/api/mt5/account")
def mt5_account() -> dict:
    """Live account snapshot from MT5 (balance, equity, profit, margin).
    Read-only: calls account_info() on the same persistent connection
    opened in lifespan() above -- never places orders, never touches
    phase_04_live.runtime.loop. Returns connected: False rather than
    raising if MT5 isn't logged in, so the frontend can show
    'unavailable' instead of crashing.
    """
    info = mt5.account_info()
    if info is None:
        return {"connected": False}
    return {
        "connected": True,
        "login": info.login,
        "server": info.server,
        "balance": info.balance,
        "equity": info.equity,
        "profit": info.profit,
        "margin": info.margin,
        "margin_free": info.margin_free,
        "currency": info.currency,
    }


def _auto_trade_all_sessions(session_engine: SessionEngine, now: datetime) -> None:
    """
    Automatically records a TRADE decision for every session occurrence
    that has not yet been decided. Mirrors the same helper in
    phase_04_live/runtime/loop.py -- keeps this dashboard's read-only
    SessionEngine instance consistent with what the running loop's own
    instance is actually doing, both now auto-TRADE by deliberate
    operator choice. Does not place orders or interact with
    phase_04_live.runtime.loop -- only calls record_decision() on this
    module's own display-only SessionEngine instance.
    """
    for state in session_engine.all_states(now=now):
        if (
            state.decision == SessionDecision.UNDECIDED
            and state.status not in (SessionStatus.CLOSED, SessionStatus.SKIPPED)
        ):
            try:
                session_engine.record_decision(
                    state.session_name,
                    SessionDecision.TRADE,
                    now=now,
                )
            except ValueError:
                continue


@app.post("/api/session-check")
def session_check() -> dict:
    """
    Real read of SessionEngine state (see phase_03_paper/sessions/engine.py).
    This creates its own SessionEngine instance, separate from any
    SessionEngine inside a running phase_04_live.runtime.loop process --
    both now auto-TRADE every session via _auto_trade_all_sessions(), so
    this display stays consistent with what the live loop's own
    SessionEngine instance is actually doing. It does not place orders
    or interact with phase_04_live.runtime.loop in any way.
    """
    now = datetime.now(timezone.utc)
    _auto_trade_all_sessions(_session_engine, now=now)
    states = _session_engine.evaluate(now=now)
    current = _session_engine.current_session(now=now)
    next_state = _session_engine.next_session(now=now)

    def _state_to_dict(state) -> dict:
        return {
            "session_name": state.session_name,
            "occurrence_id": state.occurrence_id,
            "status": state.status.value,
            "decision": state.decision.value,
            "enabled": state.enabled,
            "scheduled_start": state.scheduled_start.isoformat(),
            "scheduled_end": state.scheduled_end.isoformat(),
        }

    return {
        "checked_at": now.isoformat(),
        "current_session": _state_to_dict(current) if current else None,
        "next_session": _state_to_dict(next_state) if next_state else None,
        "minutes_until_next_session": _session_engine.minutes_until_next_session(now=now),
        "sessions": [_state_to_dict(s) for s in states],
    }


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    conversation: list[ChatMessage] = []


@app.post("/api/chat")
def chat(request: ChatRequest) -> dict:
    from frontend_dashboard.runtime_copilot import chat_with_runtime_copilot
    conversation = [m.model_dump() for m in request.conversation]
    reply = chat_with_runtime_copilot(conversation)
    return {"reply": reply}


# ---------------------------------------------------------------------------
# UI staging stubs -- in-memory only, reset on every restart/reload.
# Never touch a database, never place orders, never interact with
# phase_04_live.runtime.loop. These exist solely so the dashboard's
# "UI Stub" risk-config and start/stop controls have something to call.
#
# EXCEPTION: update_paper_risk_config() below also writes
# PAPER_RISK_CONFIG_FILE to disk -- see module docstring at top of file.
# ---------------------------------------------------------------------------

class RiskConfig(BaseModel):
    sizing_mode: str = "fixed_lot"
    fixed_lot_size: float = 0.01
    running: bool = False


class RiskConfigUpdate(BaseModel):
    sizing_mode: str
    fixed_lot_size: float


class ControlAction(BaseModel):
    action: str  # "start" | "stop"


_paper_risk_config = RiskConfig()
_live_risk_config = RiskConfig()


def _pid_alive(pid: int) -> bool:
    """
    Windows-safe liveness check for an arbitrary PID -- does not
    require the process to be a child of this one, so it survives
    uvicorn --reload restarting this server's own process tree.
    """
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    handle = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if handle:
        ctypes.windll.kernel32.CloseHandle(handle)
        return True
    return False


def _read_paper_state() -> dict | None:
    if not PAPER_PID_FILE.exists():
        return None
    try:
        return json.loads(PAPER_PID_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _write_paper_state(pid: int, log_file: str) -> None:
    STATE_DIR.mkdir(exist_ok=True)
    PAPER_PID_FILE.write_text(
        json.dumps({
            "pid": pid,
            "log_file": log_file,
            "started_at": datetime.now(timezone.utc).isoformat(),
        }),
        encoding="utf-8",
    )


def _clear_paper_state() -> None:
    try:
        PAPER_PID_FILE.unlink()
    except FileNotFoundError:
        pass


def _paper_process_running() -> tuple[bool, dict | None]:
    """
    Source of truth is the PID file on disk, not an in-memory Popen
    handle. An in-memory handle is lost every time uvicorn --reload
    restarts this server process -- that previously caused a second,
    untracked loop subprocess to launch on the next Start click while
    the first kept running orphaned. A real OS-level PID check on disk
    survives that.
    """
    state = _read_paper_state()
    if state is None:
        return False, None
    if _pid_alive(state["pid"]):
        return True, state
    _clear_paper_state()  # stale pidfile: process died without Stop
    return False, None


@app.get("/api/paper/risk-config")
def get_paper_risk_config() -> dict:
    running, _ = _paper_process_running()
    _paper_risk_config.running = running
    return _paper_risk_config.model_dump()


@app.post("/api/paper/risk-config")
def update_paper_risk_config(update: RiskConfigUpdate) -> dict:
    _paper_risk_config.sizing_mode = update.sizing_mode
    _paper_risk_config.fixed_lot_size = update.fixed_lot_size
    STATE_DIR.mkdir(exist_ok=True)
    PAPER_RISK_CONFIG_FILE.write_text(
        json.dumps({
            "sizing_mode": update.sizing_mode,
            "fixed_lot_size": update.fixed_lot_size,
        }),
        encoding="utf-8",
    )
    return _paper_risk_config.model_dump()


@app.post("/api/paper/control")
def control_paper(action: ControlAction) -> dict:
    """
    Starts/stops phase_04_live.runtime.loop as a real subprocess with
    --dry-run (no broker order is ever sent in this mode). Tracking
    is PID-file-backed (see _paper_process_running() above) so it
    survives server restarts, including uvicorn --reload.

    NOTE: dry-run activity is recorded via Phase4Persistence into
    phase_04_live.db, NOT into phase3_paper.db -- so it shows up on
    the Live Trading page's tables (tagged dry_run=True), not on this
    Paper Trading page's Trade Journal / Performance cards.
    """
    if action.action == "start":
        running, state = _paper_process_running()
        if running:
            _paper_risk_config.running = True
            return {**_paper_risk_config.model_dump(), "note": "already running", "log_file": state["log_file"]}

        LOG_DIR.mkdir(exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        log_path = LOG_DIR / f"paper_loop_{ts}.log"
        log_file = open(log_path, "w", encoding="utf-8")

        process = subprocess.Popen(
            [sys.executable, "-u", "-m", "phase_04_live.runtime.loop", "--dry-run"],
            cwd=str(PROJECT_ROOT),
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        _write_paper_state(process.pid, str(log_path))
        _paper_risk_config.running = True
        return {**_paper_risk_config.model_dump(), "log_file": str(log_path)}

    elif action.action == "stop":
        running, state = _paper_process_running()
        if running:
            pid = state["pid"]
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError:
                pass
            for _ in range(10):
                if not _pid_alive(pid):
                    break
                time.sleep(0.5)
        _clear_paper_state()
        _paper_risk_config.running = False
        return _paper_risk_config.model_dump()

    raise HTTPException(status_code=400, detail="action must be 'start' or 'stop'")


# NOTE: /api/live/* below is intentionally still a UI stub -- NOT wired
# to a real subprocess yet. Held off deliberately pending manual
# verification of the VERIFY-marked interfaces in
# phase_04_live/runtime/loop.py's module docstring, and the hardcoded
# has_opposite_direction_open=False gap in its risk-gate wiring.


@app.get("/api/live/risk-config")
def get_live_risk_config() -> dict:
    return _live_risk_config.model_dump()


@app.post("/api/live/risk-config")
def update_live_risk_config(update: RiskConfigUpdate) -> dict:
    _live_risk_config.sizing_mode = update.sizing_mode
    _live_risk_config.fixed_lot_size = update.fixed_lot_size
    return _live_risk_config.model_dump()


@app.post("/api/live/control")
def control_live(action: ControlAction) -> dict:
    _live_risk_config.running = action.action == "start"
    return _live_risk_config.model_dump()


# ---------------------------------------------------------------------------
# Static frontend (index.html) -- MUST be mounted last. FastAPI matches
# explicit @app.get/@app.post routes before falling through to a mount,
# so registering this after every /api/* route above means it can never
# shadow them. html=True makes StaticFiles serve index.html at "/".
# ---------------------------------------------------------------------------
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")