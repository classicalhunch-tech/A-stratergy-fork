$ErrorActionPreference = "Stop"
$path = ".\frontend_dashboard\server.py"
$content = Get-Content -Raw -Path $path

# 1. Imports
$old1 = @"
import subprocess
import sys
"@
$new1 = @"
import ctypes
import json
import os
import signal
import subprocess
import sys
import time
"@
if (-not $content.Contains($old1)) { Write-Error "Anchor 1 not found - aborting, nothing changed."; exit 1 }
$content = $content.Replace($old1, $new1)

# 2. STATE_DIR / PAPER_PID_FILE
$old2 = 'LOG_DIR = PROJECT_ROOT / "logs"'
$new2 = @"
LOG_DIR = PROJECT_ROOT / "logs"
STATE_DIR = PROJECT_ROOT / "state"
PAPER_PID_FILE = STATE_DIR / "paper_loop.json"
"@
if (-not $content.Contains($old2)) { Write-Error "Anchor 2 not found - aborting, check anchor 1 result manually."; exit 1 }
$content = $content.Replace($old2, $new2)

# 3. Replace paper-control block with PID-file-backed tracking
$old3 = @"
_paper_risk_config = RiskConfig()
_live_risk_config = RiskConfig()

_paper_process: subprocess.Popen | None = None


def _paper_process_running() -> bool:
    return _paper_process is not None and _paper_process.poll() is None


@app.get("/api/paper/risk-config")
def get_paper_risk_config() -> dict:
    # Reflects the REAL subprocess state -- if the loop crashed on its
    # own, this now reports that instead of a stale in-memory flag.
    _paper_risk_config.running = _paper_process_running()
    return _paper_risk_config.model_dump()


@app.post("/api/paper/risk-config")
def update_paper_risk_config(update: RiskConfigUpdate) -> dict:
    _paper_risk_config.sizing_mode = update.sizing_mode
    _paper_risk_config.fixed_lot_size = update.fixed_lot_size
    return _paper_risk_config.model_dump()


@app.post("/api/paper/control")
def control_paper(action: ControlAction) -> dict:
    """
    Starts/stops phase_04_live.runtime.loop as a real subprocess with
    --dry-run (see OrderManagerConfig.dry_run -- no broker order is
    ever sent in this mode). This is a genuine process launch, not a
    UI stub.

    NOTE: dry-run activity is recorded via Phase4Persistence into
    phase_04_live.db (expected_positions / order_attempts), NOT into
    phase3_paper.db's trades/journal_entries tables -- so it will show
    up on the Live Trading page's tables (tagged dry_run=True), not on
    this Paper Trading page's Trade Journal / Performance cards.
    """
    global _paper_process

    if action.action == "start":
        if _paper_process_running():
            _paper_risk_config.running = True
            return {**_paper_risk_config.model_dump(), "note": "already running"}

        LOG_DIR.mkdir(exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        log_path = LOG_DIR / f"paper_loop_{ts}.log"
        log_file = open(log_path, "w", encoding="utf-8")

        _paper_process = subprocess.Popen(
            [sys.executable, "-u", "-m", "phase_04_live.runtime.loop", "--dry-run"],
            cwd=str(PROJECT_ROOT),
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        _paper_risk_config.running = True
        return {**_paper_risk_config.model_dump(), "log_file": str(log_path)}

    elif action.action == "stop":
        if _paper_process_running():
            _paper_process.terminate()
            try:
                _paper_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                _paper_process.kill()
                _paper_process.wait(timeout=5)
        _paper_risk_config.running = False
        return _paper_risk_config.model_dump()

    raise HTTPException(status_code=400, detail="action must be 'start' or 'stop'")
"@

$new3 = @"
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
"@
if (-not $content.Contains($old3)) { Write-Error "Anchor 3 not found - aborting, check anchors 1-2 result manually."; exit 1 }
$content = $content.Replace($old3, $new3)

Set-Content -Path $path -Value $content -NoNewline
python -m py_compile $path
if ($LASTEXITCODE -ne 0) { Write-Error "Syntax check FAILED after patch - check server.py manually."; exit 1 }
Write-Host "server.py patched and syntax-verified: paper start/stop now PID-file-backed, survives --reload."
