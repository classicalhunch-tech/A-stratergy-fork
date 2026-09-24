"""
patch_auto_trade_sessions.py

Adds automatic, permanent TRADE decisions for every session occurrence
-- removes the manual per-session YES/NO step entirely, in both:
    - phase_04_live/runtime/loop.py (the ACTUAL trading loop)
    - frontend_dashboard/server.py (dashboard's read-only display, kept
      consistent with what the loop is really doing)

This intentionally overrides SessionEngine's own documented default:
an undecided session normally becomes SKIPPED at its start (see
phase_03_paper/sessions/engine.py's module docstring: "Phase 3 must
never assume that the user wants to trade a session"). This patch is
a deliberate choice to always trade every session, forever, with no
per-occurrence review.

Safe by construction: each old_text below must match EXACTLY ONCE in
its target file, or this aborts with an error and changes nothing.
"""

from pathlib import Path

LOOP_TARGET = Path("phase_04_live/runtime/loop.py")
SERVER_TARGET = Path("frontend_dashboard/server.py")

loop_text = LOOP_TARGET.read_text(encoding="utf-8")
server_text = SERVER_TARGET.read_text(encoding="utf-8")

loop_edits = [
    (
        "from phase_03_paper.sessions.engine import SessionEngine\n"
        "from phase_03_paper.signals.adapter import StrategyAdapter\n",
        "from phase_03_paper.models import SessionDecision, SessionStatus\n"
        "from phase_03_paper.sessions.engine import SessionEngine\n"
        "from phase_03_paper.signals.adapter import StrategyAdapter\n",
    ),
    (
        '''def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def run_live_loop(''',
        '''def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _auto_trade_all_sessions(session_engine: SessionEngine, now: datetime) -> None:
    """
    Automatically records a TRADE decision for every session occurrence
    that has not yet been decided, so the loop never waits on a manual
    per-session YES/NO. This intentionally overrides SessionEngine's own
    documented default -- an undecided session normally becomes SKIPPED
    at its start (see phase_03_paper/sessions/engine.py's module
    docstring: "Phase 3 must never assume that the user wants to trade a
    session") -- as a deliberate operator choice: every session is now
    traded automatically, forever, with no per-occurrence review.
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
                # Occurrence turned terminal between all_states() and
                # here -- nothing left to decide, skip it.
                continue


def run_live_loop(''',
    ),
    (
        '''        session_engine.evaluate(now=now)
        current_session = session_engine.current_session(now=now)''',
        '''        _auto_trade_all_sessions(session_engine, now=now)
        current_session = session_engine.current_session(now=now)''',
    ),
]

server_edits = [
    (
        "from phase_03_paper.sessions.engine import SessionEngine\n",
        "from phase_03_paper.sessions.engine import SessionEngine\n"
        "from phase_03_paper.models import SessionDecision, SessionStatus\n",
    ),
    (
        '''@app.post("/api/session-check")
def session_check() -> dict:
    """
    Real read of SessionEngine state (see phase_03_paper/sessions/engine.py).
    This creates its own SessionEngine instance, separate from any
    SessionEngine inside a running phase_04_live.runtime.loop process --
    since nothing anywhere currently calls record_decision(), this
    instance's evaluate()/current_session() output is a pure function
    of config + clock, so it matches what a live loop's own SessionEngine
    would compute at the same instant. It does not place orders or
    interact with phase_04_live.runtime.loop in any way.
    """
    now = datetime.now(timezone.utc)
    states = _session_engine.evaluate(now=now)''',
        '''def _auto_trade_all_sessions(session_engine: SessionEngine, now: datetime) -> None:
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
    states = _session_engine.evaluate(now=now)''',
    ),
]

for label, target, text, edits in [
    ("loop.py", LOOP_TARGET, loop_text, loop_edits),
    ("server.py", SERVER_TARGET, server_text, server_edits),
]:
    for i, (old, new) in enumerate(edits, start=1):
        count = text.count(old)
        if count != 1:
            raise SystemExit(
                f"ABORTED, no changes written to any file: {label} edit {i} "
                f"matched {count} times (expected exactly 1)."
            )
        text = text.replace(old, new, 1)
    if label == "loop.py":
        loop_text = text
    else:
        server_text = text

# Only write if BOTH files fully matched -- never leave one patched and
# the other not.
LOOP_TARGET.write_text(loop_text, encoding="utf-8")
SERVER_TARGET.write_text(server_text, encoding="utf-8")
print("Patched loop.py and server.py successfully -- auto-trade-all-sessions wired in.")