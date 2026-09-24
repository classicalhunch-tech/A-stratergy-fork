"""
patch_session_engine.py

One-time patch: wires /api/session-check in frontend_dashboard/server.py
to real SessionEngine state instead of the stub response.

Safe by construction: each old_text below must match EXACTLY ONCE in
the file, or this aborts with an error and changes nothing. Run once,
then delete this file (matches this project's existing patch_*.py
pattern).
"""

from pathlib import Path

TARGET = Path("frontend_dashboard/server.py")

text = TARGET.read_text(encoding="utf-8")

edits = [
    (
        "from phase_03_paper.persistence.persistence import Persistence as Phase3Persistence\n",
        "from phase_03_paper.persistence.persistence import Persistence as Phase3Persistence\n"
        "from phase_03_paper.config import DEFAULT_CONFIG as PHASE3_DEFAULT_CONFIG\n"
        "from phase_03_paper.sessions.engine import SessionEngine\n",
    ),
    (
        "PAPER_RISK_CONFIG_FILE = STATE_DIR / \"paper_risk_config.json\"\n",
        "PAPER_RISK_CONFIG_FILE = STATE_DIR / \"paper_risk_config.json\"\n\n"
        "_session_engine = SessionEngine(config=PHASE3_DEFAULT_CONFIG)\n",
    ),
    (
        '''@app.post("/api/session-check")
def session_check() -> dict:
    return {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "status": "stub",
        "note": "Session Engine read API not yet wired.",
    }''',
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
    }''',
    ),
]

for i, (old, new) in enumerate(edits, start=1):
    count = text.count(old)
    if count != 1:
        raise SystemExit(
            f"ABORTED, no changes written: edit {i} matched {count} times "
            f"(expected exactly 1). File not touched."
        )
    text = text.replace(old, new, 1)

TARGET.write_text(text, encoding="utf-8")
print("Patched frontend_dashboard/server.py successfully -- 3 edits applied.")