"""
fix_phase2_embedding.py

Safe migration for the Phase 2 dashboard now that it's embedded
in the unified multipage app (dashboard/app.py +
dashboard/pages/2_(diamond)_Phase2_Lab.py, which uses
runpy.run_path() to re-execute phase2_dashboard/app.py fresh on
every rerun).

Changes:
1. Give Phase 2 its own session-state navigation key
   ("phase2_active_page" instead of "active_page"), as a
   defensive precaution against any future key collision with
   other embedded phases.
2. Fix the Phase 2 History indentation, if it still has the
   original broken shape.
3. Fix a real bug in the History page: it was reading
   DiagnosticRecord attributes that don't exist
   ("diagnostic_key", "status", "timestamp") instead of the
   real fields ("key", "succeeded", "started_at"), so every
   History entry silently displayed "Unknown" with a blank
   timestamp regardless of what actually ran.

NOT included (removed from an earlier draft of this script):
an st.set_page_config() guard behind a PHASE2_EMBEDDED env
var. That guard is unnecessary: Streamlit's native multipage
model only executes one page's script per rerun, so
dashboard/app.py's own set_page_config() call and Phase 2's
never run in the same request. Adding a guard here would be
dead code under the runpy-based wrapper actually in use.

Run from the project root:

    python fix_phase2_embedding.py

The script only writes the file if changes are actually made.
It is safe to run more than once.
"""

from pathlib import Path


TARGET = Path("phase2_dashboard") / "app.py"


def replace_all(content: str, old: str, new: str, label: str) -> str:
    if old not in content:
        print(f"{label}: NOT FOUND")
        return content

    count = content.count(old)
    content = content.replace(old, new)
    print(f"{label}: OK ({count} replacement{'s' if count != 1 else ''})")
    return content


def main() -> None:

    if not TARGET.exists():
        print(f"ERROR: {TARGET} not found.")
        print("Run this script from the project root.")
        return

    content = TARGET.read_text(encoding="utf-8")
    original_content = content

    # ========================================================================
    # EDIT 1
    # Give Phase 2 its own session-state navigation key.
    # ========================================================================

    replacements = [
        (
            'if "active_page" not in st.session_state:',
            'if "phase2_active_page" not in st.session_state:',
        ),
        (
            'st.session_state.active_page = "Dashboard"',
            'st.session_state.phase2_active_page = "Dashboard"',
        ),
        (
            'st.session_state.active_page = page',
            'st.session_state.phase2_active_page = page',
        ),
        (
            '{st.session_state.active_page}',
            '{st.session_state.phase2_active_page}',
        ),
        (
            'active_page = st.session_state.active_page',
            'active_page = st.session_state.phase2_active_page',
        ),
    ]

    total = 0

    for old, new in replacements:
        if old in content:
            n = content.count(old)
            content = content.replace(old, new)
            total += n

    if total:
        print(f"Edit 1 (Phase 2 session-state isolation): OK ({total} replacements)")
    else:
        print("Edit 1 (Phase 2 session-state isolation): NOT FOUND")

    # ========================================================================
    # EDIT 2
    # Fix the History indentation, if it's still in the original
    # broken shape.
    # ========================================================================

    old_history = (
        '    history = []\n'
        '\n'
        '    if BACKEND_AVAILABLE:\n'
        '        try:\n'
        '            history = []\n'
        '                for diag_key in DIAGNOSTICS:\n'
        '                    history.extend(get_history(diag_key))\n'
        '                history.sort(key=lambda r: r.started_at)\n'
        '        except Exception as exc:\n'
        '            st.error(f"Unable to load research history: {exc}")'
    )

    new_history = (
        '    history = []\n'
        '\n'
        '    if BACKEND_AVAILABLE:\n'
        '        try:\n'
        '            history = []\n'
        '            for diag_key in DIAGNOSTICS:\n'
        '                history.extend(get_history(diag_key))\n'
        '            history.sort(key=lambda r: r.started_at)\n'
        '        except Exception as exc:\n'
        '            st.error(f"Unable to load research history: {exc}")'
    )

    content = replace_all(content, old_history, new_history, "Edit 2 (History indentation)")

    # ========================================================================
    # EDIT 3 -- the real bug fix
    # DiagnosticRecord's actual fields are: key, succeeded, started_at.
    # It has no diagnostic_key / status / timestamp attributes, so the
    # original code always silently fell through to the defaults below.
    # ========================================================================

    old_record_block = (
        '            if isinstance(item, dict):\n'
        '                key = item.get("diagnostic_key", "Unknown")\n'
        '                status = item.get("status", "Complete")\n'
        '                timestamp = item.get("timestamp", "")\n'
        '            else:\n'
        '                key = getattr(item, "diagnostic_key", "Unknown")\n'
        '                status = getattr(item, "status", "Complete")\n'
        '                timestamp = getattr(item, "timestamp", "")'
    )

    new_record_block = (
        '            if isinstance(item, dict):\n'
        '                key = item.get("key", item.get("diagnostic_key", "Unknown"))\n'
        '                succeeded = item.get("succeeded", True)\n'
        '                status = "Complete" if succeeded else "Needs Review"\n'
        '                timestamp = item.get("started_at", item.get("timestamp", ""))\n'
        '            else:\n'
        '                key = getattr(item, "key", getattr(item, "diagnostic_key", "Unknown"))\n'
        '                succeeded = getattr(item, "succeeded", True)\n'
        '                status = "Complete" if succeeded else "Needs Review"\n'
        '                timestamp = getattr(\n'
        '                    item, "started_at", getattr(item, "timestamp", "")\n'
        '                )\n'
        '\n'
        '            if isinstance(timestamp, (int, float)):\n'
        '                from datetime import datetime as _datetime\n'
        '                timestamp = _datetime.fromtimestamp(timestamp).strftime(\n'
        '                    "%d %b %Y %H:%M"\n'
        '                )'
    )

    content = replace_all(
        content, old_record_block, new_record_block, "Edit 3 (History record fields -- real bug fix)"
    )

    # ========================================================================
    # WRITE
    # ========================================================================

    if content == original_content:
        print()
        print("NO CHANGES MADE.")
        print("The requested fixes may already be applied.")
        return

    TARGET.write_text(content, encoding="utf-8")

    print()
    print(f"Saved changes to {TARGET}")


if __name__ == "__main__":
    main()