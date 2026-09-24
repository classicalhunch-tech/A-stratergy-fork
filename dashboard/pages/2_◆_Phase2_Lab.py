"""
dashboard/pages/2_◆_Phase2_Lab.py

Thin multipage entry point for the Phase 2 Robustness Lab.

DESIGN PRINCIPLE
----------------
This file does NOT reimplement, copy, or modify any Phase 2
dashboard logic. phase2_dashboard/app.py remains the single
source of truth for the Phase 2 UI and backend wiring.

Streamlit's native multipage mechanism only auto-discovers
pages placed directly under <entry_script>/pages/. Since the
real Phase 2 dashboard lives in phase2_dashboard/app.py (a
separate top-level package, not under dashboard/pages/), this
file re-executes that script in place using runpy.

WHY runpy AND NOT `import`
---------------------------
A plain `import phase2_dashboard.app` would only run the module
body once (on first import) because Python caches imported
modules. Streamlit reruns the active page's script on every
interaction (button click, widget change, etc.) — a cached
import would freeze the Phase 2 dashboard after its first
render and stop responding to any button.

runpy.run_path() has no such caching: it re-executes the target
file's full source, top to bottom, every single call — exactly
matching how Streamlit expects a page script to behave.

run_name="__main__" makes phase2_dashboard/app.py behave
identically to being run directly via
`streamlit run phase2_dashboard/app.py` — its own internal
PROJECT_ROOT / sys.path logic, st.set_page_config() call, and
all page routing continue to work completely unchanged.
"""

import runpy
from pathlib import Path

_PHASE2_APP_PATH = (
    Path(__file__).resolve().parents[2]
    / "phase2_dashboard"
    / "app.py"
)

runpy.run_path(
    str(_PHASE2_APP_PATH),
    run_name="__main__",
)