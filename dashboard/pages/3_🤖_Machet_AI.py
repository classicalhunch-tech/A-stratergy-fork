"""
dashboard/pages/3_🤖_Machet_AI.py

Thin multipage entry point for the Machet AI research assistant.

DESIGN PRINCIPLE
----------------
This file does NOT reimplement, copy, or modify any AI assistant
logic. phase2_dashboard/ai_assistant.py remains the single source
of truth for the assistant's UI, notifications, and Ollama wiring.

Unlike the Phase 2 Lab page, this does not need runpy: ai_assistant.py
is a plain function module with no module-level st.set_page_config()
or standalone routing of its own — it exposes one public entry point,
render_ai_assistant_page(), meant to be imported and called. Streamlit
reruns this whole page script (including this call) on every
interaction, so the assistant stays fully responsive without runpy's
caching workaround.

The root dashboard (dashboard/app.py) continues to own
st.set_page_config() — this file does not call it.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from phase2_dashboard.ai_assistant import render_ai_assistant_page

render_ai_assistant_page()