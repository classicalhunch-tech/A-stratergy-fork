"""
phase2_dashboard/backend/

Application/orchestration layer between the Phase 2 Streamlit
dashboard and the existing runner/results modules.

Public API:

    from phase2_dashboard.backend import Phase2Service

    service = Phase2Service()
    service.run_diagnostic("walk_forward")
"""

from .service import Phase2Service
from .models import DiagnosticStatus, classify
from . import formatter

__all__ = [
    "Phase2Service",
    "DiagnosticStatus",
    "classify",
    "formatter",
]