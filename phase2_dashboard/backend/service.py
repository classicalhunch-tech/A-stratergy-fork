"""
phase2_dashboard/backend/service.py

Phase2Service — the single application-facing API between the
Streamlit dashboard and the existing Phase 2 research engines.

ARCHITECTURE
------------
    dashboard (app.py)
        |
        v
    Phase2Service   <-- this file
        |
        v
    phase2_dashboard.runner   (subprocess execution, UNCHANGED)
        |
        v
    phase_02_optimization/*   (research engines, UNCHANGED)

    Phase2Service also calls phase2_dashboard.results (persistence,
    UNCHANGED) directly after every run.

THIS FIXES ONE THING
---------------------
app.py currently calls results.record_run(diagnostic_key=..., result=...,
duration_seconds=...), but results.record_run(result: DiagnosticRunResult)
only accepts a single DiagnosticRunResult object. Every dashboard run has
therefore been failing to persist, silently, via app.py's own
except-Exception fallback.

Phase2Service.run_diagnostic() below calls results.record_run(result)
correctly, once, in one place. app.py should call Phase2Service instead
of runner/results directly.

WHAT THIS FILE DOES NOT DO
---------------------------
- does not implement diagnostic logic
- does not modify runner.py or results.py
- does not calculate trading/research statistics
- does not know about Streamlit or session_state
"""

from __future__ import annotations

from typing import Optional

from phase2_dashboard import runner as _runner
from phase2_dashboard import results as _results
from phase2_dashboard.runner import DiagnosticRunResult, DiagnosticSpec
from phase2_dashboard.results import DiagnosticRecord

from .models import DiagnosticStatus, classify


class Phase2Service:
    """Application-facing API for the Phase 2 research backend."""

    # ------------------------------------------------------------------
    # DISCOVERY
    # ------------------------------------------------------------------

    def list_diagnostics(self) -> list[DiagnosticSpec]:
        """Return all registered diagnostics, in registry order."""

        return _runner.list_diagnostics()

    def get_available_diagnostics(self) -> list[str]:
        """Return just the registered diagnostic keys."""

        return [spec.key for spec in self.list_diagnostics()]

    # ------------------------------------------------------------------
    # EXECUTION
    # ------------------------------------------------------------------

    def run_diagnostic(
        self,
        key: str,
        timeout_seconds: Optional[float] = None,
    ) -> DiagnosticRunResult:
        """
        Run one diagnostic and persist the result.

        Raises KeyError if `key` is not registered (same as
        runner.run_diagnostic — not swallowed here, the caller
        decides how to display an unknown-key error).
        """

        result = _runner.run_diagnostic(key, timeout_seconds=timeout_seconds)

        # This is the fix: results.record_run takes the whole
        # DiagnosticRunResult, not separate kwargs.
        _results.record_run(result)

        return result

    def run_all(
        self,
        timeout_seconds: Optional[float] = None,
    ) -> dict[str, DiagnosticRunResult]:
        """
        Run every registered diagnostic and persist each result
        independently, in registry order.

        A diagnostic that raises (e.g. an unexpected exception
        outside the timeout/interrupt handling already covered by
        runner.run_diagnostic) does not abort the remaining
        diagnostics or erase already-collected results.
        """

        results: dict[str, DiagnosticRunResult] = {}

        for spec in self.list_diagnostics():
            try:
                results[spec.key] = self.run_diagnostic(
                    spec.key,
                    timeout_seconds=timeout_seconds,
                )
            except Exception as exc:
                # Preserve visibility of the failure without
                # inventing a fake DiagnosticRunResult. The caller
                # (dashboard) is responsible for deciding how to
                # display a diagnostic that couldn't even start.
                results[spec.key] = exc  # type: ignore[assignment]

        return results

    # ------------------------------------------------------------------
    # RESULTS
    # ------------------------------------------------------------------

    def get_latest(self, key: str) -> Optional[DiagnosticRecord]:
        return _results.get_latest(key)

    def get_history(
        self, key: Optional[str] = None
    ) -> dict[str, list[DiagnosticRecord]]:
        """
        Return history for one diagnostic, or all diagnostics if
        `key` is None.

        Always returns a dict of key -> list[DiagnosticRecord] so
        callers don't need an if/else for the single-key case.
        """

        if key is not None:
            return {key: _results.get_history(key)}

        return {
            spec.key: _results.get_history(spec.key)
            for spec in self.list_diagnostics()
        }

    def get_status(self, key: str) -> DiagnosticStatus:
        """Canonical status for one diagnostic's latest run."""

        return classify(self.get_latest(key))

    # ------------------------------------------------------------------
    # HISTORY MANAGEMENT
    # ------------------------------------------------------------------

    def clear_history(self, key: Optional[str] = None) -> None:
        """Clear history for one diagnostic, or all if key is None."""

        if key is not None:
            _results.clear_history(key)
        else:
            _results.clear_all_history()

    # ------------------------------------------------------------------
    # STATUS SUMMARY
    # ------------------------------------------------------------------

    def get_backend_status(self) -> dict[str, int]:
        """
        Suite-wide counts: total / completed / successful / failed / pending.
        Delegates to results.get_status_summary() — not recomputed here.
        """

        return _results.get_status_summary()