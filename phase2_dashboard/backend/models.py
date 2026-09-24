"""
phase2_dashboard/backend/models.py

Typed status classification for Phase 2 diagnostic results.

DESIGN PRINCIPLE
----------------
runner.DiagnosticRunResult and results.DiagnosticRecord already
carry all real diagnostic data (returncode, stdout, stderr,
duration, timestamps). This module does NOT duplicate those
fields into a new data structure.

It only adds one thing that's currently implicit and
scattered across app.py: a single canonical status enum and
a function to compute it from either a fresh run result or a
persisted record.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional, Union

from phase2_dashboard.runner import DiagnosticRunResult
from phase2_dashboard.results import DiagnosticRecord


class DiagnosticStatus(str, Enum):
    NOT_RUN = "NOT_RUN"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    TIMED_OUT = "TIMED_OUT"
    INTERRUPTED = "INTERRUPTED"


# Return codes used by runner.py to signal special outcomes.
# See phase2_dashboard/runner.py: DiagnosticRunResult.timed_out / .interrupted
_TIMED_OUT_RETURNCODE = -1
_INTERRUPTED_RETURNCODE = -2


def classify(
    record: Optional[Union[DiagnosticRunResult, DiagnosticRecord]],
) -> DiagnosticStatus:
    """
    Classify a diagnostic outcome into a DiagnosticStatus.

    Accepts either:
      - a fresh DiagnosticRunResult (from runner.run_diagnostic), or
      - a persisted DiagnosticRecord (from results.get_latest/get_history)

    Both expose `returncode`, which is the single source of truth
    for status — runner.py already encodes timeout as -1 and
    interruption as -2 (see DiagnosticRunResult.timed_out /
    .interrupted). This function does not invent new semantics,
    it just reads the existing encoding in one place instead of
    several.
    """

    if record is None:
        return DiagnosticStatus.NOT_RUN

    returncode = getattr(record, "returncode", None)

    if returncode is None:
        return DiagnosticStatus.NOT_RUN

    if returncode == _TIMED_OUT_RETURNCODE:
        return DiagnosticStatus.TIMED_OUT

    if returncode == _INTERRUPTED_RETURNCODE:
        return DiagnosticStatus.INTERRUPTED

    if returncode == 0:
        return DiagnosticStatus.SUCCESS

    return DiagnosticStatus.FAILED