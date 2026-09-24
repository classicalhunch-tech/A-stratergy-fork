"""
phase2_dashboard/backend/formatter.py

Turns backend results into dashboard-friendly representations.

DESIGN PRINCIPLE
-----------------
This module formats information the backend already received.
It does NOT calculate new trading or research statistics, and it
does NOT parse diagnostic-specific numbers out of stdout (per the
spec: raw output stays raw unless a diagnostic already returns
structured data, which none currently do — see §12).
"""

from __future__ import annotations

from typing import Optional, Union

from phase2_dashboard.runner import DiagnosticRunResult
from phase2_dashboard.results import DiagnosticRecord

from .models import DiagnosticStatus, classify

_RecordLike = Union[DiagnosticRunResult, DiagnosticRecord]


def status_label(status: DiagnosticStatus) -> str:
    """Human-readable label for a DiagnosticStatus."""

    return {
        DiagnosticStatus.NOT_RUN: "NOT RUN",
        DiagnosticStatus.RUNNING: "RUNNING",
        DiagnosticStatus.SUCCESS: "SUCCESS",
        DiagnosticStatus.FAILED: "FAILED",
        DiagnosticStatus.TIMED_OUT: "TIMED OUT",
        DiagnosticStatus.INTERRUPTED: "INTERRUPTED",
    }[status]


def format_summary(
    key: str,
    label: str,
    record: Optional[_RecordLike],
) -> dict:
    """
    Build a small dict the dashboard can render directly for one
    diagnostic card: status, duration, and a preview of output.

    Returns "Not available" style placeholders when there is no
    record yet — never fabricated numbers (spec §31).
    """

    status = classify(record)

    if record is None:
        return {
            "key": key,
            "label": label,
            "status": status_label(status),
            "duration": "Not available",
            "output_preview": "Not available",
        }

    duration = getattr(record, "duration_seconds", None)
    duration_text = (
        f"{duration:.2f}s" if isinstance(duration, (int, float)) else "Not available"
    )

    output = getattr(record, "combined_output", None)
    if output is None:
        output = getattr(record, "output", "")

    preview_lines = output.splitlines()[-15:] if output else []
    preview = "\n".join(preview_lines) if preview_lines else "(no output)"

    return {
        "key": key,
        "label": label,
        "status": status_label(status),
        "duration": duration_text,
        "output_preview": preview,
    }


def format_history(
    key: str,
    label: str,
    records: list[DiagnosticRecord],
) -> list[dict]:
    """
    Format a diagnostic's full history, newest first, for a
    dashboard "History" view. Each entry mirrors format_summary's
    shape plus a timestamp, so the two render consistently.
    """

    formatted = []

    for record in reversed(records):
        entry = format_summary(key, label, record)
        entry["started_at_display"] = getattr(
            record, "started_at_display", "Unknown time"
        )
        formatted.append(entry)

    return formatted


def format_suite_summary(
    results: dict[str, object],
) -> dict:
    """
    Format the outcome of Phase2Service.run_all() into
    "N diagnostics / N successful / N failed / N timed out"
    counts, per spec §17. Never fabricates a result for a key
    that isn't present.
    """

    total = len(results)
    successful = 0
    failed = 0
    timed_out = 0
    interrupted = 0
    errored = 0

    for value in results.values():
        if isinstance(value, Exception):
            errored += 1
            continue

        status = classify(value)

        if status is DiagnosticStatus.SUCCESS:
            successful += 1
        elif status is DiagnosticStatus.TIMED_OUT:
            timed_out += 1
        elif status is DiagnosticStatus.INTERRUPTED:
            interrupted += 1
        else:
            failed += 1

    return {
        "total": total,
        "successful": successful,
        "failed": failed,
        "timed_out": timed_out,
        "interrupted": interrupted,
        "errored": errored,
    }