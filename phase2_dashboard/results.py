"""
phase2_dashboard/results.py

Persistent storage for Phase 2 diagnostic run results.

Purpose
-------
Streamlit reruns the application frequently, and session_state
does not survive a complete application restart.

This module therefore persists diagnostic run outcomes to:

    phase2_dashboard/phase2_results.json

The stored status allows the Phase 2 dashboard to show:

    ⏳  Diagnostic has not been run
    ✅  Most recent run succeeded
    ❌  Most recent run failed

Design principles
-----------------
This module does NOT:

    - modify strategy logic
    - modify the canonical backtest
    - modify Phase 1
    - modify phase_02_optimization modules
    - modify diagnostic caches
    - recalculate diagnostic statistics

The existing Phase 2 diagnostic modules remain the source of
truth.

This module only stores and retrieves the results produced by
phase2_dashboard.runner.run_diagnostic().
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from phase2_dashboard.runner import (
    DIAGNOSTICS,
    DiagnosticRunResult,
)


# ============================================================
# STORAGE CONFIGURATION
# ============================================================

RESULTS_FILE = (
    Path(__file__).resolve().parent
    / "phase2_results.json"
)

# Keep a limited history for each diagnostic.
#
# The most recent runs are retained.
# Older runs are automatically discarded.
MAX_HISTORY_PER_KEY = 20


# ============================================================
# PERSISTED RECORD
# ============================================================

@dataclass
class DiagnosticRecord:
    """
    Persisted representation of one diagnostic run.
    """

    key: str
    label: str
    command: str
    returncode: int
    succeeded: bool
    duration_seconds: float
    started_at: float
    output: str

    @property
    def started_at_display(self) -> str:
        """
        Return the run timestamp in local time.
        """

        return time.strftime(
            "%Y-%m-%d %H:%M:%S",
            time.localtime(self.started_at),
        )


# ============================================================
# EMPTY STORE
# ============================================================

def _empty_store() -> dict[str, list[dict]]:
    """
    Return a fresh empty store containing every registered
    Phase 2 diagnostic.
    """

    return {
        key: []
        for key in DIAGNOSTICS
    }


# ============================================================
# LOAD STORE
# ============================================================

def load_all() -> dict[str, list[dict]]:
    """
    Load all persisted diagnostic histories.

    A missing, invalid, or corrupt results file is treated as
    an empty store so that dashboard startup never fails merely
    because there is no previous history.
    """

    if not RESULTS_FILE.exists():
        return _empty_store()

    try:
        with RESULTS_FILE.open(
            "r",
            encoding="utf-8",
        ) as file:

            data = json.load(file)

    except (
        OSError,
        json.JSONDecodeError,
        TypeError,
        ValueError,
    ):
        return _empty_store()

    if not isinstance(data, dict):
        return _empty_store()

    store = _empty_store()

    for key, entries in data.items():

        if (
            key in store
            and isinstance(entries, list)
        ):
            store[key] = entries

    return store


# ============================================================
# SAVE STORE
# ============================================================

def _save_all(
    store: dict[str, list[dict]],
) -> None:
    """
    Atomically save the complete diagnostic history.

    The temporary file prevents a partially written JSON file
    from replacing the existing results file.
    """

    RESULTS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path = RESULTS_FILE.with_suffix(
        ".tmp"
    )

    with temp_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            store,
            file,
            indent=2,
        )

        file.write("\n")

    temp_path.replace(
        RESULTS_FILE
    )


# ============================================================
# RECORD RUN
# ============================================================

def record_run(
    result: DiagnosticRunResult,
) -> DiagnosticRecord:
    """
    Persist one completed diagnostic run.

    Parameters
    ----------
    result:
        DiagnosticRunResult returned by runner.run_diagnostic().

    Returns
    -------
    DiagnosticRecord
        The record that was persisted.
    """

    record = DiagnosticRecord(
        key=result.key,
        label=result.label,
        command=result.command,
        returncode=result.returncode,
        succeeded=result.succeeded,
        duration_seconds=result.duration_seconds,
        started_at=result.started_at,
        output=result.combined_output,
    )

    store = load_all()

    history = store.setdefault(
        record.key,
        [],
    )

    history.append(
        asdict(record)
    )

    # Keep only the newest records.
    store[record.key] = history[
        -MAX_HISTORY_PER_KEY:
    ]

    _save_all(store)

    return record


# ============================================================
# CONVERT RAW RECORD
# ============================================================

def _record_from_dict(
    entry: object,
) -> Optional[DiagnosticRecord]:
    """
    Safely convert one raw JSON object into a DiagnosticRecord.

    Invalid or stale records are ignored instead of crashing
    the dashboard.
    """

    if not isinstance(entry, dict):
        return None

    try:
        return DiagnosticRecord(
            key=entry["key"],
            label=entry["label"],
            command=entry["command"],
            returncode=int(
                entry["returncode"]
            ),
            succeeded=bool(
                entry["succeeded"]
            ),
            duration_seconds=float(
                entry["duration_seconds"]
            ),
            started_at=float(
                entry["started_at"]
            ),
            output=str(
                entry["output"]
            ),
        )

    except (
        KeyError,
        TypeError,
        ValueError,
    ):
        return None


# ============================================================
# GET HISTORY
# ============================================================

def get_history(
    key: str,
) -> list[DiagnosticRecord]:
    """
    Return persisted history for one diagnostic.

    Results are ordered oldest to newest.

    Unknown or never-run diagnostics return an empty list.
    """

    store = load_all()

    raw_history = store.get(
        key,
        [],
    )

    records: list[DiagnosticRecord] = []

    for entry in raw_history:

        record = _record_from_dict(
            entry
        )

        if record is not None:
            records.append(record)

    return records


# ============================================================
# GET LATEST
# ============================================================

def get_latest(
    key: str,
) -> Optional[DiagnosticRecord]:
    """
    Return the most recent valid run.

    Returns None if no valid run exists.
    """

    history = get_history(key)

    if not history:
        return None

    return history[-1]


# ============================================================
# STATUS
# ============================================================

def status_icon(
    key: str,
) -> str:
    """
    Return the dashboard roadmap status.

    ⏳ = never run
    ✅ = most recent run succeeded
    ❌ = most recent run failed
    """

    latest = get_latest(key)

    if latest is None:
        return "⏳"

    if latest.succeeded:
        return "✅"

    return "❌"


# ============================================================
# CLEAR HISTORY
# ============================================================

def clear_history(
    key: str,
) -> None:
    """
    Clear dashboard history for one diagnostic.

    This function only removes the dashboard's persisted run
    history.

    It does NOT delete or modify:

        - phase_02_optimization caches
        - PBO caches
        - strategy data
        - backtest results
        - diagnostic source files
    """

    if key not in DIAGNOSTICS:
        raise KeyError(
            f"Unknown diagnostic key: {key!r}. "
            f"Known keys: {sorted(DIAGNOSTICS)}"
        )

    store = load_all()

    store[key] = []

    _save_all(store)


# ============================================================
# CLEAR ALL DASHBOARD HISTORY
# ============================================================

def clear_all_history() -> None:
    """
    Clear all dashboard diagnostic run history.

    This affects only:

        phase2_dashboard/phase2_results.json

    It does not touch any Phase 2 diagnostic cache.
    """

    _save_all(
        _empty_store()
    )


# ============================================================
# DASHBOARD SUMMARY
# ============================================================

def get_status_summary() -> dict[str, int]:
    """
    Return a simple count of diagnostic statuses.

    Returns
    -------
    dict
        Keys:

            total
            completed
            successful
            failed
            pending
    """

    total = len(DIAGNOSTICS)

    successful = 0
    failed = 0
    pending = 0

    for key in DIAGNOSTICS:

        latest = get_latest(key)

        if latest is None:
            pending += 1

        elif latest.succeeded:
            successful += 1

        else:
            failed += 1

    return {
        "total": total,
        "completed": successful + failed,
        "successful": successful,
        "failed": failed,
        "pending": pending,
    }


# ============================================================
# MANUAL SMOKE TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 80)
    print("PHASE 2 RESULTS STORE")
    print("=" * 80)

    print()
    print("Results file:")
    print(f"  {RESULTS_FILE}")

    print()
    print("Registered diagnostics:")
    print(
        f"  {len(DIAGNOSTICS)}"
    )

    print()
    print("Current diagnostic status:")
    print()

    for key, spec in DIAGNOSTICS.items():

        icon = status_icon(
            key
        )

        latest = get_latest(
            key
        )

        if latest is None:

            when = "never run"
            runs = 0

        else:

            when = (
                latest.started_at_display
            )

            runs = len(
                get_history(key)
            )

        print(
            f"  {icon}  "
            f"{spec.label:<32} "
            f"last: {when:<20} "
            f"(runs: {runs})"
        )

    summary = get_status_summary()

    print()
    print("=" * 80)
    print("STATUS SUMMARY")
    print("=" * 80)

    print(
        f"Total       : {summary['total']}"
    )
    print(
        f"Completed   : {summary['completed']}"
    )
    print(
        f"Successful  : {summary['successful']}"
    )
    print(
        f"Failed      : {summary['failed']}"
    )
    print(
        f"Pending     : {summary['pending']}"
    )

    print()
    print("Results store smoke test complete.")