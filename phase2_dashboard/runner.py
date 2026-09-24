"""
phase2_dashboard/runner.py

Phase 2 diagnostic runner.

Runs the existing phase_02_optimization modules as subprocesses
and captures their stdout/stderr for display in the Phase 2
Streamlit dashboard.

DESIGN PRINCIPLE
----------------
The existing Phase 2 diagnostic modules are the source of truth.

This runner does NOT:
    - reimplement diagnostic logic
    - modify strategy logic
    - modify the canonical backtest
    - import diagnostic internals
    - alter existing diagnostic caches

Instead, it runs each diagnostic the same way it is currently
run manually from PowerShell:

    python -m phase_02_optimization.<module_name> [args...]

The first dashboard version displays the complete diagnostic
output as text.

Structured result parsing can be added later in results.py.
"""

from __future__ import annotations

import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path


# ============================================================
# PROJECT ROOT
# ============================================================

# runner.py:
#
#     <project_root>/phase2_dashboard/runner.py
#
# parents[1] = <project_root>

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ============================================================
# DEFAULT RUNTIME SAFETY
# ============================================================

# Diagnostics are allowed to run for a long time because some
# Phase 2 tests perform many backtests.
#
# This is a safety limit, not a diagnostic rule.
#
# None means no automatic timeout.
DEFAULT_TIMEOUT_SECONDS: float | None = None


# ============================================================
# DIAGNOSTIC SPECIFICATION
# ============================================================

@dataclass(frozen=True)
class DiagnosticSpec:
    """
    Defines one Phase 2 diagnostic.
    """

    key: str
    label: str
    module: str
    args: list[str] = field(default_factory=list)
    description: str = ""


# ============================================================
# DIAGNOSTIC REGISTRY
# ============================================================

DIAGNOSTICS: dict[str, DiagnosticSpec] = {
    spec.key: spec
    for spec in [
        DiagnosticSpec(
            key="monte_carlo",
            label="Monte Carlo",
            module="phase_02_optimization.monte_carlo",
            description="Bootstrap and shuffle robustness analysis.",
        ),
        DiagnosticSpec(
            key="warmup_probe",
            label="Warmup Sensitivity",
            module="phase_02_optimization.warmup_probe",
            description="Tests behavior across different warmup lengths.",
        ),
        DiagnosticSpec(
            key="walk_forward",
            label="Walk-Forward",
            module="phase_02_optimization.walk_forward",
            description="Chronological walk-forward validation.",
        ),
        DiagnosticSpec(
            key="parameter_sensitivity",
            label="Parameter Sensitivity",
            module="phase_02_optimization.parameter_sensitivity",
            description="Tests sensitivity to key strategy parameters.",
        ),
        DiagnosticSpec(
            key="stop_buffer_diagnostic",
            label="Stop-Buffer Diagnostic",
            module="phase_02_optimization.stop_buffer_diagnostic",
            description="Tests stop-buffer sensitivity and trade flips.",
        ),
        DiagnosticSpec(
            key="spread_sensitivity",
            label="Transaction-Cost Sensitivity",
            module="phase_02_optimization.spread_sensitivity",
            description="Tests performance under increasing transaction costs.",
        ),
        DiagnosticSpec(
            key="trial_registry",
            label="Trial Registry",
            module="phase_02_optimization.trial_registry",
            description="Builds and validates the Phase 2 trial universe.",
        ),
        DiagnosticSpec(
            key="overfit_detection",
            label="DSR + PBO",
            module="phase_02_optimization.overfit_detection",
            description="Primary DSR and PBO using 10 CSCV blocks.",
        ),
        DiagnosticSpec(
            key="pbo_no_confounds",
            label="PBO Sanity: No Confounds",
            module="phase_02_optimization.overfit_sanity_no_confounds",
            description="PBO after removing known-confounded trials.",
        ),
        DiagnosticSpec(
            key="pbo_blocks_8",
            label="PBO Sanity: 8 Blocks",
            module="phase_02_optimization.overfit_sanity_blocks",
            args=["8"],
            description="PBO using 8 chronological CSCV blocks.",
        ),
    ]
}


# ============================================================
# RUN RESULT
# ============================================================

@dataclass
class DiagnosticRunResult:
    """
    Result returned after running one diagnostic.
    """

    key: str
    label: str
    command: str
    returncode: int
    stdout: str
    stderr: str
    duration_seconds: float
    started_at: float

    @property
    def succeeded(self) -> bool:
        """Return True when the diagnostic exited successfully."""

        return self.returncode == 0

    @property
    def interrupted(self) -> bool:
        """Return True when the diagnostic was interrupted."""

        return self.returncode == -2

    @property
    def timed_out(self) -> bool:
        """Return True when the diagnostic exceeded its timeout."""

        return self.returncode == -1

    @property
    def combined_output(self) -> str:
        """
        Return stdout and stderr as one displayable string.
        """

        if self.stderr.strip():
            return (
                self.stdout
                + "\n\n--- STDERR ---\n"
                + self.stderr
            )

        return self.stdout


# ============================================================
# RUN ONE DIAGNOSTIC
# ============================================================

def run_diagnostic(
    key: str,
    timeout_seconds: float | None = DEFAULT_TIMEOUT_SECONDS,
) -> DiagnosticRunResult:
    """
    Run one registered Phase 2 diagnostic.

    Parameters
    ----------
    key:
        Registered diagnostic key.

    timeout_seconds:
        Optional maximum runtime.

        None means no automatic timeout.

    Returns
    -------
    DiagnosticRunResult

    Raises
    ------
    KeyError
        If the diagnostic key is not registered.
    """

    if key not in DIAGNOSTICS:
        raise KeyError(
            f"Unknown diagnostic key: {key!r}. "
            f"Known keys: {sorted(DIAGNOSTICS)}"
        )

    spec = DIAGNOSTICS[key]

    command = [
        sys.executable,
        "-m",
        spec.module,
        *spec.args,
    ]

    command_str = " ".join(command)

    started_at = time.time()

    try:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )

        duration = time.time() - started_at

        return DiagnosticRunResult(
            key=spec.key,
            label=spec.label,
            command=command_str,
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            duration_seconds=duration,
            started_at=started_at,
        )

    except subprocess.TimeoutExpired as exc:
        duration = time.time() - started_at

        stdout = exc.stdout or ""
        stderr = exc.stderr or ""

        if not isinstance(stdout, str):
            stdout = stdout.decode(
                errors="replace"
            )

        if not isinstance(stderr, str):
            stderr = stderr.decode(
                errors="replace"
            )

        stderr += (
            "\n[runner] Diagnostic timed out."
            f" Timeout limit: {timeout_seconds}s."
        )

        return DiagnosticRunResult(
            key=spec.key,
            label=spec.label,
            command=command_str,
            returncode=-1,
            stdout=stdout,
            stderr=stderr,
            duration_seconds=duration,
            started_at=started_at,
        )

    except KeyboardInterrupt:
        duration = time.time() - started_at

        return DiagnosticRunResult(
            key=spec.key,
            label=spec.label,
            command=command_str,
            returncode=-2,
            stdout="",
            stderr=(
                "\n[runner] Diagnostic interrupted "
                "by the user."
            ),
            duration_seconds=duration,
            started_at=started_at,
        )


# ============================================================
# RUN ALL DIAGNOSTICS
# ============================================================

def run_all_diagnostics(
    timeout_seconds: float | None = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, DiagnosticRunResult]:
    """
    Run every registered Phase 2 diagnostic in registry order.

    Thin wrapper around run_diagnostic(). Does not reimplement,
    modify, or bypass any existing diagnostic-running logic —
    it simply calls run_diagnostic() once per registered spec
    and collects the results into a dict, matching the shape
    the Phase 2 dashboard's "Run Complete Suite" action expects
    (dict[key] -> DiagnosticRunResult).

    Parameters
    ----------
    timeout_seconds:
        Optional maximum runtime PER diagnostic.

        None means no automatic timeout (matches run_diagnostic's
        default behavior).

    Returns
    -------
    dict[str, DiagnosticRunResult]
        Mapping of diagnostic key -> its DiagnosticRunResult,
        in registry order.
    """

    results: dict[str, DiagnosticRunResult] = {}

    for spec in list_diagnostics():
        results[spec.key] = run_diagnostic(
            spec.key,
            timeout_seconds=timeout_seconds,
        )

    return results


# ============================================================
# LIST DIAGNOSTICS
# ============================================================

def list_diagnostics() -> list[DiagnosticSpec]:
    """
    Return all registered diagnostics in dashboard order.
    """

    return list(DIAGNOSTICS.values())


# ============================================================
# MANUAL SMOKE TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 80)
    print("PHASE 2 DIAGNOSTIC RUNNER")
    print("=" * 80)

    print()
    print("Project root:")
    print(f"  {PROJECT_ROOT}")

    print()
    print("Available diagnostics:")

    for spec in list_diagnostics():
        print(
            f"  {spec.key:<24} -> {spec.label}"
        )

    print()
    print("=" * 80)
    print("SMOKE TEST")
    print("=" * 80)

    print()
    print("Testing 'trial_registry'.")

    result = run_diagnostic(
        "trial_registry"
    )

    print()
    print(
        f"Return code : {result.returncode}"
    )
    print(
        f"Duration    : {result.duration_seconds:.2f}s"
    )
    print(
        f"Succeeded   : {result.succeeded}"
    )
    print(
        f"Timed out   : {result.timed_out}"
    )
    print(
        f"Interrupted : {result.interrupted}"
    )

    print()
    print("--- Last 20 lines of output ---")

    output_lines = result.combined_output.splitlines()

    if output_lines:
        print(
            "\n".join(output_lines[-20:])
        )
    else:
        print("(no output)")