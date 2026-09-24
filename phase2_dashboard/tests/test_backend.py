"""
phase2_dashboard/tests/test_backend.py

Tests for phase2_dashboard.backend.Phase2Service.

These tests isolate persistence to a temporary results file
(monkeypatching phase2_dashboard.results.RESULTS_FILE) so they
never touch the real phase2_results.json.

Only "test_single_diagnostic_walk_forward" runs the real
walk_forward subprocess, per spec section 24 Test 2. Every other
test monkeypatches phase2_dashboard.runner.run_diagnostic with a
fast fake so the suite stays quick and deterministic.
"""

from __future__ import annotations

import time

import pytest

from phase2_dashboard import results as results_module
from phase2_dashboard import runner as runner_module
from phase2_dashboard.backend import DiagnosticStatus, Phase2Service, formatter
from phase2_dashboard.runner import DiagnosticRunResult


@pytest.fixture(autouse=True)
def isolated_results_file(tmp_path, monkeypatch):
    """Redirect all persistence to a throwaway file for every test."""

    fake_path = tmp_path / "phase2_results_test.json"
    monkeypatch.setattr(results_module, "RESULTS_FILE", fake_path)
    yield fake_path


@pytest.fixture
def service():
    return Phase2Service()


def _fake_result(
    key: str,
    label: str,
    returncode: int = 0,
    stdout: str = "fake output",
    stderr: str = "",
) -> DiagnosticRunResult:
    now = time.time()
    return DiagnosticRunResult(
        key=key,
        label=label,
        command=f"python -m fake.{key}",
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
        duration_seconds=0.01,
        started_at=now,
    )


# ============================================================
# TEST 1 — DIAGNOSTIC DISCOVERY
# ============================================================

def test_discovery_lists_ten_diagnostics(service):
    keys = service.get_available_diagnostics()

    assert len(keys) == 10
    assert "walk_forward" in keys
    assert "monte_carlo" in keys
    assert "pbo_blocks_8" in keys


# ============================================================
# TEST 2 — SINGLE DIAGNOSTIC (real subprocess, per spec)
# ============================================================

def test_single_diagnostic_walk_forward(service):
    result = service.run_diagnostic("walk_forward")

    assert result.succeeded is True
    assert result.returncode == 0
    assert result.stdout.strip() != ""

    latest = service.get_latest("walk_forward")
    assert latest is not None
    assert latest.succeeded is True

    history = service.get_history("walk_forward")["walk_forward"]
    assert len(history) == 1


# ============================================================
# TEST 3 — FAILURE HANDLING
# ============================================================

def test_failure_is_represented_and_persisted(service, monkeypatch):
    def fake_run_diagnostic(key, timeout_seconds=None):
        return _fake_result(
            key,
            "Fake Diagnostic",
            returncode=1,
            stdout="partial output before crash",
            stderr="Traceback: something broke",
        )

    monkeypatch.setattr(runner_module, "run_diagnostic", fake_run_diagnostic)

    result = service.run_diagnostic("monte_carlo")

    assert result.succeeded is False
    assert result.returncode == 1
    assert "something broke" in result.stderr

    latest = service.get_latest("monte_carlo")
    assert latest is not None
    assert latest.succeeded is False
    assert "something broke" in latest.output

    status = service.get_status("monte_carlo")
    assert status is DiagnosticStatus.FAILED

    # Dashboard-facing formatter must not crash on a failed record.
    summary = formatter.format_summary("monte_carlo", "Monte Carlo", latest)
    assert summary["status"] == "FAILED"


# ============================================================
# TEST 4 — PERSISTENCE (run twice)
# ============================================================

def test_running_twice_keeps_both_records(service, monkeypatch):
    call_count = {"n": 0}

    def fake_run_diagnostic(key, timeout_seconds=None):
        call_count["n"] += 1
        return _fake_result(key, "Fake", stdout=f"run number {call_count['n']}")

    monkeypatch.setattr(runner_module, "run_diagnostic", fake_run_diagnostic)

    first = service.run_diagnostic("warmup_probe")
    second = service.run_diagnostic("warmup_probe")

    history = service.get_history("warmup_probe")["warmup_probe"]
    assert len(history) == 2

    latest = service.get_latest("warmup_probe")
    assert latest.output == second.combined_output

    # The older record is still accessible, not overwritten.
    assert history[0].output == first.combined_output
    assert history[1].output == second.combined_output


# ============================================================
# TEST 5 — RUN ALL
# ============================================================

def test_run_all_survives_one_failure(service, monkeypatch):
    def fake_run_diagnostic(key, timeout_seconds=None):
        if key == "spread_sensitivity":
            raise RuntimeError("simulated crash before subprocess even started")

        if key == "trial_registry":
            return _fake_result(key, "Fake", returncode=1, stderr="known failure")

        return _fake_result(key, "Fake", returncode=0)

    monkeypatch.setattr(runner_module, "run_diagnostic", fake_run_diagnostic)

    results = service.run_all()

    assert len(results) == 10

    # The raised exception is captured, not propagated, and does not
    # erase the other nine results.
    assert isinstance(results["spread_sensitivity"], RuntimeError)

    assert results["trial_registry"].succeeded is False

    successful_keys = [
        key
        for key, value in results.items()
        if not isinstance(value, Exception) and value.succeeded
    ]
    assert len(successful_keys) == 8  # 10 - 1 raised - 1 failed

    counts = formatter.format_suite_summary(results)
    assert counts["total"] == 10
    assert counts["successful"] == 8
    assert counts["failed"] == 1
    assert counts["errored"] == 1

    # Successful and failed diagnostics were persisted; the one that
    # raised before returning a result was not (nothing to persist).
    assert service.get_latest("monte_carlo") is not None
    assert service.get_latest("trial_registry") is not None
    assert service.get_latest("spread_sensitivity") is None


# ============================================================
# TEST 6 — CLEAR HISTORY
# ============================================================

def test_clear_history_removes_all_records(service, monkeypatch):
    def fake_run_diagnostic(key, timeout_seconds=None):
        return _fake_result(key, "Fake")

    monkeypatch.setattr(runner_module, "run_diagnostic", fake_run_diagnostic)

    service.run_diagnostic("monte_carlo")
    service.run_diagnostic("walk_forward")

    assert service.get_latest("monte_carlo") is not None
    assert service.get_latest("walk_forward") is not None

    service.clear_history()

    assert service.get_latest("monte_carlo") is None
    assert service.get_latest("walk_forward") is None


def test_clear_history_single_key_leaves_others(service, monkeypatch):
    def fake_run_diagnostic(key, timeout_seconds=None):
        return _fake_result(key, "Fake")

    monkeypatch.setattr(runner_module, "run_diagnostic", fake_run_diagnostic)

    service.run_diagnostic("monte_carlo")
    service.run_diagnostic("walk_forward")

    service.clear_history("monte_carlo")

    assert service.get_latest("monte_carlo") is None
    assert service.get_latest("walk_forward") is not None


# ============================================================
# STATUS CLASSIFICATION
# ============================================================

def test_status_not_run_when_never_executed(service):
    assert service.get_status("pbo_blocks_8") is DiagnosticStatus.NOT_RUN


def test_status_timed_out_and_interrupted_are_distinguished(
    service, monkeypatch
):
    def fake_run_diagnostic(key, timeout_seconds=None):
        if key == "monte_carlo":
            return _fake_result(key, "Fake", returncode=-1)  # timeout
        return _fake_result(key, "Fake", returncode=-2)  # interrupted

    monkeypatch.setattr(runner_module, "run_diagnostic", fake_run_diagnostic)

    service.run_diagnostic("monte_carlo")
    service.run_diagnostic("walk_forward")

    assert service.get_status("monte_carlo") is DiagnosticStatus.TIMED_OUT
    assert service.get_status("walk_forward") is DiagnosticStatus.INTERRUPTED