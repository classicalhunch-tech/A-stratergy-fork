"""
phase_02_optimization/test_adapter_restart_equivalence.py

PHASE 4 -- RESTART / RECOVERY EQUIVALENCE

Proves that a process crash, restart, or machine failure does not
change strategy behavior.

This test uses SECONDARY / REPLAY RECOVERY:

    1. A continuous StrategyAdapter processes candles 0..N.
    2. A separate adapter processes candles 0..K.
    3. That process is considered to crash after candle K.
    4. A genuinely FRESH StrategyAdapter is created.
    5. The fresh adapter replays candles 0..K using only information
       available at each historical candle.
    6. The recovered adapter continues from candle K+1..N.
    7. Reconstructed state and subsequent output must match the
       uninterrupted continuous run.

CRASH POINT CONVENTION
----------------------
crash_at = K = LAST candle successfully processed before the crash.

Therefore:

    Before crash : 0..K
    Replay       : 0..K
    Continue     : K+1..N

LEVELS
------
LEVEL 1 -- Event equivalence
    Continuous and recovered runs produce identical per-candle events.

LEVEL 2 -- State equivalence
    State immediately before the simulated crash equals the state
    reconstructed by the fresh adapter after replay.

LEVEL 3 -- Continuation equivalence
    After recovery, candles K+1..N produce identical per-candle output.

LEVEL 4 -- Repeated recovery
    The same crash scenario is executed repeatedly and must remain
    deterministic.

LEVEL 5 -- Full dataset
    The complete 5,000-candle dataset is tested at multiple deep
    recovery checkpoints.

FAILURE REPORTING
-----------------
Failures report:

    - dataset
    - crash point K
    - first divergent candle
    - expected events
    - actual events
    - replay or continuation phase
    - state fields that differ

The suite continues through remaining crash scenarios so that all
failures are visible in one run.

IMPORTANT
---------
This is a REPLAY-RECOVERY proof.

It does NOT implement durable persistence yet.

Persistent-state recovery from SQLite will be tested separately
later in Phase 4.

Run representative datasets:

    python -m phase_02_optimization.test_adapter_restart_equivalence

Run the full 5,000-candle dataset:

    python -m phase_02_optimization.test_adapter_restart_equivalence your_data_file.csv
"""

import sys

import numpy as np
import pandas as pd

from phase_03_paper.market.engine import Candle
from phase_03_paper.signals.adapter import StrategyAdapter


# ============================================================
# DATA LOADING
# ============================================================

def _load_csv(path: str) -> pd.DataFrame:
    """
    Load and normalize the dataset.

    Column-name normalization happens before required-column
    validation so column checks remain case-insensitive.
    """
    df = pd.read_csv(path)

    df.columns = [
        c.strip().lower()
        for c in df.columns
    ]

    if "timestamp" not in df.columns:
        raise ValueError(
            f"{path}: missing required 'timestamp' column"
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
    )

    return df


def _to_candles(df: pd.DataFrame):
    """
    Convert dataframe rows into Phase 3 Candle objects.
    """
    required = {
        "timestamp",
        "open",
        "high",
        "low",
        "close",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required candle columns: "
            f"{sorted(missing)}"
        )

    return [
        Candle(
            timestamp=row.timestamp,
            open=row.open,
            high=row.high,
            low=row.low,
            close=row.close,
        )
        for row in df.itertuples(index=False)
    ]


# ============================================================
# EVENT SIGNATURES -- LEVEL 1
# ============================================================

def _event_signature(e) -> tuple:
    """
    Build a strict deterministic event identity.
    """
    return (
        e.setup_bar_index,
        e.trigger_bar_index,
        round(e.fill_price, 8),
        round(e.initial_risk, 8),
        e.signal.signal_type,
    )


def _run_per_candle(
    adapter: StrategyAdapter,
    candles,
) -> list:
    """
    Process candles chronologically.

    One result entry is created for EVERY candle, preserving the
    exact timeline so the first divergence is precise.
    """
    out = []

    for candle in candles:
        events = adapter.on_candle(candle)

        out.append(
            {
                "events": [
                    _event_signature(event)
                    for event in events
                ]
            }
        )

    return out


def _first_divergence(
    expected: list,
    actual: list,
):
    """
    Find the first per-candle divergence.
    """
    limit = min(
        len(expected),
        len(actual),
    )

    for index in range(limit):
        if expected[index] != actual[index]:
            return (
                index,
                expected[index],
                actual[index],
            )

    if len(expected) != len(actual):
        index = limit

        expected_value = (
            expected[index]
            if index < len(expected)
            else None
        )

        actual_value = (
            actual[index]
            if index < len(actual)
            else None
        )

        return (
            index,
            expected_value,
            actual_value,
        )

    return None, None, None


# ============================================================
# STATE SNAPSHOT -- LEVEL 2
# ============================================================

def _snapshot_liquidity_level(liq) -> tuple:
    """Convert a LiquidityLevel object into a stable, comparable primitive tuple."""
    return (
        str(liq.liquidity_type),
        round(liq.price, 8),
        str(liq.formed_at),
        str(liq.status),
        liq.sweep_bar_index,
        str(liq.swept_at) if liq.swept_at is not None else None,
    )


def _snapshot_swing(swing) -> tuple:
    """Convert a Swing object into a stable, comparable primitive tuple."""
    return (
        str(swing.swing_type),
        round(swing.price, 8),
        str(swing.formed_at),
        str(swing.confirmed_at),
    )


def _snapshot_prepared_context(context) -> dict:
    """
    Convert prepared context into a deterministic comparison-safe
    representation.
    """
    if context is None:
        return {
            "present": False
        }

    snapshot = {
        "present": True
    }

    for key, value in context.items():

        if isinstance(value, pd.DataFrame):
            snapshot[key] = (
                "dataframe",
                value.to_dict("records"),
            )

        elif isinstance(value, np.ndarray):
            snapshot[key] = (
                "ndarray",
                value.tolist(),
            )

        elif isinstance(value, dict):
            snapshot[key] = (
                "dict",
                sorted(
                    value.items(),
                    key=lambda item: str(item[0]),
                ),
            )

        elif isinstance(value, set):
            snapshot[key] = (
                "set",
                sorted(
                    value,
                    key=str,
                ),
            )

        elif isinstance(value, list):
            snapshot[key] = (
                "list",
                list(value),
            )

        else:
            snapshot[key] = value

    return snapshot


def _snapshot(adapter: StrategyAdapter) -> dict:
    """
    Capture relevant internal StrategyAdapter state.
    """
    return {
        "candle_count": adapter.candle_count,
        "pending_count": adapter.pending_count,
        "errors": list(adapter.errors),

        "swing_rows_processed":
            adapter._swing_rows_processed,

        "confirmed_swings":
            list(adapter._confirmed_swings),

        "zones":
            list(adapter._zones),  # Update this similarly if zones contain objects

        "processed_break_keys":
            set(adapter._processed_break_keys),

        "liquidity_levels":
            sorted(
                [
                    _snapshot_liquidity_level(level)
                    for level in adapter._liquidity_levels
                ],
                key=str,
            ),

        "processed_swing_keys":
            set(adapter._processed_swing_keys),

        "pending_signals":
            list(adapter._pending_signals),

        "registered_setup_keys":
            set(adapter._registered_setup_keys),

        "consumed_zones":
            set(adapter._consumed_zones),

        "prepared_context":
            _snapshot_prepared_context(
                adapter._prepared_context
            ),
    }


def _diff_snapshot(
    before: dict,
    after: dict,
) -> list:
    """
    Return every state field whose reconstructed value differs.
    """
    differences = []

    all_keys = sorted(
        set(before) | set(after)
    )

    for key in all_keys:
        if before.get(key) != after.get(key):
            differences.append(key)

    return differences


# ============================================================
# LEVEL 1 + 2 + 3
# SINGLE CRASH SCENARIO
# ============================================================

def _run_crash_scenario(
    candles,
    crash_at: int,
    reference_per_candle: list,
):
    """
    Execute one crash/recovery scenario.
    """
    total_candles = len(candles)

    if crash_at < 0 or crash_at >= total_candles - 1:
        return (
            False,
            (
                f"crash_at={crash_at}: invalid crash point; "
                f"must satisfy 0 <= K < {total_candles - 1}"
            ),
        )

    pre_crash_candles = candles[:crash_at + 1]
    continuation_candles = candles[crash_at + 1:]

    reference_pre = reference_per_candle[:crash_at + 1]
    reference_post = reference_per_candle[crash_at + 1:]

    # Original adapter process
    adapter_a = StrategyAdapter()
    _run_per_candle(adapter_a, pre_crash_candles)
    snapshot_before = _snapshot(adapter_a)
    del adapter_a

    # Fresh process replay recovery
    adapter_b = StrategyAdapter()
    events_b_replay = _run_per_candle(adapter_b, pre_crash_candles)
    snapshot_after = _snapshot(adapter_b)

    # Level 1: Replay Event Equivalence
    idx, expected, actual = _first_divergence(reference_pre, events_b_replay)
    if idx is not None:
        return (
            False,
            (
                f"crash_at={crash_at}: replay divergence at candle {idx}; "
                f"phase=replay; expected={expected}; actual={actual}"
            ),
        )

    # Level 2: State Equivalence
    state_diffs = _diff_snapshot(snapshot_before, snapshot_after)
    if state_diffs:
        return (
            False,
            (
                f"crash_at={crash_at}: STATE MISMATCH after replay recovery; "
                f"fields={state_diffs}"
            ),
        )

    # Level 3: Continuation Equivalence
    events_b_continue = _run_per_candle(adapter_b, continuation_candles)
    idx, expected, actual = _first_divergence(reference_post, events_b_continue)
    if idx is not None:
        absolute_idx = crash_at + 1 + idx
        return (
            False,
            (
                f"crash_at={crash_at}: continuation divergence at candle "
                f"{absolute_idx}; phase=continuation; relative_idx={idx}; "
                f"expected={expected}; actual={actual}"
            ),
        )

    return (
        True,
        (
            f"crash_at={crash_at}: PASSED "
            f"(event + state + continuation equivalence)"
        ),
    )


# ============================================================
# LEVEL 4 -- REPEATED RECOVERY
# ============================================================

def _run_repeated_recovery(
    candles,
    crash_at: int,
    reference_per_candle: list,
    repetitions: int = 2,
):
    """
    Execute the same recovery scenario repeatedly.
    """
    for repetition in range(1, repetitions + 1):
        passed, report = _run_crash_scenario(
            candles,
            crash_at,
            reference_per_candle,
        )

        if not passed:
            return (
                False,
                (
                    f"crash_at={crash_at}: "
                    f"repeat {repetition} FAILED; {report}"
                ),
            )

    return (
        True,
        (
            f"crash_at={crash_at}: "
            f"LEVEL 4 repeated recovery PASSED "
            f"({repetitions} independent runs)"
        ),
    )


# ============================================================
# DATASET
# ============================================================

def _run_dataset(
    path: str,
    label: str,
    crash_points: list,
) -> bool:
    """
    Run restart/recovery equivalence across one dataset.
    """
    try:
        df = _load_csv(path)
        candles = _to_candles(df)
    except Exception as exc:
        print(f"[FAIL] Could not load dataset {path}: {exc}")
        return False

    if not candles:
        print(f"[FAIL] Dataset {path} contains zero candles.")
        return False

    reference_adapter = StrategyAdapter()
    reference_per_candle = _run_per_candle(reference_adapter, candles)

    total_reference_events = sum(
        len(entry["events"])
        for entry in reference_per_candle
    )

    print("=" * 70)
    print(f"DATASET: {label}")
    print(f"FILE: {path}")
    print(f"Candles: {len(candles):,}")
    print(f"Reference triggered events: {total_reference_events}")
    print("=" * 70)

    all_passed = True
    valid_crash_points = []

    for crash_at in crash_points:
        if 0 <= crash_at < len(candles) - 1:
            if crash_at not in valid_crash_points:
                valid_crash_points.append(crash_at)

    if not valid_crash_points:
        print("[FAIL] No valid crash points available for this dataset length.")
        return False

    for crash_at in valid_crash_points:
        print(f"--- Crash Point K={crash_at:,} ---")

        passed, report = _run_crash_scenario(
            candles,
            crash_at,
            reference_per_candle,
        )

        tag = "[PASS]" if passed else "[FAIL]"
        print(f"{tag} {report}")

        if not passed:
            all_passed = False

        if passed:
            repeat_passed, repeat_report = _run_repeated_recovery(
                candles,
                crash_at,
                reference_per_candle,
                repetitions=2,
            )

            repeat_tag = "[PASS]" if repeat_passed else "[FAIL]"
            print(f"{repeat_tag} {repeat_report}")

            if not repeat_passed:
                all_passed = False

        print()

    return all_passed


# ============================================================
# MAIN
# ============================================================

def main():
    target_csv = (
        sys.argv[1]
        if len(sys.argv) > 1
        else None
    )

    all_passed = True

    datasets = [
        (
            "your_data_file_1000.csv",
            "LEVEL 1-4 / 1,000 candles",
            [100, 250, 500, 750, 900],
        ),
        (
            "your_data_file_2000.csv",
            "LEVEL 1-4 / 2,000 candles",
            [250, 500, 1000, 1500, 1750],
        ),
    ]

    if target_csv is not None:
        # Dynamically scale checkpoint boundaries if custom dataset length differs
        try:
            temp_df = _load_csv(target_csv)
            n = len(temp_df)
            dynamic_points = [
                int(n * 0.1),
                int(n * 0.3),
                int(n * 0.5),
                int(n * 0.7),
                int(n * 0.85),
                int(n * 0.95),
            ]
        except Exception:
            dynamic_points = [500, 1000, 2000, 3000, 4000, 4500]

        datasets = [
            (
                target_csv,
                f"LEVEL 5 / FULL TARGET DATASET",
                dynamic_points,
            )
        ]

    for path, label, crash_points in datasets:
        passed = _run_dataset(path, label, crash_points)
        all_passed = all_passed and passed

    print("=" * 70)

    if all_passed:
        if target_csv is None:
            print("ALL RESTART / RECOVERY TESTS PASSED.")
            print()
            print("LEVELS 1-4 validated on 1,000 and 2,000 candle datasets.")
            print()
            print("NEXT: Run LEVEL 5 on the full 5,000-candle source dataset:")
            print("    python -m phase_02_optimization.test_adapter_restart_equivalence your_data_file.csv")
        else:
            print("LEVEL 5 FULL DATASET RESTART / RECOVERY TEST PASSED.")
            print()
            print("Replay recovery is behaviorally equivalent across all tested full-dataset crash points.")
    else:
        print("SOME RESTART / RECOVERY TESTS FAILED.")
        print("Replay recovery must NOT be trusted until the reported failures are resolved.")

    print("=" * 70)


if __name__ == "__main__":
    main()