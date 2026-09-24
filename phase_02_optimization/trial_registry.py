"""
phase_02_optimization/trial_registry.py

Build the exact primary trial registry used by Phase 2 DSR/PBO
overfit diagnostics.

PURPOSE
-------
DSR (Deflated Sharpe Ratio) and PBO (Probability of Backtest
Overfitting) must be based on a clearly defined history of trials.

This module reconstructs the PRIMARY parameter-sensitivity trials
that were actually evaluated in Phase 2.

It deliberately does NOT invent a Cartesian grid.

Phase 2 parameter sensitivity used a one-at-a-time design:

    one parameter changes
    all other parameters remain at baseline

This module reproduces that design exactly.

PRIMARY TRIAL UNIVERSE
----------------------
The primary DSR/PBO registry includes every parameter-sensitivity axis
that was run under that same one-at-a-time methodology -- Parameters
A, B, C, and D from the Phase 2 record, with no axis singled out for
exclusion:

    max_bars_to_retest (Parameter A):
        [10, 15, 20, 25, 30]

    reward_multiple (Parameter B):
        [1.5, 1.75, 2.0, 2.25, 2.5]

    stop_buffer (Parameter C):
        [0.0000, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005]

    entry_mode (Parameter D):
        ["midpoint", "extreme"]

The baseline:

    max_bars_to_retest = 20
    reward_multiple    = 2.0
    stop_buffer        = 0.0
    entry_mode         = "midpoint"

is included exactly once.

WHY STOP_BUFFER IS INCLUDED HERE (unlike an earlier draft of this file)
------------------------------------------------------------------------
An earlier version of this registry excluded stop_buffer on the theory
that its later diagnostic (stop_buffer_diagnostic.py, the WIN/LOSS flip
analysis) was a separate investigation rather than part of the original
search. That reasoning conflated two different things:

    1. The original stop_buffer sweep (0.0000 - 0.0005, six values) --
       this WAS run with the same one-at-a-time methodology as A and B,
       at the same time, as part of the same Parameter Sensitivity
       section (A/B/C/D) in the Phase 2 record.

    2. The later deep-dive diagnostic that investigated WHY the sweep
       looked unstable (trade-by-trade WIN->LOSS flip analysis).

Only (2) is a genuinely separate, later investigation. (1) is a primary
trial axis exactly like A and B, and excluding it here would have had
the specific effect of removing the one axis that showed instability
from the very correction (DSR) whose purpose is to penalize exactly
that kind of multiple-testing risk. That is the wrong direction for a
robustness diagnostic to err in, so stop_buffer's original six swept
values are included as primary trials below. The deeper flip-analysis
investigation remains correctly excluded (see EXCLUDED_DIAGNOSTICS).

KNOWN CONFOUND: entry_mode="extreme"
-------------------------------------
Per the Phase 2 record, when entry_mode="extreme" and stop_buffer <= 0,
the signal engine internally forces stop_buffer=0.01. This means the
entry_mode="extreme" trial is NOT a clean single-parameter change from
baseline -- it also silently changes the effective stop_buffer. This
trial is flagged with a non-null `confound` field so a later DSR/PBO
module (or a human reading the registry) does not mistake it for an
isolated single-variable trial.

TRANSACTION-COST DIAGNOSTIC
---------------------------
The transaction-cost sensitivity module is excluded from the primary
DSR/PBO trial universe. Those costs are stress assumptions applied to
completed trades after the fact; they are not alternative strategy
configurations selected during the parameter search.

WARMUP / WINDOW / STEP
----------------------
Walk-forward sensitivity is also kept separate. Those experiments
answer a different question ("how stable is performance across
temporal validation choices?") and were not part of the parameter-
search history in the same sense as A/B/C/D.

IMPORTANT DATASET LIMITATION
----------------------------
The current dataset is confirmed synthetic M5 EURUSD-like random-walk
data. Therefore DSR/PBO results from this registry are methodological
diagnostics of the current experiment. They do NOT establish real-
market profitability, live trading performance, broker execution
quality, or statistical proof of a real trading edge. The baseline
sample is also small, with only 16 closed trades.

METHODOLOGY
-----------
For each primary configuration:

    1. Run canonical strategy.backtest.run_backtest().
    2. Keep only closed WIN/LOSS trades.
    3. Extract the per-trade R-multiple sequence.
    4. Calculate expectancy.
    5. Calculate sample standard deviation.
    6. Calculate a consistent Sharpe-like statistic:

           mean(R) / sample_std(R)

The per-trade R sequence is retained because PBO needs the actual
return sequence rather than only summary statistics.

NO STRATEGY CHANGES
-------------------
This module:

    - does not modify strategy logic;
    - does not optimize parameters;
    - does not select a winning configuration;
    - does not alter the canonical backtest;
    - does not simulate broker execution.

It is read-only analysis.

OUTPUT
------
Writes:

    trial_registry_output.json

The JSON contains:

    - trial ID
    - configuration
    - which parameter axis generated the trial
    - whether it is baseline
    - known confound (if any)
    - closed-trade count
    - expectancy
    - sample standard deviation
    - Sharpe-like statistic
    - complete per-trade R sequence

The output is intended to be consumed by the later DSR/PBO module.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from strategy.backtest import run_backtest
from phase_02_optimization.walk_forward import DATA_FILE, load_dataset


OUTPUT_JSON = Path("trial_registry_output.json")


# ============================================================================
# BASELINE CONFIGURATION
# ============================================================================
# This must remain synchronized with the Phase 2 baseline.
# ============================================================================

BASELINE: dict[str, Any] = {
    "max_bars_to_retest": 20,
    "reward_multiple": 2.0,
    "stop_buffer": 0.0,
    "entry_mode": "midpoint",
}


# ============================================================================
# PRIMARY DSR/PBO TRIAL AXES
# ============================================================================
# These are ALL of the parameter-sensitivity axes (A, B, C, D) that were
# run under the one-at-a-time methodology in the Phase 2 record. No axis
# is excluded here -- see the module docstring for why stop_buffer's
# original sweep belongs in the primary universe.
# ============================================================================

PRIMARY_AXES: dict[str, list[Any]] = {
    "max_bars_to_retest": [   # Parameter A
        10,
        15,
        20,
        25,
        30,
    ],
    "reward_multiple": [      # Parameter B
        1.5,
        1.75,
        2.0,
        2.25,
        2.5,
    ],
    "stop_buffer": [          # Parameter C
        0.0000,
        0.0001,
        0.0002,
        0.0003,
        0.0004,
        0.0005,
    ],
    "entry_mode": [           # Parameter D
        "midpoint",
        "extreme",
    ],
}


# ============================================================================
# KNOWN CONFOUNDS
# ============================================================================
# Trials that are NOT clean single-parameter changes from baseline,
# despite only one axis being nominally varied. Keyed by
# (axis_name, value) so build_trial_configs() can attach the note.
# ============================================================================

KNOWN_CONFOUNDS: dict[tuple[str, Any], str] = {
    ("entry_mode", "extreme"): (
        "entry_mode='extreme' internally forces stop_buffer=0.01 when "
        "stop_buffer <= 0 (per signal engine behavior documented in the "
        "Phase 2 record). This trial therefore also changes the "
        "effective stop_buffer -- it is not an isolated single-variable "
        "change from baseline."
    ),
}


# ============================================================================
# EXPLICITLY EXCLUDED DIAGNOSTICS
# ============================================================================
# These are documented here so future edits do not accidentally fold them
# into the primary DSR/PBO universe.
# ============================================================================

EXCLUDED_DIAGNOSTICS = {
    "stop_buffer_flip_diagnostic": {
        "reason": (
            "The trade-by-trade WIN/LOSS flip investigation "
            "(stop_buffer_diagnostic.py's pairwise comparison at "
            "0.0 vs 0.0005) is a deeper investigation OF an already-"
            "included primary trial result, not a new configuration "
            "in its own right. The six stop_buffer sweep VALUES "
            "themselves ARE included as primary trials above -- only "
            "this follow-up diagnostic is excluded."
        ),
    },
    "transaction_cost_stress": {
        "reason": (
            "Transaction-cost assumptions applied to completed trades; "
            "not alternative strategy configurations."
        ),
    },
    "walk_forward_sensitivity": {
        "reason": (
            "Temporal validation sensitivity (warmup/window/step); "
            "evaluated separately from parameter-search trials."
        ),
    },
}


# ============================================================================
# DATA STRUCTURE
# ============================================================================

@dataclass
class TrialRecord:
    trial_id: int
    config: dict[str, Any]
    source_axis: str
    changed_parameter: str | None
    is_baseline: bool
    confound: str | None

    n_closed_trades: int
    expectancy_r: float
    std_r: float
    sharpe: float

    r_multiples: list[float] = field(default_factory=list)


# ============================================================================
# CONFIGURATION HELPERS
# ============================================================================

def config_key(config: dict[str, Any]) -> tuple:
    """
    Produce a deterministic hashable key for configuration deduplication.

    Sorting ensures dictionary insertion order does not affect identity.
    """
    return tuple(
        sorted(
            config.items(),
            key=lambda item: item[0],
        )
    )


def build_trial_configs() -> list[tuple[dict[str, Any], str, str | None, str | None]]:
    """
    Reconstruct the primary one-at-a-time trial universe.

    Returns:
        List of:

            (configuration, source_axis, changed_parameter, confound)

    The baseline is included exactly once. No Cartesian combinations
    are created.
    """
    seen: dict[
        tuple,
        tuple[dict[str, Any], str, str | None, str | None],
    ] = {}

    # ------------------------------------------------------------------
    # Baseline
    # ------------------------------------------------------------------
    baseline_key = config_key(BASELINE)

    seen[baseline_key] = (
        dict(BASELINE),
        "baseline",
        None,
        None,
    )

    # ------------------------------------------------------------------
    # One-at-a-time parameter trials
    # ------------------------------------------------------------------
    for axis_name, values in PRIMARY_AXES.items():

        for value in values:

            config = dict(BASELINE)
            config[axis_name] = value

            key = config_key(config)

            if key in seen:
                continue

            confound = KNOWN_CONFOUNDS.get((axis_name, value))

            seen[key] = (
                config,
                axis_name,
                axis_name,
                confound,
            )

    return list(seen.values())


def describe_config(config: dict[str, Any]) -> str:
    """
    Return a compact human-readable description of how a configuration
    differs from baseline.
    """
    differences = [
        f"{key}={value}"
        for key, value in config.items()
        if BASELINE.get(key) != value
    ]

    if not differences:
        return "BASELINE"

    return ", ".join(differences)


# ============================================================================
# STATISTICS
# ============================================================================

def sample_std(values: list[float]) -> float:
    """
    Calculate sample standard deviation.

    Returns 0.0 when fewer than two observations exist.
    """
    n = len(values)

    if n < 2:
        return 0.0

    mean = sum(values) / n

    variance = (
        sum(
            (value - mean) ** 2
            for value in values
        )
        / (n - 1)
    )

    return math.sqrt(variance)


def calculate_sharpe_like(
    r_multiples: list[float],
) -> float:
    """
    Calculate the project's consistent R-space Sharpe-like statistic.

    This is NOT a full annualized financial Sharpe ratio.

    It is intentionally:

        mean(R) / sample_std(R)

    because the registry contains trade-level R multiples rather than
    regularly spaced daily/monthly returns.
    """
    if not r_multiples:
        return 0.0

    mean_r = sum(r_multiples) / len(r_multiples)
    std_r = sample_std(r_multiples)

    if std_r <= 0:
        return 0.0

    return mean_r / std_r


# ============================================================================
# BACKTEST
# ============================================================================

def run_trial(
    df,
    config: dict[str, Any],
    source_axis: str,
    changed_parameter: str | None,
    confound: str | None,
    trial_id: int,
) -> TrialRecord:
    """
    Run one canonical backtest and convert it into a TrialRecord.
    """
    result = run_backtest(
        df,
        **config,
    )

    closed_trades = [
        trade
        for trade in result.trades
        if trade.result_status in ("WIN", "LOSS")
    ]

    r_multiples = [
        float(trade.r_multiple)
        for trade in closed_trades
    ]

    n_closed_trades = len(r_multiples)

    expectancy_r = (
        sum(r_multiples) / n_closed_trades
        if n_closed_trades
        else 0.0
    )

    std_r = sample_std(r_multiples)

    sharpe = calculate_sharpe_like(
        r_multiples
    )

    return TrialRecord(
        trial_id=trial_id,
        config=dict(config),
        source_axis=source_axis,
        changed_parameter=changed_parameter,
        is_baseline=(
            config_key(config)
            == config_key(BASELINE)
        ),
        confound=confound,
        n_closed_trades=n_closed_trades,
        expectancy_r=expectancy_r,
        std_r=std_r,
        sharpe=sharpe,
        r_multiples=r_multiples,
    )


# ============================================================================
# VALIDATION
# ============================================================================

def validate_trial_registry(
    records: list[TrialRecord],
) -> None:
    """
    Validate structural properties of the generated registry.

    Raises:
        RuntimeError if the registry is malformed.
    """
    if not records:
        raise RuntimeError(
            "Trial registry is empty."
        )

    # ------------------------------------------------------------------
    # Unique configuration check
    # ------------------------------------------------------------------
    keys = [
        config_key(record.config)
        for record in records
    ]

    if len(keys) != len(set(keys)):
        raise RuntimeError(
            "Duplicate trial configurations detected."
        )

    # ------------------------------------------------------------------
    # Exactly one baseline
    # ------------------------------------------------------------------
    baseline_records = [
        record
        for record in records
        if record.is_baseline
    ]

    if len(baseline_records) != 1:
        raise RuntimeError(
            "Trial registry must contain exactly one baseline."
        )

    # ------------------------------------------------------------------
    # Every trial must have the complete configuration schema
    # ------------------------------------------------------------------
    required_keys = set(BASELINE)

    for record in records:

        if set(record.config) != required_keys:
            raise RuntimeError(
                f"Trial {record.trial_id} has an invalid "
                f"configuration schema: {record.config}"
            )

        if record.n_closed_trades != len(
            record.r_multiples
        ):
            raise RuntimeError(
                f"Trial {record.trial_id} has inconsistent "
                "trade count and R-series length."
            )

        if not all(
            math.isfinite(value)
            for value in record.r_multiples
        ):
            raise RuntimeError(
                f"Trial {record.trial_id} contains a "
                "non-finite R multiple."
            )

        if not math.isfinite(record.expectancy_r):
            raise RuntimeError(
                f"Trial {record.trial_id} has non-finite expectancy."
            )

        if not math.isfinite(record.std_r):
            raise RuntimeError(
                f"Trial {record.trial_id} has non-finite standard deviation."
            )

        if not math.isfinite(record.sharpe):
            raise RuntimeError(
                f"Trial {record.trial_id} has non-finite Sharpe."
            )

    # ------------------------------------------------------------------
    # Trial numbering must be contiguous
    # ------------------------------------------------------------------
    expected_ids = list(
        range(
            1,
            len(records) + 1,
        )
    )

    actual_ids = [
        record.trial_id
        for record in records
    ]

    if actual_ids != expected_ids:
        raise RuntimeError(
            "Trial IDs are not contiguous."
        )


# ============================================================================
# JSON OUTPUT
# ============================================================================

def write_registry(
    records: list[TrialRecord],
) -> None:
    """
    Write the registry in a structured JSON document.

    The metadata makes the trial-universe definition explicit so a later
    DSR/PBO module does not have to infer it from the raw records.
    """
    payload = {
        "metadata": {
            "purpose": (
                "Primary Phase 2 parameter-sensitivity trial registry "
                "for DSR/PBO overfit diagnostics."
            ),
            "trial_design": (
                "One-at-a-time parameter sensitivity across Parameters "
                "A, B, C, D; no Cartesian grid."
            ),
            "baseline": dict(BASELINE),
            "primary_axes": PRIMARY_AXES,
            "known_confounds": {
                f"{axis}={value}": note
                for (axis, value), note in KNOWN_CONFOUNDS.items()
            },
            "excluded_diagnostics": EXCLUDED_DIAGNOSTICS,
            "dataset": str(DATA_FILE),
            "dataset_type": (
                "Synthetic M5 EURUSD-like random walk."
            ),
            "strategy_modified": False,
            "canonical_backtest_used": True,
            "n_primary_trials": len(records),
        },
        "trials": [
            asdict(record)
            for record in records
        ],
    }

    with OUTPUT_JSON.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            indent=2,
            allow_nan=False,
        )


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    df = load_dataset(DATA_FILE)

    trial_configs = build_trial_configs()

    print()
    print("=" * 118)
    print(
        "TRIAL REGISTRY -- PRIMARY PHASE 2 "
        "PARAMETER-SENSITIVITY SEARCH (A/B/C/D)"
    )
    print("=" * 118)

    print(
        f"Dataset                  : {DATA_FILE}"
    )

    print(
        "Dataset type             : "
        "synthetic M5 EURUSD-like random walk"
    )

    print(
        "Trial design             : "
        "one-at-a-time, no Cartesian grid, all four axes included"
    )

    print(
        f"Primary distinct trials  : {len(trial_configs)}"
    )

    print(
        "Stop-buffer flip diagnostic : EXCLUDED (see module docstring) "
        "-- but the six stop_buffer sweep VALUES ARE included as trials"
    )

    print(
        "Transaction-cost stress  : EXCLUDED from primary registry"
    )

    print(
        "Walk-forward sensitivity : EXCLUDED from primary registry"
    )

    print()

    print(
        f"{'trial#':>6} "
        f"{'source':<18} "
        f"{'configuration':<42} "
        f"{'trades':>8} "
        f"{'expectancy_R':>14} "
        f"{'std_R':>9} "
        f"{'sharpe':>9} "
        f"{'confound':>10}"
    )

    print("-" * 118)

    records: list[TrialRecord] = []

    for trial_id, (
        config,
        source_axis,
        changed_parameter,
        confound,
    ) in enumerate(
        trial_configs,
        start=1,
    ):

        record = run_trial(
            df=df,
            config=config,
            source_axis=source_axis,
            changed_parameter=changed_parameter,
            confound=confound,
            trial_id=trial_id,
        )

        records.append(record)

        print(
            f"{record.trial_id:>6} "
            f"{record.source_axis:<18} "
            f"{describe_config(record.config):<42} "
            f"{record.n_closed_trades:>8} "
            f"{record.expectancy_r:>14.4f} "
            f"{record.std_r:>9.4f} "
            f"{record.sharpe:>9.4f} "
            f"{'YES' if record.confound else '':>10}"
        )

    print("-" * 118)

    validate_trial_registry(records)

    sharpes = [
        record.sharpe
        for record in records
    ]

    mean_sharpe = (
        sum(sharpes) / len(sharpes)
    )

    std_sharpe = sample_std(
        sharpes
    )

    print(
        f"N primary trials          : {len(records)}"
    )

    print(
        f"Baseline trials           : "
        f"{sum(record.is_baseline for record in records)}"
    )

    print(
        f"Confounded trials         : "
        f"{sum(1 for record in records if record.confound)}"
    )

    print(
        f"Mean Sharpe-like value    : "
        f"{mean_sharpe:.4f}"
    )

    print(
        f"Std Sharpe-like values    : "
        f"{std_sharpe:.4f}"
    )

    print()

    print(
        "TRIAL-UNIVERSE RULE:"
    )

    print(
        "All four one-at-a-time parameter-sensitivity axes (A/B/C/D) "
        "are included in the primary DSR/PBO universe, including the "
        "original stop_buffer sweep. Only the later stop_buffer FLIP "
        "diagnostic, transaction-cost stress, and walk-forward "
        "sensitivity remain excluded (see EXCLUDED_DIAGNOSTICS)."
    )

    print()

    print(
        "CONFOUND NOTE:"
    )

    print(
        "The entry_mode='extreme' trial is flagged as confounded: it "
        "also silently changes effective stop_buffer to 0.01. Treat "
        "its Sharpe/expectancy as reflecting TWO simultaneous changes, "
        "not an isolated entry_mode effect."
    )

    print()

    print(
        "PBO INPUT:"
    )

    print(
        "Each trial retains its complete closed-trade R-multiple "
        "sequence for later CSCV analysis."
    )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "The current dataset is synthetic and the baseline contains "
        "only 16 closed trades. DSR/PBO results will therefore be "
        "diagnostic evidence about this experiment, not proof of "
        "real-market profitability."
    )

    write_registry(records)

    print()

    print(
        f"Registry written to     : {OUTPUT_JSON}"
    )

    print(
        "Registry validation      : PASSED"
    )

    print("=" * 118)
    print()


if __name__ == "__main__":
    main()