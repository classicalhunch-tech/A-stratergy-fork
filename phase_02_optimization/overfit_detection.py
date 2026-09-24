"""
phase_02_optimization/overfit_detection.py

Phase 2 final analytical stage: DSR (Deflated Sharpe Ratio) and PBO
(Probability of Backtest Overfitting) via CSCV.

Consumes:
    trial_registry_output.json

Does NOT:
    - modify strategy logic
    - modify the canonical backtest
    - select or lock a "winning" configuration
    - claim real-market profitability

IMPORTANT PERFORMANCE DESIGN
----------------------------
The PBO stage needs trade entry timestamps, which are not stored in
trial_registry_output.json. Therefore each trial must be replayed once
through the canonical backtest.

To make this resumable, completed PBO trial results are persisted to:

    pbo_backtest_cache.json

If the process is interrupted, rerunning this module resumes from the
last completed trial instead of starting the expensive replay from zero.

The cache is analytical output only. It does not modify strategy logic
or the canonical backtest.
"""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from strategy.backtest import run_backtest
from phase_02_optimization.walk_forward import DATA_FILE, load_dataset
from phase_02_optimization.trial_registry import (
    build_trial_configs,
    describe_config,
)


# ===========================================================================
# FILES / CONSTANTS
# ===========================================================================

TRIAL_REGISTRY_JSON = Path("trial_registry_output.json")

PBO_CACHE_JSON = Path(
    "phase_02_optimization/pbo_backtest_cache.json"
)

# Number of contiguous chronological blocks for CSCV.
# Must be even.
CSCV_BLOCKS = 10

EULER_MASCHERONI = 0.5772156649015329


# ===========================================================================
# SHARED MATH HELPERS
# ===========================================================================

def erfinv(y: float) -> float:
    """
    Inverse error function via Winitzki initial approximation
    + Newton refinement.
    """
    if y <= -1.0:
        return -math.inf

    if y >= 1.0:
        return math.inf

    a = 0.147

    ln1my2 = math.log(
        1.0 - y * y
    )

    term1 = (
        2.0 / (math.pi * a)
        + ln1my2 / 2.0
    )

    sign = 1.0 if y >= 0 else -1.0

    x = sign * math.sqrt(
        math.sqrt(
            term1 * term1
            - ln1my2 / a
        )
        - term1
    )

    for _ in range(50):
        err = math.erf(x) - y

        derivative = (
            2.0
            / math.sqrt(math.pi)
            * math.exp(-x * x)
        )

        if derivative == 0:
            break

        step = err / derivative

        x -= step

        if abs(step) < 1e-14:
            break

    return x


def norm_ppf(p: float) -> float:
    """Standard normal inverse CDF."""
    if p <= 0.0:
        return -math.inf

    if p >= 1.0:
        return math.inf

    return math.sqrt(2.0) * erfinv(
        2.0 * p - 1.0
    )


def norm_cdf(x: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (
        1.0
        + math.erf(
            x / math.sqrt(2.0)
        )
    )


def population_moments(
    values: list[float],
) -> tuple[float, float, float, float]:
    """
    Return:

        mean
        population standard deviation
        skewness
        kurtosis

    Kurtosis is non-excess, so normal = 3.
    """
    n = len(values)

    if n == 0:
        return (
            0.0,
            0.0,
            0.0,
            3.0,
        )

    mean = sum(values) / n

    m2 = sum(
        (v - mean) ** 2
        for v in values
    ) / n

    std = (
        math.sqrt(m2)
        if m2 > 0
        else 0.0
    )

    if std == 0:
        return (
            mean,
            0.0,
            0.0,
            3.0,
        )

    m3 = sum(
        (v - mean) ** 3
        for v in values
    ) / n

    m4 = sum(
        (v - mean) ** 4
        for v in values
    ) / n

    skew = m3 / (std ** 3)

    kurt = m4 / (std ** 4)

    return (
        mean,
        std,
        skew,
        kurt,
    )


# ===========================================================================
# PART 1: DSR
# ===========================================================================

def compute_expected_max_sharpe(
    sharpe_values: list[float],
) -> float:
    """
    Estimate expected maximum Sharpe from N zero-skill trials using
    the observed cross-trial Sharpe dispersion.
    """
    n = len(sharpe_values)

    if n < 2:
        return 0.0

    mean_sr = (
        sum(sharpe_values) / n
    )

    variance_sr = sum(
        (s - mean_sr) ** 2
        for s in sharpe_values
    ) / (n - 1)

    std_sr = math.sqrt(
        variance_sr
    )

    term_a = (
        (1.0 - EULER_MASCHERONI)
        * norm_ppf(
            1.0 - 1.0 / n
        )
    )

    term_b = (
        EULER_MASCHERONI
        * norm_ppf(
            1.0
            - 1.0
            / (n * math.e)
        )
    )

    return std_sr * (
        term_a + term_b
    )


def run_dsr(
    registry: dict,
) -> None:

    trials = registry["trials"]

    sharpe_values = [
        t["sharpe"]
        for t in trials
    ]

    n = len(sharpe_values)

    baseline_trial = next(
        t
        for t in trials
        if t["is_baseline"]
    )

    sr_hat = baseline_trial["sharpe"]

    r_multiples = (
        baseline_trial["r_multiples"]
    )

    t_obs = len(r_multiples)

    (
        mean_r,
        std_r,
        skew,
        kurt,
    ) = population_moments(
        r_multiples
    )

    sr0 = compute_expected_max_sharpe(
        sharpe_values
    )

    denom = (
        1.0
        - skew * sr_hat
        + (
            (kurt - 1.0)
            / 4.0
        )
        * (sr_hat ** 2)
    )

    print()
    print("=" * 100)
    print(
        "PART 1: DEFLATED SHARPE RATIO (DSR)"
    )
    print("=" * 100)

    print(
        f"N trials in universe             : {n}"
    )

    print(
        f"Baseline (selected) trial Sharpe  : "
        f"{sr_hat:.4f}"
    )

    print(
        f"Baseline closed trades (T)         : "
        f"{t_obs}"
    )

    print(
        f"Baseline R-series skewness         : "
        f"{skew:.4f}"
    )

    print(
        f"Baseline R-series kurtosis          : "
        f"{kurt:.4f} (normal = 3.0)"
    )

    print(
        f"E[max Sharpe | N={n} noise trials]  : "
        f"{sr0:.4f}"
    )

    print()

    if denom <= 0:
        print(
            "WARNING: non-normality correction "
            "denominator is non-positive."
        )

        print(
            f"Denominator = {denom:.4f}"
        )

        print(
            "DSR is therefore treated as undefined "
            "for this sample."
        )

    elif t_obs < 2:
        print(
            "WARNING: fewer than 2 observations; "
            "DSR cannot be estimated."
        )

    else:
        z = (
            (sr_hat - sr0)
            * math.sqrt(t_obs - 1)
            / math.sqrt(denom)
        )

        dsr = norm_cdf(z)

        print(
            f"DSR statistic (z)                  : "
            f"{z:.4f}"
        )

        print(
            f"DSR = P(true Sharpe > 0 | selection): "
            f"{dsr:.4f}"
        )

    print()

    print("READ AS:")

    print(
        "  DSR near 1.0  -> baseline's Sharpe is "
        "less consistent with pure selection noise."
    )

    print(
        "  DSR near 0.5  -> broadly consistent with "
        "what the trial search could produce by chance."
    )

    print(
        "  DSR near 0.0  -> strong selection-bias warning."
    )

    print()

    print(
        f"CAVEAT: T={t_obs} is a small sample. "
        "Skewness/kurtosis estimates are noisy. "
        "Treat DSR as directional evidence, not a "
        "precise probability."
    )

    print("=" * 100)


# ===========================================================================
# TIMESTAMP / BLOCK HELPERS
# ===========================================================================

def build_timestamp_index(
    df: pd.DataFrame,
) -> dict[pd.Timestamp, int]:

    if "timestamp" in df.columns:
        raw = df["timestamp"].tolist()
    else:
        raw = list(df.index)

    return {
        pd.Timestamp(ts): i
        for i, ts in enumerate(raw)
    }


def block_index_for(
    bar_index: int,
    n_bars: int,
    n_blocks: int,
) -> int:

    block_size = (
        n_bars / n_blocks
    )

    idx = int(
        bar_index // block_size
    )

    return min(
        idx,
        n_blocks - 1,
    )


# ===========================================================================
# PBO CACHE
# ===========================================================================

def load_pbo_cache() -> dict[str, Any]:
    """
    Load the persistent PBO cache.

    Invalid/missing cache files are treated as empty.
    """
    if not PBO_CACHE_JSON.exists():
        return {
            "version": 1,
            "dataset": str(DATA_FILE),
            "cscv_blocks": CSCV_BLOCKS,
            "trials": {},
        }

    try:
        with PBO_CACHE_JSON.open(
            "r",
            encoding="utf-8",
        ) as f:
            cache = json.load(f)

    except (
        OSError,
        json.JSONDecodeError,
        TypeError,
        ValueError,
    ):
        print(
            "\nWARNING: existing PBO cache could not "
            "be read. Starting a fresh cache."
        )

        return {
            "version": 1,
            "dataset": str(DATA_FILE),
            "cscv_blocks": CSCV_BLOCKS,
            "trials": {},
        }

    # Cache belongs to this exact analysis setup.
    if (
        cache.get("version") != 1
        or cache.get("dataset")
        != str(DATA_FILE)
        or cache.get("cscv_blocks")
        != CSCV_BLOCKS
    ):
        print(
            "\nExisting PBO cache does not match the "
            "current analysis settings."
        )

        print(
            "Starting a fresh cache."
        )

        return {
            "version": 1,
            "dataset": str(DATA_FILE),
            "cscv_blocks": CSCV_BLOCKS,
            "trials": {},
        }

    if not isinstance(
        cache.get("trials"),
        dict,
    ):
        cache["trials"] = {}

    return cache


def save_pbo_cache(
    cache: dict[str, Any],
) -> None:
    """
    Atomically save the PBO cache after every completed trial.
    """
    PBO_CACHE_JSON.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path = (
        PBO_CACHE_JSON.with_suffix(
            ".tmp"
        )
    )

    with temp_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            cache,
            f,
            indent=2,
        )

    temp_path.replace(
        PBO_CACHE_JSON
    )


def config_cache_key(
    config: dict[str, Any],
) -> str:
    """
    Stable cache key for one trial configuration.
    """
    return json.dumps(
        config,
        sort_keys=True,
        separators=(",", ":"),
    )


def cached_block_totals_valid(
    value: Any,
) -> bool:

    if not isinstance(
        value,
        list,
    ):
        return False

    if len(value) != CSCV_BLOCKS:
        return False

    return all(
        isinstance(
            x,
            (int, float),
        )
        and math.isfinite(float(x))
        for x in value
    )


# ===========================================================================
# BUILD PBO PERFORMANCE MATRIX
# ===========================================================================

def run_single_trial_for_pbo(
    df: pd.DataFrame,
    ts_index: dict[pd.Timestamp, int],
    config: dict[str, Any],
) -> list[float]:
    """
    Run exactly one canonical backtest and convert its closed-trade
    R-multiples into chronological CSCV block totals.
    """

    result = run_backtest(
        df,
        **config,
    )

    closed_trades = [
        t
        for t in result.trades
        if t.result_status
        in ("WIN", "LOSS")
    ]

    block_totals = [
        0.0
        for _ in range(CSCV_BLOCKS)
    ]

    unmatched = 0

    for trade in closed_trades:

        ts = pd.Timestamp(
            trade.entry_time
        )

        bar_idx = ts_index.get(ts)

        if bar_idx is None:
            unmatched += 1
            continue

        block_idx = block_index_for(
            bar_idx,
            len(df),
            CSCV_BLOCKS,
        )

        block_totals[
            block_idx
        ] += float(
            trade.r_multiple
        )

    if unmatched:
        print(
            f"    NOTE: {unmatched} trade(s) had "
            "entry timestamps not found in the "
            "dataset index and were excluded."
        )

    return block_totals


def build_block_performance_matrix(
    df: pd.DataFrame,
    ts_index: dict[pd.Timestamp, int],
) -> tuple[
    list[str],
    list[list[float]],
]:
    """
    Build the PBO block matrix.

    Each configuration is run at most once per cache state.

    Completed trials are saved immediately, making this operation
    resumable after interruption.
    """

    configs = build_trial_configs()

    cache = load_pbo_cache()

    labels: list[str] = []

    matrix: list[list[float]] = []

    total = len(configs)

    print()
    print("=" * 100)
    print(
        "PBO PREPARATION: BUILDING CHRONOLOGICAL "
        "TRIAL PERFORMANCE MATRIX"
    )
    print("=" * 100)

    print(
        f"Configurations required : {total}"
    )

    print(
        f"CSCV blocks             : {CSCV_BLOCKS}"
    )

    print(
        f"Cache file              : "
        f"{PBO_CACHE_JSON}"
    )

    print()

    completed_from_cache = 0
    newly_computed = 0

    for trial_number, (
        config,
        source_axis,
        changed_parameter,
        confound,
    ) in enumerate(
        configs,
        start=1,
    ):

        label = describe_config(
            config
        )

        labels.append(label)

        key = config_cache_key(
            config
        )

        cached = cache[
            "trials"
        ].get(key)

        if (
            isinstance(cached, dict)
            and cached_block_totals_valid(
                cached.get(
                    "block_totals"
                )
            )
        ):

            block_totals = [
                float(x)
                for x in cached[
                    "block_totals"
                ]
            ]

            completed_from_cache += 1

            print(
                f"[{trial_number:02d}/{total:02d}] "
                f"CACHED  {label}"
            )

        else:

            print(
                f"[{trial_number:02d}/{total:02d}] "
                f"RUNNING {label}"
            )

            print(
                "           This is one canonical "
                "backtest. Please wait..."
            )

            block_totals = (
                run_single_trial_for_pbo(
                    df,
                    ts_index,
                    config,
                )
            )

            cache[
                "trials"
            ][key] = {
                "config": config,
                "source_axis": source_axis,
                "changed_parameter": (
                    changed_parameter
                ),
                "confound": confound,
                "block_totals": block_totals,
            }

            # Save immediately so Ctrl+C does not
            # throw away completed work.
            save_pbo_cache(
                cache
            )

            newly_computed += 1

            print(
                f"           SAVED to PBO cache"
            )

        matrix.append(
            block_totals
        )

    print()

    print(
        f"Trials loaded from cache : "
        f"{completed_from_cache}"
    )

    print(
        f"Trials newly computed    : "
        f"{newly_computed}"
    )

    print(
        f"Total matrix rows        : "
        f"{len(matrix)}"
    )

    print("=" * 100)

    return labels, matrix


# ===========================================================================
# RANKING
# ===========================================================================

def rank_descending(
    values: list[float],
) -> list[float]:
    """
    Rank values with 1 = best.

    Average ranks are used for ties.
    """

    n = len(values)

    order = sorted(
        range(n),
        key=lambda i: values[i],
        reverse=True,
    )

    ranks = [0.0] * n

    i = 0

    while i < n:

        j = i

        while (
            j + 1 < n
            and values[
                order[j + 1]
            ]
            == values[
                order[i]
            ]
        ):
            j += 1

        avg_rank = (
            (i + 1 + j + 1)
            / 2.0
        )

        for k in range(
            i,
            j + 1,
        ):
            ranks[
                order[k]
            ] = avg_rank

        i = j + 1

    return ranks


# ===========================================================================
# PART 2: PBO via CSCV
# ===========================================================================

def run_pbo(
    labels: list[str],
    matrix: list[list[float]],
) -> None:

    n_trials = len(matrix)

    n_blocks = CSCV_BLOCKS

    half = n_blocks // 2

    combos = list(
        itertools.combinations(
            range(n_blocks),
            half,
        )
    )

    n_splits = len(combos)

    overfit_count = 0

    logits: list[float] = []

    print()
    print("=" * 100)
    print(
        "PART 2: PROBABILITY OF BACKTEST "
        "OVERFITTING (PBO via CSCV)"
    )
    print("=" * 100)

    print(
        f"Trials (N)                : "
        f"{n_trials}"
    )

    print(
        f"Chronological blocks (S)  : "
        f"{n_blocks}"
    )

    print(
        f"IS/OOS splits evaluated   : "
        f"{n_splits} "
        f"(C({n_blocks},{half}))"
    )

    print()

    for split_number, is_blocks in enumerate(
        combos,
        start=1,
    ):

        is_set = set(
            is_blocks
        )

        oos_set = (
            set(range(n_blocks))
            - is_set
        )

        is_perf = [
            sum(
                matrix[t][b]
                for b in is_set
            )
            for t in range(
                n_trials
            )
        ]

        oos_perf = [
            sum(
                matrix[t][b]
                for b in oos_set
            )
            for t in range(
                n_trials
            )
        ]

        is_ranks = rank_descending(
            is_perf
        )

        oos_ranks = rank_descending(
            oos_perf
        )

        best_is_trial = min(
            range(n_trials),
            key=lambda t:
                is_ranks[t],
        )

        oos_rank_of_best = (
            oos_ranks[
                best_is_trial
            ]
        )

        w = (
            oos_rank_of_best
            / (n_trials + 1.0)
        )

        w = min(
            max(w, 1e-9),
            1.0 - 1e-9,
        )

        logit = math.log(
            w / (1.0 - w)
        )

        logits.append(
            logit
        )

        if logit <= 0:
            overfit_count += 1

    pbo = (
        overfit_count
        / n_splits
    )

    print(
        f"Splits where IS-best trial "
        f"ranked <= OOS median: "
        f"{overfit_count}"
    )

    print()

    print(
        f"PBO = {pbo:.4f}"
    )

    print()

    print("READ AS:")

    print(
        "  PBO near 0.0  -> picking the "
        "in-sample winner tends to also do "
        "well out-of-sample; lower overfitting risk."
    )

    print(
        "  PBO near 0.5  -> picking the "
        "in-sample winner is roughly a coin flip "
        "out-of-sample."
    )

    print(
        "  PBO > 0.5     -> picking the "
        "in-sample winner tends to do worse "
        "than random; strong overfitting warning."
    )

    print()

    print(
        "CAVEAT: block boundaries are a chronological "
        "approximation because trade timing differs "
        "between configurations."
    )

    print(
        f"Sanity check recommendation: rerun with a "
        f"different even CSCV_BLOCKS value if the "
        f"PBO result is surprising."
    )

    print("=" * 100)


# ===========================================================================
# MAIN
# ===========================================================================

def main() -> None:

    if not TRIAL_REGISTRY_JSON.exists():
        raise SystemExit(
            f"{TRIAL_REGISTRY_JSON} not found -- "
            "run phase_02_optimization/trial_registry.py "
            "first."
        )

    with TRIAL_REGISTRY_JSON.open(
        "r",
        encoding="utf-8",
    ) as f:
        registry = json.load(f)

    # -----------------------------------------------------------------------
    # DSR
    # -----------------------------------------------------------------------

    run_dsr(
        registry
    )

    # -----------------------------------------------------------------------
    # Load dataset
    # -----------------------------------------------------------------------

    print()
    print(
        "Loading dataset for PBO..."
    )

    df = load_dataset(
        DATA_FILE
    )

    print(
        f"Dataset candles loaded : "
        f"{len(df)}"
    )

    # -----------------------------------------------------------------------
    # Timestamp index
    # -----------------------------------------------------------------------

    ts_index = build_timestamp_index(
        df
    )

    # -----------------------------------------------------------------------
    # PBO matrix
    # -----------------------------------------------------------------------

    labels, matrix = (
        build_block_performance_matrix(
            df,
            ts_index,
        )
    )

    # -----------------------------------------------------------------------
    # PBO
    # -----------------------------------------------------------------------

    run_pbo(
        labels,
        matrix,
    )

    # -----------------------------------------------------------------------
    # Final diagnostic message
    # -----------------------------------------------------------------------

    print()

    print(
        "Both DSR and PBO are diagnostic evidence "
        "about THIS experiment only."
    )

    print(
        "They do not establish real-market profitability."
    )

    print(
        "Combine them with Monte Carlo, warmup, "
        "window/step sensitivity, parameter sensitivity, "
        "and transaction-cost findings for the final "
        "Phase 2 robustness review."
    )

    print()

    print(
        f"PBO cache retained at: "
        f"{PBO_CACHE_JSON}"
    )

    print()


if __name__ == "__main__":
    main()