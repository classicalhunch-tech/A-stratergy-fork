"""
gold_orderflow/research/significance_check.py

Research-only statistical challenge for the XAUUSD quote_pressure
vs forward-return relationship.

Purpose:
    Test whether the observed relationship is distinguishable from zero
    while accounting for the dependence structure of tick data.

Methods:

1. MOVING-BLOCK BOOTSTRAP
   Resamples contiguous blocks with replacement. This preserves local
   dependence inside each block and produces an empirical distribution
   of the correlation statistic.

2. NON-OVERLAPPING CHUNK ANALYSIS
   Splits the chronological sample into independent, non-overlapping
   chunks and calculates one correlation per chunk.

   The primary output is the distribution of chunk correlations,
   including:
       - mean
       - median
       - standard deviation
       - fraction with the same sign
       - sign-test p-value

   A sign test is used rather than treating individual ticks as
   independent observations. The sign-test p-value is computed in
   log-space (via lgamma) rather than with math.comb(), because
   math.comb(n, n//2) for n in the hundreds/thousands produces an
   integer far too large to convert to a float, which raises
   OverflowError.

3. BLOCK-LEVEL CONFIDENCE INTERVAL
   The bootstrap confidence interval is reported as a robustness
   diagnostic. It is NOT interpreted as proof of a trading edge.

Important limitations:
    - Forward returns at different horizons overlap.
    - Tick observations are dependent.
    - Statistical significance does not establish profitability.
    - No transaction costs or slippage are modeled here.
    - No trading rule is tested.
    - This module does not modify the live strategy.

Research sequence:
    feed truth
        ->
    causal features
        ->
    future response
        ->
    regime robustness
        ->
    significance challenge
        ->
    costs/stress
        ->
    stronger OOS
        ->
    paper validation
        ->
    possible integration
"""

import argparse
import math

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Fixed research parameters
# ---------------------------------------------------------------------

HORIZONS = [1, 5, 20, 50]
FEATURES = ["quote_pressure"]

# Block/chunk size is deliberately fixed before inspecting results.
BLOCK_SIZE = 2000
CHUNK_SIZE = 2000

N_BOOTSTRAP = 2000

MIN_CHUNK_ROWS = 200

# Minimum number of chunks required for the chunk-level analysis.
MIN_CHUNKS = 10

RNG_SEED = 42


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------

def load_features(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(
        csv_path,
        parse_dates=["dt"],
    )

    df = (
        df.sort_values("dt")
        .drop_duplicates(
            subset=["dt"],
            keep="first",
        )
        .reset_index(drop=True)
    )

    required = [
        "dt",
        "mid",
        "quote_pressure",
    ]

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {', '.join(missing)}"
        )

    return df


# ---------------------------------------------------------------------
# Forward returns
# ---------------------------------------------------------------------

def add_forward_returns(
    df: pd.DataFrame,
    horizons: list[int],
) -> pd.DataFrame:

    out = df.copy()

    for h in horizons:
        # Strictly future mid-price.
        out[f"fwd_ret_{h}"] = (
            out["mid"].shift(-h)
            - out["mid"]
        )

    return out


# ---------------------------------------------------------------------
# Correlation helper
# ---------------------------------------------------------------------

def safe_corr(
    x: np.ndarray,
    y: np.ndarray,
) -> float:

    if len(x) < 2:
        return np.nan

    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan

    return float(
        np.corrcoef(x, y)[0, 1]
    )


# ---------------------------------------------------------------------
# Moving-block bootstrap
# ---------------------------------------------------------------------

def block_bootstrap_corr(
    x: np.ndarray,
    y: np.ndarray,
    block_size: int,
    n_boot: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Moving/circular block bootstrap.

    Each bootstrap sample is constructed from contiguous blocks,
    preserving local dependence inside each block.

    The final block may wrap around the original series.
    """

    n = len(x)

    if n < block_size:
        raise ValueError(
            "Series shorter than BLOCK_SIZE."
        )

    n_blocks_needed = int(
        np.ceil(n / block_size)
    )

    boot_corrs = np.empty(n_boot)

    for b in range(n_boot):

        starts = rng.integers(
            0,
            n,
            size=n_blocks_needed,
        )

        pieces_x = []
        pieces_y = []

        for start in starts:

            indices = (
                np.arange(
                    start,
                    start + block_size,
                )
                % n
            )

            pieces_x.append(
                x[indices]
            )

            pieces_y.append(
                y[indices]
            )

        bx = np.concatenate(
            pieces_x
        )[:n]

        by = np.concatenate(
            pieces_y
        )[:n]

        boot_corrs[b] = safe_corr(
            bx,
            by,
        )

    return boot_corrs


# ---------------------------------------------------------------------
# Non-overlapping chunk correlations
# ---------------------------------------------------------------------

def chunk_correlations(
    x: np.ndarray,
    y: np.ndarray,
    chunk_size: int,
    min_rows: int,
) -> np.ndarray:

    n = len(x)
    corrs = []

    for start in range(
        0,
        n,
        chunk_size,
    ):

        end = min(
            start + chunk_size,
            n,
        )

        cx = x[start:end]
        cy = y[start:end]

        if len(cx) < min_rows:
            continue

        corr = safe_corr(
            cx,
            cy,
        )

        if np.isfinite(corr):
            corrs.append(corr)

    return np.asarray(
        corrs,
        dtype=float,
    )


# ---------------------------------------------------------------------
# Exact binomial sign-test p-value (log-space, overflow-safe)
# ---------------------------------------------------------------------

def sign_test_pvalue(
    values: np.ndarray,
) -> tuple[float, int, int, int]:
    """
    Two-sided exact sign-test p-value, computed in log-space to avoid
    integer/float overflow from math.comb() on large n (n in the
    hundreds/thousands produces C(n, n//2) values far larger than a
    float can represent, which raises OverflowError when math.comb()'s
    huge integer result is multiplied by a float).

    Zero-valued observations are excluded because they carry no
    directional information.
    """

    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    positive = int(
        np.sum(values > 0)
    )

    negative = int(
        np.sum(values < 0)
    )

    zero = int(
        np.sum(values == 0)
    )

    n = positive + negative

    if n == 0:
        return (
            np.nan,
            positive,
            negative,
            zero,
        )

    k = min(
        positive,
        negative,
    )

    def log_binom_pmf(n_: int, i: int) -> float:
        # log of [ C(n, i) * 0.5^n ], via lgamma, so the raw
        # (potentially astronomically large) combination integer is
        # never materialized or converted to a float.
        return (
            math.lgamma(n_ + 1)
            - math.lgamma(i + 1)
            - math.lgamma(n_ - i + 1)
            - n_ * math.log(2.0)
        )

    log_terms = [
        log_binom_pmf(n, i)
        for i in range(k + 1)
    ]

    # log-sum-exp for numerical stability; the final exponentiation
    # is always safe since the summed probability is bounded by 1.
    max_log = max(log_terms)

    cumulative = math.exp(max_log) * sum(
        math.exp(t - max_log)
        for t in log_terms
    )

    p_value = min(
        1.0,
        2.0 * cumulative,
    )

    return (
        p_value,
        positive,
        negative,
        zero,
    )


# ---------------------------------------------------------------------
# Bootstrap summary
# ---------------------------------------------------------------------

def bootstrap_summary(
    observed_corr: float,
    boot_corrs: np.ndarray,
) -> dict:

    boot_corrs = boot_corrs[
        np.isfinite(boot_corrs)
    ]

    if len(boot_corrs) == 0:
        return {
            "boot_ci_low": np.nan,
            "boot_ci_high": np.nan,
            "boot_excludes_zero": False,
            "boot_mean": np.nan,
            "boot_std": np.nan,
            "boot_same_sign_fraction": np.nan,
        }

    ci_low, ci_high = np.percentile(
        boot_corrs,
        [2.5, 97.5],
    )

    same_sign_fraction = (
        np.mean(
            np.sign(boot_corrs)
            == np.sign(observed_corr)
        )
    )

    return {
        "boot_ci_low": ci_low,
        "boot_ci_high": ci_high,
        "boot_excludes_zero": (
            ci_low > 0
            or ci_high < 0
        ),
        "boot_mean": boot_corrs.mean(),
        "boot_std": boot_corrs.std(
            ddof=1
        ),
        "boot_same_sign_fraction": (
            same_sign_fraction
        ),
    }


# ---------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------

def analyze_feature_horizon(
    df: pd.DataFrame,
    feature: str,
    horizon: int,
    rng: np.random.Generator,
) -> dict:

    target_col = (
        f"fwd_ret_{horizon}"
    )

    valid = df[
        [feature, target_col]
    ].dropna()

    x = valid[feature].to_numpy(
        dtype=float
    )

    y = valid[target_col].to_numpy(
        dtype=float
    )

    n = len(x)

    observed_corr = safe_corr(
        x,
        y,
    )

    # ================================================================
    # METHOD 1: BLOCK BOOTSTRAP
    # ================================================================

    boot = block_bootstrap_corr(
        x,
        y,
        BLOCK_SIZE,
        N_BOOTSTRAP,
        rng,
    )

    boot_summary = bootstrap_summary(
        observed_corr,
        boot,
    )

    # ================================================================
    # METHOD 2: NON-OVERLAPPING CHUNKS
    # ================================================================

    chunk_corrs = chunk_correlations(
        x,
        y,
        CHUNK_SIZE,
        MIN_CHUNK_ROWS,
    )

    n_chunks = len(
        chunk_corrs
    )

    if n_chunks > 0:
        chunk_mean = float(
            chunk_corrs.mean()
        )

        chunk_median = float(
            np.median(chunk_corrs)
        )

        chunk_std = (
            float(
                chunk_corrs.std(
                    ddof=1
                )
            )
            if n_chunks > 1
            else np.nan
        )

        frac_positive = float(
            np.mean(
                chunk_corrs > 0
            )
        )

        frac_negative = float(
            np.mean(
                chunk_corrs < 0
            )
        )

    else:
        chunk_mean = np.nan
        chunk_median = np.nan
        chunk_std = np.nan
        frac_positive = np.nan
        frac_negative = np.nan

    # Exact sign test.
    (
        p_sign,
        positive_chunks,
        negative_chunks,
        zero_chunks,
    ) = sign_test_pvalue(
        chunk_corrs
    )

    sign_test_available = (
        n_chunks >= MIN_CHUNKS
    )

    if not sign_test_available:
        p_sign = np.nan

    return {
        "feature": feature,
        "horizon_ticks": horizon,
        "n": n,
        "observed_corr": observed_corr,

        # Bootstrap.
        "boot_ci_low": boot_summary[
            "boot_ci_low"
        ],
        "boot_ci_high": boot_summary[
            "boot_ci_high"
        ],
        "boot_excludes_zero": boot_summary[
            "boot_excludes_zero"
        ],
        "boot_mean": boot_summary[
            "boot_mean"
        ],
        "boot_std": boot_summary[
            "boot_std"
        ],
        "boot_same_sign_fraction": boot_summary[
            "boot_same_sign_fraction"
        ],

        # Chunk analysis.
        "n_chunks": n_chunks,
        "chunk_mean_corr": chunk_mean,
        "chunk_median_corr": chunk_median,
        "chunk_std_corr": chunk_std,
        "chunk_frac_positive": frac_positive,
        "chunk_frac_negative": frac_negative,

        # Exact sign test.
        "positive_chunks": positive_chunks,
        "negative_chunks": negative_chunks,
        "zero_chunks": zero_chunks,
        "p_sign_test": p_sign,
        "sign_test_available": sign_test_available,
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main(
    csv_path: str,
    out_csv: str,
):

    df = load_features(
        csv_path
    )

    n_loaded = len(df)

    df = add_forward_returns(
        df,
        HORIZONS,
    )

    print(
        f"rows loaded: {n_loaded}"
    )

    print(
        f"bootstrap block size: "
        f"{BLOCK_SIZE} ticks"
    )

    print(
        f"chunk size: "
        f"{CHUNK_SIZE} ticks"
    )

    print(
        f"bootstrap replicates: "
        f"{N_BOOTSTRAP}"
    )

    rng = np.random.default_rng(
        RNG_SEED
    )

    results = []

    for feature in FEATURES:

        for h in HORIZONS:

            print(
                f"\nanalyzing "
                f"feature={feature} "
                f"horizon={h} ..."
            )

            result = analyze_feature_horizon(
                df,
                feature,
                h,
                rng,
            )

            results.append(
                result
            )

    out = pd.DataFrame(
        results
    )

    pd.set_option(
        "display.width",
        220,
    )

    print(
        "\n" + "=" * 90
    )

    print(
        "SIGNIFICANCE / ROBUSTNESS SUMMARY"
    )

    print(
        "=" * 90
    )

    display_cols = [
        "feature",
        "horizon_ticks",
        "n",
        "observed_corr",
        "boot_ci_low",
        "boot_ci_high",
        "boot_excludes_zero",
        "n_chunks",
        "chunk_mean_corr",
        "chunk_median_corr",
        "chunk_frac_positive",
        "chunk_frac_negative",
        "positive_chunks",
        "negative_chunks",
        "p_sign_test",
    ]

    print(
        out[display_cols].to_string(
            index=False
        )
    )

    out.to_csv(
        out_csv,
        index=False,
    )

    print(
        f"\nwrote {out_csv}"
    )

    # ---------------------------------------------------------------
    # Multiple-testing note
    # ---------------------------------------------------------------

    n_tests = len(out)

    bonferroni_alpha = (
        0.05 / n_tests
    )

    print(
        "\n" + "=" * 90
    )

    print(
        "MULTIPLE-TESTING NOTE"
    )

    print(
        "=" * 90
    )

    print(
        f"{n_tests} horizon tests were run."
    )

    print(
        f"Bonferroni-adjusted alpha: "
        f"{bonferroni_alpha:.6f}"
    )

    print(
        "\nImportant:"
    )

    print(
        "- The horizons overlap and are therefore not independent."
    )

    print(
        "- Bonferroni is conservative and is used only as a safeguard."
    )

    print(
        "- Statistical significance is not economic significance."
    )

    print(
        "- No costs, slippage, execution or trading rule are tested."
    )

    print(
        "- No result here authorizes strategy integration."
    )

    print(
        "\nRESEARCH DECISION RULE:"
    )

    print(
        "Treat the relationship as statistically more credible only "
        "when the evidence is directionally consistent across the "
        "bootstrap distribution AND the independent chunk analysis."
    )

    print(
        "\nNEXT STEP AFTER THIS MODULE:"
    )

    print(
        "costs / slippage / execution stress testing."
    )


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--csv",
        required=True,
        help=(
            "path to quote_diagnostics.py "
            "output CSV"
        ),
    )

    parser.add_argument(
        "--out-csv",
        default=(
            "gold_orderflow/data/"
            "significance_check.csv"
        ),
    )

    args = parser.parse_args()

    main(
        args.csv,
        args.out_csv,
    )