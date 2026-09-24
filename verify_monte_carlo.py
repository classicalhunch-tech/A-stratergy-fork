"""
verify_monte_carlo.py — verify Phase 2 Monte Carlo against REAL Phase 1 output.

Usage (from project root, .venv active):

    python verify_monte_carlo.py

Before running:
    1. Replace "your_data_file.csv" with your real Phase 1 data file.
    2. Confirm the run_backtest() call matches your actual Phase 1 API.

Purpose
-------
This script verifies that the Phase 2 Monte Carlo engine correctly consumes
the REAL Phase 1 BacktestResult before walk_forward.py is built.

It does not modify strategy/*.
It only calls the canonical Phase 1 backtest and feeds its closed-trade
R-multiples into Phase 2 Monte Carlo.
"""

import math
import statistics

import pandas as pd

from strategy.backtest import run_backtest  # ADJUST ME if signature differs

from phase_02_optimization.monte_carlo import (
    MonteCarloConfig,
    extract_closed_r_multiples,
    run_monte_carlo,
)


def replay_actual_return(
    r_multiples,
    risk_per_trade_pct,
    initial_balance,
):
    """
    Compound the REAL chronological trade order once.

    This is the mathematical baseline for shuffle mode.

    With fixed-fractional compounding and no forced liquidation,
    changing only the order of the same R-multiples should not change
    the final balance due to commutativity of multiplication.
    """
    balance = initial_balance

    for r in r_multiples:
        balance *= 1.0 + risk_per_trade_pct * r

    return balance


def validate_raw_r_multiples(r_multiples):
    """Ensure the extracted R-multiple ledger contains only finite numbers."""
    for r in r_multiples:
        if isinstance(r, bool) or not isinstance(r, (int, float)):
            raise ValueError(f"Invalid R-multiple encountered: {r!r}")

        if not math.isfinite(float(r)):
            raise ValueError(f"Non-finite R-multiple encountered: {r!r}")


def main():
    # ------------------------------------------------------------------
    # 1) Load real data and run canonical Phase 1 backtest
    # ------------------------------------------------------------------

    print("=" * 70)
    print("PHASE 2 MONTE CARLO VERIFICATION")
    print("=" * 70)

    # ADJUST ME: Replace with your actual Phase 1 dataset filename.
    df = pd.read_csv(
        "your_data_file.csv",
        parse_dates=["timestamp"],
    )

    if "timestamp" in df.columns:
        df = df.set_index("timestamp")

    print(f"\nLoaded candles: {len(df)}")

    # ADJUST ME: Confirm these arguments match your actual Phase 1 API.
    backtest_result = run_backtest(
        df,
        max_bars_to_retest=20,
        reward_multiple=2.0,
        stop_buffer=0.0,
        entry_mode="midpoint",
    )

    # ------------------------------------------------------------------
    # 2) Extract closed-trade R-multiples
    # ------------------------------------------------------------------

    r_multiples = extract_closed_r_multiples(backtest_result)
    validate_raw_r_multiples(r_multiples)

    print(f"Closed trades found: {len(r_multiples)}")

    if not r_multiples:
        print("No closed trades — nothing to verify. Stop here.")
        return

    print(
        f"R-multiple range: "
        f"{min(r_multiples):.4f}R to {max(r_multiples):.4f}R"
    )

    risk = 0.01
    initial_balance = 10_000.0

    # ------------------------------------------------------------------
    # CHECK 1
    # Shuffle final balance invariance (Commutativity property)
    # ------------------------------------------------------------------

    actual = replay_actual_return(r_multiples, risk, initial_balance)

    mc_shuffle = run_monte_carlo(
        backtest_result,
        MonteCarloConfig(
            method="shuffle",
            num_simulations=2000,
            risk_per_trade_pct=risk,
            initial_balance=initial_balance,
            random_seed=42,
        ),
    )

    shuffle_final_balances = [
        run.final_balance for run in mc_shuffle.runs
    ]

    max_difference = max(
        abs(balance - actual) for balance in shuffle_final_balances
    )

    print("\n[Check 1] Shuffle final-balance invariance")
    print(f"  Actual chronological balance: {actual:.2f}")
    print(f"  Shuffle final-balance min:    {min(shuffle_final_balances):.2f}")
    print(f"  Shuffle final-balance max:    {max(shuffle_final_balances):.2f}")
    print(f"  Maximum difference:           {max_difference:.10f}")

    check_1_ok = max_difference < 1e-6
    print(
        "  -> Same trades with different order must produce "
        "the exact same final compounded balance:",
        "OK" if check_1_ok else "!!! BUG !!!",
    )

    # ------------------------------------------------------------------
    # CHECK 2
    # Compare shuffle and bootstrap distributions
    # ------------------------------------------------------------------

    mc_bootstrap = run_monte_carlo(
        backtest_result,
        MonteCarloConfig(
            method="bootstrap",
            num_simulations=2000,
            risk_per_trade_pct=risk,
            initial_balance=initial_balance,
            random_seed=42,
        ),
    )

    shuffle_spread = (
        mc_shuffle.summary["final_balance"]["p95"]
        - mc_shuffle.summary["final_balance"]["p5"]
    )

    bootstrap_spread = (
        mc_bootstrap.summary["final_balance"]["p95"]
        - mc_bootstrap.summary["final_balance"]["p5"]
    )

    print("\n[Check 2] Shuffle vs bootstrap spread")
    print(f"  Shuffle   P5-P95 spread: {shuffle_spread:.2f}")
    print(f"  Bootstrap P5-P95 spread: {bootstrap_spread:.2f}")
    print(
        "  -> Expected: Bootstrap spread >= shuffle spread "
        "because bootstrap perturbs trade composition as well as order."
    )

    # ------------------------------------------------------------------
    # CHECK 3
    # Observe ruin probability trend as risk increases
    # ------------------------------------------------------------------

    ruin_probs = []
    for risk_pct in (0.005, 0.01, 0.02):
        result = run_monte_carlo(
            backtest_result,
            MonteCarloConfig(
                method="bootstrap",
                num_simulations=1000,
                risk_per_trade_pct=risk_pct,
                initial_balance=initial_balance,
                random_seed=42,
            ),
        )
        ruin_probs.append(result.probability_of_ruin)

    print("\n[Check 3] Probability of ruin trend")
    print(f"  0.5% risk/trade: {ruin_probs[0]:.4f}")
    print(f"  1.0% risk/trade: {ruin_probs[1]:.4f}")
    print(f"  2.0% risk/trade: {ruin_probs[2]:.4f}")
    print(
        "  -> Trend diagnostic: generally expected to increase with risk size "
        "(small finite sample reversals are normal and non-fatal)."
    )

    # ------------------------------------------------------------------
    # CHECK 4
    # Observe convergence as simulation count increases
    # ------------------------------------------------------------------

    medians = []
    for n in (100, 1000, 5000):
        result = run_monte_carlo(
            backtest_result,
            MonteCarloConfig(
                method="bootstrap",
                num_simulations=n,
                risk_per_trade_pct=risk,
                initial_balance=initial_balance,
                random_seed=42,
            ),
        )
        medians.append(result.summary["final_balance"]["median"])

    mean_median = statistics.fmean(medians)
    spread_pct = (
        ((max(medians) - min(medians)) / abs(mean_median)) * 100.0
        if mean_median != 0
        else 0.0
    )

    print("\n[Check 4] Monte Carlo convergence")
    print(f"  Median final balance @ 100:  {medians[0]:.2f}")
    print(f"  Median final balance @ 1000: {medians[1]:.2f}")
    print(f"  Median final balance @ 5000: {medians[2]:.2f}")
    print(f"  Range across sample sizes:   {spread_pct:.2f}%")

    # ------------------------------------------------------------------
    # CHECK 5
    # Known synthetic ledgers
    # ------------------------------------------------------------------

    all_wins = run_monte_carlo(
        [1.0, 1.0, 1.0, 1.0],
        MonteCarloConfig(
            method="shuffle",
            num_simulations=50,
            random_seed=1,
        ),
    )

    all_losses = run_monte_carlo(
        [-1.0, -1.0, -1.0],
        MonteCarloConfig(
            method="shuffle",
            num_simulations=50,
            random_seed=1,
            ruin_drawdown_pct=0.5,
            risk_per_trade_pct=0.5,
        ),
    )

    wins_ok = all_wins.probability_of_ruin == 0.0
    losses_ok = all_losses.probability_of_ruin == 1.0

    print("\n[Check 5] Synthetic ledgers")
    print(
        f"  All wins probability_of_ruin:   {all_wins.probability_of_ruin}"
        f" -> {'OK' if wins_ok else 'FAIL'}"
    )
    print(
        f"  All losses probability_of_ruin: {all_losses.probability_of_ruin}"
        f" -> {'OK' if losses_ok else 'FAIL'}"
    )

    # ------------------------------------------------------------------
    # CHECK 6
    # Reproducibility
    # ------------------------------------------------------------------

    cfg = MonteCarloConfig(
        num_simulations=200,
        random_seed=99,
    )
    run_a = run_monte_carlo(backtest_result, cfg)
    run_b = run_monte_carlo(backtest_result, cfg)

    identical = (
        [r.final_balance for r in run_a.runs]
        == [r.final_balance for r in run_b.runs]
    )

    print("\n[Check 6] Reproducibility")
    print(f"  Identical runs with same seed: {identical}")
    print(
        "  -> Expected: True",
        "OK" if identical else "!!! BUG !!!",
    )

    # ------------------------------------------------------------------
    # FINAL SUMMARY
    # ------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)

    hard_checks = {
        "Check 1 - Shuffle final balance invariance": check_1_ok,
        "Check 5 - Synthetic all wins": wins_ok,
        "Check 5 - Synthetic all losses": losses_ok,
        "Check 6 - Reproducibility": identical,
    }

    all_passed = all(hard_checks.values())

    for name, passed in hard_checks.items():
        print(f"  {'PASS' if passed else 'FAIL'} — {name}")

    print()
    if all_passed:
        print("PHASE 2 MONTE CARLO VERIFICATION PASSED.")
        print("You are clear to proceed to walk_forward.py.")
    else:
        print("PHASE 2 MONTE CARLO VERIFICATION FAILED.")
        print("Resolve the failing check before building walk_forward.py.")


if __name__ == "__main__":
    main()