"""
phase_02_optimization/spread_sensitivity.py

Transaction-cost stress diagnostic for Parameter C.

Purpose
-------
Parameter C (stop_buffer) sensitivity testing showed:

    stop_buffer=0.0000 -> 1.2500R expectancy
    stop_buffer=0.0005 -> -0.0625R expectancy

with a progressive deterioration between those values.

The baseline stop_buffer=0.0 is also the historical best and sits at
the hard lower boundary of the tested range. Before treating that
result as meaningful, this diagnostic stress-tests the same backtest
trades against increasing illustrative transaction-cost assumptions.

IMPORTANT: DATASET IS SYNTHETIC
--------------------------------
your_data_file.csv is confirmed synthetic random-walk data
(see generate_synthetic_data.py).

Therefore this module does NOT claim:

    - that any tested cost is the real EUR/USD spread;
    - that the resulting expectancy represents live performance;
    - that the strategy has a real market edge.

The purpose is narrower:

    "How sensitive is the historical synthetic result to additional
     per-trade transaction costs?"

TRANSACTION COST VS STOP BUFFER
--------------------------------
These are deliberately kept separate.

stop_buffer:
    Changes the structural stop-loss level itself and was already
    tested by stop_buffer_diagnostic.py.

round_trip_cost:
    Is an illustrative price-unit cost deducted from each completed
    trade after the canonical backtest has finished.

This diagnostic does NOT:
    - move the stop-loss;
    - move the take-profit;
    - change signal generation;
    - change retest behavior;
    - change trade lifecycle;
    - simulate bid/ask candles;
    - simulate broker execution;
    - rerun the backtest for every cost value.

Instead, for every closed trade:

    risk_price = abs(entry_price - stop_loss)

    cost_R = round_trip_cost / risk_price

    adjusted_R = original_R - cost_R

The adjusted expectancy is then the mean adjusted_R across the
closed trades.

Because cost_R is linear in round_trip_cost for a fixed set of trades,
adjusted_expectancy(cost) is itself linear in cost:

    adjusted_expectancy(cost) = original_expectancy - cost * mean(1/risk_price)

which lets us solve for the exact breakeven cost (see
BREAKEVEN COST below) instead of only reading it off a coarse table.

IMPORTANT COST CONVENTION
-------------------------
The values in ROUND_TRIP_COST_VALUES represent a TOTAL ROUND-TRIP
price-unit cost per completed trade.

They are NOT quoted bid/ask spread values.

This distinction prevents the diagnostic from incorrectly treating
a normal quoted spread as though it were automatically the complete
round-trip transaction cost.

A proper bid/ask execution model would require separate bid/ask data
and explicit rules for entry, stop, target, and exit fills. This module
does not attempt to build that model.

Interpretation
--------------
If expectancy remains positive across reasonable illustrative costs,
the historical result is less exposed to transaction costs.

If expectancy deteriorates rapidly or becomes negative even under
small illustrative costs, the historical result is highly exposed
to execution costs.

Because the underlying dataset is synthetic and the sample is small,
neither outcome establishes real-world profitability.

This module calls strategy.backtest.run_backtest() unmodified and
performs no strategy changes.
"""

from __future__ import annotations

from strategy.backtest import run_backtest
from phase_02_optimization.walk_forward import load_dataset, DATA_FILE


# ---------------------------------------------------------------------
# Fixed backtest configuration
# ---------------------------------------------------------------------

# Parameter C baseline being stress-tested.
FIXED_STOP_BUFFER = 0.0

COMMON_KWARGS = dict(
    max_bars_to_retest=20,
    reward_multiple=2.0,
    entry_mode="midpoint",
)


# ---------------------------------------------------------------------
# Illustrative transaction-cost stress values
# ---------------------------------------------------------------------

# These are TOTAL ROUND-TRIP price-unit costs.
#
# They are deliberately illustrative because the dataset is synthetic.
# They are NOT claimed to be actual broker spreads.
ROUND_TRIP_COST_VALUES = [
    0.0000,
    0.0001,
    0.0002,
    0.0003,
    0.0005,
    0.0010,
]


def main() -> None:
    df = load_dataset(DATA_FILE)

    print()
    print("=" * 110)
    print("TRANSACTION-COST SENSITIVITY DIAGNOSTIC")
    print("=" * 110)
    print(
        f"stop_buffer held fixed at: {FIXED_STOP_BUFFER}"
    )
    print(
        "Dataset: synthetic M5 EURUSD-like random walk."
    )
    print(
        "Cost values below are ILLUSTRATIVE TOTAL ROUND-TRIP "
        "price-unit costs, not empirical broker spreads."
    )
    print()

    # ---------------------------------------------------------------
    # Run the canonical backtest once.
    # ---------------------------------------------------------------

    result = run_backtest(
        df,
        stop_buffer=FIXED_STOP_BUFFER,
        **COMMON_KWARGS,
    )

    closed_trades = [
        t
        for t in result.trades
        if t.result_status in ("WIN", "LOSS")
    ]

    if not closed_trades:
        print(
            "No closed trades at this stop_buffer -- "
            "nothing to analyze."
        )
        return

    # ---------------------------------------------------------------
    # Prepare valid trades.
    # ---------------------------------------------------------------

    trade_risks: list[tuple[object, float]] = []
    skipped = 0

    for trade in closed_trades:
        risk_price = abs(
            trade.entry_price - trade.stop_loss
        )

        if risk_price <= 0:
            skipped += 1
            continue

        trade_risks.append(
            (trade, risk_price)
        )

    if not trade_risks:
        print(
            "No trades with valid positive risk distance "
            "are available for analysis."
        )
        return

    if skipped:
        print(
            f"NOTE: skipped {skipped} trade(s) with "
            "zero/invalid risk distance."
        )
        print()

    # ---------------------------------------------------------------
    # Original expectancy.
    # ---------------------------------------------------------------

    n_trades = len(trade_risks)

    original_expectancy = (
        sum(
            trade.r_multiple
            for trade, _ in trade_risks
        )
        / n_trades
    )

    # ---------------------------------------------------------------
    # Exact breakeven cost.
    #
    # adjusted_expectancy(cost) = original_expectancy - cost * mean_inv_risk
    # is linear in cost, so we can solve directly rather than reading
    # an approximate crossing point off the swept table.
    # ---------------------------------------------------------------

    mean_inv_risk = (
        sum(1.0 / risk_price for _, risk_price in trade_risks)
        / n_trades
    )

    breakeven_cost = (
        original_expectancy / mean_inv_risk
        if mean_inv_risk > 0
        else float("inf")
    )

    # ---------------------------------------------------------------
    # Report.
    # ---------------------------------------------------------------

    print(
        f"{'round_trip_cost':>17} "
        f"{'n_trades':>10} "
        f"{'orig_expectancy_R':>19} "
        f"{'cost_adj_expectancy_R':>24} "
        f"{'delta_R':>12} "
        f"{'net_pos':>9} "
        f"{'net_neg':>9}"
    )

    print("-" * 110)

    for round_trip_cost in ROUND_TRIP_COST_VALUES:
        adjusted_r_multiples = []

        for trade, risk_price in trade_risks:

            # Convert the illustrative price-unit transaction
            # cost into R using that individual trade's risk.
            cost_r = round_trip_cost / risk_price

            adjusted_r = trade.r_multiple - cost_r

            adjusted_r_multiples.append(adjusted_r)

        adjusted_expectancy = (
            sum(adjusted_r_multiples)
            / len(adjusted_r_multiples)
        )

        delta = (
            adjusted_expectancy
            - original_expectancy
        )

        # Net economic outcome after cost, independent of the
        # original WIN/LOSS status label -- a trade recorded as WIN
        # can still be net-negative once cost is subtracted, and this
        # count is the only place that shows up.
        n_net_positive = sum(1 for r in adjusted_r_multiples if r > 0)
        n_net_negative = sum(1 for r in adjusted_r_multiples if r < 0)

        print(
            f"{round_trip_cost:>17.4f} "
            f"{len(adjusted_r_multiples):>10} "
            f"{original_expectancy:>19.4f} "
            f"{adjusted_expectancy:>24.4f} "
            f"{delta:>12.4f} "
            f"{n_net_positive:>9} "
            f"{n_net_negative:>9}"
        )

    # ---------------------------------------------------------------
    # Final interpretation.
    # ---------------------------------------------------------------

    print("-" * 110)

    print(
        "INTERPRETATION:"
    )
    print(
        "This diagnostic measures how much of the synthetic "
        "stop_buffer=0.0 result survives after subtracting "
        "illustrative round-trip transaction costs."
    )
    print()
    print(
        "It does NOT simulate bid/ask execution and does NOT "
        "estimate the actual spread of a broker or data feed."
    )
    print()
    print(
        "Because the underlying dataset is synthetic random-walk "
        "data, these adjusted expectancy values are execution-cost "
        "stress results only, not evidence of live profitability."
    )
    print()
    print(
        f"Closed trades analyzed: {n_trades}"
    )
    print(
        f"Original expectancy at stop_buffer={FIXED_STOP_BUFFER}: "
        f"{original_expectancy:.4f}R"
    )
    print(
        f"Exact breakeven round-trip cost (expectancy = 0): "
        f"{breakeven_cost:.6f} price units"
    )
    print(
        "(This is a closed-form result derived from this exact "
        f"{n_trades}-trade sample -- it moves if the sample changes, "
        "it is not a general property of the strategy.)"
    )

    print("=" * 110)
    print()


if __name__ == "__main__":
    main()