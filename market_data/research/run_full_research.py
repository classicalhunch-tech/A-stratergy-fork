"""
market_data/research/run_full_research.py

Ties together Steps 2-8 of the plan on real trade data:
  2. imbalance vs future returns
  3. multiple horizons (10s, 30s, 60s, 300s)
  4. descriptive deciles
  5. correlation screening
  6. candidate threshold inspection (not auto-selected)
  7-8. chronological OOS comparison

This produces numbers to READ, not a decision. Nothing here filters trades,
picks a threshold, or touches strategy/ or orderflow/.

Usage:
    python -m market_data.research.run_full_research \
        --file market_data/data/BTCUSDT_trades_normalized.csv \
        --time-window 60s --trade-window 500

Outputs a research CSV plus printed summary tables.
"""

from __future__ import annotations

import argparse

import pandas as pd

from .imbalance_response import build_research_table, imbalance_deciles_vs_returns, correlation_summary
from .threshold_analysis import evaluate_threshold_grid_for_inspection
from .oos_validation import compare_deciles_oos

HORIZONS = [10, 30, 60, 300]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    parser.add_argument("--time-window", default="60s")
    parser.add_argument("--trade-window", type=int, default=500)
    parser.add_argument("--n-buckets", type=int, default=10)
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument(
        "--min-trades",
        type=int,
        default=1,
        help="Minimum trades required in a window for imbalance to be reported (else NaN). "
        "Check n_trades_* distribution first -- bursty data can need this set well above 1.",
    )
    args = parser.parse_args()

    print(f"Loading {args.file} ...")
    df = pd.read_csv(args.file)
    print(f"  {len(df):,} trades")

    print(f"\nBuilding research table (time_window={args.time_window}, trade_window={args.trade_window}, "
          f"min_trades={args.min_trades}, horizons={HORIZONS}s) ...")
    research = build_research_table(
        df,
        time_window=args.time_window,
        trade_window=args.trade_window,
        return_horizons_seconds=HORIZONS,
        min_trades=args.min_trades,
    )

    n_trades_col = f"n_trades_t{args.time_window}"
    if n_trades_col in research.columns:
        dropped = research[n_trades_col].lt(args.min_trades).sum()
        total = len(research)
        print(f"  {dropped:,}/{total:,} rows ({dropped/total*100:.1f}%) have fewer than "
              f"{args.min_trades} trades in the {args.time_window} window -> imbalance set to NaN")

    imbalance_cols = [f"imbalance_t{args.time_window}", f"imbalance_n{args.trade_window}"]
    return_cols = [f"future_return_{h}s" for h in HORIZONS]

    # --- Step 5: correlation screening ---
    print("\n" + "=" * 70)
    print("CORRELATION SCREENING (Pearson r -- a screen, not a conclusion)")
    print("=" * 70)
    corr = correlation_summary(research, imbalance_cols, return_cols)
    print(corr.to_string(index=False))

    # --- Step 4: descriptive deciles, one imbalance feature x one horizon at a time ---
    for icol in imbalance_cols:
        for rcol in return_cols:
            print("\n" + "-" * 70)
            print(f"DECILES: {icol}  vs  {rcol}")
            print("-" * 70)
            try:
                deciles = imbalance_deciles_vs_returns(research, icol, rcol, n_buckets=args.n_buckets)
                print(deciles.to_string())
            except ValueError as e:
                print(f"  [skipped] {e}")

    # --- Step 6: candidate threshold inspection (not auto-selected) ---
    print("\n" + "=" * 70)
    print("CANDIDATE THRESHOLD INSPECTION (you choose, this does not pick a winner)")
    print("=" * 70)
    for icol in imbalance_cols:
        rcol = f"future_return_60s"
        print(f"\n{icol} vs {rcol}:")
        grid = evaluate_threshold_grid_for_inspection(
            research, icol, rcol, candidate_thresholds=[-0.5, -0.2, 0.0, 0.2, 0.5]
        )
        print(grid.to_string(index=False))

    # --- Steps 7-8: chronological OOS comparison ---
    print("\n" + "=" * 70)
    print(f"OUT-OF-SAMPLE COMPARISON (train_frac={args.train_frac}, chronological, no shuffling)")
    print("=" * 70)
    for icol in imbalance_cols:
        rcol = "future_return_60s"
        print(f"\n{icol} vs {rcol}:")
        try:
            oos = compare_deciles_oos(
                research, icol, rcol, n_buckets=5, train_frac=args.train_frac
            )
            print("  -- TRAIN (development) --")
            print(oos["train"].to_string())
            print("  -- TEST (out-of-sample) --")
            print(oos["test"].to_string())
        except ValueError as e:
            print(f"  [skipped] {e}")

    out_path = args.file.replace(".csv", "_research.csv")
    research.to_csv(out_path, index=False)
    print(f"\nWrote full research table -> {out_path}")
    print("\nReminder: nothing above is a recommendation. Read the shape of each")
    print("table -- monotonic? consistent between train/test? -- before drawing")
    print("any conclusion, per Steps 9-10 of the plan.")


if __name__ == "__main__":
    main()