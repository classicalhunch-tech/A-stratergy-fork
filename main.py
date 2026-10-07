"""
main.py

End-to-end multi-timeframe SMC pipeline runner.

Pipeline
--------
5M OHLC
    v
1H Macro + 15M Internal Structure (MTF context)
    v
run_backtest() [tested, validated engine]
    - Swings
    - Market Structure
    - Liquidity (Active -> Swept)
    - Supply/Demand Zones
    - Flip Zone signal generation
    - MTF Confluence Guard (per-signal filter, before execution)
    - Retest -> Trigger -> WIN/LOSS/OPEN lifecycle
    v
Final trade results

Architecture
------------
main.py is the orchestration layer only.

The strategy engines remain responsible for:
    - Swing detection
    - Market structure
    - Liquidity
    - Zones
    - Signal generation
    - Retest / trigger / exit lifecycle
    - MTF structure
    - MTF confluence
    - Structural targets (strategy/structure_targets.py)

run_backtest() accepts an optional mtf_filter_fn hook: when None it
behaves exactly as in every existing test. When provided, only
MTF-approved signals proceed to entry/retest/execution. No execution
logic is duplicated here.

Direction gate
--------------
By default the MTF filter requires the 1H (and 15M) trend to agree
with the trade direction (the original behavior). The strategy is a
counter-trend reversal from higher-timeframe zones, so
--no-direction-gate turns the direction requirement off. The 1H/15M
trend labels are then only written as tags in the --output CSV.

Trading costs
-------------
--spread, --slippage and --commission are passed straight through to
run_backtest(). All three are in PRICE UNITS (for XAUUSD, 0.30 means
30 cents) and default to 0.0. The round-trip cost deducted from each
closed trade is spread + (2 * slippage) + commission, converted to R
by dividing by that trade's initial risk (fill to stop).

Because costs are in price units, they must match the price scale of
the data. A guard refuses to run when the round-trip cost is more than
half the median 5M candle range (for example gold-scale costs on
EURUSD-scale data), because the resulting R numbers would be
meaningless.

Exit / fill options
-------------------
--reward-multiple, --fill-mode, --target-anchor, --strict-entry-fill
and --stop-atr-mult are passed straight through to run_backtest().
Defaults equal run_backtest()'s own defaults, so omitting them
reproduces the previous results exactly.

--target-anchor structure uses the nearest CONFIRMED 15M swing as the
target (see strategy/structure_targets.py). It needs --fill-mode
market_on_close. --min-rr (default 1.5) is the minimum reward to that
swing; trades below it are skipped, or with --advance-to-min-rr the
next swing that reaches it is used instead.

--stop-atr-mult widens the stop beyond the zone edge by that multiple
of the causal ATR(14) known at the fill bar's open. It needs
--fill-mode market_on_close.
"""

import argparse
import sys
import time
import traceback
from pathlib import Path

import pandas as pd

from strategy.backtest import run_backtest
from strategy.mtf_structure import build_mtf_dataset_with_structure
from strategy.confluence import build_mtf_signal_filter, build_mtf_tag_fn
from strategy.structure_targets import (
    StructureTargetConfig,
    build_structure_target_fn,
)
from strategy.swings import find_swings
from dashboard.mtf_context import MTFConfig


# =====================================================================
# Constants
# =====================================================================

REQUIRED_OHLC_COLUMNS = {"open", "high", "low", "close"}

# Multi-timeframe settings (one place to change them)
MTF_MACRO_TF = "1h"
MTF_INTERNAL_TF = "15min"

# Minimum reward to the structural target, in R (spec value).
DEFAULT_MIN_RR = 1.5

# Costs may not exceed this fraction of the median 5M candle range.
# Real gold: median range ~4.35, a 0.50 cost is ~0.11 of it -- fine.
# Gold-scale costs on EURUSD-scale data is ~1500x -- refused.
MAX_COST_TO_RANGE_RATIO = 0.5

# Errors we treat as EXPECTED pipeline conditions (bad input, bad data).
# These get a clean one-line message because the user can act on them.
EXPECTED_ERRORS = (FileNotFoundError, ValueError)


# =====================================================================
# Step 1: Load and clean raw 5M OHLC data
# =====================================================================

def load_5m_data(csv_path: str) -> pd.DataFrame:
    """
    Load, clean, and validate the raw 5M OHLC CSV.

    Expected format:
        timestamp,open,high,low,close[,volume]

    The timestamp is stored as the DatetimeIndex used by the rest
    of the strategy architecture.
    """

    path = Path(csv_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Data file not found: {path}\n"
            f"Pass the correct path with --data <file.csv>."
        )

    df = pd.read_csv(path, parse_dates=True, index_col=0)

    # Normalize column names
    df.columns = [str(column).strip().lower() for column in df.columns]

    # Validate timestamp index
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")

    invalid_timestamp_count = int(df.index.isna().sum())
    if invalid_timestamp_count:
        print(f"   WARNING: removing {invalid_timestamp_count} invalid timestamp row(s).")
        df = df[df.index.notna()]

    df = df.sort_index()

    duplicate_count = int(df.index.duplicated().sum())
    if duplicate_count:
        print(f"   WARNING: {duplicate_count} duplicate timestamp(s) found -- keeping the first occurrence.")
        df = df[~df.index.duplicated(keep="first")]

    # Defensive copy: everything above is boolean-mask filtering, which
    # can leave df as a view. Make "this is my own frame now" explicit.
    df = df.copy()

    missing = REQUIRED_OHLC_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"5M dataset is missing required columns: {sorted(missing)}")

    for column in sorted(REQUIRED_OHLC_COLUMNS):
        df[column] = pd.to_numeric(df[column], errors="coerce")

    invalid_ohlc_mask = df[list(REQUIRED_OHLC_COLUMNS)].isna().any(axis=1)
    invalid_ohlc_count = int(invalid_ohlc_mask.sum())
    if invalid_ohlc_count:
        print(f"   WARNING: removing {invalid_ohlc_count} row(s) with invalid OHLC values.")
        df = df[~invalid_ohlc_mask]

    if df.empty:
        raise ValueError("5M dataset is empty after loading and cleaning.")

    if not df.index.is_monotonic_increasing:
        raise ValueError("5M dataset index is not chronologically sorted.")

    print(f"   Loaded {len(df)} rows of 5M data.")
    print(f"   Data range: {df.index[0]} -> {df.index[-1]}")

    return df


# =====================================================================
# Step 1b: Guard against cost / price-scale mismatch
# =====================================================================

def check_cost_scale(
    df_5m: pd.DataFrame,
    spread: float,
    slippage: float,
    commission: float,
) -> None:
    """
    Refuse to run when trading costs are on a different price scale
    than the data.

    Costs are in price units. If the round-trip cost is more than
    MAX_COST_TO_RANGE_RATIO of the median 5M candle range, every trade
    would be charged many R of cost and the expectancy would be
    meaningless (e.g. gold-scale 0.30 spread on EURUSD-scale data
    produced -2650R). Fail loudly instead of printing that number.
    """

    round_trip_cost = spread + (2.0 * slippage) + commission

    if round_trip_cost <= 0:
        return

    median_range = float((df_5m["high"] - df_5m["low"]).median())

    if round_trip_cost > MAX_COST_TO_RANGE_RATIO * median_range:
        raise ValueError(
            f"Round-trip cost {round_trip_cost} (spread + 2*slippage + "
            f"commission) is more than {MAX_COST_TO_RANGE_RATIO:.0%} of "
            f"the median 5M candle range ({median_range:.6f}). Costs are "
            f"in price units, and this data is on a different price scale "
            f"than the costs assume. Use real gold data or scale the costs."
        )


# =====================================================================
# Step 2: Build MTF structural context
# =====================================================================

def build_mtf_context(df_5m: pd.DataFrame) -> pd.DataFrame:
    """
    Build the lookahead-safe MTF dataset.

    Higher-timeframe structure:
        1H = macro context
        15M = internal context

    The actual MTF logic remains inside strategy.mtf_structure.
    """

    df_enriched = build_mtf_dataset_with_structure(
        df_5m,
        macro_swings_fn=find_swings,
        internal_swings_fn=find_swings,
        config=MTFConfig(
            macro_tf=MTF_MACRO_TF,
            internal_tf=MTF_INTERNAL_TF,
        ),
    )

    print("   MTF enrichment complete.")

    mtf_columns = [
        column for column in df_enriched.columns
        if column.startswith("macro_") or column.startswith("internal_")
    ]

    print(f"   MTF columns ({len(mtf_columns)}):")
    for column in mtf_columns:
        print(f"      - {column}")

    return df_enriched


# =====================================================================
# Step 3: Run the tested backtest engine, gated by MTF confluence
# =====================================================================

def run_mtf_gated_backtest(
    df_5m: pd.DataFrame,
    df_enriched: pd.DataFrame,
    allow_neutral_internal: bool = True,
    spread: float = 0.0,
    slippage: float = 0.0,
    commission: float = 0.0,
    gate_direction: bool = True,
    reward_multiple: float = 2.0,
    fill_mode: str = "touch",
    target_anchor: str = "zone",
    strict_entry_fill: bool = False,
    target_fn=None,
    stop_atr_mult: float = 0.0,
):
    """
    Run the SAME tested run_backtest() engine used by Phase 1/2/3,
    with an MTF confluence filter gating which signals are allowed
    to proceed to entry/retest/execution.

    No execution logic is duplicated here or in confluence.py --
    the filter is a predicate checked per-signal at the exact point
    a fresh signal would otherwise be registered as pending. The
    trade lifecycle itself (retest, trigger, SL/TP, WIN/LOSS/OPEN)
    is untouched, tested code.

    spread, slippage and commission are in price units and are passed
    straight through to run_backtest(). All default to 0.0.

    gate_direction=False removes the 1H/15M trend-direction
    requirement (counter-trend entries allowed).

    reward_multiple, fill_mode, target_anchor, strict_entry_fill,
    target_fn and stop_atr_mult are passed straight through to
    run_backtest(); defaults equal run_backtest()'s own defaults.
    """

    mtf_filter_fn = build_mtf_signal_filter(
        df_enriched,
        allow_neutral_internal=allow_neutral_internal,
        gate_direction=gate_direction,
    )

    result = run_backtest(
        df_5m,
        reward_multiple=reward_multiple,
        mtf_filter_fn=mtf_filter_fn,
        spread=spread,
        slippage=slippage,
        commission=commission,
        strict_entry_fill=strict_entry_fill,
        fill_mode=fill_mode,
        target_anchor=target_anchor,
        target_fn=target_fn,
        stop_atr_mult=stop_atr_mult,
    )

    closed_count = sum(
        1 for t in result.trades
        if t.result_status in {"WIN", "LOSS"}
    )

    print("--------------------------------------------------")
    print("MTF-GATED BACKTEST COMPLETE")
    print("--------------------------------------------------")
    print(
        "Costs (price units):              "
        f"spread={spread} slippage={slippage} commission={commission}"
    )
    print(
        "Direction gate:                   "
        f"{'ON' if gate_direction else 'OFF (counter-trend allowed)'}"
    )
    print(
        "Exit / fill settings:             "
        f"reward={reward_multiple}R fill_mode={fill_mode} "
        f"target_anchor={target_anchor} strict_entry_fill={strict_entry_fill} "
        f"stop_atr_mult={stop_atr_mult}"
    )
    print(f"Signals generated (MTF-approved): {result.total_signals_generated}")
    print(f"Rejected by MTF confluence:       {result.total_mtf_rejected}")
    print(f"Triggered:                        {result.total_trades_triggered}")
    print(f"Invalidated:                      {result.total_invalidated}")
    print(f"Expired:                          {result.total_expired}")
    print(f"Closed trades (WIN/LOSS):         {closed_count}")
    print(f"Win rate:                         {result.win_rate:.2f}%")
    print(f"Expectancy:                       {result.expectancy:.4f}R")

    if result.errors:
        print(f"\nWARNING: {len(result.errors)} error(s) during replay:")
        for err in result.errors[:10]:
            print(f"   {err}")

    return result


# =====================================================================
# Output helpers
# =====================================================================

def _signal_time(df_5m: pd.DataFrame, trade):
    """
    Timestamp of the bar where the signal was registered (the break
    bar), or None if it cannot be determined.
    """

    bar_index = getattr(trade.signal, "bar_index", None)

    try:
        return df_5m.index[int(bar_index)]
    except (TypeError, ValueError, IndexError):
        return None


def build_trade_rows(result, df_5m: pd.DataFrame, df_enriched: pd.DataFrame):
    """
    One row per closed trade, tagged with the 1H macro and 15M internal
    trend at signal time. The tags are for reporting only.
    """

    tag_fn = build_mtf_tag_fn(df_enriched)

    closed = [t for t in result.trades if t.result_status in {"WIN", "LOSS"}]

    rows = []

    for t in closed:

        signal_ts = _signal_time(df_5m, t)

        if signal_ts is not None:
            macro_tag, internal_tag = tag_fn(signal_ts)
        else:
            macro_tag, internal_tag = None, None

        if t.initial_risk > 0:
            planned_rr = abs(t.take_profit - t.fill_price) / t.initial_risk
        else:
            planned_rr = None

        rows.append(
            {
                "setup_time": t.setup_time,
                "signal_time": signal_ts,
                "entry_time": t.entry_time,
                "exit_time": t.exit_time,
                "direction": getattr(t.direction, "value", t.direction),
                "macro_trend_at_signal": macro_tag,
                "internal_trend_at_signal": internal_tag,
                "entry_price": t.fill_price,
                "exit_price": t.exit_price,
                "stop_loss": t.stop_loss,
                "take_profit": t.take_profit,
                "initial_risk": t.initial_risk,
                "planned_rr": planned_rr,
                "result_status": t.result_status,
                "r_multiple": t.r_multiple,
                "bars_held": t.bars_held,
                "session": t.session,
            }
        )

    return rows


# =====================================================================
# Main entry point
# =====================================================================

def main():

    parser = argparse.ArgumentParser(
        description="Run the multi-timeframe SMC pipeline (MTF-gated backtest)."
    )

    parser.add_argument(
        "--data",
        default="your_data_file.csv",
        help="Path to the 5M OHLC CSV (default: your_data_file.csv)",
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Optional path for saving closed trades (with 1H/15M trend tags) as CSV.",
    )

    parser.add_argument(
        "--no-mtf-neutral",
        action="store_true",
        help="Reject signals when internal (15M) trend is neutral, instead of allowing them.",
    )

    parser.add_argument(
        "--no-direction-gate",
        action="store_true",
        help=(
            "Do not require the 1H/15M trend to agree with the trade "
            "direction (counter-trend entries allowed). Trends are "
            "still written as tags in --output."
        ),
    )

    parser.add_argument(
        "--spread",
        type=float,
        default=0.0,
        help="Spread per round trip in price units (XAUUSD: 0.30 = 30 cents). Default 0.0.",
    )

    parser.add_argument(
        "--slippage",
        type=float,
        default=0.0,
        help="Adverse slippage per side in price units. Applied on entry and exit. Default 0.0.",
    )

    parser.add_argument(
        "--commission",
        type=float,
        default=0.0,
        help="Total round-trip commission in price units. Default 0.0.",
    )

    parser.add_argument(
        "--reward-multiple",
        type=float,
        default=2.0,
        help="Target as a multiple of risk (default 2.0, the original fixed target).",
    )

    parser.add_argument(
        "--fill-mode",
        choices=["touch", "market_on_close"],
        default="touch",
        help=(
            "touch (default): fill on zone-edge touch. "
            "market_on_close: trigger on touch, fill at the next bar's open."
        ),
    )

    parser.add_argument(
        "--target-anchor",
        choices=["zone", "fill", "structure"],
        default="zone",
        help=(
            "zone (default): target stays where the zone-midpoint entry put it. "
            "fill: target = reward-multiple x real risk. "
            "structure: target = nearest confirmed 15M swing (see --min-rr). "
            "fill and structure require --fill-mode market_on_close."
        ),
    )

    parser.add_argument(
        "--min-rr",
        type=float,
        default=None,
        help=(
            f"With --target-anchor structure: minimum reward to the 15M swing, "
            f"in R (default {DEFAULT_MIN_RR}). Trades below it are skipped."
        ),
    )

    parser.add_argument(
        "--advance-to-min-rr",
        action="store_true",
        help=(
            "With --target-anchor structure: instead of skipping a trade whose "
            "nearest swing is below --min-rr, use the nearest swing that reaches it."
        ),
    )

    parser.add_argument(
        "--stop-atr-mult",
        type=float,
        default=0.0,
        help=(
            "Widen the stop beyond the zone edge by this multiple of the "
            "causal ATR(14) known at the fill bar's open. Default 0.0 (off). "
            "Requires --fill-mode market_on_close."
        ),
    )

    parser.add_argument(
        "--strict-entry-fill",
        action="store_true",
        help="Fill only if price actually reaches the entry (cannot combine with market_on_close).",
    )

    args = parser.parse_args()

    # Fail fast, before the expensive MTF build.
    if args.target_anchor != "zone" and args.fill_mode != "market_on_close":
        parser.error("--target-anchor fill/structure requires --fill-mode market_on_close")

    if args.stop_atr_mult < 0:
        parser.error("--stop-atr-mult must be >= 0")

    if args.stop_atr_mult > 0 and args.fill_mode != "market_on_close":
        parser.error("--stop-atr-mult > 0 requires --fill-mode market_on_close")

    if args.strict_entry_fill and args.fill_mode == "market_on_close":
        parser.error("--strict-entry-fill cannot be combined with --fill-mode market_on_close")

    if args.target_anchor != "structure" and (
        args.min_rr is not None or args.advance_to_min_rr
    ):
        parser.error("--min-rr and --advance-to-min-rr only apply with --target-anchor structure")

    if args.min_rr is not None and args.min_rr <= 0:
        parser.error("--min-rr must be > 0")

    min_rr = args.min_rr if args.min_rr is not None else DEFAULT_MIN_RR

    start_time = time.perf_counter()

    print("==================================================")
    print("RUNNING TRUE MTF SMC PIPELINE (MTF-gated backtest)")
    print("==================================================")

    try:

        print("\n1. Loading raw 5M OHLC dataset...")
        df_5m = load_5m_data(args.data)

        check_cost_scale(
            df_5m,
            spread=args.spread,
            slippage=args.slippage,
            commission=args.commission,
        )

        target_fn = None
        if args.target_anchor == "structure":
            target_fn = build_structure_target_fn(
                df_5m,
                tf=MTF_INTERNAL_TF,
                config=StructureTargetConfig(
                    min_rr=min_rr,
                    advance_to_min_rr=args.advance_to_min_rr,
                ),
            )
            print(
                f"   Structural target: nearest confirmed {MTF_INTERNAL_TF} swing, "
                f"min {min_rr}R, "
                f"{'advance to next swing' if args.advance_to_min_rr else 'skip if below min'}."
            )

        print("\n2. Building 1H macro and 15M internal structure...")
        df_enriched = build_mtf_context(df_5m)

        print("\n3. Running MTF-gated backtest (tested engine + confluence guard)...")
        result = run_mtf_gated_backtest(
            df_5m,
            df_enriched,
            allow_neutral_internal=not args.no_mtf_neutral,
            spread=args.spread,
            slippage=args.slippage,
            commission=args.commission,
            gate_direction=not args.no_direction_gate,
            reward_multiple=args.reward_multiple,
            fill_mode=args.fill_mode,
            target_anchor=args.target_anchor,
            strict_entry_fill=args.strict_entry_fill,
            target_fn=target_fn,
            stop_atr_mult=args.stop_atr_mult,
        )

        if args.output:
            rows = build_trade_rows(result, df_5m, df_enriched)
            pd.DataFrame(rows).to_csv(args.output, index=False)
            print(f"\nSaved {len(rows)} closed trade(s) to {args.output}")

    except EXPECTED_ERRORS as exc:
        print(f"\nPIPELINE ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    except Exception:
        print(
            "\nUNEXPECTED ERROR -- this looks like a bug, not a data problem:",
            file=sys.stderr,
        )
        traceback.print_exc()
        sys.exit(1)

    elapsed = time.perf_counter() - start_time
    print(f"\nTotal runtime: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
