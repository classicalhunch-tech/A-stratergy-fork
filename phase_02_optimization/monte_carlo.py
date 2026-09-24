"""
Monte Carlo robustness analysis for Phase 2.

Purpose
-------
This module answers "how stable is the observed performance?", not
"can I make it more profitable?". It consumes the closed-trade R-multiple
ledger produced by the canonical Phase 1 backtest engine
(strategy.backtest.run_backtest -> BacktestResult) and resamples the
*order* (and optionally the *composition*) of those trades many times to
build a distribution of possible equity paths.

Design rules (Phase 2 architectural constraints)
-------------------------------------------------
- This module knows nothing about BOS/CHoCH/zones/liquidity/signals.
- It never calls the backtest engine and never touches strategy/*.
- Its only required input is a sequence of realized R-multiples (one
  float per CLOSED trade) plus, optionally, entry/exit timestamps to
  estimate calendar duration for a CAGR-style figure.
- Every run is reproducible: pass random_seed to get identical results
  across runs, per the project's determinism/reproducibility rule.

Two independent resampling methods are supported (selectable via
MonteCarloConfig.method):

    "bootstrap"  Sample len(r_multiples) trades WITH replacement.
                 This is the classic Monte Carlo: it also perturbs which
                 trades appear and how often, not just their order.
    "shuffle"    A random permutation of the exact same trades, WITHOUT
                 replacement. This isolates pure trade-*order* sensitivity
                 (sequence-of-returns risk) while holding the sample fixed.

Position sizing model
----------------------
A simple fixed-fractional model is used: each trade risks
`risk_per_trade_pct` of the *current* equity, and the realized R-multiple
scales that risk. This compounds, which is the standard way to expose
sequence-of-returns risk (the same trades in a different order produce
a different final balance and a different drawdown path).

    equity_after = equity_before * (1 + risk_per_trade_pct * r_multiple)

If a trade's R-multiple would push equity to zero or below, equity is
floored at 0.0 and treated as a ruin event for the remainder of that run
(no further compounding from a wiped-out account).
"""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence


# --------------------------------------------------------------------------
# Extracting a clean R-multiple ledger from Phase 1 output
# --------------------------------------------------------------------------

def _normalize_status(value: Any) -> str:
    """
    Best-effort normalization of a TradeResult.result_status value to an
    uppercase string, regardless of whether it's a plain string, an Enum
    (with a .value or .name), or something else printable.

    This mirrors the _normalize_enum_like() approach already used in the
    dashboard, so status comparisons behave the same way here.
    """
    if value is None:
        return ""
    enum_value = getattr(value, "value", None)
    if isinstance(enum_value, str):
        return enum_value.upper()
    enum_name = getattr(value, "name", None)
    if isinstance(enum_name, str):
        return enum_name.upper()
    return str(value).upper()


def extract_closed_r_multiples(
    backtest_result: Any,
    valid_statuses: Sequence[str] = ("WIN", "LOSS"),
) -> list[float]:
    """
    Pull the R-multiple of every CLOSED trade out of a BacktestResult
    (or any object exposing a `.trades` list of TradeResult-like objects).

    Trades whose result_status is not in `valid_statuses` (e.g. OPEN, or
    anything unrecognized) are skipped — an OPEN trade at the end of the
    dataset has no realized R-multiple yet and must not be fed into the
    simulation as if it were closed.

    A trade is also skipped (not silently coerced to 0.0) if r_multiple is
    missing or not a real number, since that would misrepresent the ledger.

    Returns
    -------
    list[float]
        One realized R-multiple per closed, valid trade, in the same
        order they appear in backtest_result.trades (i.e. chronological,
        since that's how the Phase 1 engine records them).
    """
    trades = getattr(backtest_result, "trades", backtest_result)
    valid = {s.upper() for s in valid_statuses}

    r_multiples: list[float] = []
    for trade in trades:
        status = _normalize_status(getattr(trade, "result_status", None))
        if status not in valid:
            continue

        r_value = getattr(trade, "r_multiple", None)
        if isinstance(r_value, bool) or not isinstance(r_value, (int, float)):
            continue

        r_multiples.append(float(r_value))

    return r_multiples


def _trade_duration_days(trade: Any) -> Optional[float]:
    """
    Best-effort per-trade duration in days from entry_time/exit_time,
    used only to produce an approximate CAGR-style figure. Returns None
    if either timestamp is missing or not comparable — callers must treat
    a None average duration as "CAGR cannot be estimated" rather than
    guessing.
    """
    entry_time = getattr(trade, "entry_time", None)
    exit_time = getattr(trade, "exit_time", None)
    if entry_time is None or exit_time is None:
        return None
    try:
        delta = exit_time - entry_time
        seconds = delta.total_seconds()
    except (AttributeError, TypeError):
        return None
    if seconds < 0:
        return None
    return seconds / 86400.0


def estimate_avg_trade_duration_days(backtest_result: Any) -> Optional[float]:
    """
    Average closed-trade duration in days, used to convert a per-trade
    sequence into a rough annualized (CAGR-style) return. Returns None if
    no trade exposes usable entry_time/exit_time — in that case CAGR is
    simply omitted from the results rather than fabricated.
    """
    trades = getattr(backtest_result, "trades", backtest_result)
    durations = [
        d for d in (_trade_duration_days(t) for t in trades) if d is not None
    ]
    if not durations:
        return None
    return statistics.fmean(durations)


# --------------------------------------------------------------------------
# Configuration and per-run / aggregate result shapes
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class MonteCarloConfig:
    """
    method:                 "bootstrap" (with replacement) or "shuffle"
                             (permutation, no replacement).
    num_simulations:        how many equity paths to generate.
    initial_balance:        starting account equity for every run.
    risk_per_trade_pct:     fraction of CURRENT equity risked per trade
                             (e.g. 0.01 for 1%). Applied as a compounding
                             fixed-fractional model — see module docstring.
    ruin_drawdown_pct:      a run is flagged as "ruined" if its
                             peak-to-trough drawdown meets or exceeds this
                             fraction (e.g. 0.50 for a 50% drawdown), OR if
                             equity is floored at 0.
    avg_trade_duration_days: optional; if provided (or derivable from the
                             ledger's timestamps), used to compute a
                             CAGR-style annualized return per run. If
                             neither is available, cagr_pct is None on
                             every run and omitted from the summary.
    random_seed:            seed for reproducibility. None means
                             non-deterministic (a fresh seed each call).
    """

    method: str = "bootstrap"
    num_simulations: int = 1000
    initial_balance: float = 10_000.0
    risk_per_trade_pct: float = 0.01
    ruin_drawdown_pct: float = 0.50
    avg_trade_duration_days: Optional[float] = None
    random_seed: Optional[int] = None

    def __post_init__(self) -> None:
        if self.method not in ("bootstrap", "shuffle"):
            raise ValueError(
                f"method must be 'bootstrap' or 'shuffle', got {self.method!r}"
            )
        if self.num_simulations <= 0:
            raise ValueError("num_simulations must be positive")
        if self.initial_balance <= 0:
            raise ValueError("initial_balance must be positive")
        if not (0 < self.risk_per_trade_pct < 1):
            raise ValueError("risk_per_trade_pct must be between 0 and 1")
        if not (0 < self.ruin_drawdown_pct <= 1):
            raise ValueError("ruin_drawdown_pct must be between 0 and 1")


@dataclass(frozen=True)
class SimulationRun:
    """Full metrics captured for a single simulated equity path."""

    final_balance: float
    total_return_pct: float
    max_drawdown_pct: float
    max_win_streak: int
    max_loss_streak: int
    cagr_pct: Optional[float]
    ruined: bool
    cagr_overflow: bool
    equity_curve: tuple[float, ...] = field(repr=False)


@dataclass(frozen=True)
class MonteCarloResult:
    """Aggregate output across all simulated runs."""

    config: MonteCarloConfig
    num_trades_per_run: int
    runs: tuple[SimulationRun, ...]
    probability_of_ruin: float
    summary: dict[str, dict[str, float]]
    errors: tuple[str, ...] = ()


# --------------------------------------------------------------------------
# Core simulation
# --------------------------------------------------------------------------

def _resample(
    r_multiples: Sequence[float], method: str, rng: random.Random
) -> list[float]:
    if method == "bootstrap":
        return [rng.choice(r_multiples) for _ in r_multiples]

    # "shuffle": permutation without replacement
    shuffled = list(r_multiples)
    rng.shuffle(shuffled)
    return shuffled


def _compute_cagr_pct(
    growth: float,
    years: float,
) -> tuple[Optional[float], bool]:
    """
    Safely annualize `growth` over `years` using log-space arithmetic.

    Equivalent to:

        ((growth ** (1 / years)) - 1) * 100

    but avoids constructing an enormous intermediate power directly.

    Returns
    -------
    (cagr_pct, overflowed)

    cagr_pct is None when the figure cannot be computed because:
        - years <= 0
        - growth <= 0
        - annualization exceeds floating-point range

    overflowed is True only for the floating-point overflow case.
    """
    if years <= 0 or growth <= 0:
        return None, False

    try:
        exponent = math.log(growth) / years
        return math.expm1(exponent) * 100.0, False
    except OverflowError:
        return None, True


def _simulate_single_run(
    r_sequence: Sequence[float],
    config: MonteCarloConfig,
) -> SimulationRun:
    balance = config.initial_balance
    peak = balance
    equity_curve = [balance]

    max_drawdown_pct = 0.0
    ruined = False

    current_win_streak = 0
    current_loss_streak = 0
    max_win_streak = 0
    max_loss_streak = 0

    for r in r_sequence:
        if balance <= 0.0:
            # Already wiped out earlier in this run; nothing left to risk.
            equity_curve.append(0.0)
            continue

        balance = balance * (1.0 + config.risk_per_trade_pct * r)
        balance = max(balance, 0.0)
        equity_curve.append(balance)

        peak = max(peak, balance)
        drawdown_pct = 0.0 if peak <= 0 else (peak - balance) / peak
        max_drawdown_pct = max(max_drawdown_pct, drawdown_pct)

        if balance <= 0.0 or max_drawdown_pct >= config.ruin_drawdown_pct:
            ruined = True

        if r > 0:
            current_win_streak += 1
            current_loss_streak = 0
        elif r < 0:
            current_loss_streak += 1
            current_win_streak = 0
        else:
            # r == 0 breaks both streaks without starting a new one
            current_win_streak = 0
            current_loss_streak = 0

        max_win_streak = max(max_win_streak, current_win_streak)
        max_loss_streak = max(max_loss_streak, current_loss_streak)

    total_return_pct = (
        (balance - config.initial_balance) / config.initial_balance
    ) * 100.0

    cagr_pct: Optional[float] = None
    cagr_overflow = False

    if (
        config.avg_trade_duration_days
        and len(r_sequence) > 0
        and balance > 0
    ):
        total_days = (
            config.avg_trade_duration_days
            * len(r_sequence)
        )
        years = total_days / 365.25

        if years > 0:
            growth = balance / config.initial_balance
            cagr_pct, cagr_overflow = _compute_cagr_pct(
                growth,
                years,
            )

    return SimulationRun(
        final_balance=balance,
        total_return_pct=total_return_pct,
        max_drawdown_pct=max_drawdown_pct * 100.0,
        max_win_streak=max_win_streak,
        max_loss_streak=max_loss_streak,
        cagr_pct=cagr_pct,
        ruined=ruined,
        cagr_overflow=cagr_overflow,
        equity_curve=tuple(equity_curve),
    )


def _percentile_summary(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {}

    sorted_values = sorted(values)

    def pct(p: float) -> float:
        if len(sorted_values) == 1:
            return sorted_values[0]

        k = (len(sorted_values) - 1) * p
        f = int(k)
        c = min(f + 1, len(sorted_values) - 1)

        if f == c:
            return sorted_values[f]

        return (
            sorted_values[f]
            + (sorted_values[c] - sorted_values[f]) * (k - f)
        )

    return {
        "mean": statistics.fmean(sorted_values),
        "median": pct(0.50),
        "stdev": (
            statistics.pstdev(sorted_values)
            if len(sorted_values) > 1
            else 0.0
        ),
        "min": sorted_values[0],
        "p5": pct(0.05),
        "p25": pct(0.25),
        "p75": pct(0.75),
        "p95": pct(0.95),
        "max": sorted_values[-1],
    }


def run_monte_carlo(
    trade_ledger: Any,
    config: Optional[MonteCarloConfig] = None,
) -> MonteCarloResult:
    """
    Run a Monte Carlo robustness analysis over a Phase 1 closed-trade
    ledger.

    Parameters
    ----------
    trade_ledger:
        Either a BacktestResult (or anything with a `.trades` attribute
        of TradeResult-like objects), OR a plain iterable of realized
        R-multiples (floats) for testing/flexibility. Passing raw floats
        skips status filtering — the caller is responsible for having
        already excluded OPEN/INVALIDATED/EXPIRED trades in that case.

    config:
        MonteCarloConfig; defaults are used if omitted.

    Returns
    -------
    MonteCarloResult
        Per-run results plus percentile summaries across final balance,
        total return, max drawdown, win/loss streaks, and (if derivable)
        CAGR, along with the empirical probability of ruin.
    """
    config = config or MonteCarloConfig()
    errors: list[str] = []

    if hasattr(trade_ledger, "trades") or (
        not isinstance(trade_ledger, Iterable)
    ):
        r_multiples = extract_closed_r_multiples(trade_ledger)
    else:
        # Assume it's already a sequence of R-multiples.
        r_multiples = [float(r) for r in trade_ledger]

    if not r_multiples:
        raise ValueError(
            "No closed trades with a usable r_multiple were found — "
            "Monte Carlo requires at least one realized WIN/LOSS trade."
        )

    avg_duration = config.avg_trade_duration_days

    if avg_duration is None and hasattr(trade_ledger, "trades"):
        avg_duration = estimate_avg_trade_duration_days(trade_ledger)

        if avg_duration is not None:
            config = MonteCarloConfig(
                method=config.method,
                num_simulations=config.num_simulations,
                initial_balance=config.initial_balance,
                risk_per_trade_pct=config.risk_per_trade_pct,
                ruin_drawdown_pct=config.ruin_drawdown_pct,
                avg_trade_duration_days=avg_duration,
                random_seed=config.random_seed,
            )

    rng = random.Random(config.random_seed)

    runs: list[SimulationRun] = []

    for _ in range(config.num_simulations):
        sequence = _resample(
            r_multiples,
            config.method,
            rng,
        )
        runs.append(
            _simulate_single_run(
                sequence,
                config,
            )
        )

    probability_of_ruin = (
        sum(1 for r in runs if r.ruined) / len(runs)
    )

    summary: dict[str, dict[str, float]] = {
        "final_balance": _percentile_summary(
            [r.final_balance for r in runs]
        ),
        "total_return_pct": _percentile_summary(
            [r.total_return_pct for r in runs]
        ),
        "max_drawdown_pct": _percentile_summary(
            [r.max_drawdown_pct for r in runs]
        ),
        "max_win_streak": _percentile_summary(
            [float(r.max_win_streak) for r in runs]
        ),
        "max_loss_streak": _percentile_summary(
            [float(r.max_loss_streak) for r in runs]
        ),
    }

    cagr_values = [
        r.cagr_pct
        for r in runs
        if r.cagr_pct is not None
    ]

    overflow_count = sum(
        1
        for r in runs
        if r.cagr_overflow
    )

    if cagr_values:
        summary["cagr_pct"] = _percentile_summary(
            cagr_values
        )

    if overflow_count:
        errors.append(
            f"cagr_pct: {overflow_count}/{len(runs)} runs produced an "
            "annualization overflow (avg_trade_duration_days is too short "
            "relative to num_trades for the compounded growth to be "
            "meaningfully annualized) and were excluded."
        )
    elif not cagr_values:
        errors.append(
            "cagr_pct omitted: no avg_trade_duration_days provided and none "
            "could be derived from entry_time/exit_time on the ledger."
        )

    return MonteCarloResult(
        config=config,
        num_trades_per_run=len(r_multiples),
        runs=tuple(runs),
        probability_of_ruin=probability_of_ruin,
        summary=summary,
        errors=tuple(errors),
    )