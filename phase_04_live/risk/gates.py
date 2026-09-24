"""
phase_04_live/risk/gates.py

Phase 4 Step 3 -- Pre-Trade Risk Gate
=====================================

Purpose
-------
A single, ordered set of pre-trade safety checks that must ALL pass
before a triggered signal is allowed to reach position sizing.

This module does NOT:
    - place orders
    - call MT5
    - compute position size
    - read broker/account state directly

It is a pure decision layer: given plain, already-gathered numbers
about the current risk state, it returns whether trading is currently
allowed and why not, if not.

Callers (e.g. a future live runtime coordinator) are responsible for
gathering RiskGateInput from PositionManager / Journal / PerformanceEngine
/ a persisted EmergencyKillSwitch, and applying evaluate_risk_gate()
BEFORE calling compute_position_size().

Check order (first blocking check wins as .reason; ALL are evaluated
and reported in .blocking_issues, for audit purposes):

    1. Emergency kill switch
    2. Position limits
    3. Daily loss limit
    4. Drawdown limit

Safety behavior
---------------
A configured limit is enabled.

If an enabled limit requires a measurement and that measurement is
missing, the gate FAILS CLOSED and blocks the trade.

A limit configured as None is explicitly DISABLED.

Configuration limits must be positive when provided.
Invalid non-positive limits raise ValueError rather than being silently
corrected.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

from strategy.signals import SignalType


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PositionLimitConfig:
    """
    Position-count limits.

    max_open_positions:
        Overall cap across all symbols.
        None explicitly disables the overall cap.

    max_open_positions_per_symbol:
        Cap for the signal's own symbol.
        None explicitly disables the per-symbol cap.

    allow_hedging:
        If False, an opposite-direction open position on the same
        symbol blocks a new signal.
    """

    max_open_positions: Optional[int] = 1
    max_open_positions_per_symbol: Optional[int] = 1
    allow_hedging: bool = False

    def __post_init__(self) -> None:
        if (
            self.max_open_positions is not None
            and self.max_open_positions <= 0
        ):
            raise ValueError(
                "max_open_positions must be positive or None."
            )

        if (
            self.max_open_positions_per_symbol is not None
            and self.max_open_positions_per_symbol <= 0
        ):
            raise ValueError(
                "max_open_positions_per_symbol must be positive or None."
            )


@dataclass(frozen=True)
class DailyLossLimitConfig:
    """
    Daily loss circuit breaker.

    Exactly one of max_daily_loss_r / max_daily_loss_amount would
    normally be set; both may be set (whichever trips first blocks).

    Both None explicitly disables the daily loss check entirely.

    Limits must be positive when configured.
    """

    max_daily_loss_r: Optional[float] = None
    max_daily_loss_amount: Optional[float] = None

    def __post_init__(self) -> None:
        if (
            self.max_daily_loss_r is not None
            and self.max_daily_loss_r <= 0
        ):
            raise ValueError(
                "max_daily_loss_r must be positive or None."
            )

        if (
            self.max_daily_loss_amount is not None
            and self.max_daily_loss_amount <= 0
        ):
            raise ValueError(
                "max_daily_loss_amount must be positive or None."
            )


@dataclass(frozen=True)
class DrawdownLimitConfig:
    """
    Account drawdown circuit breaker, measured from equity peak.

    Both None explicitly disables the drawdown check entirely.

    Limits must be positive when configured.
    """

    max_drawdown_r: Optional[float] = None
    max_drawdown_pct: Optional[float] = None

    def __post_init__(self) -> None:
        if (
            self.max_drawdown_r is not None
            and self.max_drawdown_r <= 0
        ):
            raise ValueError(
                "max_drawdown_r must be positive or None."
            )

        if (
            self.max_drawdown_pct is not None
            and self.max_drawdown_pct <= 0
        ):
            raise ValueError(
                "max_drawdown_pct must be positive or None."
            )


# ---------------------------------------------------------------------------
# Input / Output
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RiskGateInput:
    """
    Plain, already-gathered risk state for one prospective signal.

    This module does not know how these numbers were computed -- the
    caller derives them from PositionManager, Journal, PerformanceEngine,
    and the EmergencyKillSwitch.

    daily_realized_r / daily_realized_amount:
        Realized closed-trade result so far today.

        For daily_realized_r and daily_realized_amount:
            negative = loss
            zero = break-even
            positive = profit

        Populate the field when the corresponding daily-loss limit
        is enabled.

    current_drawdown_r / current_drawdown_pct:
        Current drawdown from equity peak.

        These are expressed as POSITIVE values representing the
        distance below the peak.

        Example:
            3.5 means 3.5R below peak
            3.5 means 3.5% below peak

        Populate the field when the corresponding drawdown limit
        is enabled.

    kill_switch_engaged:
        Emergency kill switch state.
    """

    symbol: str
    direction: SignalType

    open_positions_count: int
    open_positions_count_for_symbol: int
    has_opposite_direction_open: bool

    daily_realized_r: Optional[float] = None
    daily_realized_amount: Optional[float] = None

    current_drawdown_r: Optional[float] = None
    current_drawdown_pct: Optional[float] = None

    kill_switch_engaged: bool = False


@dataclass(frozen=True)
class RiskGateResult:
    """
    Result of the complete pre-trade risk evaluation.

    allowed:
        True only when no blocking issue was found.

    reason:
        The FIRST blocking issue in fixed check order.
        This is the primary reason a caller should display/log.

    blocking_issues:
        Every blocking issue found in fixed check order.
        Intended for audit logging and diagnostics, not control flow.
    """

    allowed: bool
    reason: Optional[str]
    blocking_issues: Tuple[str, ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def _check_kill_switch(data: RiskGateInput) -> Tuple[str, ...]:
    """Check the emergency kill switch."""

    if data.kill_switch_engaged:
        return (
            "Emergency kill switch is ENGAGED. "
            "No new trades permitted.",
        )

    return ()


def _check_position_limits(
    data: RiskGateInput,
    config: PositionLimitConfig,
) -> Tuple[str, ...]:
    """Check overall, per-symbol, and hedging limits."""

    issues = []

    if (
        config.max_open_positions is not None
        and data.open_positions_count >= config.max_open_positions
    ):
        issues.append(
            "Open position cap reached: "
            f"{data.open_positions_count} >= "
            f"{config.max_open_positions} (overall max)."
        )

    if (
        config.max_open_positions_per_symbol is not None
        and data.open_positions_count_for_symbol
        >= config.max_open_positions_per_symbol
    ):
        issues.append(
            f"Open position cap reached for '{data.symbol}': "
            f"{data.open_positions_count_for_symbol} >= "
            f"{config.max_open_positions_per_symbol} "
            "(per-symbol max)."
        )

    if not config.allow_hedging and data.has_opposite_direction_open:
        issues.append(
            f"Opposite-direction position already open on "
            f"'{data.symbol}' and hedging is disabled."
        )

    return tuple(issues)


def _check_daily_loss(
    data: RiskGateInput,
    config: DailyLossLimitConfig,
) -> Tuple[str, ...]:
    """
    Check configured daily loss limits.

    Fail closed when an enabled limit has no corresponding measurement.
    """

    issues = []

    if config.max_daily_loss_r is not None:
        if data.daily_realized_r is None:
            issues.append(
                "Daily loss R limit is enabled but "
                "daily_realized_r is unavailable."
            )
        elif data.daily_realized_r <= -config.max_daily_loss_r:
            issues.append(
                "Daily loss limit reached: realized "
                f"{data.daily_realized_r:.4f}R <= "
                f"-{config.max_daily_loss_r:.4f}R."
            )

    if config.max_daily_loss_amount is not None:
        if data.daily_realized_amount is None:
            issues.append(
                "Daily loss amount limit is enabled but "
                "daily_realized_amount is unavailable."
            )
        elif data.daily_realized_amount <= -config.max_daily_loss_amount:
            issues.append(
                "Daily loss limit reached: realized "
                f"{data.daily_realized_amount:.2f} <= "
                f"-{config.max_daily_loss_amount:.2f}."
            )

    return tuple(issues)


def _check_drawdown(
    data: RiskGateInput,
    config: DrawdownLimitConfig,
) -> Tuple[str, ...]:
    """
    Check configured drawdown limits.

    Fail closed when an enabled limit has no corresponding measurement.
    """

    issues = []

    if config.max_drawdown_r is not None:
        if data.current_drawdown_r is None:
            issues.append(
                "Drawdown R limit is enabled but "
                "current_drawdown_r is unavailable."
            )
        elif data.current_drawdown_r >= config.max_drawdown_r:
            issues.append(
                "Drawdown limit reached: "
                f"{data.current_drawdown_r:.4f}R >= "
                f"{config.max_drawdown_r:.4f}R."
            )

    if config.max_drawdown_pct is not None:
        if data.current_drawdown_pct is None:
            issues.append(
                "Drawdown percentage limit is enabled but "
                "current_drawdown_pct is unavailable."
            )
        elif data.current_drawdown_pct >= config.max_drawdown_pct:
            issues.append(
                "Drawdown limit reached: "
                f"{data.current_drawdown_pct:.2f}% >= "
                f"{config.max_drawdown_pct:.2f}%."
            )

    return tuple(issues)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def evaluate_risk_gate(
    data: RiskGateInput,
    position_config: PositionLimitConfig,
    daily_loss_config: DailyLossLimitConfig,
    drawdown_config: DrawdownLimitConfig,
) -> RiskGateResult:
    """
    Evaluate all pre-trade risk checks in fixed order:

        1. Emergency kill switch
        2. Position limits
        3. Daily loss limit
        4. Drawdown limit

    Every check is evaluated.

    The first blocking issue becomes `reason`.

    All blocking issues are returned in `blocking_issues`.

    If no issues are found, the result is:

        allowed=True
        reason=None
        blocking_issues=()
    """

    all_issues = []

    # 1. Emergency kill switch
    all_issues.extend(_check_kill_switch(data))

    # 2. Position limits
    all_issues.extend(
        _check_position_limits(
            data,
            position_config,
        )
    )

    # 3. Daily loss limit
    all_issues.extend(
        _check_daily_loss(
            data,
            daily_loss_config,
        )
    )

    # 4. Drawdown limit
    all_issues.extend(
        _check_drawdown(
            data,
            drawdown_config,
        )
    )

    if not all_issues:
        return RiskGateResult(
            allowed=True,
            reason=None,
            blocking_issues=(),
        )

    return RiskGateResult(
        allowed=False,
        reason=all_issues[0],
        blocking_issues=tuple(all_issues),
    )