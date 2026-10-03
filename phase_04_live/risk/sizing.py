"""
phase_04_live/risk/sizing.py

Converts a triggered AdapterSignalEvent (from Phase 3's StrategyAdapter,
reused as-is) into a real MT5 lot size, given account info and a
Phase 4-only risk configuration.

CONFIRMED CONTRACT (verify against source before ever changing this
module -- do not re-derive these from memory):
    AdapterSignalEvent.initial_risk == abs(fill_price - signal.stop_loss)
        -- confirmed in strategy/retest_engine.py:289
        -- a PRICE-DISTANCE in the instrument's own price units,
           NOT a monetary amount and NOT an R-multiple.
    TradeSignal.signal_type is SignalType.LONG or SignalType.SHORT
        -- confirmed in strategy/signals.py:56-58, exactly two members.

SIZING MODES (LiveRiskConfig.sizing_mode):
    "fixed_lot" (DEFAULT):
        Every trade uses config.fixed_lot_size, full stop. Nothing
        about the setup's risk distance changes the size. This is
        the safe, predictable default -- "when not set it trades
        with a specific one set."
    "risk_percent":
        Dynamic sizing: lots are derived from
        account_balance * risk_percent_per_trade, divided by the
        setup's actual price-distance risk (event.initial_risk).
        This mode is intended to be switched on LATER, from the
        dashboard -- once reward_multiple/risk:reward controls exist
        there. Fully implemented now so the dashboard has something
        real to switch to, not a placeholder.

LiveRiskConfig IS MUTABLE ON PURPOSE. Unlike a normal frozen config,
this one is meant to be adjusted at runtime by the dashboard --
switching sizing_mode, changing fixed_lot_size, adjusting
risk_percent_per_trade or reward_multiple_override -- without
restarting the live process. Because of this, validation is NOT done
once at construction (there is no safe "once" for a value that can
change later); call config.validate() immediately before each use in
compute_position_size() (already done for you inside that function).

SAFETY PRINCIPLE (applies to BOTH modes):
    Lot size is ALWAYS rounded DOWN to the broker's volume_step.
    Rounding up would silently risk more than configured. If the
    rounded-down size is below the broker's volume_min, the trade is
    REJECTED (PositionSizeResult.accepted = False) rather than forced
    up to volume_min.

THIS MODULE DOES NOT:
    - place orders
    - know about MT5 order tickets
    - manage open positions
    - decide entry/exit logic (that remains strategy/ and Phase 3's
      StrategyAdapter, untouched)
    - apply reward_multiple_override to anything yet -- it is stored
      on the config and exposed on PositionSizeResult so the future
      order-manager (which sets take_profit) can read it, but this
      module only sizes LOTS, it does not compute TP prices.

Deliberately NOT built on top of phase_03_paper.config.Phase3Config --
see that module's own docstring: "A configuration change must never
be capable of enabling real orders... Phase 4 will use a separate
LiveExecutor implementation." Position sizing is exactly that kind of
live-only, money-touching concern.

Everything here is pure and testable without a live MT5 connection
EXCEPT get_symbol_trading_specs() and get_account_info(), the two
functions that must talk to the terminal. Callers should already be
connected (see phase_04_live.main's explicit connect()-before-first-
cycle pattern) before calling either.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from phase_03_paper.signals.adapter import AdapterSignalEvent
from strategy.signals import SignalType


_VALID_SIZING_MODES = {"fixed_lot", "risk_percent"}


# ---------------------------------------------------------------------------
# Phase 4-only risk configuration -- MUTABLE, see module docstring.
# ---------------------------------------------------------------------------

@dataclass
class LiveRiskConfig:
    """
    Live-only, runtime-adjustable risk sizing configuration. No
    relationship to Phase3Config's PaperExecutionSettings by design.

    Meant to be constructed once with safe defaults, then mutated
    later by the dashboard (change .sizing_mode, .fixed_lot_size,
    .risk_percent_per_trade, .reward_multiple_override directly --
    this is a plain mutable dataclass, not frozen).
    """

    # Which sizing strategy to use. Defaults to the safe, predictable
    # fixed-lot mode -- dynamic risk-based sizing is opt-in, switched
    # on later from the dashboard.
    sizing_mode: str = "fixed_lot"

    # --- fixed_lot mode ---
    fixed_lot_size: float = 0.01

    # --- risk_percent mode (dormant until sizing_mode is switched) ---
    risk_percent_per_trade: float = 0.005
    max_risk_percent_per_trade: float = 0.02

    # Reserved for the future dashboard risk:reward control. Not yet
    # applied anywhere in THIS module (lot sizing only) -- carried
    # through on PositionSizeResult so the future order manager can
    # read it when computing take_profit. None means "use the
    # strategy signal's own take_profit, unmodified."
    reward_multiple_override: Optional[float] = None

    # Applies to BOTH modes: hard cap on lot size regardless of how
    # it was calculated. None means no cap beyond the broker's own
    # volume_max.
    max_lots_per_trade: Optional[float] = None

    def validate(self) -> None:
        """
        Check invariants. Called immediately before each use inside
        compute_position_size(), NOT once at construction -- since
        this config can be mutated at any time by the dashboard,
        there is no single safe moment to validate it "once".
        """
        if self.sizing_mode not in _VALID_SIZING_MODES:
            raise ValueError(
                f"sizing_mode must be one of {_VALID_SIZING_MODES}, "
                f"got {self.sizing_mode!r}"
            )

        if self.sizing_mode == "fixed_lot":
            if self.fixed_lot_size <= 0:
                raise ValueError(
                    f"fixed_lot_size must be > 0, got {self.fixed_lot_size}"
                )

        if self.sizing_mode == "risk_percent":
            if not (0.0 < self.risk_percent_per_trade <= self.max_risk_percent_per_trade):
                raise ValueError(
                    f"risk_percent_per_trade ({self.risk_percent_per_trade}) must be "
                    f"> 0 and <= max_risk_percent_per_trade "
                    f"({self.max_risk_percent_per_trade})"
                )

        if self.max_lots_per_trade is not None and self.max_lots_per_trade <= 0:
            raise ValueError(
                f"max_lots_per_trade must be > 0 if set, got {self.max_lots_per_trade}"
            )

        if self.reward_multiple_override is not None and self.reward_multiple_override <= 0:
            raise ValueError(
                f"reward_multiple_override must be > 0 if set, got "
                f"{self.reward_multiple_override}"
            )


# ---------------------------------------------------------------------------
# Broker symbol specs -- pulled live, never hardcoded (varies by
# broker and even by account type on the SAME broker).
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SymbolTradingSpecs:
    """Broker-reported trading parameters for one symbol."""

    symbol: str
    contract_size: float      # e.g. 100 for XAUUSD == 100 oz per lot
    volume_min: float
    volume_max: float
    volume_step: float
    point: float
    digits: int


def get_symbol_trading_specs(symbol: str) -> SymbolTradingSpecs:
    """
    Pull live trading specs for `symbol` directly from MT5.

    Raises RuntimeError if the symbol isn't found or MT5 isn't
    connected.
    """
    import MetaTrader5 as mt5

    info = mt5.symbol_info(symbol)

    if info is None:
        raise RuntimeError(
            f"symbol_info() returned None for '{symbol}': {mt5.last_error()}"
        )

    return SymbolTradingSpecs(
        symbol=symbol,
        contract_size=float(info.trade_contract_size),
        volume_min=float(info.volume_min),
        volume_max=float(info.volume_max),
        volume_step=float(info.volume_step),
        point=float(info.point),
        digits=int(info.digits),
    )


# ---------------------------------------------------------------------------
# Account info -- includes leverage, which fixed_lot mode doesn't need
# today but is exposed now so it's available the moment dashboard
# controls (or a future margin guard) need it, without another round
# of "go check what field MT5 actually calls this."
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AccountInfo:
    balance: float
    equity: float
    margin_free: float
    leverage: int
    currency: str


def get_account_info() -> AccountInfo:
    """
    Pull live account info from MT5, including leverage.

    Raises RuntimeError if MT5 isn't connected or account info is
    unavailable. Deliberately separate from PaperExecutionSettings'
    starting_balance -- that field simulates a static paper balance;
    this reads the real, live, changing account state.
    """
    import MetaTrader5 as mt5

    account = mt5.account_info()

    if account is None:
        raise RuntimeError(f"account_info() returned None: {mt5.last_error()}")

    return AccountInfo(
        balance=float(account.balance),
        equity=float(account.equity),
        margin_free=float(account.margin_free),
        leverage=int(account.leverage),
        currency=str(account.currency),
    )


# ---------------------------------------------------------------------------
# Sizing result -- explicit, not a bare float. A rejected trade (too
# small to size safely) must be unambiguous, not "0.0 lots", which
# could be misread as "size computed as zero" rather than "refused".
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PositionSizeResult:
    accepted: bool
    lots: Optional[float]
    direction: Optional[SignalType]
    dollar_risk: Optional[float]
    sizing_mode: str
    reward_multiple_override: Optional[float]
    reason: Optional[str] = None  # populated when accepted is False


def _round_down_to_step(raw_lots: float, volume_step: float) -> float:
    """Round DOWN to the nearest valid volume_step -- never up."""
    steps = math.floor(raw_lots / volume_step)
    return round(steps * volume_step, 8)  # avoid float artifacts


def compute_position_size(
    event: AdapterSignalEvent,
    account: AccountInfo,
    config: LiveRiskConfig,
    specs: SymbolTradingSpecs,
) -> PositionSizeResult:
    """
    Convert a triggered AdapterSignalEvent into a broker-valid lot
    size, using whichever mode config.sizing_mode currently selects.

    fixed_lot mode:
        lots = config.fixed_lot_size, rounded down to volume_step,
        clamped to [volume_min, volume_max] and any configured
        max_lots_per_trade. dollar_risk is computed for visibility/
        logging only -- it does NOT drive the lot size in this mode.

    risk_percent mode:
        dollar_risk = account.balance * config.risk_percent_per_trade
        raw_lots = dollar_risk / (event.initial_risk * specs.contract_size)
        lots = floor(raw_lots / volume_step) * volume_step

    Both modes reject (rather than round up to) any size below
    specs.volume_min -- see module docstring's SAFETY PRINCIPLE.
    """
    config.validate()

    if event.initial_risk <= 0:
        return PositionSizeResult(
            accepted=False, lots=None, direction=None, dollar_risk=None,
            sizing_mode=config.sizing_mode,
            reward_multiple_override=config.reward_multiple_override,
            reason=f"initial_risk must be > 0, got {event.initial_risk}",
        )

    direction = event.signal.signal_type

    if config.sizing_mode == "fixed_lot":
        lots = _round_down_to_step(config.fixed_lot_size, specs.volume_step)
        # Informational only in this mode -- does not drive lots.
        dollar_risk = config.fixed_lot_size * specs.contract_size * event.initial_risk

    else:  # "risk_percent" -- validated above, no other value possible
        if account.balance <= 0:
            return PositionSizeResult(
                accepted=False, lots=None, direction=direction, dollar_risk=None,
                sizing_mode=config.sizing_mode,
                reward_multiple_override=config.reward_multiple_override,
                reason=f"account.balance must be > 0, got {account.balance}",
            )

        dollar_risk = account.balance * config.risk_percent_per_trade
        raw_lots = dollar_risk / (event.initial_risk * specs.contract_size)
        lots = _round_down_to_step(raw_lots, specs.volume_step)

    # Applies to BOTH modes.
    lots = min(lots, specs.volume_max)
    if config.max_lots_per_trade is not None:
        lots = min(lots, config.max_lots_per_trade)
    lots = round(lots, 8)

    if lots < specs.volume_min:
        return PositionSizeResult(
            accepted=False,
            lots=None,
            direction=direction,
            dollar_risk=dollar_risk,
            sizing_mode=config.sizing_mode,
            reward_multiple_override=config.reward_multiple_override,
            reason=(
                f"Calculated size {lots} lots is below broker minimum "
                f"{specs.volume_min} lots for '{specs.symbol}' "
                f"(mode={config.sizing_mode}). Trade rejected, not sized "
                f"up, to avoid exceeding intended risk."
            ),
        )

    return PositionSizeResult(
        accepted=True,
        lots=lots,
        direction=direction,
        dollar_risk=dollar_risk,
        sizing_mode=config.sizing_mode,
        reward_multiple_override=config.reward_multiple_override,
        reason=None,
    )
