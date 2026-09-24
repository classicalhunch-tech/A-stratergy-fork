"""
phase_04_live/orders/order_manager.py

Converts a triggered AdapterSignalEvent + PositionSizeResult into an
MT5 market-order request and, optionally, sends it.

IMPORTANT -- WHY THIS DOES NOT USE event.fill_price AS THE ORDER PRICE:

AdapterSignalEvent.fill_price is a SIMULATED fill computed by
strategy/retest_engine.py from the closed candle the strategy just
processed. It is the same fill_price used by backtest.py for R-multiple
calculations.

By the time a live cycle reaches this module, real market price may have
moved.

Therefore this module:

    1. Fetches ONE current live tick.
    2. Uses the correct side of that tick:
           BUY  -> ask
           SELL -> bid
    3. Checks drift from the strategy's expected fill_price.
    4. Rejects stale signals instead of chasing price.
    5. Resolves a broker-safe filling mode via the compatibility layer.
    6. Builds the order using that SAME tick snapshot.
    7. Validates that SL/TP are still on the correct side of the
       actual execution price.
    8. In dry_run mode, returns the exact request without sending it.
    9. Only calls mt5.order_send() when dry_run=False.

The same tick snapshot is deliberately used for both the staleness
decision and the actual request price. This avoids a race where one tick
passes the staleness check but a second, later tick is used to build the
order.

SL/TP:

The strategy's signal.stop_loss and signal.take_profit are used exactly
as calculated by the existing strategy. They are NOT shifted to the
current market price.

This preserves the validated Phase 1-3 strategy behavior.

However, because live price may have moved since the strategy generated
the signal, this module verifies that those existing SL/TP levels are
still valid relative to the actual live execution price.

FILLING MODE:

This module NEVER hardcodes an ORDER_FILLING_* constant.

Before building a request, it calls
phase_04_live.broker.compatibility.check_symbol_compatibility() to
resolve a broker-safe filling mode for the symbol.

If the symbol is not compatible (blocking issues exist, or no safe
filling mode can be resolved), the order is rejected before any
request is constructed. This is a fail-closed gate, consistent with
the compatibility module's "NEVER GUESS BROKER CAPABILITIES" rule.

THIS MODULE DOES NOT:

    - calculate position size
    - change strategy entry logic
    - recalculate strategy SL/TP
    - manage open positions
    - reconcile broker state
    - contain dashboard/UI logic
    - decide whether a strategy signal should exist

Those responsibilities remain in their respective Phase 3/4 modules.

SAFETY:

OrderManagerConfig.dry_run defaults to True.

When dry_run=True:

    - the current tick is fetched
    - spread is checked
    - staleness is checked
    - symbol/filling compatibility is checked
    - SL/TP validity is checked
    - the complete MT5 request is constructed
    - the request is logged and returned
    - mt5.order_send() is NOT called

Real order submission requires explicitly setting dry_run=False.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Optional

import MetaTrader5 as mt5

from phase_03_paper.signals.adapter import AdapterSignalEvent
from strategy.signals import SignalType

from phase_04_live.risk.sizing import (
    PositionSizeResult,
    SymbolTradingSpecs,
)

from phase_04_live.broker.compatibility import (
    SymbolCompatibilityReport,
    check_symbol_compatibility,
)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class OrderManagerConfig:
    """
    Mutable live-order configuration.

    These values may eventually be changed by the dashboard without
    restarting the live process, so validation is performed immediately
    before order processing.
    """

    magic_number: int = 404_001

    # Maximum allowed difference between the strategy's expected
    # fill_price and the current live execution price, expressed
    # in broker points.
    max_deviation_points: int = 50

    # Broker-side execution deviation passed to MT5.
    #
    # This is separate from max_deviation_points:
    #
    # max_deviation_points
    #     = strategy-signal staleness protection
    #
    # broker_deviation_points
    #     = broker execution tolerance
    broker_deviation_points: int = 20

    # Maximum spread allowed before an order is rejected.
    # This is an execution safety guard, not a strategy condition.
    max_spread_points: int = 500

    comment: str = "phase4-live"

    # Safety boundary:
    # True  -> construct/log request only
    # False -> allow mt5.order_send()
    dry_run: bool = True

    def validate(self) -> None:
        """
        Validate mutable configuration immediately before use.
        """

        if (
            not isinstance(self.magic_number, int)
            or self.magic_number < 0
        ):
            raise ValueError(
                "magic_number must be a non-negative int, "
                f"got {self.magic_number!r}"
            )

        if (
            not isinstance(self.max_deviation_points, int)
            or self.max_deviation_points < 0
        ):
            raise ValueError(
                "max_deviation_points must be a non-negative int, "
                f"got {self.max_deviation_points!r}"
            )

        if (
            not isinstance(self.broker_deviation_points, int)
            or self.broker_deviation_points < 0
        ):
            raise ValueError(
                "broker_deviation_points must be a non-negative int, "
                f"got {self.broker_deviation_points!r}"
            )

        if (
            not isinstance(self.max_spread_points, int)
            or self.max_spread_points < 0
        ):
            raise ValueError(
                "max_spread_points must be a non-negative int, "
                f"got {self.max_spread_points!r}"
            )

        if not self.comment or not self.comment.strip():
            raise ValueError("comment must not be empty")

        if not isinstance(self.dry_run, bool):
            raise ValueError(
                "dry_run must be a bool, "
                f"got {type(self.dry_run).__name__}"
            )


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OrderRequest:
    """
    Immutable representation of the exact MT5 request context.

    price is the current live market price captured from the tick used
    for the staleness check.

    filling_mode is the broker-resolved ORDER_FILLING_* constant
    produced by phase_04_live.broker.compatibility. It is never
    hardcoded here.
    """

    symbol: str
    order_type: int
    volume: float
    price: float
    sl: float
    tp: float
    deviation: int
    magic: int
    comment: str
    filling_mode: int

    def as_mt5_request(self) -> dict:
        """
        Convert the immutable request model into the dictionary expected
        by mt5.order_send().
        """

        return {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": self.volume,
            "type": self.order_type,
            "price": self.price,
            "sl": self.sl,
            "tp": self.tp,
            "deviation": self.deviation,
            "magic": self.magic,
            "comment": self.comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self.filling_mode,
        }


@dataclass(frozen=True)
class OrderResult:
    """
    Final result of the order-processing attempt.

    accepted=True means:

        - dry_run=True:
              the request passed all local checks and was constructed.

        - dry_run=False:
              the broker accepted the order.

    It does NOT mean a dry-run order exists at the broker.
    """

    accepted: bool
    dry_run: bool
    request: Optional[OrderRequest]
    retcode: Optional[int]
    ticket: Optional[int]
    reason: Optional[str] = None
    deal: Optional[int] = None


# ---------------------------------------------------------------------------
# Tick snapshot
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _ExecutionTick:
    """
    Immutable snapshot of the single tick used by one order attempt.

    Keeping the tick and selected execution price together prevents
    accidental use of a second, later tick during request construction.
    """

    bid: float
    ask: float
    execution_price: float


def _get_execution_tick(
    symbol: str,
    direction: SignalType,
    specs: SymbolTradingSpecs,
    max_spread_points: int,
) -> _ExecutionTick:
    """
    Fetch exactly one current tick, validate its prices and spread,
    then select the execution price.

    LONG  -> ask
    SHORT -> bid

    Raises RuntimeError when the live tick is invalid.
    """

    tick = mt5.symbol_info_tick(symbol)

    if tick is None:
        raise RuntimeError(
            f"symbol_info_tick() returned None for '{symbol}': "
            f"{mt5.last_error()}"
        )

    bid = float(tick.bid)
    ask = float(tick.ask)

    if bid <= 0 or ask <= 0:
        raise RuntimeError(
            f"Invalid live tick prices for '{symbol}': "
            f"bid={bid}, ask={ask}"
        )

    if ask < bid:
        raise RuntimeError(
            f"Invalid live tick for '{symbol}': "
            f"ask={ask} is below bid={bid}"
        )

    if specs.point <= 0:
        raise RuntimeError(
            f"Invalid symbol point size for '{symbol}': "
            f"{specs.point}"
        )

    spread_points = (ask - bid) / specs.point

    if spread_points > max_spread_points:
        raise RuntimeError(
            f"Spread too wide for '{symbol}': "
            f"{spread_points:.1f} points "
            f"(bid={bid}, ask={ask}), "
            f"max_spread_points={max_spread_points}. "
            f"Order aborted for safety."
        )

    if direction == SignalType.LONG:
        execution_price = ask

    elif direction == SignalType.SHORT:
        execution_price = bid

    else:
        raise ValueError(
            f"Unsupported signal direction: {direction!r}"
        )

    return _ExecutionTick(
        bid=bid,
        ask=ask,
        execution_price=execution_price,
    )


# ---------------------------------------------------------------------------
# Staleness check
# ---------------------------------------------------------------------------

def _check_price_staleness(
    event: AdapterSignalEvent,
    current_price: float,
    specs: SymbolTradingSpecs,
    config: OrderManagerConfig,
) -> Optional[str]:
    """
    Return a rejection reason if current live price has moved too far
    from the strategy's expected fill_price.

    Otherwise return None.
    """

    if specs.point <= 0:
        return (
            f"Invalid symbol point size for '{specs.symbol}': "
            f"{specs.point}"
        )

    drift_points = (
        abs(current_price - event.fill_price)
        / specs.point
    )

    if drift_points > config.max_deviation_points:
        return (
            f"Price drifted {drift_points:.1f} points from strategy's "
            f"expected fill_price {event.fill_price} "
            f"(current={current_price}), exceeding "
            f"max_deviation_points={config.max_deviation_points}. "
            f"Signal treated as stale; order rejected rather than chased."
        )

    return None


# ---------------------------------------------------------------------------
# Symbol / filling-mode compatibility gate
# ---------------------------------------------------------------------------

def _resolve_filling_mode(
    symbol: str,
) -> tuple[Optional[int], Optional[str]]:
    """
    Resolve a broker-safe ORDER_FILLING_* constant for symbol via the
    Phase 4 compatibility layer.

    Returns (filling_mode, rejection_reason).

    Exactly one of the two return values is None:

        - filling_mode is not None  -> safe to proceed
        - rejection_reason is not None -> order must be rejected

    This function NEVER falls back to a hardcoded filling mode. If the
    compatibility layer cannot resolve a safe mode, or reports any
    blocking issue for the symbol, the order is rejected here before
    any request is constructed.
    """

    report: SymbolCompatibilityReport = check_symbol_compatibility(
        symbol
    )

    if not report.is_compatible:
        return (
            None,
            f"Symbol '{symbol}' failed compatibility check: "
            f"{'; '.join(report.blocking_issues) or 'unknown reason'}",
        )

    if report.resolved_order_filling_mode is None:
        return (
            None,
            f"Symbol '{symbol}' has no resolved order filling mode; "
            "refusing to guess a broker-safe filling policy.",
        )

    return (report.resolved_order_filling_mode, None)


# ---------------------------------------------------------------------------
# SL / TP validation
# ---------------------------------------------------------------------------

def _check_stop_and_target(
    direction: SignalType,
    current_price: float,
    stop_loss: float,
    take_profit: float,
) -> Optional[str]:
    """
    Verify that existing strategy SL/TP levels remain valid relative
    to the current live execution price.

    LONG:

        stop_loss < current_price < take_profit

    SHORT:

        take_profit < current_price < stop_loss

    This does not modify strategy levels. It only rejects a live order
    when market movement has made the existing levels invalid.
    """

    if stop_loss <= 0 or take_profit <= 0:
        return (
            f"Invalid SL/TP bounds: "
            f"stop_loss={stop_loss}, "
            f"take_profit={take_profit}"
        )

    if direction == SignalType.LONG:

        if stop_loss >= current_price:
            return (
                f"Invalid LONG stop_loss: {stop_loss} "
                f"must be below execution price {current_price}."
            )

        if current_price >= take_profit:
            return (
                f"Invalid LONG take_profit: {take_profit} "
                f"must be above execution price {current_price}."
            )

    elif direction == SignalType.SHORT:

        if take_profit >= current_price:
            return (
                f"Invalid SHORT take_profit: {take_profit} "
                f"must be below execution price {current_price}."
            )

        if current_price >= stop_loss:
            return (
                f"Invalid SHORT stop_loss: {stop_loss} "
                f"must be above execution price {current_price}."
            )

    else:
        return (
            f"Unsupported signal direction: {direction!r}"
        )

    return None


# ---------------------------------------------------------------------------
# Order construction
# ---------------------------------------------------------------------------

def _build_order_request(
    event: AdapterSignalEvent,
    sizing: PositionSizeResult,
    specs: SymbolTradingSpecs,
    config: OrderManagerConfig,
    execution_price: float,
    filling_mode: int,
) -> OrderRequest:
    """
    Build an MT5 request using the already-selected current execution
    price and the already-resolved broker-safe filling mode.

    This function deliberately does NOT fetch another tick and does
    NOT resolve filling mode itself.
    """

    if not sizing.accepted or sizing.lots is None:
        raise ValueError(
            "Order request construction requires an accepted "
            f"PositionSizeResult (reason={sizing.reason!r})."
        )

    if sizing.direction == SignalType.LONG:
        order_type = mt5.ORDER_TYPE_BUY

    elif sizing.direction == SignalType.SHORT:
        order_type = mt5.ORDER_TYPE_SELL

    else:
        raise ValueError(
            f"Unsupported signal direction: "
            f"{sizing.direction!r}"
        )

    return OrderRequest(
        symbol=specs.symbol,
        order_type=order_type,
        volume=sizing.lots,
        price=execution_price,
        sl=float(event.signal.stop_loss),
        tp=float(event.signal.take_profit),
        deviation=config.broker_deviation_points,
        magic=config.magic_number,
        comment=config.comment.strip(),
        filling_mode=filling_mode,
    )


# ---------------------------------------------------------------------------
# Main order pipeline
# ---------------------------------------------------------------------------

def place_order_for_signal(
    event: AdapterSignalEvent,
    sizing: PositionSizeResult,
    specs: SymbolTradingSpecs,
    config: OrderManagerConfig,
    on_before_send: Optional[Callable[[OrderRequest], None]] = None,
) -> OrderResult:
    """
    Full order pipeline:

        1. validate configuration
        2. verify sizing was accepted
        3. resolve a broker-safe filling mode (fail closed if unsafe)
        4. fetch ONE live tick
        5. validate spread
        6. select current execution price
        7. reject stale strategy signals
        8. validate strategy SL/TP against current price
        9. construct exact MT5 request
       10. if dry_run -> return request without sending
       11. otherwise -> mt5.order_send()

    This is the ONLY function in this module that can submit a real
    broker order.
    """

    config.validate()

    if not sizing.accepted or sizing.lots is None:
        return OrderResult(
            accepted=False,
            dry_run=config.dry_run,
            request=None,
            retcode=None,
            ticket=None,
            reason=(
                "Sizing rejected this trade: "
                f"{sizing.reason}"
            ),
        )

    # ------------------------------------------------------------------
    # 0. Symbol / filling-mode compatibility gate
    #
    # This runs before any tick is fetched or request is built. A
    # symbol with no safe filling mode must never reach order
    # construction, regardless of sizing or price conditions.
    # ------------------------------------------------------------------

    filling_mode, filling_rejection_reason = _resolve_filling_mode(
        specs.symbol
    )

    if filling_rejection_reason is not None:
        return OrderResult(
            accepted=False,
            dry_run=config.dry_run,
            request=None,
            retcode=None,
            ticket=None,
            reason=filling_rejection_reason,
        )

    try:
        execution_tick = _get_execution_tick(
            symbol=specs.symbol,
            direction=sizing.direction,
            specs=specs,
            max_spread_points=config.max_spread_points,
        )

    except (RuntimeError, ValueError) as exc:
        return OrderResult(
            accepted=False,
            dry_run=config.dry_run,
            request=None,
            retcode=None,
            ticket=None,
            reason=str(exc),
        )

    current_price = execution_tick.execution_price

    # ------------------------------------------------------------------
    # 1. Signal staleness gate
    # ------------------------------------------------------------------

    staleness_reason = _check_price_staleness(
        event=event,
        current_price=current_price,
        specs=specs,
        config=config,
    )

    if staleness_reason is not None:
        return OrderResult(
            accepted=False,
            dry_run=config.dry_run,
            request=None,
            retcode=None,
            ticket=None,
            reason=staleness_reason,
        )

    # ------------------------------------------------------------------
    # 2. Live SL/TP validity gate
    # ------------------------------------------------------------------

    stop_target_reason = _check_stop_and_target(
        direction=sizing.direction,
        current_price=current_price,
        stop_loss=float(event.signal.stop_loss),
        take_profit=float(event.signal.take_profit),
    )

    if stop_target_reason is not None:
        return OrderResult(
            accepted=False,
            dry_run=config.dry_run,
            request=None,
            retcode=None,
            ticket=None,
            reason=stop_target_reason,
        )

    # ------------------------------------------------------------------
    # 3. Build request from the SAME tick snapshot and the
    #    already-resolved filling mode
    # ------------------------------------------------------------------

    try:
        request = _build_order_request(
            event=event,
            sizing=sizing,
            specs=specs,
            config=config,
            execution_price=current_price,
            filling_mode=filling_mode,
        )

    except ValueError as exc:
        return OrderResult(
            accepted=False,
            dry_run=config.dry_run,
            request=None,
            retcode=None,
            ticket=None,
            reason=str(exc),
        )

    # ------------------------------------------------------------------
    # 4. Dry-run safety boundary
    # ------------------------------------------------------------------

    if config.dry_run:
        logger.info(
            "[DRY RUN] Would send order: %s",
            request.as_mt5_request(),
        )

        return OrderResult(
            accepted=True,
            dry_run=True,
            request=request,
            retcode=None,
            ticket=None,
            reason=(
                "dry_run=True -- order built but NOT sent to broker."
            ),
        )

    # ------------------------------------------------------------------
    # 5. Real broker submission
    # ------------------------------------------------------------------

    # Crash-risk boundary: if the process dies between order_send()
    # being called and its response arriving, the broker may have
    # accepted the order with nothing local recording the attempt.
    # on_before_send() is the caller's hook (guard_and_place_order())
    # to durably record the exact request BEFORE that window opens.
    # Not called in dry_run -- no broker contact happens there.

    if on_before_send is not None:
        on_before_send(request)

    result = mt5.order_send(
        request.as_mt5_request()
    )

    if result is None:
        return OrderResult(
            accepted=False,
            dry_run=False,
            request=request,
            retcode=None,
            ticket=None,
            reason=(
                f"order_send() returned None: "
                f"{mt5.last_error()}"
            ),
        )

    if result.retcode != mt5.TRADE_RETCODE_DONE:
        retcode_desc = {
            mt5.TRADE_RETCODE_REQUOTE: "Requote / Price changed",
            mt5.TRADE_RETCODE_PRICE_OFF: "No prices / Off quotes",
            mt5.TRADE_RETCODE_INVALID_PRICE: "Invalid price structure",
            mt5.TRADE_RETCODE_INVALID_STOPS: "Invalid Stop Loss or Take Profit",
            mt5.TRADE_RETCODE_TRADE_DISABLED: "Trading disabled for symbol",
        }.get(
            result.retcode,
            "Broker rejected order",
        )

        return OrderResult(
            accepted=False,
            dry_run=False,
            request=request,
            retcode=result.retcode,
            ticket=None,
            reason=(
                f"{retcode_desc}: retcode={result.retcode}, "
                f"comment={result.comment!r}"
            ),
        )

    logger.info(
        "[LIVE ORDER SENT] %s %s lots @ %.5f, ticket=%s",
        specs.symbol,
        request.volume,
        request.price,
        result.order,
    )

    return OrderResult(
        accepted=True,
        dry_run=False,
        request=request,
        retcode=result.retcode,
        ticket=int(result.order),
        deal=int(result.deal) if getattr(result, "deal", 0) else None,
        reason=None,
    )
