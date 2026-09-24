"""
phase_03_paper/trading/paper_engine.py

PaperTradeEngine — Phase 3 simulated trade execution and lifecycle.

Responsibilities (per the Phase 3 spec):
    1. accept an already-TRIGGERED strategy signal
    2. simulate whether the order is accepted/rejected
    3. simulate the entry fill (spread + slippage, always adverse)
    4. create an OPEN paper trade
    5. process future candles
    6. detect stop-loss or take-profit (stop wins on same-candle collision)
    7. close the paper trade
    8. calculate the final R result (using ACTUAL entry, not requested)
    9. record audit events via the existing AuditLog
    10. prevent duplicate processing of the same logical signal

Explicitly does NOT:
    - detect retests / decide whether a setup has triggered
      (strategy/signals.py owns that)
    - send real broker orders
    - simulate genuine tick-level latency (candle-level only)
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Hashable, List, Optional
from uuid import uuid4

from phase_03_paper.config import Phase3Config, PaperExecutionSettings
from phase_03_paper.events.audit import AuditLog
from phase_03_paper.market.engine import Candle
from strategy.signals import SignalStatus, SignalType, TradeSignal


# ============================================================
# ENUMS
# ============================================================


class PaperTradeStatus(Enum):
    PENDING = "PENDING"
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    REJECTED = "REJECTED"


class ExitReason(str, Enum):
    STOP_HIT = "STOP_HIT"
    TARGET_HIT = "TARGET_HIT"


# ============================================================
# PAPER TRADE MODEL
# ============================================================


@dataclass
class PaperTrade:
    """
    Enough information to reconstruct exactly what happened to one
    simulated trade.
    """

    trade_id: str
    direction: SignalType
    session: str

    requested_entry: float
    stop: float
    target: float

    entry: Optional[float] = None

    opened_at: Optional[datetime] = None
    closed_at: Optional[datetime] = None

    exit_price: Optional[float] = None
    exit_reason: Optional[ExitReason] = None

    result_r: float = 0.0

    status: PaperTradeStatus = PaperTradeStatus.PENDING

    # Stable identity of the originating signal, used for duplicate
    # protection. Not part of the "public" trade record but kept for
    # debugging/traceability.
    signal_key: Optional[Hashable] = None


# ============================================================
# PAPER TRADE ENGINE
# ============================================================


class PaperTradeEngine:
    """
    Simulates paper trade execution and trade lifecycle.

    Never sends real broker orders. Only accepts signals that are
    already SignalStatus.TRIGGERED — retest/trigger detection remains
    strategy/signals.py's responsibility.
    """

    def __init__(
        self,
        config: Phase3Config,
        audit_log: AuditLog,
        rng: Optional[random.Random] = None,
    ) -> None:
        self._config = config
        self._audit_log = audit_log
        self._rng = rng if rng is not None else random.Random()

        self._open: Dict[str, PaperTrade] = {}
        self._closed: List[PaperTrade] = []
        self._rejected: List[PaperTrade] = []

        # Stable signal identity -> processed, even after the
        # resulting trade is closed or rejected.
        self._processed_signal_keys: set = set()

    # --------------------------------------------------------------
    # Public API
    # --------------------------------------------------------------

    def on_signal(
        self,
        signal: TradeSignal,
        now: Optional[datetime] = None,
    ) -> Optional[PaperTrade]:
        """
        Accept a TRIGGERED signal and simulate its execution.

        Returns the created PaperTrade (OPEN or REJECTED), or None if
        the signal was ignored (not TRIGGERED, or a duplicate of an
        already-processed signal).
        """

        if signal.status != SignalStatus.TRIGGERED:
            return None

        key = self._signal_key(signal)

        if key in self._processed_signal_keys:
            return None

        self._processed_signal_keys.add(key)

        now_utc = self._ensure_utc(
            now if now is not None else datetime.now(timezone.utc)
        )

        trade = PaperTrade(
            trade_id=str(uuid4()),
            direction=signal.signal_type,
            session=signal.session,
            requested_entry=signal.entry_price,
            stop=signal.stop_loss,
            target=signal.take_profit,
            signal_key=key,
        )

        execution = self._config.paper_execution

        is_rejected = self._rng.random() < execution.rejection_probability

        if is_rejected:
            trade.status = PaperTradeStatus.REJECTED
            self._rejected.append(trade)

            self._audit_log.record(
                event_type="TRADE_REJECTED",
                source="PaperTradeEngine",
                message=(
                    f"Signal rejected: {trade.direction.value} "
                    f"requested_entry={trade.requested_entry}"
                ),
                related_id=trade.trade_id,
            )

            return trade

        adverse_amount = self._calculate_adverse_amount(execution)

        if trade.direction == SignalType.LONG:
            actual_entry = trade.requested_entry + adverse_amount
        else:
            actual_entry = trade.requested_entry - adverse_amount

        trade.entry = actual_entry
        trade.opened_at = now_utc
        trade.status = PaperTradeStatus.OPEN

        self._open[trade.trade_id] = trade

        self._audit_log.record(
            event_type="TRADE_OPENED",
            source="PaperTradeEngine",
            message=(
                f"Trade opened: {trade.direction.value} "
                f"entry={actual_entry} stop={trade.stop} "
                f"target={trade.target}"
            ),
            related_id=trade.trade_id,
        )

        return trade

    def on_candle(self, candle: Candle) -> List[PaperTrade]:
        """
        Advance all open trades by one candle. Closes any trade whose
        stop or target is reached. If a single candle touches both,
        the stop always wins (conservative — a single OHLC candle
        cannot tell us the true intrabar path).

        Returns the list of trades closed by THIS call.
        """

        closed_now: List[PaperTrade] = []

        for trade in list(self._open.values()):

            stop_hit = False
            target_hit = False

            if trade.direction == SignalType.LONG:
                if candle.low <= trade.stop:
                    stop_hit = True
                if candle.high >= trade.target:
                    target_hit = True
            else:
                if candle.high >= trade.stop:
                    stop_hit = True
                if candle.low <= trade.target:
                    target_hit = True

            if not stop_hit and not target_hit:
                continue

            if stop_hit:
                exit_reason = ExitReason.STOP_HIT
                exit_price = trade.stop
            else:
                exit_reason = ExitReason.TARGET_HIT
                exit_price = trade.target

            trade.exit_reason = exit_reason
            trade.exit_price = exit_price
            trade.closed_at = self._ensure_utc(candle.timestamp)
            trade.status = PaperTradeStatus.CLOSED
            trade.result_r = self._calculate_result_r(trade)

            del self._open[trade.trade_id]
            self._closed.append(trade)
            closed_now.append(trade)

            self._audit_log.record(
                event_type=exit_reason.value,
                source="PaperTradeEngine",
                message=(
                    f"Trade {exit_reason.value} at {exit_price} "
                    f"(result_r={trade.result_r:.4f})"
                ),
                related_id=trade.trade_id,
            )

            self._audit_log.record(
                event_type="TRADE_CLOSED",
                source="PaperTradeEngine",
                message=(
                    f"Trade closed via {exit_reason.value}, "
                    f"result_r={trade.result_r:.4f}"
                ),
                related_id=trade.trade_id,
            )

        return closed_now

    # --------------------------------------------------------------
    # Queries
    # --------------------------------------------------------------

    def open_trades(self) -> List[PaperTrade]:
        return list(self._open.values())

    def closed_trades(self) -> List[PaperTrade]:
        return list(self._closed)

    def rejected_trades(self) -> List[PaperTrade]:
        return list(self._rejected)

    def all_trades(self) -> List[PaperTrade]:
        return list(self._open.values()) + list(self._closed) + list(self._rejected)

    @property
    def audit_log(self) -> AuditLog:
        return self._audit_log

    # --------------------------------------------------------------
    # Internal helpers
    # --------------------------------------------------------------

    @staticmethod
    def _signal_key(signal: TradeSignal) -> Hashable:
        """
        Stable identity for duplicate-signal protection.

        Prefers the strategy's stable zone identity. Falls back to a
        deterministic combination of bar_index / setup_bar_index /
        direction when zone_stable_key is unavailable.
        """
        if signal.zone_stable_key is not None:
            return (
                "zone",
                signal.zone_stable_key,
                signal.signal_type.value,
            )

        return (
            "fallback",
            signal.bar_index,
            signal.setup_bar_index,
            signal.signal_type.value,
        )
    @staticmethod
    def _ensure_utc(dt: datetime) -> datetime:
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    def _calculate_adverse_amount(
        self,
        execution: PaperExecutionSettings,
    ) -> float:
        """
        Conservative fill: half the simulated spread, plus a slippage
        component. Both always move the fill against the trader —
        never favorably.
        """

        half_spread = execution.simulated_spread / 2.0

        if execution.slippage_model == "none":
            slippage = 0.0
        elif execution.slippage_model == "random":
            slippage = self._rng.uniform(0.0, execution.slippage_fixed_amount)
        else:
            slippage = execution.slippage_fixed_amount

        return half_spread + slippage

    @staticmethod
    def _calculate_result_r(trade: PaperTrade) -> float:
        if trade.entry is None or trade.exit_price is None:
            return 0.0

        risk = abs(trade.entry - trade.stop)

        if risk <= 0:
            return 0.0

        if trade.direction == SignalType.LONG:
            reward = trade.exit_price - trade.entry
        else:
            reward = trade.entry - trade.exit_price

        return reward / risk


__all__ = [
    "PaperTradeStatus",
    "ExitReason",
    "PaperTrade",
    "PaperTradeEngine",
]