"""
phase_04_live/execution/executor.py

Broker order executor for Phase 4 live trading.

Design:
    - Stateless (except for deduplication tracking)
    - Idempotent (same request always produces the same order)
    - Guard-before-execute (kill-switch + dedup checks before broker call)
    - Fail-closed (errors never silently proceed)
    - Non-blocking (returns immediately after broker submission)
    - Audit-friendly (every decision is recorded with reason and timestamp)

The executor does NOT:
    - manage positions
    - calculate risk
    - simulate fills
    - make session decisions
    - decide whether a signal is valid
    
The executor's ONE job: safely transform a SIGNAL_TRIGGERED event into
a broker order, with idempotency and kill-switch protection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional

from phase_04_live.execution.models import (
    ExecutionGuardStatus,
    ExecutionRequest,
    ExecutionResult,
    OrderExecutionStatus,
)
from phase_04_live.recovery.safe_mode import SafeModeGate
from phase_04_live.risk.kill_switch import EmergencyKillSwitch


logger = logging.getLogger(__name__)


@dataclass
class ExecutionConfig:
    """Configuration for the broker executor."""
    max_order_size: float = 0.1  # Max notional per order
    order_timeout_seconds: float = 30.0
    enable_dry_run: bool = False  # If True, no actual broker orders


class BrokerExecutor:
    """
    Execute orders against a broker API.
    
    Usage:
        executor = BrokerExecutor(
            kill_switch=kill_switch,
            safe_mode_gate=safe_mode_gate,
            config=ExecutionConfig(),
            order_sink=mt5_submit_order,
        )
        
        request = ExecutionRequest(...)
        result = executor.execute(request)
        
        if result.succeeded:
            # Order was placed; monitor for fills
            monitor_order(result.broker_order_id)
    """

    def __init__(
        self,
        kill_switch: EmergencyKillSwitch,
        safe_mode_gate: SafeModeGate,
        config: Optional[ExecutionConfig] = None,
        order_sink: Optional[Callable[[ExecutionRequest], Optional[str]]] = None,
    ):
        self._kill_switch = kill_switch
        self._safe_mode_gate = safe_mode_gate
        self._config = config or ExecutionConfig()
        self._order_sink = order_sink
        
        # Deduplication: track what we've already submitted
        # key: idempotency_key, value: broker_order_id
        self._submitted_orders: Dict[str, str] = {}
        
        # Rejection tracking: prevent retrying obviously broken requests
        self._rejected_keys: set = set()

    def execute(self, request: ExecutionRequest) -> ExecutionResult:
        """
        Safely execute a single order request.
        
        Returns an ExecutionResult capturing what happened.
        """
        now = datetime.now(timezone.utc)
        idempotency_key = request.compute_idempotency_key()
        
        # ============================================================
        # 1. Deduplication check: have we already submitted this?
        # ============================================================
        
        if idempotency_key in self._submitted_orders:
            broker_order_id = self._submitted_orders[idempotency_key]
            logger.info(
                "[EXECUTOR] Dedup hit: signal %s already submitted as order %s",
                request.signal_id,
                broker_order_id,
            )
            return ExecutionResult(
                request=request,
                status=OrderExecutionStatus.ACKNOWLEDGED,
                broker_order_id=broker_order_id,
                reason="Duplicate request (already submitted)",
                executed_at=now,
            )
        
        if idempotency_key in self._rejected_keys:
            logger.warning(
                "[EXECUTOR] Rejection hit: signal %s already rejected",
                request.signal_id,
            )
            return ExecutionResult(
                request=request,
                status=OrderExecutionStatus.REJECTED,
                broker_order_id=None,
                reason="Request was previously rejected",
                executed_at=now,
            )
        
        # ============================================================
        # 2. Guard checks: is it safe to send this order?
        # ============================================================
        
        guard_status = self._check_guards(request)
        if not guard_status.allowed:
            logger.warning(
                "[EXECUTOR] Guard rejected: %s | signal=%s",
                guard_status.reason,
                request.signal_id,
            )
            self._rejected_keys.add(idempotency_key)
            return ExecutionResult(
                request=request,
                status=OrderExecutionStatus.REJECTED,
                broker_order_id=None,
                reason=guard_status.reason,
                executed_at=now,
            )
        
        # ============================================================
        # 3. Submit to broker
        # ============================================================
        
        try:
            if self._config.enable_dry_run:
                logger.info(
                    "[EXECUTOR] DRY RUN: would submit %s order for %s "
                    "at %.2f with risk %.2f",
                    request.order_type,
                    request.symbol,
                    request.entry_price,
                    request.initial_risk,
                )
                broker_order_id = f"DRY-{idempotency_key}"
            else:
                if self._order_sink is None:
                    raise RuntimeError(
                        "No order_sink configured; cannot submit to broker"
                    )
                broker_order_id = self._order_sink(request)
                
                if broker_order_id is None:
                    raise RuntimeError("order_sink returned None (no order ID)")
            
            # Record successful submission
            self._submitted_orders[idempotency_key] = broker_order_id
            
            logger.info(
                "[EXECUTOR] Order placed: %s | signal=%s | "
                "broker_id=%s | price=%.2f | risk=%.2f",
                request.order_type,
                request.signal_id,
                broker_order_id,
                request.entry_price,
                request.initial_risk,
            )
            
            return ExecutionResult(
                request=request,
                status=OrderExecutionStatus.PLACED,
                broker_order_id=broker_order_id,
                reason=None,
                executed_at=now,
            )
        
        except Exception as exc:
            error_msg = f"{type(exc).__name__}: {exc}"
            logger.error(
                "[EXECUTOR] Submission failed: %s | signal=%s",
                error_msg,
                request.signal_id,
            )
            return ExecutionResult(
                request=request,
                status=OrderExecutionStatus.ERROR,
                broker_order_id=None,
                reason=error_msg,
                executed_at=now,
            )

    def _check_guards(self, request: ExecutionRequest) -> ExecutionGuardStatus:
        """
        Verify the order is safe to send.
        
        Checks:
            1. Kill switch is not engaged (fresh read, not cached)
            2. Order size is within limits
            3. Required fields are present and valid
        """
        
        # --------------------------------------------------------
        # 1. Kill switch check (fresh read every time)
        # --------------------------------------------------------
        
        kill_switch_status = self._safe_mode_gate.check()
        if not kill_switch_status.allowed:
            return ExecutionGuardStatus(
                allowed=False,
                reason=f"Kill switch engaged: {kill_switch_status.reason}",
            )
        
        # --------------------------------------------------------
        # 2. Order size validation
        # --------------------------------------------------------
        
        notional = request.entry_price * abs(request.initial_risk)
        if notional > self._config.max_order_size:
            return ExecutionGuardStatus(
                allowed=False,
                reason=(
                    f"Order notional {notional:.2f} exceeds limit "
                    f"{self._config.max_order_size:.2f}"
                ),
            )
        
        # --------------------------------------------------------
        # 3. Request completeness
        # --------------------------------------------------------
        
        if not request.signal_id or not request.signal_id.strip():
            return ExecutionGuardStatus(
                allowed=False,
                reason="Missing signal_id",
            )
        
        if not request.symbol or not request.symbol.strip():
            return ExecutionGuardStatus(
                allowed=False,
                reason="Missing symbol",
            )
        
        if request.order_type not in ("BUY", "SELL"):
            return ExecutionGuardStatus(
                allowed=False,
                reason=f"Invalid order_type: {request.order_type}",
            )
        
        if request.entry_price <= 0:
            return ExecutionGuardStatus(
                allowed=False,
                reason=f"Invalid entry_price: {request.entry_price}",
            )
        
        if request.initial_risk <= 0:
            return ExecutionGuardStatus(
                allowed=False,
                reason=f"Invalid initial_risk: {request.initial_risk}",
            )
        
        return ExecutionGuardStatus(allowed=True, reason=None)

    @property
    def order_count(self) -> int:
        """Number of orders submitted so far."""
        return len(self._submitted_orders)

    @property
    def rejection_count(self) -> int:
        """Number of rejected requests."""
        return len(self._rejected_keys)


__all__ = ["BrokerExecutor", "ExecutionConfig"]
