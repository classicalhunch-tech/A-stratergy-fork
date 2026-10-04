"""
phase_04_live/execution/models.py

Order execution models for Phase 4 live trading.

Design principles:
    - Idempotent by construction: every order has a stable, deterministic ID
    - Immutable snapshots: execution state never changes retroactively
    - Minimal: only tracks what's essential for deduplication and reconciliation
    - Audit-friendly: every execution decision is recorded with reason and timestamp
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


class OrderExecutionStatus(str, Enum):
    """Current execution state of an order."""
    PENDING = "PENDING"           # Created, waiting for broker ACK
    ACKNOWLEDGED = "ACKNOWLEDGED"  # Broker received, status unknown
    PLACED = "PLACED"              # Confirmed in broker system
    FILLED = "FILLED"              # Partially or fully executed
    REJECTED = "REJECTED"          # Broker rejected
    CANCELLED = "CANCELLED"        # Manually cancelled
    ERROR = "ERROR"                # Execution system error


@dataclass(frozen=True)
class ExecutionRequest:
    """
    Immutable execution request derived from a triggered signal.
    
    This is what the broker executor receives and acts on.
    """
    # Signal identity
    signal_id: str                  # From AdapterSignalEvent
    symbol: str                     # Trading symbol (e.g., "XAUUSD")
    
    # Order details
    order_type: str                 # "BUY" or "SELL"
    entry_price: float              # Execution price
    initial_risk: float             # Stop distance in PRICE units: abs(entry - stop). NOT account currency.
    
    # Timing
    signal_generated_at: datetime   # When the signal was created
    request_created_at: datetime    # When this request was formed (UTC)
    
    # Idempotency key: stable hash of signal + price + risk
    # Allows safe replay without duplicate orders
    idempotency_key: Optional[str] = None

    # Protective and sizing fields. Optional so older callers keep working.
    # They are NOT part of the idempotency key.
    stop_loss: Optional[float] = None    # Stop price
    take_profit: Optional[float] = None  # Target price
    volume: Optional[float] = None       # Order size in lots
    magic: Optional[int] = None          # Broker magic number (order tag)
    
    def compute_idempotency_key(self) -> str:
        """
        Deterministic hash of signal essentials.
        
        Same signal + price + risk always produces the same key, so
        if this request is replayed, the executor recognizes it and
        deduplicates rather than placing a second order.
        """
        if self.idempotency_key is not None:
            return self.idempotency_key
        
        msg = f"{self.signal_id}|{self.symbol}|{self.order_type}|{self.entry_price}|{self.initial_risk}"
        return hashlib.sha256(msg.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class ExecutionResult:
    """
    Immutable record of a single execution attempt.
    
    Never mutated; new events create new ExecutionResult instances.
    """
    request: ExecutionRequest
    status: OrderExecutionStatus
    broker_order_id: Optional[str]  # Broker's order ID if placed
    reason: Optional[str]            # Status reason (error message, rejection reason, etc.)
    executed_at: datetime            # When this result was recorded
    
    @property
    def succeeded(self) -> bool:
        """Order is in a terminal success state."""
        return self.status in (OrderExecutionStatus.PLACED, OrderExecutionStatus.FILLED)
    
    @property
    def failed(self) -> bool:
        """Order is in a terminal failure state."""
        return self.status in (OrderExecutionStatus.REJECTED, OrderExecutionStatus.ERROR, OrderExecutionStatus.CANCELLED)
    
    @property
    def terminal(self) -> bool:
        """Order will not change state without external interaction."""
        return self.failed or self.status == OrderExecutionStatus.FILLED


@dataclass(frozen=True)
class ExecutionGuardStatus:
    """
    Result of a guard check before order placement.
    
    Guards ensure the order is safe to send to the broker.
    """
    allowed: bool
    reason: Optional[str]  # Why the order is blocked (if allowed=False)


__all__ = [
    "OrderExecutionStatus",
    "ExecutionRequest",
    "ExecutionResult",
    "ExecutionGuardStatus",
]
