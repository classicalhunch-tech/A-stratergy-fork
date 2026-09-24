"""
phase_04_live/execution/idempotency.py

Duplicate / idempotency protection for Phase 4 order submission.

Purpose
-------
Prevent the same strategy signal from accidentally producing more
than one live order attempt -- whether from a duplicate AdapterSignalEvent
in one runtime session, or from the SAME signal being reprocessed
after a restart (e.g. replaying recent history to rebuild adapter
state).

This module does NOT modify order_manager.place_order_for_signal().
That function remains the only function in the codebase that can
submit a real broker order. This module wraps it with a guard check
performed BEFORE it is called.

Identity
--------
AdapterSignalEvent.trigger_bar_index and event.signal.bar_index are
adapter-lifetime-relative counters -- they reset to 0 whenever a
StrategyAdapter instance is recreated (e.g. after a process restart),
so they are NOT safe as a durable identity.

TradeSignal.setup_timestamp, by contrast, is a real market timestamp
(the liquidity-sweep candle's time) -- deterministic given the same
market data, and therefore stable across a restart that replays the
same historical candles. This mirrors the same style of composite
identity strategy/backtest.py's _make_setup_key() already uses for
duplicate-setup protection within one backtest run.

Persistence
-----------
IdempotencyStore is a small pluggable interface. InMemoryIdempotencyStore
is the only implementation provided here and does NOT survive a
process restart on its own -- true restart-durability requires
backing it with the SQLite persistence layer (Phase 4 Step 7 in the
project roadmap). The interface is deliberately store-agnostic so
that wiring happens without changing any caller of guard_and_place_order().

This module does NOT:

    - submit orders directly
    - modify OrderManager
    - make risk decisions
    - persist anything durable on its own (yet)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Hashable, Optional, Protocol, Tuple

import pandas as pd

from phase_03_paper.signals.adapter import AdapterSignalEvent
from phase_04_live.orders.order_manager import (
    OrderManagerConfig,
    OrderRequest,
    OrderResult,
    place_order_for_signal,
    SymbolTradingSpecs,
)
from phase_04_live.risk.sizing import PositionSizeResult


ExecutionIdentity = Tuple[str, Hashable, Hashable, Hashable]


def build_execution_identity(event: AdapterSignalEvent) -> ExecutionIdentity:
    """
    Build a durable, restart-safe identity for one signal's execution
    attempt.

    Composed of:
        signal_type      -- "LONG" or "SHORT"
        setup_timestamp  -- real market timestamp of the liquidity sweep
        zone_stable_key  -- the matched zone's stable identity (see
                             strategy.signals._stable_cache_key)
        triggered_at_time -- real market timestamp the retest triggered

    All four are derived from actual market data, not from adapter
    process state, so replaying the same historical candles after a
    restart reproduces the identical identity for the identical
    real-world trade opportunity.
    """

    signal = event.signal

    signal_type = getattr(signal.signal_type, "value", signal.signal_type)

    setup_timestamp = signal.setup_timestamp
    if setup_timestamp is not None:
        setup_timestamp = pd.Timestamp(setup_timestamp)

    triggered_at_time = signal.triggered_at_time
    if triggered_at_time is not None:
        triggered_at_time = pd.Timestamp(triggered_at_time)

    return (
        str(signal_type),
        setup_timestamp,
        signal.zone_stable_key,
        triggered_at_time,
    )


def _json_default(value):
    """Handle pd.Timestamp (used in ExecutionIdentity) for json.dumps."""
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return str(value)


def encode_execution_identity(identity: ExecutionIdentity) -> str:
    """
    Canonical, deterministic string encoding of an ExecutionIdentity
    tuple. Single source of truth: phase_04_live.persistence.store
    uses this same function for both execution_identities and
    order_attempts primary keys, so an attempt recorded there can
    always be traced back to the identity guard_and_place_order()
    already checked via IdempotencyStore.
    """
    return json.dumps(identity, default=_json_default, sort_keys=False)


class OrderAttemptRecorder(Protocol):
    """
    Minimal pluggable interface for durably recording an OrderRequest
    immediately before it is sent to the broker. Structurally
    satisfied by phase_04_live.persistence.store.Phase4Persistence
    (record_order_attempt()) -- same structural-typing style as
    IdempotencyStore below.
    """

    def record_order_attempt(
        self,
        identity_key: str,
        request: OrderRequest,
        attempted_at: datetime,
    ) -> None:
        ...


class IdempotencyStore(Protocol):
    """
    Minimal pluggable interface for tracking which execution
    identities have already been attempted.

    has() and record() must be safe to call in that order for the
    same key without race conditions in a single-threaded runtime
    (guard_and_place_order() calls has() then record() sequentially,
    never concurrently, for a given identity).
    """

    def has(self, identity: ExecutionIdentity) -> bool:
        ...

    def record(self, identity: ExecutionIdentity) -> None:
        ...


class InMemoryIdempotencyStore:
    """
    In-memory-only IdempotencyStore.

    Does NOT survive a process restart. Suitable for single-session
    duplicate protection and for tests. A durable, SQLite-backed
    store implementing the same IdempotencyStore interface is
    required before this can protect against restart-triggered
    reprocessing in a live environment (see module docstring).
    """

    def __init__(self) -> None:
        self._seen: set = set()

    def has(self, identity: ExecutionIdentity) -> bool:
        return identity in self._seen

    def record(self, identity: ExecutionIdentity) -> None:
        self._seen.add(identity)


def guard_and_place_order(
    event: AdapterSignalEvent,
    sizing: PositionSizeResult,
    specs: SymbolTradingSpecs,
    config: OrderManagerConfig,
    store: IdempotencyStore,
    attempt_store: Optional[OrderAttemptRecorder] = None,
) -> OrderResult:
    """
    Idempotency-guarded wrapper around place_order_for_signal().

    If this event's execution identity has already been recorded,
    the order is NOT submitted -- an OrderResult reflecting local
    rejection is returned instead, with no broker contact attempted.

    Otherwise, the identity is recorded BEFORE place_order_for_signal()
    is called, so that even if place_order_for_signal() raises, this
    identity is not retried by a caller that catches the exception and
    calls guard_and_place_order() again for the same event. A caller
    that genuinely needs to retry after a raised exception must use a
    new event/identity, not silently loop this call.
    """

    identity = build_execution_identity(event)

    if store.has(identity):
        return OrderResult(
            accepted=False,
            dry_run=config.dry_run,
            request=None,
            retcode=None,
            ticket=None,
            reason=(
                "Duplicate execution attempt blocked by idempotency "
                f"guard: identity {identity!r} was already attempted. "
                "No broker contact was made for this call."
            ),
        )

    store.record(identity)

    on_before_send = None

    if attempt_store is not None:
        identity_key = encode_execution_identity(identity)

        def on_before_send(request: OrderRequest) -> None:
            attempt_store.record_order_attempt(
                identity_key,
                request,
                datetime.now(timezone.utc),
            )

    return place_order_for_signal(
        event,
        sizing,
        specs,
        config,
        on_before_send=on_before_send,
    )
