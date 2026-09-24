"""
phase_03_paper/execution/stage2_runner.py

Stage 2 orchestration: Triggered Signal -> PaperTradeEngine.

PURPOSE
-------
This module is the thin integration layer between:

    RuntimeCoordinator
        ->
    Triggered Strategy Events
        ->
    PaperTradeEngine

It does NOT contain:

- strategy logic
- session logic
- dashboard/UI logic
- broker execution logic
- position-sizing logic
- journal/persistence logic

Its only responsibility is to preserve the correct execution order
between the runtime coordinator and the paper-trading engine.


CAUSAL EXECUTION ORDER
----------------------

Every Stage 2 tick follows this exact sequence:

    1. coordinator.tick(now)
    2. paper_engine.on_candle(candle)
    3. coordinator.triggered_events()
    4. paper_engine.on_signal(signal)

This ordering is NON-NEGOTIABLE.

WHY THE ORDER MATTERS
---------------------

The existing strategy/backtest architecture follows the rule that
a trade opened on candle N must NOT be evaluated for stop/target using
candle N itself.

Therefore:

    Candle N
        |
        +--> evaluate existing open trades
        |
        +--> collect new triggered signals
        |
        +--> open new paper trades
        |
    Candle N+1
        |
        +--> newly opened trades may now be evaluated

Calling paper_engine.on_signal() before paper_engine.on_candle()
would allow a newly opened trade to be evaluated against its own
trigger candle and would violate the project's causal execution rule.


ARCHITECTURAL BOUNDARY
----------------------

RuntimeCoordinator coordinates runtime activity.

PaperTradeEngine manages simulated trade execution.

This module connects those two responsibilities without moving either
responsibility into the other component.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Protocol

from phase_03_paper.runtime.coordinator import RuntimeStatus
from phase_03_paper.trading.paper_engine import (
    PaperTrade,
    PaperTradeEngine,
    PaperTradeStatus,
)


class Stage2Coordinator(Protocol):
    """
    Minimal coordinator contract required by Stage 2.

    Using a protocol keeps this integration layer independent from the
    concrete RuntimeCoordinator implementation and makes deterministic
    testing easier.
    """

    def tick(self, now: Optional[datetime] = None) -> RuntimeStatus:
        ...

    def triggered_events(self) -> list:
        ...


@dataclass
class Stage2TickResult:
    """
    Result of one complete Stage 2 execution cycle.

    status:
        Runtime status returned by the coordinator.

    closed_trades:
        Existing paper trades closed while processing this candle.

    opened_trades:
        New paper trades successfully opened from this tick's
        triggered signal events.
    """

    status: RuntimeStatus
    closed_trades: List[PaperTrade] = field(default_factory=list)
    opened_trades: List[PaperTrade] = field(default_factory=list)


def run_tick_with_paper_execution(
    coordinator: Stage2Coordinator,
    paper_engine: PaperTradeEngine,
    now: Optional[datetime] = None,
) -> Stage2TickResult:
    """
    Execute exactly one Stage 2 paper-trading cycle.

    Execution order:

        1. Advance RuntimeCoordinator.
        2. Manage existing paper trades against the new candle.
        3. Drain newly triggered strategy events.
        4. Submit those events to PaperTradeEngine.

    The ordering guarantees that a trade opened on candle N is not
    evaluated against candle N itself.

    Parameters
    ----------
    coordinator:
        Runtime coordinator or compatible object implementing the
        Stage2Coordinator protocol.

    paper_engine:
        PaperTradeEngine responsible for simulated trade execution.

    now:
        Optional runtime timestamp passed to the coordinator and used
        when opening new paper trades.

    Returns
    -------
    Stage2TickResult
        Contains runtime status, trades closed during this candle,
        and trades opened from signals triggered on this candle.
    """

    # --------------------------------------------------------------
    # 1. Advance the runtime.
    # --------------------------------------------------------------
    status = coordinator.tick(now)

    # --------------------------------------------------------------
    # 2. Evaluate EXISTING open trades against the new candle.
    #
    # This MUST happen before new signals are submitted.
    # --------------------------------------------------------------
    candle = status.latest_candle

    if candle is not None:
        closed_trades = paper_engine.on_candle(candle)
    else:
        closed_trades = []

    # --------------------------------------------------------------
    # 3. Drain signals triggered during this runtime tick.
    # --------------------------------------------------------------
    events = coordinator.triggered_events()

    # --------------------------------------------------------------
    # 4. Open NEW paper trades.
    #
    # These trades were not present during step 2, so they cannot
    # be evaluated against their own trigger candle.
    # --------------------------------------------------------------
    opened_trades: List[PaperTrade] = []

    for event in events:
        trade = paper_engine.on_signal(
            event.signal,
            now=now,
        )

        if trade is not None and trade.status == PaperTradeStatus.OPEN:
            opened_trades.append(trade)

    # --------------------------------------------------------------
    # Return a complete description of this Stage 2 cycle.
    # --------------------------------------------------------------
    return Stage2TickResult(
        status=status,
        closed_trades=closed_trades,
        opened_trades=opened_trades,
    )


__all__ = [
    "Stage2Coordinator",
    "Stage2TickResult",
    "run_tick_with_paper_execution",
]