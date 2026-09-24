"""
phase_03_paper/tests/test_stage2_integration.py

Stage 2 integration test:
Triggered Strategy Signal -> PaperTradeEngine.

PURPOSE
-------
This test verifies the causal boundary between the Phase 3 runtime
signal layer and the paper-trading execution layer.

It intentionally uses a deterministic hand-built scenario instead of
the real market dataset.

The test proves that:

    1. A triggered signal reaches the PaperTradeEngine.
    2. A triggered signal opens exactly one paper trade.
    3. A newly opened trade is NOT evaluated against its own trigger
       candle.
    4. An existing trade is evaluated on subsequent candles.
    5. A later candle can close the trade.
    6. Duplicate submission of the same signal does not create a
       second paper trade.

ARCHITECTURAL SCOPE
-------------------
This test does NOT test the real RuntimeCoordinator or StrategyAdapter.

Those components have their own tests.

This test specifically validates the integration function:

    run_tick_with_paper_execution()

and its execution ordering.

CAUSAL EXECUTION ORDER
----------------------

Every Stage 2 tick must follow:

    1. coordinator.tick(now)
    2. paper_engine.on_candle(candle)
    3. coordinator.triggered_events()
    4. paper_engine.on_signal(signal)

The ordering is critical.

A trade opened on candle N must NOT be evaluated against candle N.
It becomes eligible for stop/target evaluation beginning with candle
N+1.

Therefore:

    Candle N
        |
        +--> evaluate existing trades
        |
        +--> collect triggered signals
        |
        +--> open new trades
        |
    Candle N+1
        |
        +--> newly opened trades can now be evaluated
"""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from typing import List

from phase_03_paper.config import Phase3Config
from phase_03_paper.events.audit import AuditLog
from phase_03_paper.execution.stage2_runner import (
    run_tick_with_paper_execution,
)
from phase_03_paper.market.engine import Candle
from phase_03_paper.runtime.coordinator import RuntimeStatus
from phase_03_paper.trading.paper_engine import (
    PaperTradeEngine,
    PaperTradeStatus,
)
from strategy.signals import (
    SignalStatus,
    SignalType,
    TradeSignal,
)


class _FakeAdapterSignalEvent:
    """
    Minimal stand-in for AdapterSignalEvent.

    Stage 2 only requires the event to expose:

        event.signal
    """

    def __init__(self, signal: TradeSignal):
        self.signal = signal


class _ScriptedCoordinator:
    """
    Deterministic coordinator used only by this test.

    It allows the test to control exactly:

        - which candle is returned on each tick
        - which triggered events are returned on each tick

    This keeps the test focused on Stage 2 execution ordering.

    Only the methods required by run_tick_with_paper_execution()
    are implemented:

        tick()
        triggered_events()
    """

    def __init__(
        self,
        candles: List[Candle],
        events_by_index: dict[int, list],
    ):
        self._candles = candles
        self._events_by_index = events_by_index
        self._index = -1

    def tick(
        self,
        now: datetime | None = None,
    ) -> RuntimeStatus:
        """Return the next deterministic candle."""

        self._index += 1

        if self._index >= len(self._candles):
            raise RuntimeError(
                "Scripted coordinator received more ticks "
                "than available test candles."
            )

        candle = self._candles[self._index]

        return RuntimeStatus(
            tick_count=self._index + 1,
            last_tick_at=now,
            last_error=None,
            latest_candle=candle,
        )

    def triggered_events(self) -> list:
        """Return events scheduled for the current tick."""

        return self._events_by_index.get(
            self._index,
            [],
        )


def _make_candle(
    timestamp: datetime,
    open_price: float,
    high: float,
    low: float,
    close: float,
) -> Candle:
    """Create a deterministic timezone-aware test candle."""

    return Candle(
        timestamp=timestamp,
        open=open_price,
        high=high,
        low=low,
        close=close,
    )


def _make_long_signal(
    entry: float,
    stop: float,
    target: float,
) -> TradeSignal:
    """Create a deterministic TRIGGERED long signal."""

    signal = TradeSignal(
        signal_type=SignalType.LONG,
        entry_price=entry,
        stop_loss=stop,
        take_profit=target,
    )

    signal.status = SignalStatus.TRIGGERED

    return signal


class TestStage2SignalToPaperExecution(unittest.TestCase):
    """
    Stage 2 integration tests.

    Signal -> PaperTradeEngine.
    """

    @staticmethod
    def _build_engine() -> PaperTradeEngine:
        """Create a fresh PaperTradeEngine for each test."""

        config = Phase3Config()
        audit_log = AuditLog()

        return PaperTradeEngine(
            config=config,
            audit_log=audit_log,
        )

    def test_trade_opens_and_is_not_evaluated_on_its_own_trigger_candle(
        self,
    ):
        """
        Verify the same-candle-exclusion rule.

        Scenario:

            Candle 0:
                Signal triggers.
                Trade opens.
                Candle range reaches the target.
                Target MUST be ignored.

            Candle 1:
                Quiet candle.
                Trade remains open.

            Candle 2:
                Target is reached.
                Trade closes.

        This proves that a newly opened trade is not evaluated against
        the candle that created the trade.
        """

        base = datetime(
            2026,
            1,
            1,
            12,
            0,
            tzinfo=timezone.utc,
        )

        # ----------------------------------------------------------
        # Candle 0: trigger candle
        #
        # High = 1.2000, which is above target = 1.1500.
        #
        # The trade must NOT close here because the trade is only
        # created after this candle has already been processed.
        # ----------------------------------------------------------

        candle_0 = _make_candle(
            timestamp=base,
            open_price=1.1000,
            high=1.2000,
            low=1.0900,
            close=1.1050,
        )

        # ----------------------------------------------------------
        # Candle 1: quiet candle
        # ----------------------------------------------------------

        candle_1 = _make_candle(
            timestamp=base + timedelta(minutes=5),
            open_price=1.1050,
            high=1.1080,
            low=1.1020,
            close=1.1060,
        )

        # ----------------------------------------------------------
        # Candle 2: later target hit
        # ----------------------------------------------------------

        candle_2 = _make_candle(
            timestamp=base + timedelta(minutes=10),
            open_price=1.1060,
            high=1.2000,
            low=1.1040,
            close=1.1990,
        )

        signal = _make_long_signal(
            entry=1.1000,
            stop=1.0950,
            target=1.1500,
        )

        coordinator = _ScriptedCoordinator(
            candles=[
                candle_0,
                candle_1,
                candle_2,
            ],
            events_by_index={
                0: [
                    _FakeAdapterSignalEvent(signal),
                ],
            },
        )

        paper_engine = self._build_engine()

        # ==========================================================
        # TICK 0
        # ==========================================================

        result_0 = run_tick_with_paper_execution(
            coordinator=coordinator,
            paper_engine=paper_engine,
            now=base,
        )

        self.assertEqual(
            len(result_0.opened_trades),
            1,
            "Exactly one trade should open from the triggered signal.",
        )

        self.assertEqual(
            len(result_0.closed_trades),
            0,
            "The newly opened trade must not close on its trigger candle.",
        )

        opened_trade = result_0.opened_trades[0]

        self.assertEqual(
            opened_trade.status,
            PaperTradeStatus.OPEN,
        )

        self.assertEqual(
            len(paper_engine.open_trades()),
            1,
            "The opened trade must remain active after candle 0.",
        )

        # ==========================================================
        # TICK 1
        # ==========================================================

        result_1 = run_tick_with_paper_execution(
            coordinator=coordinator,
            paper_engine=paper_engine,
            now=base + timedelta(minutes=5),
        )

        self.assertEqual(
            len(result_1.closed_trades),
            0,
            "The quiet candle must not close the trade.",
        )

        self.assertEqual(
            len(paper_engine.open_trades()),
            1,
            "The trade must remain open after candle 1.",
        )

        # ==========================================================
        # TICK 2
        # ==========================================================

        result_2 = run_tick_with_paper_execution(
            coordinator=coordinator,
            paper_engine=paper_engine,
            now=base + timedelta(minutes=10),
        )

        self.assertEqual(
            len(result_2.closed_trades),
            1,
            "Exactly one trade should close on the later target candle.",
        )

        self.assertEqual(
            len(paper_engine.open_trades()),
            0,
            "No open trades should remain after the target is reached.",
        )

        closed_trade = result_2.closed_trades[0]

        self.assertEqual(
            closed_trade.trade_id,
            opened_trade.trade_id,
            "The trade that closes must be the same trade that opened.",
        )

        self.assertEqual(
            closed_trade.status,
            PaperTradeStatus.CLOSED,
        )

        print()
        print("=" * 70)
        print("STAGE 2 — SAME-CANDLE EXCLUSION TEST PASSED")
        print("=" * 70)

    def test_duplicate_signal_does_not_open_second_trade(self):
        """
        Verify duplicate signal protection.

        The same signal is submitted on two consecutive ticks.

        Expected:

            First submission:
                1 trade opens.

            Second submission:
                0 additional trades open.

        Final engine state:

            Total trades = 1
        """

        base = datetime(
            2026,
            1,
            1,
            12,
            0,
            tzinfo=timezone.utc,
        )

        candle_0 = _make_candle(
            timestamp=base,
            open_price=1.1000,
            high=1.1010,
            low=1.0990,
            close=1.1005,
        )

        candle_1 = _make_candle(
            timestamp=base + timedelta(minutes=5),
            open_price=1.1005,
            high=1.1010,
            low=1.1000,
            close=1.1005,
        )

        signal = _make_long_signal(
            entry=1.1000,
            stop=1.0950,
            target=1.1500,
        )

        coordinator = _ScriptedCoordinator(
            candles=[
                candle_0,
                candle_1,
            ],
            events_by_index={
                0: [
                    _FakeAdapterSignalEvent(signal),
                ],
                1: [
                    _FakeAdapterSignalEvent(signal),
                ],
            },
        )

        paper_engine = self._build_engine()

        # ----------------------------------------------------------
        # First signal submission.
        # ----------------------------------------------------------

        result_0 = run_tick_with_paper_execution(
            coordinator=coordinator,
            paper_engine=paper_engine,
            now=base,
        )

        # ----------------------------------------------------------
        # Duplicate signal submission.
        # ----------------------------------------------------------

        result_1 = run_tick_with_paper_execution(
            coordinator=coordinator,
            paper_engine=paper_engine,
            now=base + timedelta(minutes=5),
        )

        self.assertEqual(
            len(result_0.opened_trades),
            1,
            "The first signal should open exactly one trade.",
        )

        self.assertEqual(
            len(result_1.opened_trades),
            0,
            "The duplicate signal must not open another trade.",
        )

        self.assertEqual(
            len(paper_engine.all_trades()),
            1,
            "The engine must contain exactly one trade.",
        )

        print()
        print("=" * 70)
        print("STAGE 2 — DUPLICATE SIGNAL TEST PASSED")
        print("=" * 70)


if __name__ == "__main__":
    unittest.main()