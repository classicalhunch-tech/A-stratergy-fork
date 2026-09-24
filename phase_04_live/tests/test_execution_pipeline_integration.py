"""
phase_04_live/tests/test_execution_pipeline_integration.py

Phase 4 Execution Pipeline Integration Test
============================================

Purpose
-------
Proves that a REAL triggered strategy signal can travel through the
complete Phase 4 execution boundary, end to end, using the same
validated Stage 1 dataset and plumbing:

    MarketDataEngine
          |
          v
    RuntimeCoordinator
          |
          v
    StrategyAdapter
          |
          v
    Existing strategy/
          |
          v
    Triggered AdapterSignalEvent
          |
          v
    compute_position_size()
          |
          v
    place_order_for_signal()
          |
          v
    Dry-run MT5 request
          |
          X
    mt5.order_send() NEVER called

This test deliberately reuses the exact CSV dataset and loader/source
classes already validated by
phase_03_paper.tests.test_stage1_integration.

The purpose is to prove the NEXT stage of an already-proven pipeline,
not to re-validate Stage 1 using a second synthetic fixture.

What this test does NOT do
--------------------------
Per the Phase 4 roadmap, this test does NOT:

    - turn dry_run off
    - send a real broker order
    - add position reconciliation
    - add automatic retries
    - modify strategy logic
    - duplicate strategy logic
    - call live-only get_symbol_trading_specs()
    - call live-only get_account_info()

NOTE: this test DOES exercise check_symbol_compatibility(), because
place_order_for_signal() calls it internally and it makes its own
mt5.symbol_info() call (a separate MT5 reference living in
phase_04_live.broker.compatibility, not phase_04_live.orders.order_manager).
Since there is no live MT5 terminal connection in this environment,
mt5.symbol_info() must be mocked here too, alongside symbol_info_tick
and order_send, or the compatibility check fails closed with
"No IPC connection" regardless of how good the signal/sizing/order
request is. The mock below returns a symbol_info() consistent with
TEST_SPECS -- it does NOT re-validate the compatibility logic itself
(that lives in phase_04_live/broker/test_compatibility.py).

AccountInfo and SymbolTradingSpecs are constructed directly here
because compute_position_size() only needs those resulting dataclasses.

Mocking boundary
----------------
The following direct MT5 calls are mocked:

    - mt5.order_send                              (order_manager module)
    - mt5.symbol_info_tick                        (order_manager module)
    - mt5.symbol_info                             (compatibility module)

Everything upstream of these calls runs for real:

    historical CSV
        -> MarketDataEngine
        -> RuntimeCoordinator
        -> StrategyAdapter
        -> strategy/
        -> real triggered signal
        -> real Phase 4 risk sizing
        -> real Phase 4 OrderManager
        -> real Phase 4 compatibility check (against a mocked symbol_info)

This ensures the test proves a genuine end-to-end execution path
rather than manually constructing a fake strategy signal.

NOTE ON SL/TP:
AdapterSignalEvent does NOT carry stop_loss/take_profit directly --
only signal, setup_bar_index, trigger_bar_index, fill_price, and
initial_risk (see phase_03_paper/signals/adapter.py). The real
strategy stop/target live on event.signal.stop_loss and
event.signal.take_profit, exactly as OrderManager itself reads them.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import MetaTrader5 as mt5

from phase_03_paper.market.engine import MarketDataEngine
from phase_03_paper.notifications.engine import NotificationEngine
from phase_03_paper.runtime.coordinator import RuntimeCoordinator
from phase_03_paper.sessions.engine import SessionDecision, SessionEngine
from phase_03_paper.signals.adapter import StrategyAdapter

# Reuse the already-validated Stage 1 dataset loader and CSV source.
from phase_03_paper.tests.test_stage1_integration import (
    CANDLE_LIMIT,
    DATA_FILE,
    _CSVMarketDataSource,
    _load_data,
)

from phase_04_live.broker.compatibility import SYMBOL_FILLING_IOC
from phase_04_live.orders.order_manager import (
    OrderManagerConfig,
    OrderResult,
    place_order_for_signal,
)
from phase_04_live.risk.sizing import (
    AccountInfo,
    LiveRiskConfig,
    PositionSizeResult,
    SymbolTradingSpecs,
    compute_position_size,
)
from strategy.signals import SignalType


ORDER_MANAGER_MODULE = "phase_04_live.orders.order_manager"
COMPATIBILITY_MODULE = "phase_04_live.broker.compatibility"


# ----------------------------------------------------------------------
# Controlled Phase 4 test environment
# ----------------------------------------------------------------------
#
# These specifications intentionally do not come from a live MT5
# connection. This test is proving the execution pipeline, not live
# broker discovery.
#
# The dataset uses EURUSD-style prices around 1.08, so these specs
# provide a realistic 5-decimal FX execution environment.
# ----------------------------------------------------------------------

TEST_SPECS = SymbolTradingSpecs(
    symbol="EURUSD",
    contract_size=100_000.0,
    volume_min=0.01,
    volume_max=100.0,
    volume_step=0.01,
    point=0.00001,
    digits=5,
)

TEST_ACCOUNT = AccountInfo(
    balance=10_000.0,
    equity=10_000.0,
    margin_free=10_000.0,
    leverage=100,
    currency="USD",
)


def _tick_near(
    price: float,
    half_spread: float = 0.00001,
):
    """
    Build a mock MT5 tick tightly centered on `price`.

    With a 5-decimal FX symbol:

        bid = price - 1 point
        ask = price + 1 point

    This produces a 2-point total spread.

    The spread remains comfortably below the configured OrderManager
    limit while still exercising the real bid/ask execution-price
    selection logic. Deterministic from `price` alone, so it can be
    safely recomputed later (see section 9) instead of relying on a
    mutable loop variable.
    """
    return SimpleNamespace(
        bid=price - half_spread,
        ask=price + half_spread,
    )


def _mock_symbol_info():
    """
    Build a mock MT5 symbol_info() result for check_symbol_compatibility().

    This test is proving pipeline connectivity, not broker compatibility
    logic itself (that is covered separately by
    phase_04_live/broker/test_compatibility.py). The values here are
    chosen to resolve as unambiguously compatible and consistent with
    TEST_SPECS:

        - trade_mode=4 (FULL): both LONG and SHORT signals are tradable
        - trade_exemode=MARKET: a real, common MT5 execution mode
        - filling_mode explicitly advertises IOC, matching what
          OrderManager currently hardcodes (ORDER_FILLING_IOC), so the
          compatibility layer does not emit a filling-mismatch warning
        - trade_stops_level / trade_freeze_level = 0: no broker-imposed
          minimum distance to interfere with the test's tight synthetic
          spread/prices
    """
    return SimpleNamespace(
        visible=True,
        trade_mode=4,  # FULL
        trade_exemode=mt5.SYMBOL_TRADE_EXECUTION_MARKET,
        volume_min=TEST_SPECS.volume_min,
        volume_max=TEST_SPECS.volume_max,
        volume_step=TEST_SPECS.volume_step,
        point=TEST_SPECS.point,
        digits=TEST_SPECS.digits,
        trade_stops_level=0,
        trade_freeze_level=0,
        filling_mode=SYMBOL_FILLING_IOC,
    )


class TestPhase4ExecutionPipelineIntegration(unittest.TestCase):
    """
    End-to-end integration test:

        real historical candles
            -> real strategy signal
            -> real risk sizing
            -> real OrderManager
            -> real compatibility check (against mocked symbol_info)
            -> dry-run MT5 request

    No real broker order is submitted.
    """

    data_file: str = DATA_FILE
    candle_limit: int = CANDLE_LIMIT

    def test_triggered_signal_reaches_order_manager_as_dry_run_request(
        self,
    ):
        # ==============================================================
        # 1. Load the exact dataset already validated by Stage 1.
        # ==============================================================

        try:
            df = _load_data(
                path=self.data_file,
                limit=self.candle_limit,
            )
        except Exception as exc:
            self.fail(
                f"Failed to load dataset from {self.data_file}: {exc}"
            )

        self.assertGreater(
            len(df),
            0,
            "No valid candles were loaded.",
        )

        # ==============================================================
        # 2. Build the same Phase 3 pipeline already validated by
        #    Stage 1.
        # ==============================================================

        source = _CSVMarketDataSource(df)

        market_data_engine = MarketDataEngine(source)
        adapter = StrategyAdapter()
        session_engine = SessionEngine()
        notification_engine = NotificationEngine()

        coordinator = RuntimeCoordinator(
            session_engine=session_engine,
            notification_engine=notification_engine,
            market_data_engine=market_data_engine,
            strategy_adapter=adapter,
        )

        # Phase 4 execution configuration.
        #
        # dry_run=True is mandatory for this integration test.
        order_config = OrderManagerConfig(
            dry_run=True,
        )

        # Fixed-lot sizing keeps this integration test focused on
        # pipeline connectivity rather than dynamic risk policy.
        risk_config = LiveRiskConfig(
            sizing_mode="fixed_lot",
            fixed_lot_size=0.01,
        )

        all_triggered = []
        sizing_results: list[PositionSizeResult] = []
        order_results: list[OrderResult] = []

        # ==============================================================
        # 3. Stream every historical candle through the real Phase 3
        #    runtime.
        #
        #    Session approval follows the same pattern used by the
        #    already-validated Stage 1 integration test.
        # ==============================================================

        for timestamp, _row in df.iterrows():
            now = timestamp.to_pydatetime()

            for state in session_engine.evaluate(now):
                if state.decision == SessionDecision.UNDECIDED:
                    session_engine.record_decision(
                        state.session_name,
                        SessionDecision.TRADE,
                        note=(
                            "auto-approved by phase4 "
                            "integration test"
                        ),
                        now=now,
                    )

            coordinator.tick(now=now)

            new_events = coordinator.triggered_events()

            all_triggered.extend(new_events)

            # ==========================================================
            # 4. Push each REAL triggered event through Phase 4.
            #
            #    Strategy
            #       -> Risk
            #       -> OrderManager
            #       -> Compatibility check (mocked symbol_info)
            #       -> Dry-run MT5 request
            # ==========================================================

            for event in new_events:
                sizing = compute_position_size(
                    event=event,
                    account=TEST_ACCOUNT,
                    config=risk_config,
                    specs=TEST_SPECS,
                )

                sizing_results.append(sizing)

                # Keep the mocked execution price close to the
                # strategy fill price so OrderManager's real
                # staleness/deviation validation is exercised.
                mock_tick = _tick_near(event.fill_price)

                with (
                    patch(
                        f"{ORDER_MANAGER_MODULE}.mt5.order_send"
                    ) as mock_send,
                    patch(
                        f"{ORDER_MANAGER_MODULE}.mt5.symbol_info_tick"
                    ) as mock_get_tick,
                    patch(
                        f"{COMPATIBILITY_MODULE}.mt5.symbol_info"
                    ) as mock_symbol_info,
                ):
                    mock_get_tick.return_value = mock_tick
                    mock_symbol_info.return_value = _mock_symbol_info()

                    result = place_order_for_signal(
                        event=event,
                        sizing=sizing,
                        specs=TEST_SPECS,
                        config=order_config,
                    )

                    order_results.append(result)

                    # The integration test must never reach the broker.
                    mock_send.assert_not_called()

        # ==============================================================
        # 5. Diagnostics
        # ==============================================================

        print()
        print("=" * 76)
        print("PHASE 4 -- SIGNAL -> RISK -> ORDER INTEGRATION TEST")
        print("=" * 76)
        print(f"Candles supplied      : {len(df)}")
        print(f"Triggered events      : {len(all_triggered)}")
        print(f"Sizing results        : {len(sizing_results)}")
        print(f"Order manager results : {len(order_results)}")

        for event, sizing, order in zip(
            all_triggered,
            sizing_results,
            order_results,
        ):
            print("-" * 76)

            print(
                f"trigger_bar={event.trigger_bar_index} "
                f"fill={event.fill_price:.8f} "
                f"risk={event.initial_risk:.8f} "
                f"direction={event.signal.signal_type}"
            )

            print(
                "  sizing: "
                f"accepted={sizing.accepted} "
                f"lots={sizing.lots} "
                f"reason={sizing.reason}"
            )

            print(
                "  order : "
                f"accepted={order.accepted} "
                f"dry_run={order.dry_run} "
                f"reason={order.reason}"
            )

            if order.request is not None:
                print(
                    "  request: "
                    f"{order.request.as_mt5_request()}"
                )

        print("=" * 76)

        # ==============================================================
        # 6. Core integration assertions
        # ==============================================================

        # Never allow the test to pass vacuously. We need at least one
        # genuine strategy signal to prove the Phase 4 stages.
        self.assertGreater(
            len(all_triggered),
            0,
            "No triggered signal was produced -- cannot prove the "
            "risk/order stages without a real signal.",
        )

        self.assertEqual(
            len(sizing_results),
            len(all_triggered),
        )

        self.assertEqual(
            len(order_results),
            len(all_triggered),
        )

        # ==============================================================
        # 7. Validate the risk stage
        # ==============================================================

        for sizing in sizing_results:
            self.assertTrue(
                sizing.accepted,
                (
                    "Fixed-lot sizing unexpectedly rejected: "
                    f"{sizing.reason}"
                ),
            )

            self.assertIn(
                sizing.direction,
                (
                    SignalType.LONG,
                    SignalType.SHORT,
                ),
            )

            self.assertEqual(
                sizing.lots,
                0.01,
            )

        # ==============================================================
        # 8. Validate the OrderManager stage
        # ==============================================================

        for order in order_results:
            self.assertTrue(
                order.accepted,
                (
                    "OrderManager unexpectedly rejected a valid "
                    f"dry-run request: {order.reason}"
                ),
            )

            self.assertTrue(
                order.dry_run,
            )

            self.assertIsNotNone(
                order.request,
            )

            self.assertIsNone(
                order.retcode,
            )

            self.assertIsNone(
                order.ticket,
            )

        # ==============================================================
        # 9. Validate that important values survived the entire
        #    strategy -> risk -> execution boundary.
        #
        #    FIX: AdapterSignalEvent has no .stop_loss/.take_profit of
        #    its own -- those live on event.signal.stop_loss /
        #    event.signal.take_profit, exactly as OrderManager itself
        #    reads them. Also, the expected tick is recomputed per
        #    event here (deterministic from event.fill_price) rather
        #    than reusing whatever `mock_tick` was last left holding
        #    by the loop in section 4 -- that would only happen to be
        #    correct when exactly one signal fires.
        # ==============================================================

        for event, sizing, order in zip(
            all_triggered,
            sizing_results,
            order_results,
        ):
            request = order.request

            self.assertIsNotNone(request)

            # Execution symbol must match the controlled Phase 4 specs.
            self.assertEqual(
                request.symbol,
                TEST_SPECS.symbol,
            )

            # Risk sizing must determine the requested volume.
            self.assertEqual(
                request.volume,
                sizing.lots,
            )

            # Strategy-generated stop and target must survive unchanged
            # into the final order request -- read from event.signal,
            # matching what OrderManager itself reads.
            self.assertEqual(
                request.sl,
                event.signal.stop_loss,
            )

            self.assertEqual(
                request.tp,
                event.signal.take_profit,
            )

            # Execution price must come from the mocked live execution
            # tick for THIS event, not directly from the historical
            # strategy fill_price, and not from a stale tick belonging
            # to a different event.
            expected_tick = _tick_near(event.fill_price)

            expected_price = (
                expected_tick.ask
                if event.signal.signal_type == SignalType.LONG
                else expected_tick.bid
            )

            self.assertEqual(
                request.price,
                expected_price,
            )

        # ==============================================================
        # 10. Final human-readable pass markers
        # ==============================================================

        print()
        print("[PASS] Real triggered signal(s) reached Phase 4 risk sizing")
        print("[PASS] Risk sizing accepted and produced valid lot size(s)")
        print("[PASS] Compatibility check passed against mocked symbol_info")
        print("[PASS] OrderManager accepted the dry-run execution request")
        print("[PASS] Strategy SL/TP reached the final order request")
        print("[PASS] Execution price came from the mocked bid/ask tick")
        print("[PASS] mt5.order_send() was never called")
        print()


if __name__ == "__main__":
    unittest.main(verbosity=2)