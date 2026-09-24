"""
phase_03_paper/runtime/test_coordinator_market_smoke.py

Phase 3 — Runtime Coordinator + Market Data Smoke Test

Purpose
-------
Verify the wiring between RuntimeCoordinator and MarketDataEngine.

This test proves that:
    1. No candle exists before the first tick.
    2. Each tick advances the market feed by exactly one candle.
    3. latest_candle() reflects the most recently consumed candle.
    4. RuntimeStatus reflects the coordinator's tick count and latest candle.
    5. An exhausted market-data source is not treated as an error.
    6. The last valid candle remains available after source exhaustion.

This test does NOT re-test MarketDataEngine's own normalization,
validation, or stream behavior. Those responsibilities are covered by:
    phase_03_paper/market/test_engine_smoke.py
"""

from __future__ import annotations

from typing import Iterator

from phase_03_paper.market.engine import MarketDataEngine
from phase_03_paper.notifications.engine import NotificationEngine
from phase_03_paper.runtime.coordinator import RuntimeCoordinator
from phase_03_paper.sessions.engine import SessionEngine


class FakeMarketDataSource:
    """Minimal two-candle source used only by this smoke test."""

    def stream(self) -> Iterator[dict]:
        yield {
            "timestamp": "2026-01-02T10:00:00+00:00",
            "open": "100.0",
            "high": "105.0",
            "low": "99.0",
            "close": "103.0",
            "volume": "1500",
        }

        yield {
            "timestamp": "2026-01-02T10:01:00+00:00",
            "open": "103.0",
            "high": "107.0",
            "low": "102.0",
            "close": "106.0",
            "volume": "1800",
        }


def main() -> None:
    print("=" * 70)
    print("PHASE 3 RUNTIME COORDINATOR + MARKET DATA SMOKE TEST")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Arrange
    # ------------------------------------------------------------------

    market_engine = MarketDataEngine(FakeMarketDataSource())
    session_engine = SessionEngine()
    notification_engine = NotificationEngine()

    coordinator = RuntimeCoordinator(
        session_engine=session_engine,
        notification_engine=notification_engine,
        market_data_engine=market_engine,
    )

    # ------------------------------------------------------------------
    # Initial state
    # ------------------------------------------------------------------

    initial_status = coordinator.status()

    assert initial_status.tick_count == 0, (
        f"Expected initial tick_count=0, got {initial_status.tick_count}"
    )
    assert initial_status.last_tick_at is None, (
        "Expected no last_tick_at before the first tick"
    )
    assert initial_status.last_error is None, (
        f"Expected no initial error, got {initial_status.last_error}"
    )
    assert initial_status.latest_candle is None, (
        "Expected no candle before the first tick"
    )

    # ------------------------------------------------------------------
    # Tick 1
    # ------------------------------------------------------------------

    first_status = coordinator.tick()

    assert first_status.tick_count == 1, (
        f"Expected tick_count=1 after first tick, got {first_status.tick_count}"
    )
    assert first_status.healthy, (
        f"Expected first tick to be healthy, got error: {first_status.last_error}"
    )

    first_candle = coordinator.latest_candle()
    assert first_candle is not None, "Expected exactly one candle after first tick"
    assert first_candle.close == 103.0, (
        f"Expected first candle close=103.0, got {first_candle.close}"
    )
    assert first_status.latest_candle == first_candle, (
        "RuntimeStatus.latest_candle should match coordinator.latest_candle()"
    )
    assert market_engine.candle_count() == 1, (
        f"Expected exactly 1 market candle after first tick, got {market_engine.candle_count()}"
    )

    # ------------------------------------------------------------------
    # Tick 2
    # ------------------------------------------------------------------

    second_status = coordinator.tick()

    assert second_status.tick_count == 2, (
        f"Expected tick_count=2 after second tick, got {second_status.tick_count}"
    )
    assert second_status.healthy, (
        f"Expected second tick to be healthy, got error: {second_status.last_error}"
    )

    second_candle = coordinator.latest_candle()
    assert second_candle is not None, "Expected a candle after second tick"
    assert second_candle.close == 106.0, (
        f"Expected second candle close=106.0, got {second_candle.close}"
    )
    assert second_candle.timestamp > first_candle.timestamp, (
        "Expected the second candle timestamp to be later than the first candle timestamp"
    )
    assert second_status.latest_candle == second_candle, (
        "RuntimeStatus.latest_candle should match the second latest candle"
    )
    assert market_engine.candle_count() == 2, (
        f"Expected exactly 2 market candles after second tick, got {market_engine.candle_count()}"
    )

    # ------------------------------------------------------------------
    # Tick 3 — Source Exhausted
    # ------------------------------------------------------------------

    third_status = coordinator.tick()

    assert third_status.tick_count == 3, (
        f"Expected tick_count=3 after third tick, got {third_status.tick_count}"
    )
    assert third_status.healthy, (
        f"An exhausted market-data source should not make runtime unhealthy. Got error: {third_status.last_error}"
    )
    assert market_engine.candle_count() == 2, (
        f"Expected candle count to remain 2 after source exhaustion, got {market_engine.candle_count()}"
    )

    third_candle = coordinator.latest_candle()
    assert third_candle is not None, (
        "Expected the last valid candle to remain available after source exhaustion"
    )
    assert third_candle.close == 106.0, (
        f"Expected latest_candle() to remain second candle after exhaustion, got {third_candle.close}"
    )
    assert third_status.latest_candle == third_candle, (
        "RuntimeStatus.latest_candle should remain the last valid candle"
    )

    # ------------------------------------------------------------------
    # Final status check
    # ------------------------------------------------------------------

    final_status = coordinator.status()
    assert final_status.tick_count == 3
    assert final_status.healthy
    assert final_status.latest_candle is not None
    assert final_status.latest_candle.close == 106.0

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    print()
    print("Initial tick count  :", initial_status.tick_count)
    print("After tick 1        :", first_status.tick_count)
    print("After tick 2        :", second_status.tick_count)
    print("After tick 3        :", third_status.tick_count)
    print("Market candles consumed:", market_engine.candle_count())
    print("Final latest candle close:", final_status.latest_candle.close)
    print("Runtime healthy     :", final_status.healthy)
    print()
    print("PASS: RuntimeCoordinator + MarketDataEngine smoke test completed successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()