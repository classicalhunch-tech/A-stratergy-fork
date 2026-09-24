"""
phase_03_paper/trading/test_paper_engine_smoke.py

Smoke test for PaperTradeEngine.

This test verifies the PaperTradeEngine independently of:
    - the strategy
    - the session engine
    - the RuntimeCoordinator
    - the future Strategy Adapter

Covers:
    1.  Non-TRIGGERED signal is ignored.
    2.  Valid TRIGGERED LONG signal opens correctly.
    3.  Valid TRIGGERED SHORT signal opens correctly.
    4.  Duplicate submission creates only one trade.
    5.  LONG stop detection.
    6.  LONG target detection.
    7.  SHORT stop detection.
    8.  SHORT target detection.
    9.  A candle touching both stop and target resolves as STOP_HIT.
    10. Rejection probability produces a REJECTED trade deterministically.
    11. Spread/slippage moves the simulated entry adversely.
    12. R calculation is correct using the actual simulated entry.
    13. Closed trades move from open trades to closed trades.
    14. Rejected trades are stored separately.
    15. Audit events are generated correctly.
    16. Multiple open trades can be processed by one candle.
    17. Datetimes remain UTC-aware.

All inputs are deterministic.

The test does not depend on DEFAULT_CONFIG values.
Randomness is injected through random.Random.

NOTE ON ORDERING:
    on_candle() advances every currently-open trade against the SAME
    candle, because they share one underlying market feed. The LONG
    trades in this file live around the ~1.10 price band and the SHORT
    trades live around the ~1.20 price band. If a long trade and a
    short trade were left open at the same time, a candle built to hit
    the long trade's stop (~1.09) would also fall below the short
    trade's target (1.1900) and close it early. To avoid that
    cross-contamination, each single-trade test fully opens and closes
    its trade before the next one opens. Concurrent-open behavior is
    still explicitly exercised in TEST 16, using two trades on a
    matching price band.
"""

from __future__ import annotations

import random
from datetime import datetime, timezone

from phase_03_paper.config import Phase3Config, PaperExecutionSettings
from phase_03_paper.events.audit import AuditLog
from phase_03_paper.market.engine import Candle
from phase_03_paper.trading.paper_engine import (
    ExitReason,
    PaperTradeEngine,
    PaperTradeStatus,
)
from strategy.signals import SignalStatus, SignalType, TradeSignal


# ============================================================
# DETERMINISTIC TEST CONFIGURATION
# ============================================================

# spread = 0.0002
# spread / 2 = 0.0001
# fixed slippage = 0.0001
#
# Total adverse movement:
#     0.0001 + 0.0001 = 0.0002

_NO_REJECTION_CONFIG = Phase3Config(
    paper_execution=PaperExecutionSettings(
        starting_balance=10_000.0,
        simulated_spread=0.0002,
        slippage_model="fixed",
        slippage_fixed_amount=0.0001,
        latency_model="fixed",
        latency_fixed_ms=150.0,
        rejection_probability=0.0,
    )
)

_ALWAYS_REJECT_CONFIG = Phase3Config(
    paper_execution=PaperExecutionSettings(
        starting_balance=10_000.0,
        simulated_spread=0.0002,
        slippage_model="fixed",
        slippage_fixed_amount=0.0001,
        latency_model="fixed",
        latency_fixed_ms=150.0,
        rejection_probability=1.0,
    )
)

_ADVERSE_AMOUNT = 0.0002


# ============================================================
# TEST HELPERS
# ============================================================


def _utc(hour: int, minute: int) -> datetime:
    """Return a deterministic UTC-aware datetime."""
    return datetime(
        2026,
        1,
        1,
        hour,
        minute,
        tzinfo=timezone.utc,
    )


def _build_signal(
    signal_type: SignalType,
    entry_price: float,
    stop_loss: float,
    take_profit: float,
    zone_stable_key: str,
    status: SignalStatus = SignalStatus.TRIGGERED,
    bar_index: int = 1,
    setup_bar_index: int = 0,
    session: str = "TestSession",
) -> TradeSignal:
    """Build a minimal deterministic TradeSignal for isolated testing."""
    return TradeSignal(
        signal_type=signal_type,
        entry_price=entry_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        status=status,
        bar_index=bar_index,
        setup_bar_index=setup_bar_index,
        zone_stable_key=zone_stable_key,
        session=session,
    )


def _make_candle(
    hour: int,
    minute: int,
    open_: float,
    high: float,
    low: float,
    close: float,
) -> Candle:
    """Build a deterministic UTC-aware Candle."""
    return Candle(
        timestamp=_utc(hour, minute),
        open=open_,
        high=high,
        low=low,
        close=close,
    )


# ============================================================
# SMOKE TEST
# ============================================================


def run_smoke_test() -> None:

    # ========================================================
    # TEST 1
    # Non-TRIGGERED signal is ignored.
    # ========================================================

    audit_log = AuditLog()

    engine = PaperTradeEngine(
        config=_NO_REJECTION_CONFIG,
        audit_log=audit_log,
        rng=random.Random(1),
    )

    pending_signal = _build_signal(
        SignalType.LONG,
        1.1000,
        1.0950,
        1.1100,
        zone_stable_key="ignored-1",
        status=SignalStatus.PENDING_RETEST,
        bar_index=1,
        setup_bar_index=0,
        session="TestSession",
    )

    result = engine.on_signal(
        pending_signal,
        now=_utc(6, 0),
    )

    assert result is None
    assert engine.all_trades() == []


    # ========================================================
    # TEST 2
    # Valid TRIGGERED LONG signal opens correctly.
    # ========================================================

    long_signal = _build_signal(
        SignalType.LONG,
        1.1000,
        1.0950,
        1.1100,
        zone_stable_key="long-open-1",
        status=SignalStatus.TRIGGERED,
        bar_index=2,
        setup_bar_index=0,
        session="TestSession",
    )

    long_trade = engine.on_signal(
        long_signal,
        now=_utc(6, 5),
    )

    assert long_trade is not None
    assert long_trade.status == PaperTradeStatus.OPEN
    assert long_trade.direction == SignalType.LONG

    assert long_trade.requested_entry == 1.1000

    assert abs(
        long_trade.entry - (1.1000 + _ADVERSE_AMOUNT)
    ) < 1e-9

    assert long_trade in engine.open_trades()

    # LONG adverse fill must be above requested entry.
    assert long_trade.entry > long_trade.requested_entry

    # Open timestamp must be explicitly UTC-aware.
    assert long_trade.opened_at is not None
    assert long_trade.opened_at.tzinfo == timezone.utc


    # ========================================================
    # TEST 4
    # Duplicate submission creates only one trade.
    #
    # Run this here, while only long_trade is open, so the
    # open-trade count check isn't affected by trades opened
    # later in the file.
    # ========================================================

    duplicate_result = engine.on_signal(
        long_signal,
        now=_utc(6, 6),
    )

    assert duplicate_result is None

    # Only the original LONG trade exists so far.
    assert len(engine.open_trades()) == 1


    # ========================================================
    # TEST 5 + TEST 12
    # LONG stop detection + R calculation.
    #
    # No other trade is open yet, so this candle can only
    # affect long_trade.
    # ========================================================

    stop_hit_candle = _make_candle(
        7,
        0,
        open_=1.0950,
        high=1.0960,
        low=1.0940,
        close=1.0945,
    )

    closed_now = engine.on_candle(stop_hit_candle)

    assert long_trade in closed_now
    assert long_trade.status == PaperTradeStatus.CLOSED
    assert long_trade.exit_reason == ExitReason.STOP_HIT
    assert long_trade.exit_price == 1.0950

    # Hand calculation:
    #
    # actual entry = 1.1002
    # stop         = 1.0950
    #
    # risk         = 0.0052
    # reward       = -0.0052
    #
    # R = -1.0

    assert abs(
        long_trade.result_r - (-1.0)
    ) < 1e-9

    # Closed trade must no longer be open.
    assert long_trade not in engine.open_trades()
    assert long_trade in engine.closed_trades()

    # Closed timestamp must be explicitly UTC-aware.
    assert long_trade.closed_at is not None
    assert long_trade.closed_at.tzinfo == timezone.utc


    # ========================================================
    # TEST 3
    # Valid TRIGGERED SHORT signal opens correctly.
    #
    # Opened only now, after long_trade is fully closed, so
    # this trade's ~1.20 price band never overlaps with a
    # candle built for the ~1.10 long band.
    # ========================================================

    short_signal = _build_signal(
        SignalType.SHORT,
        1.2000,
        1.2050,
        1.1900,
        zone_stable_key="short-open-1",
        status=SignalStatus.TRIGGERED,
        bar_index=3,
        setup_bar_index=0,
        session="TestSession",
    )

    short_trade = engine.on_signal(
        short_signal,
        now=_utc(7, 3),
    )

    assert short_trade is not None
    assert short_trade.status == PaperTradeStatus.OPEN
    assert short_trade.direction == SignalType.SHORT

    assert short_trade.requested_entry == 1.2000

    assert abs(
        short_trade.entry - (1.2000 - _ADVERSE_AMOUNT)
    ) < 1e-9

    # SHORT adverse fill must be below requested entry.
    assert short_trade.entry < short_trade.requested_entry


    # ========================================================
    # TEST 7
    # SHORT stop detection.
    # ========================================================

    short_stop_candle = _make_candle(
        7,
        5,
        open_=1.2030,
        high=1.2060,
        low=1.2010,
        close=1.2040,
    )

    closed_now = engine.on_candle(short_stop_candle)

    assert short_trade in closed_now
    assert short_trade.status == PaperTradeStatus.CLOSED
    assert short_trade.exit_reason == ExitReason.STOP_HIT
    assert short_trade.exit_price == 1.2050

    assert short_trade not in engine.open_trades()
    assert short_trade in engine.closed_trades()


    # ========================================================
    # TEST 6
    # LONG target detection.
    # ========================================================

    long_target_signal = _build_signal(
        SignalType.LONG,
        1.1000,
        1.0950,
        1.1100,
        zone_stable_key="long-target-1",
        bar_index=4,
        setup_bar_index=0,
        session="TestSession",
    )

    long_target_trade = engine.on_signal(
        long_target_signal,
        now=_utc(7, 10),
    )

    long_target_candle = _make_candle(
        7,
        15,
        open_=1.1090,
        high=1.1110,
        low=1.1080,
        close=1.1095,
    )

    closed_now = engine.on_candle(long_target_candle)

    assert long_target_trade in closed_now
    assert long_target_trade.status == PaperTradeStatus.CLOSED
    assert long_target_trade.exit_reason == ExitReason.TARGET_HIT
    assert long_target_trade.exit_price == 1.1100

    # Hand calculation:
    #
    # actual entry = 1.1002
    # stop         = 1.0950
    # risk         = 0.0052
    #
    # reward       = 1.1100 - 1.1002
    #              = 0.0098
    #
    # R            = 0.0098 / 0.0052
    #              = 1.8846153846153846

    assert abs(
        long_target_trade.result_r - 1.8846153846153846
    ) < 1e-9


    # ========================================================
    # TEST 8
    # SHORT target detection.
    # ========================================================

    short_target_signal = _build_signal(
        SignalType.SHORT,
        1.2000,
        1.2050,
        1.1900,
        zone_stable_key="short-target-1",
        bar_index=5,
        setup_bar_index=0,
        session="TestSession",
    )

    short_target_trade = engine.on_signal(
        short_target_signal,
        now=_utc(7, 20),
    )

    short_target_candle = _make_candle(
        7,
        25,
        open_=1.1920,
        high=1.1950,
        low=1.1890,
        close=1.1900,
    )

    closed_now = engine.on_candle(short_target_candle)

    assert short_target_trade in closed_now
    assert short_target_trade.status == PaperTradeStatus.CLOSED
    assert short_target_trade.exit_reason == ExitReason.TARGET_HIT
    assert short_target_trade.exit_price == 1.1900


    # ========================================================
    # TEST 9
    # Candle touching both stop and target resolves as STOP_HIT.
    # ========================================================

    both_touch_signal = _build_signal(
        SignalType.LONG,
        1.1000,
        1.0950,
        1.1100,
        zone_stable_key="both-touch-1",
        bar_index=6,
        setup_bar_index=0,
        session="TestSession",
    )

    both_touch_trade = engine.on_signal(
        both_touch_signal,
        now=_utc(7, 30),
    )

    both_touch_candle = _make_candle(
        7,
        35,
        open_=1.1000,
        high=1.1120,
        low=1.0940,
        close=1.1000,
    )

    closed_now = engine.on_candle(both_touch_candle)

    assert both_touch_trade in closed_now
    assert both_touch_trade.status == PaperTradeStatus.CLOSED
    assert both_touch_trade.exit_reason == ExitReason.STOP_HIT


    # ========================================================
    # TEST 16
    # Multiple open trades processed by the same candle.
    #
    # Both trades share the same ~1.10 price band, so a single
    # shared candle can legitimately affect both.
    # ========================================================

    multi_a_signal = _build_signal(
        SignalType.LONG,
        1.1000,
        1.0950,
        1.1100,
        zone_stable_key="multi-a",
        bar_index=7,
        setup_bar_index=0,
        session="TestSession",
    )

    multi_b_signal = _build_signal(
        SignalType.LONG,
        1.1000,
        1.0960,
        1.1050,
        zone_stable_key="multi-b",
        bar_index=8,
        setup_bar_index=0,
        session="TestSession",
    )

    multi_a_trade = engine.on_signal(
        multi_a_signal,
        now=_utc(8, 0),
    )

    multi_b_trade = engine.on_signal(
        multi_b_signal,
        now=_utc(8, 0),
    )

    assert multi_a_trade is not None
    assert multi_b_trade is not None

    shared_candle = _make_candle(
        8,
        5,
        open_=1.1020,
        high=1.1060,
        low=1.1010,
        close=1.1040,
    )

    closed_now = engine.on_candle(shared_candle)

    # multi_b target = 1.1050, so it closes.
    assert multi_b_trade in closed_now
    assert multi_b_trade.exit_reason == ExitReason.TARGET_HIT

    # multi_a target = 1.1100, so it remains open.
    assert multi_a_trade not in closed_now
    assert multi_a_trade in engine.open_trades()


    # ========================================================
    # TEST 10 + TEST 14
    # Rejection probability creates a REJECTED trade.
    # ========================================================

    reject_audit_log = AuditLog()

    reject_engine = PaperTradeEngine(
        config=_ALWAYS_REJECT_CONFIG,
        audit_log=reject_audit_log,
        rng=random.Random(2),
    )

    reject_signal = _build_signal(
        SignalType.LONG,
        1.1000,
        1.0950,
        1.1100,
        zone_stable_key="reject-1",
        bar_index=1,
        setup_bar_index=0,
        session="TestSession",
    )

    rejected_trade = reject_engine.on_signal(
        reject_signal,
        now=_utc(9, 0),
    )

    assert rejected_trade is not None
    assert rejected_trade.status == PaperTradeStatus.REJECTED

    assert rejected_trade in reject_engine.rejected_trades()
    assert rejected_trade not in reject_engine.open_trades()
    assert rejected_trade not in reject_engine.closed_trades()

    # A rejected order was never filled.
    assert rejected_trade.entry is None


    # ========================================================
    # TEST 15
    # Audit events are generated correctly.
    # ========================================================

    opened_events = audit_log.events_of_type(
        "TRADE_OPENED"
    )

    stop_events = audit_log.events_of_type(
        "STOP_HIT"
    )

    target_events = audit_log.events_of_type(
        "TARGET_HIT"
    )

    closed_events = audit_log.events_of_type(
        "TRADE_CLOSED"
    )

    # Seven trades were opened on the main engine:
    #
    # 1. long_trade
    # 2. short_trade
    # 3. long_target_trade
    # 4. short_target_trade
    # 5. both_touch_trade
    # 6. multi_a_trade
    # 7. multi_b_trade
    #
    # Therefore:
    #   opened = 7
    #   closed = 6
    #   still open = 1

    assert len(opened_events) == 7
    assert len(closed_events) == 6

    # Three trades ended through STOP_HIT:
    #   long_trade
    #   short_trade
    #   both_touch_trade

    assert len(stop_events) == 3

    # Three trades ended through TARGET_HIT:
    #   long_target_trade
    #   short_target_trade
    #   multi_b_trade

    assert len(target_events) == 3

    # Verify specific trade/audit relationships.

    assert any(
        event.related_id == long_trade.trade_id
        for event in stop_events
    )

    assert any(
        event.related_id == short_trade.trade_id
        for event in stop_events
    )

    assert any(
        event.related_id == both_touch_trade.trade_id
        for event in stop_events
    )

    assert any(
        event.related_id == long_target_trade.trade_id
        for event in target_events
    )

    assert any(
        event.related_id == short_target_trade.trade_id
        for event in target_events
    )

    assert any(
        event.related_id == multi_b_trade.trade_id
        for event in target_events
    )


    # ========================================================
    # FINAL STATE CHECKS
    # ========================================================

    # Main engine:
    #
    # 7 opened
    # 6 closed
    # 1 still open

    assert len(engine.closed_trades()) == 6
    assert len(engine.open_trades()) == 1

    assert multi_a_trade in engine.open_trades()

    # Rejected engine:
    #
    # 1 rejected
    # 0 open
    # 0 closed

    assert len(reject_engine.rejected_trades()) == 1
    assert len(reject_engine.open_trades()) == 0
    assert len(reject_engine.closed_trades()) == 0


if __name__ == "__main__":
    run_smoke_test()
    print("All PaperTradeEngine smoke tests passed.")