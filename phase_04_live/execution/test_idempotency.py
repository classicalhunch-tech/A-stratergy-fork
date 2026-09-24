"""
phase_04_live/execution/test_idempotency.py

Tests for the idempotency guard around place_order_for_signal().

place_order_for_signal() itself is mocked/patched everywhere -- these
tests never touch MT5 or construct a real OrderManagerConfig/specs,
since the guard's own logic (identity building, block/allow decision,
record-before-call ordering) is what's under test, not order
construction itself (already covered by order_manager's own tests).
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from phase_03_paper.signals.adapter import AdapterSignalEvent
from phase_04_live.execution.idempotency import (
    InMemoryIdempotencyStore,
    build_execution_identity,
    guard_and_place_order,
)
from phase_04_live.orders.order_manager import OrderResult
from strategy.signals import SignalStatus, SignalType, TradeSignal


def _make_signal(
    *,
    signal_type=SignalType.LONG,
    setup_timestamp="2026-09-18 12:00:00",
    zone_stable_key="zone-abc",
    triggered_at_time="2026-09-18 12:10:00",
    entry=4100.0,
    stop=4090.0,
    target=4120.0,
):
    return TradeSignal(
        signal_type=signal_type,
        entry_price=entry,
        stop_loss=stop,
        take_profit=target,
        status=SignalStatus.TRIGGERED,
        setup_timestamp=pd.Timestamp(setup_timestamp) if setup_timestamp else None,
        zone_stable_key=zone_stable_key,
        triggered_at_time=pd.Timestamp(triggered_at_time) if triggered_at_time else None,
    )


def _make_event(signal=None, *, trigger_bar_index=10, fill_price=4100.0):
    if signal is None:
        signal = _make_signal()

    return AdapterSignalEvent(
        signal=signal,
        setup_bar_index=5,
        trigger_bar_index=trigger_bar_index,
        fill_price=fill_price,
        initial_risk=10.0,
    )


def _fake_config(dry_run=True):
    config = MagicMock()
    config.dry_run = dry_run
    return config


def _fake_order_result(ticket=999):
    return OrderResult(
        accepted=True,
        dry_run=True,
        request=MagicMock(),
        retcode=None,
        ticket=ticket,
        reason=None,
    )


# ============================================================
# IDENTITY CONSTRUCTION
# ============================================================

def test_identity_reflects_signal_fields():
    signal = _make_signal(
        signal_type=SignalType.SHORT,
        setup_timestamp="2026-09-18 09:00:00",
        zone_stable_key="zone-xyz",
        triggered_at_time="2026-09-18 09:15:00",
    )
    event = _make_event(signal)

    identity = build_execution_identity(event)

    assert identity[0] == "SHORT"
    assert identity[1] == pd.Timestamp("2026-09-18 09:00:00")
    assert identity[2] == "zone-xyz"
    assert identity[3] == pd.Timestamp("2026-09-18 09:15:00")


def test_identity_ignores_adapter_lifetime_counters():
    """
    trigger_bar_index and fill_price are adapter-process-relative and
    must NOT affect identity -- two events with identical real-market
    signal fields but different bar indices/fill prices (as would
    happen across a restart replaying the same history) must produce
    the SAME identity.
    """
    signal_a = _make_signal()
    signal_b = _make_signal()  # same real-market fields

    event_a = _make_event(signal_a, trigger_bar_index=10, fill_price=4100.0)
    event_b = _make_event(signal_b, trigger_bar_index=99999, fill_price=4250.5)

    assert build_execution_identity(event_a) == build_execution_identity(event_b)


def test_different_setup_timestamp_produces_different_identity():
    event_a = _make_event(_make_signal(setup_timestamp="2026-09-18 12:00:00"))
    event_b = _make_event(_make_signal(setup_timestamp="2026-09-18 13:00:00"))

    assert build_execution_identity(event_a) != build_execution_identity(event_b)


# ============================================================
# DUPLICATE BLOCKING
# ============================================================

def test_duplicate_identity_blocks_second_call_without_broker_contact():
    store = InMemoryIdempotencyStore()
    event = _make_event()
    config = _fake_config()

    with patch(
        "phase_04_live.execution.idempotency.place_order_for_signal",
        return_value=_fake_order_result(ticket=111),
    ) as mock_place:

        first = guard_and_place_order(event, MagicMock(), MagicMock(), config, store)
        second = guard_and_place_order(event, MagicMock(), MagicMock(), config, store)

    assert mock_place.call_count == 1  # NOT called a second time
    assert first.ticket == 111
    assert second.accepted is False
    assert second.ticket is None
    assert second.request is None
    assert "duplicate" in second.reason.lower()


def test_distinct_identities_both_reach_place_order_for_signal():
    store = InMemoryIdempotencyStore()
    config = _fake_config()

    event_a = _make_event(_make_signal(setup_timestamp="2026-09-18 12:00:00"))
    event_b = _make_event(_make_signal(setup_timestamp="2026-09-18 13:00:00"))

    with patch(
        "phase_04_live.execution.idempotency.place_order_for_signal",
        side_effect=[_fake_order_result(ticket=1), _fake_order_result(ticket=2)],
    ) as mock_place:

        result_a = guard_and_place_order(event_a, MagicMock(), MagicMock(), config, store)
        result_b = guard_and_place_order(event_b, MagicMock(), MagicMock(), config, store)

    assert mock_place.call_count == 2
    assert result_a.ticket == 1
    assert result_b.ticket == 2


def test_duplicate_rejection_uses_configured_dry_run_value():
    store = InMemoryIdempotencyStore()
    event = _make_event()
    config = _fake_config(dry_run=False)

    with patch(
        "phase_04_live.execution.idempotency.place_order_for_signal",
        return_value=_fake_order_result(),
    ):
        guard_and_place_order(event, MagicMock(), MagicMock(), config, store)

    second = guard_and_place_order(event, MagicMock(), MagicMock(), config, store)

    assert second.dry_run is False


# ============================================================
# RECORD-BEFORE-CALL SAFETY
# ============================================================

def test_identity_recorded_even_if_place_order_raises():
    """
    The identity must be recorded BEFORE place_order_for_signal() is
    called, so a caller that catches an exception and retries the
    exact same event does not resubmit.
    """
    store = InMemoryIdempotencyStore()
    event = _make_event()
    config = _fake_config()

    identity = build_execution_identity(event)
    assert store.has(identity) is False

    with patch(
        "phase_04_live.execution.idempotency.place_order_for_signal",
        side_effect=RuntimeError("simulated failure"),
    ):
        with pytest.raises(RuntimeError):
            guard_and_place_order(event, MagicMock(), MagicMock(), config, store)

    assert store.has(identity) is True


# ============================================================
# IN-MEMORY STORE BEHAVIOR
# ============================================================

def test_in_memory_store_has_and_record():
    store = InMemoryIdempotencyStore()
    key = ("LONG", pd.Timestamp("2026-09-18 12:00:00"), "zone-1", pd.Timestamp("2026-09-18 12:10:00"))

    assert store.has(key) is False
    store.record(key)
    assert store.has(key) is True


def test_in_memory_store_is_fresh_per_instance():
    store_a = InMemoryIdempotencyStore()
    store_b = InMemoryIdempotencyStore()

    key = ("LONG", pd.Timestamp("2026-09-18 12:00:00"), "zone-1", pd.Timestamp("2026-09-18 12:10:00"))
    store_a.record(key)

    assert store_a.has(key) is True
    assert store_b.has(key) is False