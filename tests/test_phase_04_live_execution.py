from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from types import SimpleNamespace

from phase_03_paper.signals.adapter import AdapterSignalEvent
from phase_04_live.events.event_types import LiveEvent, LiveEventType
from phase_04_live.execution.consumer import ExecutionEventConsumer
from phase_04_live.execution.executor import BrokerExecutor, ExecutionConfig
from phase_04_live.execution.models import ExecutionRequest, OrderExecutionStatus
from phase_04_live.recovery.safe_mode import SafeModeGate
from phase_04_live.risk.kill_switch import EmergencyKillSwitch


class _Direction(Enum):
    """Plain (non-str) enum, like a real signal type may be."""
    LONG = "LONG"
    SHORT = "SHORT"


def _make_request(**overrides) -> ExecutionRequest:
    now = datetime.now(timezone.utc)
    values = dict(
        signal_id="signal-42",
        symbol="XAUUSD",
        order_type="BUY",
        entry_price=2050.0,
        initial_risk=0.5,
        signal_generated_at=now,
        request_created_at=now,
    )
    values.update(overrides)
    return ExecutionRequest(**values)


def _make_executor(tmp_path: Path, **config_kwargs):
    kill_switch = EmergencyKillSwitch(tmp_path / "kill_switch.json")
    safe_gate = SafeModeGate(kill_switch)

    calls = []

    def order_sink(request: ExecutionRequest):
        calls.append(request)
        return f"broker-order-{len(calls)}"

    executor = BrokerExecutor(
        kill_switch=kill_switch,
        safe_mode_gate=safe_gate,
        config=ExecutionConfig(**config_kwargs),
        order_sink=order_sink,
    )
    return executor, calls


def test_broker_executor_rejects_when_kill_switch_engaged(tmp_path: Path):
    kill_switch = EmergencyKillSwitch(tmp_path / "kill_switch.json")
    kill_switch.engage("manual stop for regression test")
    safe_gate = SafeModeGate(kill_switch)

    calls = []

    def order_sink(request: ExecutionRequest):
        calls.append(request.signal_id)
        return "broker-order-1"

    executor = BrokerExecutor(
        kill_switch=kill_switch,
        safe_mode_gate=safe_gate,
        config=ExecutionConfig(max_order_size=1000.0),
        order_sink=order_sink,
    )

    result = executor.execute(_make_request())

    assert result.status == OrderExecutionStatus.REJECTED
    assert result.reason is not None and "Kill switch engaged" in result.reason
    assert calls == []


def test_broker_executor_deduplicates_same_request(tmp_path: Path):
    kill_switch = EmergencyKillSwitch(tmp_path / "kill_switch.json")
    safe_gate = SafeModeGate(kill_switch)

    calls = []

    def order_sink(request: ExecutionRequest):
        calls.append(request.signal_id)
        return f"broker-order-{len(calls)}"

    executor = BrokerExecutor(
        kill_switch=kill_switch,
        safe_mode_gate=safe_gate,
        config=ExecutionConfig(max_order_size=1000.0),
        order_sink=order_sink,
    )

    request = _make_request()
    first = executor.execute(request)
    second = executor.execute(request)

    assert first.status == OrderExecutionStatus.PLACED
    assert second.status == OrderExecutionStatus.ACKNOWLEDGED
    assert calls == ["signal-42"]


def test_execution_consumer_executes_signal_trigger_and_emits_order_event(tmp_path: Path):
    kill_switch = EmergencyKillSwitch(tmp_path / "kill_switch.json")
    safe_gate = SafeModeGate(kill_switch)

    executor = BrokerExecutor(
        kill_switch=kill_switch,
        safe_mode_gate=safe_gate,
        config=ExecutionConfig(max_order_size=1000.0, enable_dry_run=True),
        order_sink=None,
    )
    consumer = ExecutionEventConsumer(broker_executor=executor, symbol="XAUUSD")

    signal = SimpleNamespace(direction="LONG")
    event = LiveEvent(
        event_type=LiveEventType.SIGNAL_TRIGGERED,
        occurred_at=datetime.now(timezone.utc),
        payload=AdapterSignalEvent(
            signal=signal,
            setup_bar_index=10,
            trigger_bar_index=11,
            fill_price=2051.25,
            initial_risk=0.5,
        ),
    )

    processed = consumer.consume([event])

    assert len(processed) == 2
    assert processed[0].event_type == LiveEventType.SIGNAL_TRIGGERED
    assert processed[1].event_type == LiveEventType.ORDER_PLACED
    assert processed[1].payload.status == OrderExecutionStatus.PLACED


def test_executor_rejects_volume_above_limit(tmp_path: Path):
    executor, calls = _make_executor(tmp_path, max_order_size=0.1)

    result = executor.execute(_make_request(volume=0.5))

    assert result.status == OrderExecutionStatus.REJECTED
    assert result.reason is not None and "exceeds" in result.reason
    assert calls == []


def test_executor_rejects_missing_stop_when_required(tmp_path: Path):
    executor, calls = _make_executor(tmp_path, require_stop_loss=True)

    result = executor.execute(_make_request())

    assert result.status == OrderExecutionStatus.REJECTED
    assert result.reason is not None and "stop_loss" in result.reason
    assert calls == []


def test_executor_rejects_stop_on_wrong_side(tmp_path: Path):
    executor, calls = _make_executor(tmp_path)

    result = executor.execute(_make_request(stop_loss=2055.0))

    assert result.status == OrderExecutionStatus.REJECTED
    assert result.reason is not None and "stop_loss" in result.reason
    assert calls == []


def test_executor_rejects_take_profit_on_wrong_side(tmp_path: Path):
    executor, calls = _make_executor(tmp_path)

    result = executor.execute(
        _make_request(stop_loss=2049.5, take_profit=2049.0)
    )

    assert result.status == OrderExecutionStatus.REJECTED
    assert result.reason is not None and "take_profit" in result.reason
    assert calls == []


def test_executor_places_valid_protected_order(tmp_path: Path):
    executor, calls = _make_executor(
        tmp_path, max_order_size=0.1, require_stop_loss=True
    )

    result = executor.execute(
        _make_request(
            volume=0.05,
            stop_loss=2049.5,
            take_profit=2051.25,
        )
    )

    assert result.status == OrderExecutionStatus.PLACED
    assert len(calls) == 1
    assert calls[0].volume == 0.05
    assert calls[0].stop_loss == 2049.5


def test_consumer_passes_stop_and_target_and_handles_enum_direction(tmp_path: Path):
    executor, calls = _make_executor(tmp_path, max_order_size=1000.0)
    consumer = ExecutionEventConsumer(broker_executor=executor, symbol="XAUUSD")

    signal = SimpleNamespace(
        signal_type=_Direction.SHORT,
        stop_loss=2052.0,
        take_profit=2046.0,
    )
    event = LiveEvent(
        event_type=LiveEventType.SIGNAL_TRIGGERED,
        occurred_at=datetime.now(timezone.utc),
        payload=AdapterSignalEvent(
            signal=signal,
            setup_bar_index=20,
            trigger_bar_index=21,
            fill_price=2050.0,
            initial_risk=2.0,
        ),
    )

    processed = consumer.consume([event])

    assert len(processed) == 2
    assert processed[1].event_type == LiveEventType.ORDER_PLACED
    assert len(calls) == 1
    assert calls[0].order_type == "SELL"
    assert calls[0].stop_loss == 2052.0
    assert calls[0].take_profit == 2046.0
