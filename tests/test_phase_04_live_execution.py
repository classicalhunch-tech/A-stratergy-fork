from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from phase_03_paper.signals.adapter import AdapterSignalEvent
from phase_04_live.events.event_types import LiveEvent, LiveEventType
from phase_04_live.execution.consumer import ExecutionEventConsumer
from phase_04_live.execution.executor import BrokerExecutor, ExecutionConfig
from phase_04_live.execution.models import ExecutionRequest, OrderExecutionStatus
from phase_04_live.recovery.safe_mode import SafeModeGate
from phase_04_live.risk.kill_switch import EmergencyKillSwitch


def _make_request() -> ExecutionRequest:
    now = datetime.now(timezone.utc)
    return ExecutionRequest(
        signal_id="signal-42",
        symbol="XAUUSD",
        order_type="BUY",
        entry_price=2050.0,
        initial_risk=0.5,
        signal_generated_at=now,
        request_created_at=now,
    )


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
