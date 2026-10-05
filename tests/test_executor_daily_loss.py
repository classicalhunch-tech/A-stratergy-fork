from datetime import datetime, timezone

from phase_04_live.execution.executor import (
    BrokerExecutor,
    ExecutionConfig,
)
from phase_04_live.execution.models import (
    ExecutionRequest,
    OrderExecutionStatus,
)
from phase_04_live.recovery.safe_mode import SafeModeGate
from phase_04_live.risk.daily_loss_limit import DailyLossLimit
from phase_04_live.risk.kill_switch import EmergencyKillSwitch


def _request(signal_id: str) -> ExecutionRequest:
    now = datetime.now(timezone.utc)

    return ExecutionRequest(
        signal_id=signal_id,
        symbol="XAUUSD",
        order_type="BUY",
        entry_price=2000.0,
        initial_risk=10.0,
        signal_generated_at=now,
        request_created_at=now,
        stop_loss=1990.0,
        take_profit=2020.0,
        volume=0.01,
    )


def _build(
    tmp_path,
    equity_provider=None,
    with_limit=True,
):
    kill_switch = EmergencyKillSwitch(tmp_path / "kill_switch.json")
    gate = SafeModeGate(kill_switch)

    limit = (
        DailyLossLimit(
            path=tmp_path / "daily_loss.json",
            kill_switch=kill_switch,
            max_loss_amount=100.0,
        )
        if with_limit
        else None
    )

    executor = BrokerExecutor(
        kill_switch=kill_switch,
        safe_mode_gate=gate,
        config=ExecutionConfig(
            enable_dry_run=True,
            require_stop_loss=True,
        ),
        daily_loss_limit=limit,
        equity_provider=equity_provider,
    )

    return executor, kill_switch


def test_without_limit_orders_behave_as_before(tmp_path):
    executor, _ = _build(tmp_path, with_limit=False)

    result = executor.execute(_request("sig-1"))

    assert result.status == OrderExecutionStatus.PLACED
    assert result.broker_order_id.startswith("DRY-")


def test_order_allowed_when_loss_is_within_limit(tmp_path):
    equity = {"value": 10_000.0}
    executor, kill_switch = _build(
        tmp_path,
        equity_provider=lambda: equity["value"],
    )

    first = executor.execute(_request("sig-1"))
    equity["value"] = 9_950.0
    second = executor.execute(_request("sig-2"))

    assert first.status == OrderExecutionStatus.PLACED
    assert second.status == OrderExecutionStatus.PLACED
    assert kill_switch.is_engaged() is False


def test_breach_rejects_order_and_engages_kill_switch(tmp_path):
    equity = {"value": 10_000.0}
    executor, kill_switch = _build(
        tmp_path,
        equity_provider=lambda: equity["value"],
    )

    executor.execute(_request("sig-1"))
    equity["value"] = 9_800.0
    result = executor.execute(_request("sig-2"))

    assert result.status == OrderExecutionStatus.REJECTED
    assert "Daily loss limit" in result.reason
    assert kill_switch.is_engaged() is True


def test_orders_stay_blocked_after_breach_even_if_equity_recovers(tmp_path):
    equity = {"value": 10_000.0}
    executor, _ = _build(
        tmp_path,
        equity_provider=lambda: equity["value"],
    )

    executor.execute(_request("sig-1"))
    equity["value"] = 9_800.0
    executor.execute(_request("sig-2"))

    equity["value"] = 10_000.0
    result = executor.execute(_request("sig-3"))

    assert result.status == OrderExecutionStatus.REJECTED
    assert "Kill switch engaged" in result.reason


def test_limit_without_equity_provider_rejects(tmp_path):
    executor, _ = _build(tmp_path, equity_provider=None)

    result = executor.execute(_request("sig-1"))

    assert result.status == OrderExecutionStatus.REJECTED
    assert "equity_provider" in result.reason


def test_equity_read_failure_rejects(tmp_path):
    def broken_provider():
        raise RuntimeError("MT5 down")

    executor, _ = _build(tmp_path, equity_provider=broken_provider)

    result = executor.execute(_request("sig-1"))

    assert result.status == OrderExecutionStatus.REJECTED
    assert "Could not read account equity" in result.reason
