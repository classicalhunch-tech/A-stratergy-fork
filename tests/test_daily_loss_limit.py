from datetime import datetime, timedelta, timezone

import pytest

from phase_04_live.risk.daily_loss_limit import DailyLossLimit
from phase_04_live.risk.kill_switch import EmergencyKillSwitch


NOW = datetime(2026, 10, 5, 8, 0, tzinfo=timezone.utc)


def _make(tmp_path, **kwargs):
    kill_switch = EmergencyKillSwitch(tmp_path / "kill_switch.json")
    limit = DailyLossLimit(
        path=tmp_path / "daily_loss.json",
        kill_switch=kill_switch,
        **kwargs,
    )
    return limit, kill_switch


def test_first_check_sets_baseline_and_is_not_breached(tmp_path):
    limit, kill_switch = _make(tmp_path, max_loss_amount=100.0)

    status = limit.check(10_000.0, now=NOW)

    assert status.breached is False
    assert status.start_equity == 10_000.0
    assert status.loss == 0.0
    assert kill_switch.is_engaged() is False


def test_small_loss_is_allowed(tmp_path):
    limit, kill_switch = _make(tmp_path, max_loss_amount=100.0)

    limit.check(10_000.0, now=NOW)
    status = limit.check(9_950.0, now=NOW + timedelta(hours=1))

    assert status.breached is False
    assert status.loss == 50.0
    assert kill_switch.is_engaged() is False


def test_loss_at_limit_breaches_and_engages_kill_switch(tmp_path):
    limit, kill_switch = _make(tmp_path, max_loss_amount=100.0)

    limit.check(10_000.0, now=NOW)
    status = limit.check(9_900.0, now=NOW + timedelta(hours=1))

    assert status.breached is True
    assert "Daily loss limit reached" in status.reason
    assert kill_switch.is_engaged() is True


def test_kill_switch_stays_engaged_after_equity_recovers(tmp_path):
    limit, kill_switch = _make(tmp_path, max_loss_amount=100.0)

    limit.check(10_000.0, now=NOW)
    limit.check(9_800.0, now=NOW + timedelta(hours=1))
    status = limit.check(10_000.0, now=NOW + timedelta(hours=2))

    assert status.breached is False
    assert kill_switch.is_engaged() is True


def test_percent_limit(tmp_path):
    limit, kill_switch = _make(tmp_path, max_loss_percent=1.0)

    limit.check(10_000.0, now=NOW)

    ok = limit.check(9_900.01, now=NOW + timedelta(hours=1))
    assert ok.breached is False

    hit = limit.check(9_900.0, now=NOW + timedelta(hours=2))
    assert hit.breached is True
    assert kill_switch.is_engaged() is True


def test_smaller_of_amount_and_percent_applies(tmp_path):
    limit, _ = _make(
        tmp_path,
        max_loss_amount=500.0,
        max_loss_percent=1.0,
    )

    status = limit.check(10_000.0, now=NOW)

    assert status.limit == 100.0


def test_baseline_survives_restart(tmp_path):
    first, _ = _make(tmp_path, max_loss_amount=100.0)
    first.check(10_000.0, now=NOW)

    # New objects, same files: simulates a restart mid-day.
    second, kill_switch = _make(tmp_path, max_loss_amount=100.0)
    status = second.check(9_850.0, now=NOW + timedelta(hours=3))

    assert status.start_equity == 10_000.0
    assert status.breached is True
    assert kill_switch.is_engaged() is True


def test_new_day_gets_a_new_baseline(tmp_path):
    limit, kill_switch = _make(tmp_path, max_loss_amount=100.0)

    limit.check(10_000.0, now=NOW)
    status = limit.check(9_500.0, now=NOW + timedelta(days=1))

    assert status.start_equity == 9_500.0
    assert status.breached is False
    assert kill_switch.is_engaged() is False


def test_corrupt_state_fails_closed(tmp_path):
    limit, kill_switch = _make(tmp_path, max_loss_amount=100.0)
    (tmp_path / "daily_loss.json").write_text("not json at all")

    status = limit.check(10_000.0, now=NOW)

    assert status.breached is True
    assert "Failing closed" in status.reason
    assert kill_switch.is_engaged() is True


@pytest.mark.parametrize(
    "bad", [float("nan"), float("inf"), -5.0, None, "abc"]
)
def test_invalid_equity_blocks_without_latching(tmp_path, bad):
    limit, kill_switch = _make(tmp_path, max_loss_amount=100.0)

    status = limit.check(bad, now=NOW)

    assert status.breached is True
    assert "Invalid equity" in status.reason
    assert kill_switch.is_engaged() is False


def test_requires_at_least_one_limit(tmp_path):
    kill_switch = EmergencyKillSwitch(tmp_path / "kill_switch.json")

    with pytest.raises(ValueError):
        DailyLossLimit(
            path=tmp_path / "daily_loss.json",
            kill_switch=kill_switch,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_loss_amount": 0},
        {"max_loss_amount": -1},
        {"max_loss_amount": float("nan")},
        {"max_loss_percent": 0},
        {"max_loss_percent": 101},
    ],
)
def test_invalid_limit_values_are_rejected(tmp_path, kwargs):
    kill_switch = EmergencyKillSwitch(tmp_path / "kill_switch.json")

    with pytest.raises(ValueError):
        DailyLossLimit(
            path=tmp_path / "daily_loss.json",
            kill_switch=kill_switch,
            **kwargs,
        )
