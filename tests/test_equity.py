from types import SimpleNamespace

import pytest

from phase_04_live.risk.equity import equity_from_account_state


def test_returns_equity_when_connected():
    state = SimpleNamespace(connected=True, equity=10_250.5)

    assert equity_from_account_state(state) == 10_250.5


def test_disconnected_state_raises():
    state = SimpleNamespace(connected=False, equity=None)

    with pytest.raises(RuntimeError):
        equity_from_account_state(state)


def test_connected_state_without_equity_raises():
    state = SimpleNamespace(connected=True, equity=None)

    with pytest.raises(RuntimeError):
        equity_from_account_state(state)
