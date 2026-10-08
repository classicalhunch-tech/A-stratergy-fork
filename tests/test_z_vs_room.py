import math

import numpy as np
import pandas as pd
import pytest

from strategy.signals import SignalType
from strategy.structure_targets import StructureTargetConfig, StructureTargets
from strategy.swings import Swing, SwingType
from tools import z_vs_room as zr

Z = "Z_against_both"


def row(label, room_r, run_r, stopped, risk=2.0):
    return {
        "label": label,
        "room_r": room_r,
        "run_r": run_r,
        "stopped": stopped,
        "risk": risk,
    }


def trade(net, z, room, win=None):
    return {"net": net, "z": z, "room": room, "win": net > 0 if win is None else win}


# ---------------------------------------------------------------- room_passes

def test_room_passes_boundary_and_missing_swing():
    assert zr.room_passes(1.5, 1.5)
    assert zr.room_passes(2.0, 1.5)
    assert not zr.room_passes(1.49, 1.5)
    assert not zr.room_passes(None, 1.0)


# ------------------------------------------------------------ resolved_trades

def test_resolved_trades_net_of_cost_and_censoring():
    rows = [
        row(Z, 3.0, run_r=2.5, stopped=False),     # target hit: +2R
        row("A_counter_trend", 1.0, run_r=0.5, stopped=True),  # stop: -1R
        row("B_with_trend", 1.0, run_r=0.5, stopped=False),    # open: dropped
    ]

    out = zr.resolved_trades(rows, target=2.0, cost=0.5)

    assert len(out) == 2

    # cost 0.5 / risk 2.0 = 0.25R
    assert out[0]["net"] == pytest.approx(2.0 - 0.25)
    assert out[0]["win"] and out[0]["z"]
    assert out[1]["net"] == pytest.approx(-1.0 - 0.25)
    assert not out[1]["win"] and not out[1]["z"]


# ----------------------------------------------------------------- cell_stats

def test_cell_stats_and_empty_cell():
    n, win, mean, se = zr.cell_stats(
        [trade(1.0, True, 2.0), trade(-1.0, True, 2.0), trade(3.0, True, 2.0)]
    )

    assert n == 3
    assert win == pytest.approx(100.0 * 2 / 3)
    assert mean == pytest.approx(1.0)
    assert se == pytest.approx(math.sqrt(4.0 / 3))  # sample SD 2, n 3

    n, win, mean, se = zr.cell_stats([])
    assert n == 0 and win != win and mean != mean


# ----------------------------------------------------------------- welch_diff

def test_welch_diff_known_values():
    a = (10, 50.0, 1.0, 0.3)
    b = (12, 40.0, 0.4, 0.4)

    diff, se, t = zr.welch_diff(a, b)

    assert diff == pytest.approx(0.6)
    assert se == pytest.approx(0.5)
    assert t == pytest.approx(1.2)


def test_welch_diff_needs_two_trades_in_each_cell():
    one = (1, 100.0, 1.0, float("nan"))
    ok = (10, 50.0, 0.0, 0.2)

    for pair in ((one, ok), (ok, one), ((0, float("nan"), float("nan"), float("nan")), ok)):
        diff, se, t = zr.welch_diff(*pair)
        assert diff != diff and t != t


# ------------------------------------------------------------- pooled_effects

def test_pooled_effects_recovers_known_coefficients():
    rng = np.random.default_rng(7)
    trades = []

    for _ in range(400):
        z = bool(rng.integers(0, 2))
        passes = bool(rng.integers(0, 2))
        room = 3.0 if passes else 0.5
        net = -0.2 + 0.5 * z + 1.0 * passes + rng.normal(0, 0.3)
        trades.append(trade(net, z, room))

    out = zr.pooled_effects(trades, threshold=1.5)

    assert out["z"][0] == pytest.approx(0.5, abs=0.1)
    assert out["room"][0] == pytest.approx(1.0, abs=0.1)
    assert out["z"][1] > 0 and out["room"][1] > 0

    # same coefficients as plain least squares
    X = np.column_stack(
        [
            np.ones(len(trades)),
            [1.0 if t["z"] else 0.0 for t in trades],
            [1.0 if zr.room_passes(t["room"], 1.5) else 0.0 for t in trades],
        ]
    )
    y = np.array([t["net"] for t in trades])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]

    assert out["z"][0] == pytest.approx(beta[1])
    assert out["room"][0] == pytest.approx(beta[2])


def test_pooled_effects_not_estimable_when_a_factor_never_varies():
    only_z = [trade(float(i), True, 3.0) for i in range(10)]
    assert zr.pooled_effects(only_z, 1.5) is None

    no_pass = [trade(float(i), i % 2 == 0, 0.5) for i in range(10)]
    assert zr.pooled_effects(no_pass, 1.5) is None

    assert zr.pooled_effects([trade(1.0, True, 3.0)] * 3, 1.5) is None


# -------------------------------------------------------------------- verdict

@pytest.mark.parametrize(
    "t_z,t_rz,t_rn,expected",
    [
        (3.0, 0.5, 0.5, "Z is the edge"),
        (3.0, 0.5, 2.5, "BOTH"),
        (0.5, 2.5, 2.5, "ROOM is the edge"),
        (0.5, 0.5, 2.5, "room leans"),
        (0.5, 0.5, 0.5, "NEITHER"),
        (float("nan"), float("nan"), float("nan"), "NEITHER"),
        (-3.0, -3.0, -3.0, "NEITHER"),  # a large NEGATIVE t is not evidence for
    ],
)
def test_verdict(t_z, t_rz, t_rn, expected):
    assert expected in zr.verdict(t_z, t_rz, t_rn, crit=2.0)


# ------------------------------------------------------ significant_notes / flips

def test_significant_notes_reports_both_directions():
    pooled = {"z": (-0.4, 0.2), "room": (0.1, 0.5)}  # t -2.0 (not > 2), +0.2

    notes = zr.significant_notes(
        {"a": -3.16, "b": 1.5, "c": float("nan"), "d": 2.4}, pooled, crit=1.9
    )

    assert notes == ["a t -3.16", "d t +2.40", "pooled Z effect t -2.00"]
    assert zr.significant_notes({"a": 0.1}, None, crit=2.0) == []


def test_sign_flips_detects_each_effect_separately():
    head = {"z": (-0.4, 0.2), "room": (-0.1, 0.2)}
    tail = {"z": (0.3, 0.2), "room": (0.4, 0.2)}

    assert zr.sign_flips(head, tail) == ["Z", "room"]
    assert zr.sign_flips(tail, tail) == []
    assert zr.sign_flips({"z": (0.3, 0.2), "room": (-0.1, 0.2)}, tail) == ["room"]
    assert zr.sign_flips(None, tail) == []


# ------------------------------------------------- room uses the real swing

T0 = pd.Timestamp("2026-01-05 12:00")
H = pd.Timedelta(hours=1)


def test_tiny_min_rr_returns_nearest_swing_whatever_its_distance():
    swings = [
        Swing(SwingType.HIGH, 100.5, T0 - 2 * H - H, T0 - 2 * H),
        Swing(SwingType.HIGH, 110.0, T0 - H - H, T0 - H),
    ]

    # risk 4: nearest swing is 0.125R away
    room_fn = StructureTargets(
        swings, StructureTargetConfig(min_rr=zr.TINY_MIN_RR)
    ).target

    price = room_fn(SignalType.LONG, 100.0, 96.0, T0)

    assert price == 100.5
    assert abs(price - 100.0) / 4.0 == pytest.approx(0.125)

    # the spec's rule would have skipped it
    skip_fn = StructureTargets(swings).target
    assert skip_fn(SignalType.LONG, 100.0, 96.0, T0) is None


def test_room_function_is_causal_like_the_exit():
    swings = [Swing(SwingType.LOW, 90.0, T0, T0 + H)]  # not yet confirmed

    room_fn = StructureTargets(
        swings, StructureTargetConfig(min_rr=zr.TINY_MIN_RR)
    ).target

    assert room_fn(SignalType.SHORT, 100.0, 102.0, T0) is None
