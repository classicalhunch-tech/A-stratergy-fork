
"""
phase_04_live/clock/synchronizer.py

Compares the local PC clock against MT5 server time.

MT5''s Python API has no direct "get server time" call. The
established way to get it is the timestamp on the most recent tick
(symbol_info_tick().time), which is what this module uses as the
broker/server time reference.

IMPORTANT: MT5''s tick.time, when converted via
datetime.fromtimestamp(tz=timezone.utc), does NOT give true UTC --
it gives the broker server''s LOCAL wall-clock time, mislabeled as
UTC. This broker runs on Africa/Nairobi (EAT, UTC+3), confirmed via
direct measurement (~3h offset observed) and consistent with
phase_03_paper''s SessionEngine, which already uses Africa/Nairobi as
its reference timezone. So this module re-labels the wall-clock
numbers as Africa/Nairobi, then converts to true UTC, before
computing drift. Skipping this step would show a permanent false
~3-hour "drift" on every single check, regardless of whether the
local PC clock is actually correct.
"""

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import MetaTrader5 as mt5

BROKER_TIMEZONE = ZoneInfo("Africa/Nairobi")


def get_broker_server_time(symbol: str) -> datetime:
    """
    Returns the server-side timestamp of the most recent tick for
    `symbol`, correctly converted to true UTC.

    Raises RuntimeError if no tick is available (e.g. symbol not
    selected, market closed with no cached tick, or MT5 not connected).
    """
    tick = mt5.symbol_info_tick(symbol)

    if tick is None:
        raise RuntimeError(
            f"symbol_info_tick() returned None for '{symbol}': "
            f"{mt5.last_error()}"
        )

    # tick.time''s numbers are the broker''s Africa/Nairobi wall-clock
    # time, incorrectly labeled UTC by fromtimestamp(). Strip that
    # incorrect label, re-label as Africa/Nairobi, then convert to
    # true UTC.
    mislabeled = datetime.fromtimestamp(tick.time, tz=timezone.utc)
    naive = mislabeled.replace(tzinfo=None)
    nairobi_time = naive.replace(tzinfo=BROKER_TIMEZONE)

    return nairobi_time.astimezone(timezone.utc)


def compute_drift_seconds(symbol: str) -> float:
    """
    Returns local_now - server_time (both true UTC), in seconds.

    Positive: local clock is ahead of the server.
    Negative: local clock is behind the server.
    """
    server_time = get_broker_server_time(symbol)
    local_now = datetime.now(timezone.utc)

    return (local_now - server_time).total_seconds()
