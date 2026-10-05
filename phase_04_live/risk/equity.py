"""
phase_04_live/risk/equity.py

Account-equity readers used as the equity_provider for the daily
loss limit.

Both functions RAISE when equity cannot be read. The executor treats
any exception from its equity_provider as "do not trade", so an
unreadable account always fails closed.

This module deliberately does not import MetaTrader5 or the broker
package at import time, so it can be imported (and tested) on machines
without MT5.

Two ways to supply equity:

    read_mt5_equity()
        Reads mt5.account_info() directly. MT5 must already be
        initialized (the live runner does this in live_source.connect()).

    equity_from_account_state(state)
        Converts an AccountState snapshot, for example:

            lambda: equity_from_account_state(get_account_state(connection))
"""

from __future__ import annotations


def equity_from_account_state(state) -> float:
    """
    Return equity from an AccountState-like object.

    Raises RuntimeError when the snapshot is disconnected or has no
    equity value.
    """

    if not getattr(state, "connected", False):
        raise RuntimeError(
            "Account state is disconnected; equity unavailable."
        )

    equity = getattr(state, "equity", None)

    if equity is None:
        raise RuntimeError(
            "Account state has no equity value."
        )

    return float(equity)


def read_mt5_equity() -> float:
    """
    Read current account equity straight from MetaTrader 5.

    Equity includes floating profit/loss on open positions, so open
    losses count toward the daily limit.

    Raises RuntimeError when MT5 cannot provide account information.
    """

    import MetaTrader5 as mt5

    account = mt5.account_info()

    if account is None:
        raise RuntimeError(
            f"mt5.account_info() returned None: {mt5.last_error()}"
        )

    return float(account.equity)
