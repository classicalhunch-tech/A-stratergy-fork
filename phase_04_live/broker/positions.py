"""
phase_04_live/broker/positions.py

Real MT5 position-state reader for the pre-trade risk gate.

Purpose
-------
phase_04_live/risk/gates.py's RiskGateInput needs three numbers that
nothing in this codebase previously computed from real broker state:
open_positions_count, open_positions_count_for_symbol, and
has_opposite_direction_open. Before this file, phase_04_live/runtime/
loop.py hardcoded these (0, 0, False) -- meaning the hedge-limit
check in evaluate_risk_gate() could never actually block a real
opposite-direction position, regardless of PositionLimitConfig.

This module closes that gap using mt5.positions_get() directly. It
does NOT place, close, or modify any order or position -- read-only,
same as BrokerConnection's own stated boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass

import MetaTrader5 as mt5

from strategy.signals import SignalType


@dataclass(frozen=True)
class PositionState:
    """
    Real, freshly-queried position counts for one prospective signal.

    open_positions_count:
        Total open positions across ALL symbols in this MT5 account.

    open_positions_count_for_symbol:
        Open positions on `symbol` specifically.

    has_opposite_direction_open:
        True if there is an open position on `symbol` whose direction
        is opposite to `signal_direction`.
    """

    open_positions_count: int
    open_positions_count_for_symbol: int
    has_opposite_direction_open: bool


def get_position_state(
    symbol: str,
    signal_direction: SignalType,
) -> PositionState:
    """
    Query MT5 for real, current open positions and compute the three
    values RiskGateInput needs.

    MT5 position type mapping:
        mt5.POSITION_TYPE_BUY  (0) -> SignalType.LONG
        mt5.POSITION_TYPE_SELL (1) -> SignalType.SHORT

    Raises:
        RuntimeError:
            If mt5.positions_get() itself fails (returns None with a
            real MT5 error, as opposed to an empty tuple which means
            "connected, zero positions" -- these are different and
            must not be treated the same way. A failed read must not
            silently be treated as "zero positions", since that could
            let a hedge-check pass when the true position state is
            actually unknown.
    """

    all_positions = mt5.positions_get()

    if all_positions is None:
        error = mt5.last_error()
        # error == (1, 'Success') is MT5's way of saying "connected,
        # legitimately zero positions" rather than an actual failure.
        if error[0] != 1:
            raise RuntimeError(
                f"mt5.positions_get() failed: {error}"
            )
        all_positions = ()

    symbol_positions = [
        p for p in all_positions if p.symbol == symbol
    ]

    opposite_type = (
        mt5.POSITION_TYPE_SELL
        if signal_direction == SignalType.LONG
        else mt5.POSITION_TYPE_BUY
    )

    has_opposite_direction_open = any(
        p.type == opposite_type for p in symbol_positions
    )

    return PositionState(
        open_positions_count=len(all_positions),
        open_positions_count_for_symbol=len(symbol_positions),
        has_opposite_direction_open=has_opposite_direction_open,
    )


__all__ = ["PositionState", "get_position_state"]