"""
phase_04_live/recovery/safe_mode.py

Safe-mode gate for Phase 4 live trading.

Wraps the existing EmergencyKillSwitch to answer ONE question the
live loop needs before acting on a newly triggered signal: is new
trading currently allowed?

This does NOT:
    - engage or reset the kill switch itself (that's restart.py /
      manual operator action / risk gate's job)
    - decide what happens to a blocked signal (the caller -- e.g.
      LiveEventEngine -- decides: suppress it, log it, alert on it)
    - affect whether StrategyAdapter keeps receiving candles. Per
      Phase 3's own design (StrategyAdapter receives EVERY candle so
      structure stays continuous), safe mode only gates whether a
      TRIGGERED signal is acted upon, never whether the adapter
      keeps processing market data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from phase_04_live.risk.kill_switch import EmergencyKillSwitch, KillSwitchState


@dataclass(frozen=True)
class SafeModeStatus:
    """
    Snapshot of whether new trading is currently allowed.

    allowed:
        True means new signals may be acted upon (sent toward
        idempotency/order placement). False means they must be
        suppressed.

    reason:
        The kill switch's own reason for its current state, when
        engaged. None when allowed=True.
    """
    allowed: bool
    reason: Optional[str]


class SafeModeGate:
    """
    Thin, side-effect-free check of the existing EmergencyKillSwitch,
    phrased for the live loop's actual question: "is it safe to act
    on a new triggered signal right now?"

    Usage:
        gate = SafeModeGate(kill_switch)
        status = gate.check()
        if not status.allowed:
            # suppress the signal, do not call guard_and_place_order()
            ...
    """

    def __init__(self, kill_switch: EmergencyKillSwitch):
        self._kill_switch = kill_switch

    def check(self) -> SafeModeStatus:
        """
        Fresh read of the kill switch's current state (not cached --
        another process/operator may have changed it since the last
        check).
        """
        state: KillSwitchState = self._kill_switch.current_state()

        if state.engaged:
            return SafeModeStatus(allowed=False, reason=state.reason)

        return SafeModeStatus(allowed=True, reason=None)

    def is_trading_allowed(self) -> bool:
        """Convenience boolean form of check()."""
        return not self._kill_switch.is_engaged()


__all__ = ["SafeModeStatus", "SafeModeGate"]
