"""
phase_04_live/risk/daily_loss_limit.py

Phase 4 -- Daily Loss Limit
===========================

Purpose
-------
Stop all new trading for the rest of the day once the account has lost
more than a configured amount since the first equity reading of that day.

How it works
------------
- check(equity) is called with the CURRENT account equity.
- The first reading of each day is saved to disk as that day's baseline.
  The baseline survives restarts, so restarting the bot cannot reset
  the day's loss allowance.
- When (baseline - equity) reaches the limit, the EmergencyKillSwitch is
  engaged. It stays engaged until a human calls kill_switch.reset(...).
  A new day does NOT clear it automatically.

Limits
------
- max_loss_amount: absolute loss in account currency.
- max_loss_percent: percent of the day's baseline equity (0 < p <= 100).
- If both are given, the SMALLER resulting limit applies.

Fail-closed behavior
--------------------
- Invalid equity reading (NaN, inf, negative, not a number): the order
  is blocked, but the kill switch is NOT latched (a bad reading is not
  proof of a loss).
- Corrupt or unreadable state file: the kill switch is engaged and the
  order is blocked.

The "day" is a calendar day in day_timezone (default UTC). The baseline
is the FIRST equity reading seen that day, so call check() regularly
(at startup and every cycle) so the baseline is as early as possible.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone, tzinfo
from pathlib import Path
from typing import Optional

from phase_04_live.risk.kill_switch import EmergencyKillSwitch


EPSILON = 1e-9


@dataclass(frozen=True)
class DailyLossStatus:
    """
    Result of one daily-loss check.

    breached:
        True means new trading must be blocked right now.
    """

    breached: bool
    reason: Optional[str]
    day: Optional[str]
    start_equity: Optional[float]
    current_equity: Optional[float]
    loss: Optional[float]
    limit: Optional[float]


def _is_positive_finite(value) -> bool:
    if isinstance(value, bool):
        return False
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number > 0


def _is_valid_equity(value) -> bool:
    """A usable equity reading: finite and not negative."""
    if isinstance(value, bool):
        return False
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number >= 0


class DailyLossLimit:
    """
    File-backed daily loss limit tied to the emergency kill switch.

    Usage:
        limit = DailyLossLimit(
            path=Path("state/daily_loss.json"),
            kill_switch=kill_switch,
            max_loss_amount=200.0,
        )
        status = limit.check(current_equity)
        if status.breached:
            ...  # block the order
    """

    def __init__(
        self,
        path: Path,
        kill_switch: EmergencyKillSwitch,
        max_loss_amount: Optional[float] = None,
        max_loss_percent: Optional[float] = None,
        day_timezone: str = "UTC",
    ):
        if max_loss_amount is None and max_loss_percent is None:
            raise ValueError(
                "Set max_loss_amount and/or max_loss_percent."
            )

        if max_loss_amount is not None and not _is_positive_finite(
            max_loss_amount
        ):
            raise ValueError(
                "max_loss_amount must be a finite number > 0."
            )

        if max_loss_percent is not None:
            if (
                not _is_positive_finite(max_loss_percent)
                or float(max_loss_percent) > 100.0
            ):
                raise ValueError(
                    "max_loss_percent must be > 0 and <= 100."
                )

        self._path = path
        self._kill_switch = kill_switch
        self._max_loss_amount = (
            float(max_loss_amount)
            if max_loss_amount is not None
            else None
        )
        self._max_loss_percent = (
            float(max_loss_percent)
            if max_loss_percent is not None
            else None
        )

        # "UTC" avoids needing the IANA timezone database (which
        # Windows does not ship). Any other name uses zoneinfo.
        if day_timezone == "UTC":
            self._tz: tzinfo = timezone.utc
        else:
            from zoneinfo import ZoneInfo

            self._tz = ZoneInfo(day_timezone)

    # ------------------------------------------------------------------
    # State file
    # ------------------------------------------------------------------

    def _read_state(self) -> Optional[dict]:
        """
        Return the saved baseline, or None if no file exists yet.

        Raises ValueError (or a JSON/OS error) if the file exists but
        cannot be trusted.
        """

        if not self._path.exists():
            return None

        raw = json.loads(self._path.read_text(encoding="utf-8"))

        if not isinstance(raw, dict):
            raise ValueError("State must be a JSON object.")

        day = raw.get("day")
        start_equity = raw.get("start_equity")

        if not isinstance(day, str) or not day:
            raise ValueError("State 'day' must be a non-empty string.")

        if not _is_positive_finite(start_equity):
            raise ValueError(
                "State 'start_equity' must be a finite number > 0."
            )

        return {"day": day, "start_equity": float(start_equity)}

    def _write_state(self, state: dict) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(state, indent=2),
            encoding="utf-8",
        )

    # ------------------------------------------------------------------
    # Kill switch
    # ------------------------------------------------------------------

    def _engage(self, reason: str, moment: datetime) -> None:
        """
        Engage the kill switch once. If it is already engaged (for any
        reason) the existing reason and timestamp are left untouched.
        """

        if not self._kill_switch.is_engaged():
            self._kill_switch.engage(reason, now=moment)

    # ------------------------------------------------------------------
    # Limit calculation
    # ------------------------------------------------------------------

    def _limit_for(self, start_equity: float) -> float:
        limits = []

        if self._max_loss_amount is not None:
            limits.append(self._max_loss_amount)

        if self._max_loss_percent is not None:
            limits.append(
                start_equity * self._max_loss_percent / 100.0
            )

        return min(limits)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(
        self,
        equity: float,
        now: Optional[datetime] = None,
    ) -> DailyLossStatus:
        """
        Compare current equity with the day's baseline.

        Engages the kill switch and returns breached=True when the loss
        has reached the limit.
        """

        moment = (
            now or datetime.now(timezone.utc)
        ).astimezone(timezone.utc)

        day = moment.astimezone(self._tz).date().isoformat()

        if not _is_valid_equity(equity):
            return DailyLossStatus(
                breached=True,
                reason=f"Invalid equity reading: {equity!r}",
                day=day,
                start_equity=None,
                current_equity=None,
                loss=None,
                limit=None,
            )

        current = float(equity)

        try:
            state = self._read_state()

            if state is None or state["day"] != day:
                state = {"day": day, "start_equity": current}
                self._write_state(state)

        except Exception as exc:
            reason = (
                "Daily loss state file is corrupted or unusable at "
                f"{self._path}. Failing closed. Error: {exc}"
            )
            self._engage(reason, moment)
            return DailyLossStatus(
                breached=True,
                reason=reason,
                day=day,
                start_equity=None,
                current_equity=current,
                loss=None,
                limit=None,
            )

        start_equity = state["start_equity"]
        loss = start_equity - current
        limit = self._limit_for(start_equity)

        if loss >= limit - EPSILON:
            reason = (
                f"Daily loss limit reached: lost {loss:.2f} "
                f"(limit {limit:.2f}) since day start equity "
                f"{start_equity:.2f} on {day}."
            )
            self._engage(reason, moment)
            return DailyLossStatus(
                breached=True,
                reason=reason,
                day=day,
                start_equity=start_equity,
                current_equity=current,
                loss=loss,
                limit=limit,
            )

        return DailyLossStatus(
            breached=False,
            reason=None,
            day=day,
            start_equity=start_equity,
            current_equity=current,
            loss=loss,
            limit=limit,
        )
