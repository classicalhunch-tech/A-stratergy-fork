
"""
phase_04_live/risk/kill_switch.py

Phase 4 -- Emergency Kill Switch
================================

Purpose
-------
A durable, file-backed manual/automatic full-stop that can block ALL
new trading regardless of what any other Phase 4 check says.

Design goals:
    - state MUST survive a process restart/crash (file-backed, not
      just an in-memory flag)
    - engaging/resetting is explicit and auditable (required reason
      + timestamp)
    - reading state is cheap and side-effect-free (called on every
      signal via evaluate_risk_gate())
    - a corrupted/malformed/unreadable state file fails CLOSED
      (engaged=True)
    - a file that has never existed at all (first-ever run) defaults
      to not-engaged
    - invalid state data is never silently normalized

This component stores only the CURRENT kill-switch state.
Historical engage/reset events should be recorded by the Phase 4
Journal/audit layer rather than duplicated here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class KillSwitchState:
    """
    Current emergency kill-switch state.

    engaged:
        True means new trading must be blocked.

    reason:
        Reason supplied when the current state was changed.
        None for the untouched first-ever state.

    changed_at:
        ISO-8601 UTC timestamp of the most recent state change,
        or None for the untouched first-ever state.
    """

    engaged: bool
    reason: Optional[str]
    changed_at: Optional[str]


class EmergencyKillSwitch:
    """
    File-backed emergency kill switch.

    Usage:
        switch = EmergencyKillSwitch(
            Path("state/kill_switch.json")
        )

        switch.engage("manual stop -- reviewing broker fills")
        switch.is_engaged()  # True

        switch.reset("review completed -- resuming")
        switch.is_engaged()  # False

    Safety behavior:
        - Missing file on first-ever run -> not engaged.
        - Existing corrupted/malformed/unreadable file -> engaged.
        - Invalid configuration/state is never silently corrected.
    """

    def __init__(self, path: Path):
        self._path = path

    # ------------------------------------------------------------------
    # Internal state handling
    # ------------------------------------------------------------------

    def _fail_closed(self, reason: str) -> KillSwitchState:
        """
        Return a safe engaged state when persisted state cannot be trusted.
        """

        return KillSwitchState(
            engaged=True,
            reason=reason,
            changed_at=None,
        )

    def _read(self) -> KillSwitchState:
        """
        Read the current persisted state.

        Missing file is treated as the intentional first-ever state.

        Any existing file that cannot be trusted fails closed.
        """

        if not self._path.exists():
            return KillSwitchState(
                engaged=False,
                reason=None,
                changed_at=None,
            )

        try:
            raw = json.loads(
                self._path.read_text(encoding="utf-8")
            )

            if not isinstance(raw, dict):
                raise ValueError(
                    "Kill switch state must be a JSON object."
                )

            engaged = raw["engaged"]

            if not isinstance(engaged, bool):
                raise ValueError(
                    "Kill switch 'engaged' must be a boolean."
                )

            reason = raw.get("reason")
            changed_at = raw.get("changed_at")

            if reason is not None and not isinstance(reason, str):
                raise ValueError(
                    "Kill switch 'reason' must be a string or null."
                )

            if changed_at is not None and not isinstance(changed_at, str):
                raise ValueError(
                    "Kill switch 'changed_at' must be a string or null."
                )

            return KillSwitchState(
                engaged=engaged,
                reason=reason,
                changed_at=changed_at,
            )

        except Exception as exc:
            return self._fail_closed(
                "Kill switch state file is corrupted, malformed, "
                f"or unreadable at {self._path}. "
                f"Failing closed (engaged). Error: {exc}"
            )

    def _write(self, state: KillSwitchState) -> None:
        """
        Persist the current state.

        The parent directory is created when necessary.
        """

        self._path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._path.write_text(
            json.dumps(
                {
                    "engaged": state.engaged,
                    "reason": state.reason,
                    "changed_at": state.changed_at,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    # ------------------------------------------------------------------
    # Public state access
    # ------------------------------------------------------------------

    def is_engaged(self) -> bool:
        """
        Return whether the kill switch is currently engaged.

        This performs a fresh state read so that changes made by another
        process are visible without requiring an object restart.
        """

        return self._read().engaged

    def current_state(self) -> KillSwitchState:
        """
        Return the complete current persisted state.
        """

        return self._read()

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------

    def engage(
        self,
        reason: str,
        now: Optional[datetime] = None,
    ) -> None:
        """
        Engage the emergency kill switch.

        A non-empty, non-whitespace reason is required.

        The timestamp is stored as ISO-8601 UTC.
        """

        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(
                "engage() requires a non-empty reason."
            )

        timestamp = (
            now or datetime.now(timezone.utc)
        ).astimezone(timezone.utc)

        self._write(
            KillSwitchState(
                engaged=True,
                reason=reason.strip(),
                changed_at=timestamp.isoformat(),
            )
        )

    def reset(
        self,
        reason: str,
        now: Optional[datetime] = None,
    ) -> None:
        """
        Reset/disengage the emergency kill switch.

        A non-empty, non-whitespace reason is required.

        The timestamp is stored as ISO-8601 UTC.
        """

        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(
                "reset() requires a non-empty reason."
            )

        timestamp = (
            now or datetime.now(timezone.utc)
        ).astimezone(timezone.utc)

        self._write(
            KillSwitchState(
                engaged=False,
                reason=reason.strip(),
                changed_at=timestamp.isoformat(),
            )
        )

