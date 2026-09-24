"""
phase_04_live/risk/test_kill_switch.py

Tests for the Phase 4 emergency kill switch (kill_switch.py).
"""

import json
import unittest
from datetime import datetime, timezone
from pathlib import Path
import tempfile

from phase_04_live.risk.kill_switch import EmergencyKillSwitch


class TestEmergencyKillSwitch(unittest.TestCase):
    def setUp(self):
        """Create a temporary directory for isolated state files per test."""
        self._temp_dir = tempfile.TemporaryDirectory()
        self._state_path = Path(self._temp_dir.name) / "kill_switch.json"
        self._switch = EmergencyKillSwitch(self._state_path)

    def tearDown(self):
        self._temp_dir.cleanup()

    def test_missing_file_defaults_to_not_engaged(self):
        """A file that has never existed defaults to not-engaged (False)."""
        self.assertFalse(self._switch.is_engaged())
        state = self._switch.current_state()
        self.assertFalse(state.engaged)
        self.assertIsNone(state.reason)
        self.assertIsNone(state.changed_at)

    def test_engage_persists_state_and_requires_reason(self):
        """Engaging the switch updates state, records timestamp, and writes to disk."""
        with self.assertRaises(ValueError):
            self._switch.engage("")  # Empty reason must raise ValueError

        with self.assertRaises(ValueError):
            self._switch.engage("   ")  # Whitespace-only reason must raise ValueError

        test_time = datetime(2026, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        self._switch.engage("Manual broker maintenance check", now=test_time)

        self.assertTrue(self._switch.is_engaged())
        state = self._switch.current_state()
        self.assertTrue(state.engaged)
        self.assertEqual(state.reason, "Manual broker maintenance check")
        self.assertEqual(state.changed_at, "2026-06-15T12:00:00+00:00")

        # Verify it successfully persisted to disk for process survival
        self.assertTrue(self._state_path.exists())
        raw_data = json.loads(self._state_path.read_text(encoding="utf-8"))
        self.assertTrue(raw_data["engaged"])
        self.assertEqual(raw_data["reason"], "Manual broker maintenance check")

    def test_reset_disengages_state(self):
        """Resetting the switch clears the engaged flag and logs the reset reason."""
        self._switch.engage("Initial emergency stop")
        self.assertTrue(self._switch.is_engaged())

        test_time = datetime(2026, 6, 15, 13, 30, 0, tzinfo=timezone.utc)
        self._switch.reset("Resuming normal trading after review", now=test_time)

        self.assertFalse(self._switch.is_engaged())
        state = self._switch.current_state()
        self.assertFalse(state.engaged)
        self.assertEqual(state.reason, "Resuming normal trading after review")
        self.assertEqual(state.changed_at, "2026-06-15T13:30:00+00:00")

    def test_corrupted_state_file_fails_closed(self):
        """A malformed or corrupted JSON file must fail closed (engaged=True)."""
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        self._state_path.write_text("NOT VALID JSON {{{", encoding="utf-8")

        self.assertTrue(self._switch.is_engaged())
        state = self._switch.current_state()
        self.assertTrue(state.engaged)
        self.assertIn("corrupted", state.reason.lower())

    def test_invalid_json_types_fail_closed(self):
        """A JSON file with invalid internal types must fail closed."""
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        # 'engaged' should be a boolean, not a string
        self._state_path.write_text(
            json.dumps({"engaged": "yes_please", "reason": 123, "changed_at": []}),
            encoding="utf-8",
        )

        self.assertTrue(self._switch.is_engaged())
        state = self._switch.current_state()
        self.assertTrue(state.engaged)


if __name__ == "__main__":
    unittest.main(verbosity=2)