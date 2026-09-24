"""
phase_04_live/recovery/test_recovery_state_correlation.py

Tests for the correlation-aware branch of decide_position_recovery(),
added without touching its pre-existing (uncorrelated) test coverage.
"""

import unittest
from datetime import datetime, timezone

from phase_04_live.reconciliation.discrepancies import PositionDiscrepancy
from phase_04_live.recovery.correlation import CorrelationResult
from phase_04_live.recovery.recovery_state import (
    RecoveryAction,
    decide_position_recovery,
)


def _unexpected_discrepancy(ticket=555):
    return PositionDiscrepancy(
        ticket=ticket,
        outcome="UNEXPECTED_AT_BROKER",
        expected=None,
        broker=None,
        detail="no local expectation",
        as_of=datetime.now(timezone.utc),
    )


class TestDecidePositionRecoveryCorrelation(unittest.TestCase):
    def test_no_correlation_arg_preserves_original_behavior(self):
        disc = _unexpected_discrepancy()
        result = decide_position_recovery(disc)
        self.assertEqual(result.action, RecoveryAction.REQUIRE_MANUAL_REVIEW)
        self.assertFalse(result.should_track_as_expected_position)

    def test_matching_correlation_adopts_instead_of_review(self):
        disc = _unexpected_discrepancy()
        correlation = CorrelationResult(
            matched=disc, candidate_count=1, detail="matched ticket 555"
        )
        result = decide_position_recovery(disc, correlation=correlation)
        self.assertEqual(result.action, RecoveryAction.ADOPT_EXISTING_STATE)
        self.assertTrue(result.should_track_as_expected_position)
        self.assertIn("matched ticket 555", result.reason)

    def test_correlation_matching_a_different_discrepancy_does_not_adopt(self):
        disc_a = _unexpected_discrepancy(ticket=555)
        disc_b = _unexpected_discrepancy(ticket=999)
        correlation = CorrelationResult(
            matched=disc_b, candidate_count=1, detail="matched ticket 999"
        )
        result = decide_position_recovery(disc_a, correlation=correlation)
        self.assertEqual(result.action, RecoveryAction.REQUIRE_MANUAL_REVIEW)


if __name__ == "__main__":
    unittest.main(verbosity=2)
