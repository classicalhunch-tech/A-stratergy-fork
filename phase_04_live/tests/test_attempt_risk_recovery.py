import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from phase_04_live.persistence.store import Phase4Persistence
from phase_04_live.recovery import restart as restart_module


class AttemptRiskStoreTests(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "t.db"

    def tearDown(self):
        self.tmp.cleanup()

    def test_roundtrip_and_missing(self):
        p = Phase4Persistence(self.db)
        self.assertIsNone(p.get_attempt_risk("k1"))
        p.record_attempt_risk("k1", 25.0)
        self.assertEqual(p.get_attempt_risk("k1"), 25.0)

    def test_never_overwrites_existing(self):
        p = Phase4Persistence(self.db)
        p.record_attempt_risk("k1", 25.0)
        p.record_attempt_risk("k1", 99.0)
        self.assertEqual(p.get_attempt_risk("k1"), 25.0)

    def test_survives_reopen(self):
        Phase4Persistence(self.db).record_attempt_risk("k1", 12.5)
        self.assertEqual(
            Phase4Persistence(self.db).get_attempt_risk("k1"), 12.5
        )

    def test_existing_database_without_table_gets_it(self):
        conn = sqlite3.connect(self.db)
        conn.execute("CREATE TABLE unrelated (x INTEGER)")
        conn.commit()
        conn.close()

        p = Phase4Persistence(self.db)
        p.record_attempt_risk("k1", 7.0)
        self.assertEqual(p.get_attempt_risk("k1"), 7.0)


def _run_restart(attempt_risk):
    adoption = SimpleNamespace(
        expected_position=object(),
        identity_key="k1",
        ticket=555,
    )
    resync = SimpleNamespace(
        correlated_adoptions=(adoption,),
        decisions=(),
        connected=True,
    )
    persistence = MagicMock()
    persistence.get_attempt_risk.return_value = attempt_risk
    kill_switch = MagicMock()
    kill_switch.is_engaged.return_value = False

    with patch.object(
        restart_module, "resync_positions", return_value=resync
    ):
        restart_module.restart(MagicMock(), persistence, kill_switch)

    return persistence


class RestartRecordsRiskTests(unittest.TestCase):

    def test_adopted_position_gets_risk_under_position_id(self):
        persistence = _run_restart(25.0)
        persistence.get_attempt_risk.assert_called_once_with("k1")
        persistence.record_trade_risk.assert_called_once_with(555, 25.0)

    def test_risk_is_recorded_before_attempt_is_resolved(self):
        persistence = _run_restart(25.0)
        names = [c[0] for c in persistence.mock_calls]
        self.assertLess(
            names.index("record_trade_risk"),
            names.index("resolve_order_attempt"),
        )

    def test_no_recorded_risk_means_nothing_is_guessed(self):
        persistence = _run_restart(None)
        persistence.record_trade_risk.assert_not_called()
        persistence.resolve_order_attempt.assert_called_once_with("k1")


if __name__ == "__main__":
    unittest.main()
