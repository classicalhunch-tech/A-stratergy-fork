"""
tests/test_mtf_causality.py

Look-ahead check for the multi-timeframe context.

For many cut points N, build the MTF context from rows[:N+1] only,
and compare macro_trend / internal_trend at the last few timestamps
against the same timestamps in the context built from ALL rows.
If the future changes the past, the values differ (look-ahead).

Run:  python tests/test_mtf_causality.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main import load_5m_data
from strategy.mtf_structure import build_mtf_dataset_with_structure
from strategy.swings import find_swings
from dashboard.mtf_context import MTFConfig

DATA_PATH = ROOT / "data" / "master_historical_data.csv"
TOTAL_ROWS = 6000
FIRST_CUT = 1500
NUM_CUTS = 60
LOOKBACK_ROWS = 5
COLUMNS = ["macro_trend", "internal_trend"]


def _build(df):
    return build_mtf_dataset_with_structure(
        df,
        macro_swings_fn=find_swings,
        internal_swings_fn=find_swings,
        config=MTFConfig(macro_tf="1h", internal_tf="15min"),
    )


def _norm(value):
    return None if pd.isna(value) else str(value)


def run_check(total_rows=TOTAL_ROWS):
    df_all = load_5m_data(str(DATA_PATH))
    df = df_all.iloc[-total_rows:].copy()
    n = len(df)

    first_cut = min(FIRST_CUT, n // 3)
    cuts = sorted({int(x) for x in np.linspace(first_cut, n - 1, NUM_CUTS)})

    df_full = _build(df)

    comparisons = 0
    mismatches = []

    for cut in cuts:
        df_cut = df.iloc[: cut + 1]
        enriched = _build(df_cut)

        for k in range(LOOKBACK_ROWS + 1):
            pos = cut - k
            if pos < 0:
                continue
            ts = df.index[pos]

            for col in COLUMNS:
                value_cut = _norm(enriched[col].loc[ts])
                value_full = _norm(df_full[col].loc[ts])
                comparisons += 1
                if value_cut != value_full:
                    mismatches.append(
                        (str(ts), col, value_cut, value_full, cut)
                    )

    return comparisons, mismatches, len(cuts)


def test_mtf_has_no_lookahead():
    comparisons, mismatches, _ = run_check()
    assert not mismatches, (
        f"{len(mismatches)} look-ahead mismatches out of "
        f"{comparisons} comparisons. First 10: {mismatches[:10]}"
    )


if __name__ == "__main__":
    comparisons, mismatches, num_cuts = run_check()
    print("-" * 50)
    print(f"Cut points tested: {num_cuts}")
    print(f"Comparisons made:  {comparisons}")
    print(f"Mismatches:        {len(mismatches)}")
    for ts, col, value_cut, value_full, cut in mismatches[:20]:
        print(f"  cut={cut} ts={ts} {col}: cut={value_cut} full={value_full}")
    print("RESULT:", "PASS (no look-ahead found)" if not mismatches else "FAIL (look-ahead detected)")
    sys.exit(0 if not mismatches else 1)
