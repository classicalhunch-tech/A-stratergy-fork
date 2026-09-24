"""
phase_02_optimization/test_zone_equivalence.py

Proves that compute_zone_for_break() -- called once per structure
break, in causal order -- produces EXACTLY the same zones as
find_zones(df, breaks) called once on the complete break list.

find_zones() remains the untouched correctness reference. This test
must pass at every dataset size before compute_zone_for_break() is
used anywhere outside this file (i.e. before any adapter change).

Run:
    python -m phase_02_optimization.test_zone_equivalence
"""

import random
import pandas as pd

from strategy.swings import find_swings
from strategy.structure import analyze_structure
from strategy.zones import find_zones, compute_zone_for_break


def _zones_equal(z1, z2) -> bool:
    return (
        z1.zone_type == z2.zone_type
        and z1.price_top == z2.price_top
        and z1.price_bottom == z2.price_bottom
        and z1.origin_start == z2.origin_start
        and z1.origin_end == z2.origin_end
        and z1.formed_from_break_at == z2.formed_from_break_at
    )


def _run_check(df: pd.DataFrame, label: str) -> bool:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    all_swings = find_swings(df)
    all_breaks, _, _ = analyze_structure(df, all_swings)

    # --- Reference: batch find_zones() ---
    reference_zones = find_zones(df, all_breaks)

    # --- Incremental: compute_zone_for_break() once per break ---
    index_pos = {ts: pos for pos, ts in enumerate(df.index)}

    incremental_zones = []
    for brk in all_breaks:
        break_idx = index_pos.get(brk.broken_at)
        if break_idx is None:
            continue
        zone = compute_zone_for_break(df, brk, break_idx)
        if zone is not None:
            incremental_zones.append(zone)

    if len(reference_zones) != len(incremental_zones):
        print(
            f"[FAIL] {label}: zone count differs -- "
            f"reference={len(reference_zones)} "
            f"incremental={len(incremental_zones)}"
        )
        return False

    for i, (ref, inc) in enumerate(zip(reference_zones, incremental_zones)):
        if not _zones_equal(ref, inc):
            print(f"[FAIL] {label}: zone {i} differs -- ref={ref} inc={inc}")
            return False

    print(f"[PASS] {label}: {len(reference_zones)} zones, identical.")
    return True


def _load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp")
    return df


def _make_synthetic(n: int, seed: int) -> pd.DataFrame:
    random.seed(seed)
    rows = []
    price = 1.1000
    ts = pd.Timestamp("2026-01-01", tz="UTC")
    for _ in range(n):
        o = price
        move = random.uniform(-0.002, 0.002)
        c = o + move
        h = max(o, c) + random.uniform(0, 0.0015)
        l = min(o, c) - random.uniform(0, 0.0015)
        rows.append(
            {"timestamp": ts, "open": o, "high": h, "low": l, "close": c}
        )
        price = c
        ts = ts + pd.Timedelta(minutes=5)
    df = pd.DataFrame(rows).set_index("timestamp")
    return df


def main():
    all_passed = True

    for size, path in [
        (500, "your_data_file_500.csv"),
        (1000, "your_data_file_1000.csv"),
        (2000, "your_data_file_2000.csv"),
        (5000, "your_data_file.csv"),
    ]:
        df = _load_csv(path)
        ok = _run_check(df, f"real data, {size} candles ({path})")
        all_passed = all_passed and ok

    for seed in range(4):
        df = _make_synthetic(1000, seed=seed)
        ok = _run_check(df, f"synthetic seed={seed}, 1000 candles")
        all_passed = all_passed and ok

    print()
    if all_passed:
        print("ALL ZONE EQUIVALENCE TESTS PASSED.")
    else:
        print("SOME ZONE EQUIVALENCE TESTS FAILED -- do not integrate.")


if __name__ == "__main__":
    main()