"""Simple PBO substitute: do in-sample ranks predict out-of-sample ranks?

Input CSV: one row per strategy/variant with an in-sample and an out-of-sample metric
(e.g. mean net R on dev and on holdout). Reports Spearman rank correlation and
the out-of-sample percentile rank of the in-sample winner.

    python tools/rank_stability.py --csv variants.csv --is-col dev_mean_r --oos-col hold_mean_r
"""
from __future__ import annotations

import argparse
import math

import numpy as np


def _ranks(a: np.ndarray) -> np.ndarray:
    order = a.argsort(kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    ranks[order] = np.arange(1, len(a) + 1)
    for v in np.unique(a):  # average ties
        m = a == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    return ranks


def rank_stability(is_vals, oos_vals) -> dict:
    a = np.asarray(is_vals, float)
    b = np.asarray(oos_vals, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    a, b = a[ok], b[ok]
    n = len(a)
    res = {"n": n, "spearman": float("nan"), "p_value": float("nan"),
           "winner_oos_percentile": float("nan"), "verdict": "undefined (need >= 3 strategies)"}
    if n < 3:
        return res
    ra, rb = _ranks(a), _ranks(b)
    if ra.std() == 0 or rb.std() == 0:
        return res
    rho = float(np.corrcoef(ra, rb)[0, 1])
    # t-approximation p-value (two-sided) via normal approx of t for moderate n
    t = rho * math.sqrt((n - 2) / max(1e-12, 1 - rho ** 2))
    p = float(2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2)))))
    win = int(np.argmax(a))
    pct = float((rb[win] - 1) / (n - 1))  # 0 = worst OOS, 1 = best
    if rho < 0.2 or pct < 0.5:
        verdict = "PROBLEM: in-sample ranking does not carry out of sample"
    elif rho < 0.5:
        verdict = "weak rank persistence"
    else:
        verdict = "rankings persist"
    res.update(spearman=rho, p_value=p, winner_oos_percentile=pct, verdict=verdict)
    return res


def main(argv=None) -> int:
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--is-col", required=True)
    ap.add_argument("--oos-col", required=True)
    a = ap.parse_args(argv)
    df = pd.read_csv(a.csv)
    r = rank_stability(df[a.is_col], df[a.oos_col])
    print(f"strategies            : {r['n']}")
    print(f"Spearman rho (IS,OOS) : {r['spearman']:.3f}  (p~{r['p_value']:.3f})")
    print(f"IS winner OOS pctile  : {r['winner_oos_percentile']:.2f}")
    print(f"verdict               : {r['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
