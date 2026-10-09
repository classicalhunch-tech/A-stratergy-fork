"""Backtest Inflation Factor: BIF = dev Z / holdout Z.

Z is whatever headline statistic you pre-registered (coefficient or t-stat) --
use the same quantity for dev and holdout. Import `bif` and print it at the end
of the holdout run in session_vs_outside.py.
"""
from __future__ import annotations

import argparse
import math


def bif(dev_z: float, holdout_z: float, high: float = 2.0) -> dict:
    out = {"dev_z": dev_z, "holdout_z": holdout_z, "bif": float("nan"), "verdict": ""}
    if any(map(lambda v: v is None or math.isnan(v), (dev_z, holdout_z))):
        out["verdict"] = "undefined (missing input)"
    elif holdout_z == 0:
        out.update(bif=math.inf if dev_z != 0 else float("nan"),
                   verdict="holdout Z is zero: dev result did not replicate")
    else:
        v = dev_z / holdout_z
        out["bif"] = v
        if holdout_z < 0 < dev_z or dev_z < 0 < holdout_z:
            out["verdict"] = "SIGN FLIP: dev result did not replicate (BIF negative)"
        elif v >= high:
            out["verdict"] = f"BIF >= {high}: dev result likely selection-inflated"
        elif v > 1.25:
            out["verdict"] = "some shrinkage, normal after selection"
        else:
            out["verdict"] = "holdout matches dev"
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-z", type=float, required=True)
    ap.add_argument("--holdout-z", type=float, required=True)
    ap.add_argument("--high", type=float, default=2.0)
    a = ap.parse_args(argv)
    r = bif(a.dev_z, a.holdout_z, a.high)
    print(f"BIF = {r['bif']:.3f}  ({r['verdict']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
