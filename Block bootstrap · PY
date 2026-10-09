"""Block bootstrap CI for mean net R (numpy only).

Drop `block_bootstrap_ci` into tools/session_vs_outside.py (as the handoff asks)
or import it from here. Suggested CLI wiring is at the bottom of this file.
"""
from __future__ import annotations

import numpy as np


def block_bootstrap_ci(values, block_size: int = 5, n_boot: int = 10000,
                       alpha: float = 0.10, seed: int | None = None,
                       return_dist: bool = False):
    """CI for the mean via moving-block resampling (overlapping block starts).

    Blocks are consecutive runs of `block_size` trades (overlapping starts),
    drawn with replacement until the series length is reached, then trimmed to N.
    Returns (lower, upper) or (lower, upper, dist). Empty input -> NaNs.
    """
    x = np.asarray(values, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n == 0:
        out = (float("nan"), float("nan"))
        return out + (np.array([]),) if return_dist else out
    b = max(1, min(int(block_size), n))
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / b))
    starts = rng.integers(0, n - b + 1, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(b)[None, None, :]).reshape(n_boot, -1)[:, :n]
    means = x[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return (float(lo), float(hi), means) if return_dist else (float(lo), float(hi))


# ---- CLI wiring for session_vs_outside.py (paste into its argparse + main) ----
# ap.add_argument("--bootstrap", action="store_true")
# ap.add_argument("--bootstrap-block-size", type=int, default=5)
# ap.add_argument("--bootstrap-n", type=int, default=10000)
# ...
# if args.bootstrap:
#     lo, hi = block_bootstrap_ci(df[args.returns_col].to_numpy(),
#                                 args.bootstrap_block_size, args.bootstrap_n, alpha=0.10)
#     print(f"block bootstrap 90% CI: [{lo:.4f}, {hi:.4f}]  (cluster-robust CI above)")
