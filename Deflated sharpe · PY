"""Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014).

Per-trade Sharpe (mean / sd of net R). Not annualized unless --periods-per-year given.

NOTE on the formula: the handoff's last line
    DSR = Phi((SR_obs - E_max * SE_SR * sqrt(T)) / SE_SR)
mixes units (E_max is a z-score, SR_obs is in Sharpe units). The published form is
    SR0 = sqrt(Var[SR_trials]) * E_max_z
    DSR = Phi((SR_obs - SR0) / SE_SR)
which is what is implemented here. Var[SR_trials] defaults to the null sampling
variance of a Sharpe estimate, 1/(T-1); pass --sr-var if you have the real
cross-trial variance of Sharpe ratios.

Correlated trials: if the N variants you tried are highly correlated (e.g. 5 thresholds x
2 targets x 2 splits), the raw count over-penalises. Pass --effective-trials to give the
effective number of independent trials; it replaces --n-trials in E[max z].
"""
from __future__ import annotations

import argparse
import math

import numpy as np

EULER_GAMMA = 0.5772156649015329


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_ppf(p: float) -> float:
    """Inverse normal CDF (Acklam's rational approximation, ~1e-9 accuracy)."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00]
    plow = 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > 1 - plow:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5]) * q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


try:  # prefer scipy when present
    from scipy.stats import norm as _norm
    norm_cdf = lambda x: float(_norm.cdf(x))  # noqa: E731
    norm_ppf = lambda p: float(_norm.ppf(p))  # noqa: E731
except Exception:  # pragma: no cover
    pass


def expected_max_z(n_trials: int) -> float:
    """Expected maximum of N standard-normal trials."""
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    if n_trials == 1:
        return 0.0
    return ((1 - EULER_GAMMA) * norm_ppf(1 - 1.0 / n_trials)
            + EULER_GAMMA * norm_ppf(1 - 1.0 / (n_trials * math.e)))


def sharpe_stats(returns) -> dict:
    x = np.asarray(returns, dtype=float)
    x = x[~np.isnan(x)]
    t = len(x)
    out = {"T": t, "sharpe": float("nan"), "skew": float("nan"),
           "excess_kurt": float("nan")}
    if t < 2:
        return out
    sd = x.std(ddof=1)
    if sd == 0 or not np.isfinite(sd):
        return out
    z = (x - x.mean()) / x.std(ddof=0)
    out.update(sharpe=float(x.mean() / sd),
               skew=float((z ** 3).mean()),
               excess_kurt=float((z ** 4).mean() - 3.0))
    return out


def sharpe_se(sr: float, t: int, skew: float, excess_kurt: float) -> float:
    var = (1 - skew * sr + ((excess_kurt + 2.0) / 4.0) * sr ** 2) / (t - 1)
    return math.sqrt(var) if var > 0 else float("nan")


def deflated_sharpe(returns, n_trials: int = 1, sr_var: float | None = None,
                    effective_trials: int | None = None) -> dict:
    """effective_trials, when given, replaces n_trials in the expected-max term."""
    n_used = n_trials if effective_trials is None else effective_trials
    s = sharpe_stats(returns)
    res = dict(s, n_trials=n_trials, effective_trials=effective_trials, n_used=n_used,
               se=float("nan"), sr0=float("nan"),
               e_max_z=float("nan"), dsr=float("nan"))
    if not np.isfinite(s["sharpe"]):
        return res
    t = s["T"]
    se = sharpe_se(s["sharpe"], t, s["skew"], s["excess_kurt"])
    e_max = expected_max_z(n_used)
    v = sr_var if sr_var is not None else 1.0 / (t - 1)
    sr0 = math.sqrt(v) * e_max
    dsr = norm_cdf((s["sharpe"] - sr0) / se) if np.isfinite(se) and se > 0 else float("nan")
    res.update(se=se, sr0=sr0, e_max_z=e_max, dsr=dsr)
    return res


def interpret(dsr: float) -> str:
    if not np.isfinite(dsr):
        return "undefined (too few trades or zero variance)"
    if dsr > 0.95:
        return "strong evidence the edge is real"
    if dsr > 0.90:
        return "moderate evidence"
    return "likely selection bias / not distinguishable from the best of N noise trials"


def format_report(r: dict) -> str:
    return "\n".join([
        f"trades (T)            : {r['T']}",
        f"Sharpe (per trade)    : {r['sharpe']:.4f}",
        f"skewness              : {r['skew']:.4f}",
        f"excess kurtosis       : {r['excess_kurt']:.4f}",
        f"SE of Sharpe          : {r['se']:.4f}",
        f"N trials (raw)        : {r['n_trials']}",
        f"N effective           : {r['effective_trials'] if r['effective_trials'] is not None else '(not given; raw N used)'}",
        f"E[max z] under null   : {r['e_max_z']:.4f}",
        f"SR0 (hurdle)          : {r['sr0']:.4f}",
        f"DSR                   : {r['dsr']:.4f}",
        f"interpretation        : {interpret(r['dsr'])}",
    ])


def main(argv=None) -> int:
    import pandas as pd
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--trades", required=True)
    ap.add_argument("--returns-col", default="net_r")
    ap.add_argument("--n-trials", type=int, required=True)
    ap.add_argument("--effective-trials", type=int, default=None,
                    help="effective number of independent trials; overrides --n-trials "
                         "when provided")
    ap.add_argument("--sr-var", type=float, default=None,
                    help="variance of Sharpe ratios across trials (default 1/(T-1))")
    a = ap.parse_args(argv)
    df = pd.read_csv(a.trades)
    print(format_report(deflated_sharpe(df[a.returns_col].to_numpy(), a.n_trials, a.sr_var,
                                         a.effective_trials)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
