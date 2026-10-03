"""Default bias detection at a kernel output, without a declared mechanism, direction or suspect node.

Units are independent draws from the declared input / state source.  For each unit and each measured
output, e = K - RN(K_R) over a coordinate set fixed on the development units (complete references only).
The detector applies a fixed family of tests chosen in advance, so nothing about the mechanism has to be
supplied, and reports two targets separately:

vector mean, E[e] != 0
  * omnibus      : U-statistic sum_{i != j} <e_i, e_j> / (n (n - 1)), unbiased for |E[e]|^2; it sees any
                   nonzero mean vector, including patterns whose coordinate sum is zero.  p-value by
                   random sign flips of the units (exact when the null unit distribution is symmetric)
  * total        : the mean over coordinates (projection on the all-ones direction), all units
  * learned      : a direction learned on the development units, tested on the confirmation units
  * rows / cols  : the learned direction over row sums / column sums when the output is a matrix

alignment with the output (state-dependent, can be nonzero when E[e] = 0)
  * aligned      : <e, K> / |K| per unit
  * relative     : mean of e / K per unit (multiplicative bias)

Holm is applied within each target family.  Each t-type test carries diagnostics (skewness, excess
kurtosis, effective number of coordinates of its direction, largest single-unit share); a detection whose
diagnostics show a projection concentrated on few coordinates together with a skewed or heavy-tailed
score is reported as exploratory, not as a confirmed bias.  Rare large values that never appear in the
sample cannot be diagnosed from the data; that limit is stated in every report.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

ALPHA = 0.05
FLIPS = 999
GATE = {"effective_coordinates": 10.0, "abs_skewness": 1.0, "excess_kurtosis": 3.0}


def _t_test(scores: np.ndarray) -> dict:
    from scipy.stats import t as tdist

    n = scores.size
    mean = float(scores.mean())
    sd = float(scores.std(ddof=1)) if n > 1 else 0.0
    if sd == 0.0:
        return {"n": n, "mean": mean, "sd": 0.0, "t": math.inf if mean else 0.0, "p": 0.0 if mean else 1.0}
    t = mean / (sd / math.sqrt(n))
    return {"n": n, "mean": mean, "sd": sd, "t": float(t), "p": float(2 * tdist.sf(abs(t), n - 1))}


def _diagnostics(scores: np.ndarray, direction: Optional[np.ndarray]) -> dict:
    c = scores - scores.mean()
    m2 = float((c ** 2).mean())
    skew = float((c ** 3).mean() / m2 ** 1.5) if m2 > 0 else 0.0
    kurt = float((c ** 4).mean() / m2 ** 2 - 3.0) if m2 > 0 else 0.0
    share = float((c ** 2).max() / (c ** 2).sum()) if m2 > 0 else 0.0
    eff = None
    if direction is not None:
        w2 = np.asarray(direction, dtype=np.float64) ** 2
        eff = float(w2.sum() ** 2 / (w2 ** 2).sum()) if w2.sum() > 0 else 0.0
    supported = not (eff is not None and eff < GATE["effective_coordinates"]
                     and (abs(skew) > GATE["abs_skewness"] or kurt > GATE["excess_kurtosis"]))
    return {"skewness": skew, "excess_kurtosis": kurt, "largest_unit_share": share,
            "effective_coordinates": eff, "tail_assumption_supported": supported}


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else None


def omnibus(e: np.ndarray, rng, flips: int = FLIPS) -> dict:
    """U-statistic for |E[e]|^2 with a sign-flip p-value and the Chen-Qin normal approximation."""

    from scipy.stats import norm

    n = e.shape[0]
    g = e @ e.T
    off = g - np.diag(np.diag(g))
    stat = float(off.sum() / (n * (n - 1)))
    signs = rng.choice([-1.0, 1.0], size=(flips, n))
    flipped = np.einsum("bi,ij,bj->b", signs, off, signs) / (n * (n - 1))
    p_flip = float((1 + (flipped >= stat).sum()) / (flips + 1))
    tr2 = float((off ** 2).sum() / (n * (n - 1)))  # estimates tr(Sigma^2) under the null
    sd = math.sqrt(2 * tr2 / (n * (n - 1))) if tr2 > 0 else 0.0
    p_norm = float(norm.sf(stat / sd)) if sd > 0 else (0.0 if stat > 0 else 1.0)
    return {"n": n, "estimate_of_squared_mean_norm": stat, "z": stat / sd if sd > 0 else math.inf,
            "p": p_flip, "p_normal_approximation": p_norm,
            "note": "one-sided; sign-flip p is exact when the null unit distribution is symmetric"}


def _holm(rows: list) -> None:
    order = sorted(range(len(rows)), key=lambda i: rows[i]["p"])
    stop = False
    for rank, i in enumerate(order):
        rows[i]["holm_reject"] = (not stop) and rows[i]["p"] <= ALPHA / (len(rows) - rank)
        stop = stop or not rows[i]["holm_reject"]


def detect(e: np.ndarray, k: np.ndarray, n_dev: int, shape: Optional[tuple] = None, seed: int = 0) -> dict:
    """e, k: (units, coordinates) residual mid values and actual outputs on the fixed coordinate set
    (original order).  ``shape`` is the output's shape over these coordinates when it is a matrix."""

    rng = np.random.default_rng(seed)
    n, d = e.shape
    dev, conf = slice(0, n_dev), slice(n_dev, n)
    vec = []
    r = omnibus(e, rng)
    r["test"] = "omnibus"
    vec.append(r)
    ones = np.full(d, 1 / math.sqrt(d))
    s = e @ ones
    vec.append({"test": "total", **_t_test(s), "diagnostics": _diagnostics(s, ones)})
    w = _unit(e[dev].mean(axis=0))
    if w is not None:
        s = e[conf] @ w
        vec.append({"test": "learned", **_t_test(s), "diagnostics": _diagnostics(s, w)})
    if shape is not None and len(shape) == 2 and shape[0] > 1 and shape[1] > 1:
        m = e.reshape(n, *shape)
        for axis, name in ((2, "rows"), (1, "cols")):
            g = m.sum(axis=axis) / math.sqrt(m.shape[axis])  # group sums, unit-normalized
            wg = _unit(g[dev].mean(axis=0))
            if wg is not None:
                s = g[conf] @ wg
                vec.append({"test": name, **_t_test(s), "diagnostics": _diagnostics(s, wg)})
    ali = []
    kn = np.linalg.norm(k, axis=1)
    s = np.where(kn > 0, (e * k).sum(axis=1) / np.where(kn > 0, kn, 1.0), 0.0)
    ali.append({"test": "aligned", **_t_test(s), "diagnostics": _diagnostics(s, None)})
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(k != 0, e / k, np.nan)
    s = np.nanmean(rel, axis=1)
    s = s[np.isfinite(s)]
    ali.append({"test": "relative", **_t_test(s), "diagnostics": _diagnostics(s, None)})
    for fam in (vec, ali):
        _holm(fam)
        for row in fam:
            supported = row.get("diagnostics", {}).get("tail_assumption_supported", True)
            if row["holm_reject"]:
                row["verdict"] = "DETECTED" if supported else "EXPLORATORY_ONLY"
            else:
                row["verdict"] = "NOT_CONFIRMED"

    def summary(fam):
        if any(r["verdict"] == "DETECTED" for r in fam):
            return "DETECTED"
        if any(r["verdict"] == "EXPLORATORY_ONLY" for r in fam):
            return "EXPLORATORY_ONLY"
        return "NOT_CONFIRMED"

    return {"units": n, "development_units": n_dev, "coordinates": d,
            "vector_mean": {"verdict": summary(vec), "tests": vec},
            "alignment": {"verdict": summary(ali), "tests": ali},
            "limits": ["rare values that do not appear among the units cannot be diagnosed from the data",
                       "NOT_CONFIRMED is not evidence of a zero mean; see the calibrated detection rates"]}
