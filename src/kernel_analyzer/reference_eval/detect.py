"""Default bias detection at a measurement point, without a declared mechanism, direction or suspect node.

Units are independent draws from the declared input / state source.  For each unit the residual of every
coordinate in the set fixed on the development units is known only as an interval [lo, hi] (the reference
is an enclosure), so every test is endpoint-conservative: a test rejects only if it rejects for every
residual inside the box.  Midpoints are used only to define directions on the development units and for
the tail diagnostics; they never produce a detection by themselves.

vector mean, E[e] != 0 (fixed-direction mean)
  * omnibus      : U-statistic T(e) = sum_{i != j} <e_i, e_j> / (n (n - 1)), unbiased for |E[e]|^2; it sees
                   any nonzero mean vector, including patterns whose coordinate sum is zero.  Box-conservative
                   sign-flip p-value: a lower bound of T over the box is compared with an upper bound of T for
                   every flipped sample, so the p-value is at least the sign-flip p-value of every residual in
                   the box (exact sign-flip test when the boxes have zero width)
  * total        : the projection on the all-ones direction, all units
  * learned      : a direction learned on the development units, tested on the confirmation units
  * rows / cols  : the learned direction over row sums / column sums when the output is a matrix

alignment with the reference (state-dependent, can be nonzero when E[e] = 0)
  * aligned      : <e, K> / |K| per unit
  * relative     : mean of e / K per unit (multiplicative bias)

The projection tests use strict outward projection bounds [l, h] per unit (directed products, exact sum)
and the endpoint-conservative t inference of analysis._summarize.  Holm is applied within each family.
Each t-type test carries diagnostics (skewness, excess kurtosis, effective number of coordinates of its
direction, largest single-unit share) computed on the midpoint scores; a detection whose diagnostics show a
projection concentrated on few coordinates together with a skewed or heavy-tailed score is reported as
exploratory, not as a confirmed bias.  Rare large values that never appear in the sample cannot be
diagnosed from the data; that limit is stated in every report.  Floating-point evaluation of the omnibus
statistic itself is not directed (relative error ~1e-13), which is negligible against its sampling spread.

Version 2 (interval inputs).  Version 1 (frozen for blind_test_v1 at e43616c) took midpoints and could
report a detection where every residual interval contained zero.  Version 2.1: a test with fewer than two
units or a numerical failure (non-finite bounds, overflow) reports CANNOT_JUDGE and takes no part in Holm;
per-unit projection bounds and the omnibus flip count are recorded for offline recomputation.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from . import intervals as iv
from .analysis import _summarize

VERSION = "2.1"
ALPHA = 0.05
FLIPS = 999
GATE = {"effective_coordinates": 10.0, "abs_skewness": 1.0, "excess_kurtosis": 3.0}
QUESTIONS = {"vector_mean": "fixed-direction mean: E[e] != 0 at the measurement point",
             "alignment": "reference-dependent alignment: e correlates with the reference (scaling / shrinkage)"}


def _diagnostics(scores: np.ndarray, direction: Optional[np.ndarray]) -> dict:
    c = scores - scores.mean()
    m2 = float((c ** 2).mean())
    skew = float((c ** 3).mean() / m2 ** 1.5) if m2 > 0 else 0.0
    kurt = float((c ** 4).mean() / m2 ** 2 - 3.0) if m2 > 0 else 0.0
    share = float((c ** 2).max() / (c ** 2).sum()) if m2 > 0 else 0.0
    eff = None
    if direction is not None:
        w2 = np.asarray(direction, dtype=np.float64) ** 2
        if w2.ndim > 1:
            w2 = w2.mean(axis=0)
        eff = float(w2.sum() ** 2 / (w2 ** 2).sum()) if w2.sum() > 0 else 0.0
    supported = not (eff is not None and eff < GATE["effective_coordinates"]
                     and (abs(skew) > GATE["abs_skewness"] or kurt > GATE["excess_kurtosis"]))
    return {"skewness": skew, "excess_kurtosis": kurt, "largest_unit_share": share,
            "effective_coordinates": eff, "tail_assumption_supported": supported}


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else None


def _box(lo, hi):
    """Midpoint m and a radius r with |e - m| <= r for every e in [lo, hi] (r rounded up)."""

    m = 0.5 * (lo + hi)
    r = np.maximum(iv.up(hi - m), iv.up(m - lo))
    return m, np.maximum(r, 0.0)


def omnibus(lo: np.ndarray, hi: np.ndarray, rng, flips: int = FLIPS) -> dict:
    """Box-conservative sign-flip test of the U-statistic for |E[e]|^2 (one-sided).

    With m, r the midpoints and radii, U = sum_i m_i, R = sum_i r_i:
      T(e)      >= T_min    = (sum_c max(0, |U_c| - R_c)^2 - sum_ic (|m_ic| + r_ic)^2) / (n (n - 1))
      T(s * e)  <= T_max(s) = ((sqrt(s' G s) + |R|)^2 - sum_ic max(0, |m_ic| - r_ic)^2) / (n (n - 1)),
    G = m m' (Cauchy-Schwarz for the cross term), for every e in the box.  Hence #{s: T(s e) >= T(e)}
    <= #{s: T_max(s) >= T_min} and the reported p-value bounds the sign-flip p-value of every residual in
    the box.  With zero radii both bounds equal the plain U-statistic."""

    n = lo.shape[0]
    if n < 2:
        return {"test": "omnibus", "n": n, "verdict": "CANNOT_JUDGE", "p": None,
                "reason": f"{n} unit(s): the U-statistic needs at least 2"}
    if not (np.isfinite(lo).all() and np.isfinite(hi).all()):
        return {"test": "omnibus", "n": n, "verdict": "CANNOT_JUDGE", "p": None, "reason": "non-finite residual bounds"}
    denom = n * (n - 1)
    signs = rng.choice([-1.0, 1.0], size=(flips, n))
    with np.errstate(all="ignore"):  # overflow is caught below and reported as CANNOT_JUDGE
        m, r = _box(lo, hi)
        big_u, big_r = m.sum(axis=0), r.sum(axis=0)
        t_min = float((np.maximum(0.0, np.abs(big_u) - big_r) ** 2).sum() - ((np.abs(m) + r) ** 2).sum()) / denom
        shrink = float((np.maximum(0.0, np.abs(m) - r) ** 2).sum())
        g = m @ m.T
        quad = np.maximum(np.einsum("bi,ij,bj->b", signs, g, signs), 0.0)
        t_max = ((np.sqrt(quad) + np.linalg.norm(big_r)) ** 2 - shrink) / denom
    if not (math.isfinite(t_min) and np.isfinite(t_max).all()):
        return {"test": "omnibus", "n": n, "verdict": "CANNOT_JUDGE", "p": None,
                "reason": "overflow in the U-statistic bounds"}
    exceed = int((t_max >= t_min).sum())
    p = float((1 + exceed) / (flips + 1))
    t_mid = float((big_u @ big_u - (m ** 2).sum()) / denom)
    return {"test": "omnibus", "n": n, "lower_bound_of_statistic": t_min, "midpoint_statistic": t_mid,
            "p": p, "flips": flips, "flips_with_upper_bound_at_least_lower_bound": exceed,
            "note": "one-sided, box-conservative sign-flip; exact sign-flip test when the null unit "
                    "distribution is symmetric and the boxes have zero width; the flips are the first draws "
                    "of numpy default_rng(seed) (choice of -1 / +1, shape (flips, units))"}


def _projection_test(name, lo, hi, w, diag_direction=None) -> dict:
    l, h = iv.project_bounds(lo, hi, w)
    r = _summarize(name, name, l, h, ALPHA)
    bounds = [[float(a), float(b)] for a, b in zip(l, h)]  # for offline recomputation
    if "p_value_two_sided_conservative" not in r:  # too few units or a numerical failure: never a detection
        return {"test": name, "n": r["n"], "verdict": "CANNOT_JUDGE", "p": None, "reason": r["reason"],
                "per_unit_bounds": bounds}
    sign = {"DETECTED_POSITIVE": "+", "DETECTED_NEGATIVE": "-"}.get(r["verdict"], "")
    return {"test": name, "n": r["n"], "p": r["p_value_two_sided_conservative"],
            "estimate_interval": [r["lower_bound_of_E_l"], r["upper_bound_of_E_h"]],
            "mean_midpoint_projection": r["mean_projection"], "interval_sign": sign,
            "units_with_box_containing_zero": r["zero_or_ambiguous"],
            "diagnostics": _diagnostics(0.5 * (l + h), diag_direction), "per_unit_bounds": bounds}


def _holm(rows: list) -> None:
    """Holm over the tests that could judge; the others keep CANNOT_JUDGE and take no part."""

    tested = [r for r in rows if r.get("p") is not None]
    order = sorted(range(len(tested)), key=lambda i: tested[i]["p"])
    stop = False
    for rank, i in enumerate(order):
        tested[i]["holm_reject"] = (not stop) and tested[i]["p"] <= ALPHA / (len(tested) - rank)
        stop = stop or not tested[i]["holm_reject"]


def detect(lo: np.ndarray, hi: np.ndarray, k: np.ndarray, n_dev: int, shape: Optional[tuple] = None,
           seed: int = 0, flips: int = FLIPS, return_directions: bool = False) -> dict:
    """lo, hi: (units, coordinates) residual interval endpoints on the fixed coordinate set (original order);
    point residuals pass lo = hi.  k: the reference the alignment tests compare with (same shape).
    ``shape`` is the output's shape over these coordinates when it is a matrix.  With
    ``return_directions`` the directions actually used are returned under "directions" (numpy arrays)."""

    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    if np.any(lo > hi):
        raise ValueError("lo > hi")
    rng = np.random.default_rng(seed)
    n, d = lo.shape
    dev, conf = slice(0, n_dev), slice(n_dev, n)
    mid = 0.5 * (lo + hi)
    directions = {}
    vec = [omnibus(lo, hi, rng, flips)]
    ones = np.full(d, 1 / math.sqrt(d))
    directions["total"] = ones
    vec.append(_projection_test("total", lo, hi, ones, ones))
    w = _unit(mid[dev].mean(axis=0))
    if w is not None:
        directions["learned"] = w
        vec.append(_projection_test("learned", lo[conf], hi[conf], w, w))
    if shape is not None and len(shape) == 2 and shape[0] > 1 and shape[1] > 1:
        mm = mid.reshape(n, *shape)
        for axis, name in ((2, "rows"), (1, "cols")):
            size = shape[axis - 1]
            g = mm.sum(axis=axis) / math.sqrt(size)  # group sums, unit-normalized (midpoints: direction only)
            wg = _unit(g[dev].mean(axis=0))
            if wg is not None:
                # the group direction pulled back to the coordinates: w_c = wg[group(c)] / sqrt(size)
                full = (np.repeat(wg, shape[1]) if axis == 2 else np.tile(wg, shape[0])) / math.sqrt(size)
                directions[name] = full
                vec.append(_projection_test(name, lo[conf], hi[conf], full, wg))
    ali = []
    k = np.asarray(k, dtype=np.float64)
    kn = np.linalg.norm(k, axis=1, keepdims=True)
    keep = kn[:, 0] > 0
    wa = np.divide(k, kn, out=np.zeros_like(k), where=kn > 0)
    directions["aligned"] = wa
    ali.append(_projection_test("aligned", lo[keep], hi[keep], wa[keep]))
    nnz = (k != 0).sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        wr = np.where(k != 0, 1.0 / (k * np.maximum(nnz, 1)), 0.0)
    keep = nnz[:, 0] > 0
    directions["relative"] = wr
    ali.append(_projection_test("relative", lo[keep], hi[keep], wr[keep]))
    for fam in (vec, ali):
        _holm(fam)
        for row in fam:
            if row.get("p") is None:
                continue
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
        if fam and all(r["verdict"] == "CANNOT_JUDGE" for r in fam):
            return "CANNOT_JUDGE"
        return "NOT_CONFIRMED"

    out = {"detector_version": VERSION, "seed": seed, "units": n, "development_units": n_dev, "coordinates": d,
           "max_interval_width": float((hi - lo).max()) if lo.size else 0.0,
           "vector_mean": {"question": QUESTIONS["vector_mean"], "verdict": summary(vec), "tests": vec},
           "alignment": {"question": QUESTIONS["alignment"], "verdict": summary(ali), "tests": ali},
           "limits": ["rare values that do not appear among the units cannot be diagnosed from the data",
                      "t-type tests assume an approximately normal mean projection; the diagnostics flag "
                      "concentrated, skewed or heavy-tailed scores but cannot exclude unseen rare values",
                      "NOT_CONFIRMED is not evidence of a zero mean; see the calibrated detection rates"]}
    if return_directions:
        out["directions"] = directions
    return out
