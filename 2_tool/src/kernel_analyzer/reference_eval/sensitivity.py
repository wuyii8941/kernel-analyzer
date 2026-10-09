"""Bounded route and sensitivity of one rule (DSL v2 rc3 02 8.2-8.5, 04 W6).

For a rule the decision layer has per-unit projection bounds l_i <= a_i <= h_i (a_i = <K_i - G_i, w_i>).

* ``bounded_route``: with a magnitude bound M that holds for |a_i| over the declared population (never taken from the
  sample), the endpoints are truncated to [-M, M] (still l'_i <= a_i <= h'_i) and Hoeffding gives, with probability at
  least 1 - alpha, mu in [mean(l') - r_n, mean(h') + r_n], r_n = M sqrt(2 ln(2 / alpha) / n).  It needs no sample
  variance (s = 0 is allowed).  Empty truncated boxes mean the numerical evidence contradicts M: an error, no interval.
* ``approximate_mde``: the effect the endpoint-conservative t route detects with the given power at the frozen design
  (n, alpha) and the observed spread, plus the mean endpoint width.  A sensitivity of the design, not an excluded
  effect (only the equivalence axis excludes effects).
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np


def bounded_route(l, h, M: float, alpha: float) -> dict:
    l, h = np.asarray(l, dtype=np.float64), np.asarray(h, dtype=np.float64)
    n = l.size
    if n < 1 or not (np.isfinite(l).all() and np.isfinite(h).all()) or not (math.isfinite(M) and M > 0):
        return {"verdict": "NOT_ESTABLISHED", "reason": "no units, non-finite bounds or no valid M"}
    lt, ht = np.maximum(l, -M), np.minimum(h, M)
    if (lt > ht).any():
        return {"verdict": "ERROR", "reason": f"{int((lt > ht).sum())} unit(s) lie outside [-M, M]: the numerical "
                                              "evidence contradicts the declared bound"}
    r = M * math.sqrt(2.0 * math.log(2.0 / alpha) / n)
    lo, hi = float(lt.mean()) - r, float(ht.mean()) + r
    width = float((ht - lt).mean())
    return {"method": "Hoeffding on endpoints truncated to [-M, M] (rc3 02 8.3)", "M": M, "n": int(n),
            "half_width_r_n": r, "interval": [lo, hi],
            "verdict": "DETECTED_POSITIVE" if lo > 0 else ("DETECTED_NEGATIVE" if hi < 0 else "NOT_CONFIRMED"),
            "guaranteed_detectable_effect": 2.0 * r + width,
            "guaranteed_detectable_effect_meaning": "|mu| above this is detected with probability >= 1 - alpha/2 "
                                                    "(2 r_n + mean truncated endpoint width)"}


def approximate_mde(l, h, alpha: float, power: float = 0.8) -> dict:
    from scipy.stats import t

    l, h = np.asarray(l, dtype=np.float64), np.asarray(h, dtype=np.float64)
    n = l.size
    if n < 2 or not (np.isfinite(l).all() and np.isfinite(h).all()):
        return {"mde": None, "reason": "fewer than 2 units or non-finite bounds"}
    sd = float(np.std(0.5 * (l + h), ddof=1))
    width = float((h - l).mean())
    q = float(t.ppf(1 - alpha / 2, n - 1) + t.ppf(power, n - 1))
    return {"mde": q * sd / math.sqrt(n) + width, "power": power, "alpha": alpha, "n": int(n), "sd": sd,
            "mean_endpoint_width": width,
            "meaning": "sensitivity of the frozen design for the approximate route; not an excluded effect"}


def sensitivity_fields(l, h, alpha: float, M: Optional[float] = None, basis: Optional[str] = None) -> dict:
    out = {"mde_approximate": approximate_mde(l, h, alpha)}
    if M is not None:
        out["bounded"] = {**bounded_route(l, h, M, alpha), "M_basis": basis or "declared"}
    return out
