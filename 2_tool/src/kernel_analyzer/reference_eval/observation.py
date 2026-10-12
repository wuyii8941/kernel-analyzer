"""Observation layers (DSL v2 rc3 02 1.2: the measurement point D_m).  A layer maps the actual output K, its reference
enclosure [r_lo, r_hi] (K_R in modes A / B, the specification f in black-box mode) and, where needed, a declared input,
to the quantities the bias query observes.  The residual K - reference is the same at every layer; what changes is the
frame the directions and the scale are taken in.

* ``output``: the output itself (the default; directions R2 / R3 follow the reference value).
* ``increment`` (rc3 02 1.2, "formula update": the ideal response on the same old state): for a state or parameter
  written from a declared old value x (an in-place update, or an output that replaces an input), the actual increment
  dK = K - x and the reference increment dG = [r_lo - x, r_hi - x].  Directions R2 / R3 follow dG, so a positive R3
  reads "the written increment is scaled down against the reference increment" (retention below one) and a positive R2
  "increments are pulled toward zero".  Per unit the layer also reports, on the coordinates whose reference increment is
  certainly nonzero (the interval excludes 0) and not below the declared floor:
    - dropped: dK == 0 exactly (the update was not written at all);
    - sign flips: dK has the opposite sign of dG;
    - retention: <dK, m> / <m, m> with m the midpoint of dG (1: the reference increment fully written);
    - error energy relative to the increment energy: ||K - mid(r)||^2 / ||m||^2.
  Coordinates below the floor are counted, never used for a ratio (an update near zero gives an unstable ratio).

The floor is declared before the data: an absolute value, or a factor times |x| (relative to the old value).

``output_unit`` gives the same frame quantities without the increment's coordinate selection, for an output (or an
increment) against a point reference: the per-unit descriptions that sit next to the direction rules R2 / R3 (it
decides nothing; the decisions are the rules over independent units).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

LAYERS = ("output", "increment")


def floor_of(spec: Optional[dict], old: np.ndarray) -> np.ndarray:
    """The near-zero floor per coordinate from its declaration: {"absolute": a} or {"relative_to_old": c} (c |x|),
    both may be given (the larger applies); None: 0 (no floor)."""
    old = np.asarray(old, np.float64)
    f = np.zeros_like(old)
    if spec:
        if spec.get("absolute") is not None:
            f = np.maximum(f, float(spec["absolute"]))
        if spec.get("relative_to_old") is not None:
            f = np.maximum(f, float(spec["relative_to_old"]) * np.abs(old))
    return f


def increment_unit(k, r_lo, r_hi, old, ok, floor=None) -> dict:
    """The increment-layer quantities of one unit (arrays over the coordinates of the output's frame)."""
    k, r_lo, r_hi, old = (np.asarray(a, np.float64) for a in (k, r_lo, r_hi, old))
    ok = np.asarray(ok, bool)
    d_lo, d_hi = r_lo - old, r_hi - old          # float64 subtraction of float64 endpoints; widened below
    d_lo, d_hi = np.nextafter(d_lo, -np.inf), np.nextafter(d_hi, np.inf)
    dk = k - old
    fl = np.zeros_like(old) if floor is None else np.asarray(floor, np.float64)
    nonzero = ok & ((d_lo > 0) | (d_hi < 0))
    m = 0.5 * (d_lo + d_hi)
    above = nonzero & (np.minimum(np.abs(d_lo), np.abs(d_hi)) > fl)
    near = ok & ~above
    dropped = above & (dk == 0)
    flips = above & (dk != 0) & (np.sign(dk) != np.sign(m))
    mm = float(np.dot(m[above], m[above]))
    resid = k - 0.5 * (r_lo + r_hi)
    out = {"coordinates": int(ok.sum()), "reference_increment_nonzero": int(nonzero.sum()),
           "above_floor": int(above.sum()), "near_zero_or_below_floor": int(near.sum()),
           "dropped": int(dropped.sum()), "sign_flips": int(flips.sum()),
           "dropped_fraction": float(dropped.sum() / above.sum()) if above.any() else None,
           "sign_flip_fraction": float(flips.sum() / above.sum()) if above.any() else None,
           "retention": float(np.dot(dk[above], m[above]) / mm) if mm > 0 else None,
           "error_energy_over_increment_energy": float(np.dot(resid[above], resid[above]) / mm) if mm > 0 else None,
           "mean_abs_reference_increment": float(np.abs(m[above]).mean()) if above.any() else None,
           "mean_abs_written_increment": float(np.abs(dk[above]).mean()) if above.any() else None}
    return out


def output_unit(k, r) -> dict:
    """The frame quantities of one unit: the actual values k against a point reference r (an enclosure midpoint, a
    specification or another implementation), e = k - r:
      - relative_rms: sqrt(mean(e^2) / mean(r^2));
      - alignment: <e, r> / <r, r>, the error's component along the reference (negative: the output scaled down);
      - mean_error_over_rms_reference: mean(e) / rms(r), the fixed-direction counterpart;
      - scale_ratio: <k, r> / <r, r> (for increments: the retention over every coordinate, without a floor);
      - cosine: <k, r> / (|k| |r|);
      - scaling_share: alignment^2 / relative_rms^2, the share of the error energy that is a uniform scaling of r."""
    k, r = np.asarray(k, np.float64), np.asarray(r, np.float64)
    e = k - r
    rr = max((r * r).sum(), 1e-300)
    kf, rf = k.ravel(), r.ravel()
    out = {"relative_rms": float(np.sqrt((e * e).mean() / max((r * r).mean(), 1e-300))),
           "alignment": float((e * r).sum() / rr),
           "mean_error_over_rms_reference": float(e.mean() / max(np.sqrt((r * r).mean()), 1e-300)),
           "scale_ratio": float(np.dot(kf, rf) / max(np.dot(rf, rf), 1e-300)),
           "cosine": float(np.dot(kf, rf) / max(np.linalg.norm(kf) * np.linalg.norm(rf), 1e-300))}
    out["scaling_share"] = out["alignment"] ** 2 / out["relative_rms"] ** 2 if out["relative_rms"] > 0 else None
    return out


def summarize_increment(units: list) -> dict:
    """Across units: totals, and the per-unit distribution of the ratio quantities (each unit is one independent
    sample; coordinates are not units)."""
    keys = ("dropped_fraction", "sign_flip_fraction", "retention", "error_energy_over_increment_energy")
    out = {"units": len(units)}
    for k in ("coordinates", "reference_increment_nonzero", "above_floor", "near_zero_or_below_floor", "dropped",
              "sign_flips"):
        out[k] = int(sum(u[k] for u in units))
    for k in keys:
        v = np.array([u[k] for u in units if u[k] is not None], np.float64)
        out[k] = ({"mean": float(v.mean()), "min": float(v.min()), "max": float(v.max()),
                   "sd": float(v.std(ddof=1)) if v.size > 1 else None, "n": int(v.size)} if v.size else None)
    return out
