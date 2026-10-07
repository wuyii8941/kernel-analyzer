"""Shared pieces of phase 1 of the essential-bug round (docs/protocol_essential_bugs_20261007.md).

- float64 enclosures of the specification's exact values (Fractions, rigorous decimal intervals, +-inf, NaN);
- the numerical contract of protocol section 6 (tau_64, tau_32, exact equality, bf16 recorded only);
- ``fr_run``: one Triton candidate through the tool's mode B (``kernel_analyzer.check.run``) with f supplied by the
  independent specification, returning K, K_R and f per element and seed (the FR group).
"""
from __future__ import annotations

import math
import sys
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "src", ROOT / "specs/phase1", ROOT / "scripts/essential"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

TAU = {"float64": 1e-9, "float32": 2.0 ** -12}


# ------------------------------------------------------------------------------------------------ enclosures

def _down_up(x: float, exact) -> tuple:
    """the float64 neighbours of an exact value: largest float <= exact, smallest float >= exact."""
    fx = Fraction(x)
    if fx == exact:
        return x, x
    if fx > exact:
        return math.nextafter(x, -math.inf), x
    return x, math.nextafter(x, math.inf)


def enclose(v) -> tuple:
    """(lo, hi) float64 enclosing a specification value: Fraction / int (exact), an Interval with Decimal endpoints
    (the CE spec), +-inf, NaN."""
    if hasattr(v, "lo") and hasattr(v, "hi"):                        # spec_cross_entropy.Interval
        lo = _down_up(float(v.lo), Fraction(v.lo))[0]
        hi = _down_up(float(v.hi), Fraction(v.hi))[1]
        return lo, hi
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return v, v
        return v, v
    if isinstance(v, Decimal):
        if v.is_nan():
            return math.nan, math.nan
        f = Fraction(v)
        return _down_up(float(v), f)
    f = Fraction(v)
    return _down_up(float(f), f)


def enclose_array(nested, shape=None):
    """nested lists of spec values -> (lo, hi) float64 arrays."""
    flat = []

    def walk(t):
        if isinstance(t, (list, tuple)) and not (hasattr(t, "lo")):
            for e in t:
                walk(e)
        else:
            flat.append(enclose(t))

    walk(nested)
    lo = np.array([a for a, _ in flat], dtype=np.float64)
    hi = np.array([b for _, b in flat], dtype=np.float64)
    if shape is not None:
        lo, hi = lo.reshape(shape), hi.reshape(shape)
    return lo, hi


# ------------------------------------------------------------------------------------------------ contract

def deviation(k, f_lo, f_hi, dtype, exact=False):
    """Protocol section 6, per element: True where K is outside f's enclosure widened by the screening threshold
    (exact inputs: no widening); NaN / inf compared by class.  Returns (deviates, judged) boolean arrays; bf16
    candidates are never judged (recorded only)."""
    k = np.asarray(k, dtype=np.float64)
    f_lo, f_hi = np.asarray(f_lo, np.float64), np.asarray(f_hi, np.float64)
    judged = np.full(k.shape, dtype in TAU or exact)
    spec_nan, k_nan = np.isnan(f_lo), np.isnan(k)
    spec_inf = np.isinf(f_lo) & (f_lo == f_hi)
    special = spec_nan | spec_inf | k_nan | np.isinf(k)
    if exact:
        tol = np.zeros_like(k)
    else:
        mag = np.maximum(np.abs(np.where(np.isfinite(f_lo), f_lo, 0)), np.abs(np.where(np.isfinite(f_hi), f_hi, 0)))
        tol = TAU.get(dtype, np.inf) * (1.0 + mag)
    with np.errstate(invalid="ignore"):
        out = (k < f_lo - tol) | (k > f_hi + tol)
    same_class = (spec_nan & k_nan) | (spec_inf & (k == f_lo))
    dev = np.where(special, ~same_class, out)
    return dev & judged, judged


# ------------------------------------------------------------------------------------------------ FR group

def fr_run(name, setup, make_inputs, launch, spec, seeds=(0, 1, 2)):
    """``launch(inp)`` -> {output: tensor}; ``spec(inp)`` -> {output: (lo, hi)} float64 arrays in the outputs'
    logical shapes.  Returns (report, keep) where keep[output] is a list over seeds of dicts with K, K_R lo/hi and
    the tool's ok mask (logical element order)."""
    from kernel_analyzer.check import Case, run

    class _Case(Case):
        def setup(self):
            setup()

        def inputs(self, seed):
            return make_inputs(seed)

        def launch(self, inp):
            return launch(inp)

        def spec(self, inp):
            return spec(inp)

    c = _Case()
    c.name, c.implementation, c.specification = name, name, "independent specification (specs/phase1)"
    keep = {}
    seeds = list(seeds)
    report = run(c, dev=seeds[:1], conf=seeds[1:], keep=keep)
    return report, keep
