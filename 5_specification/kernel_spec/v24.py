"""Access to the v2.4 executable references (specification_v2/algorithm_specs) for the adapters.

The v2.4 code has three interval implementations (Decimal intervals in phase1 CE, rigorous.Interval in phase2, the
mpmath-based rigorous_mp). Adapters convert every result to the shared values of kernel_spec.num: exact endpoints are
read out exactly and re-enclosed outward. rigorous_mp sets the global mpmath interval precision to 200 bits on
import; it is set back to 256 here so that every enclosure in kernel_spec uses one precision.
"""
from __future__ import annotations

import importlib
import sys
from decimal import Decimal
from fractions import Fraction

from mpmath import iv

from .num import IV, NotEstablished, from_interval

ALG = __import__("pathlib").Path(__file__).resolve().parents[1] / "specification_v2" / "algorithm_specs"
for _p in ("phase1", "phase2", "phase3"):
    if str(ALG / _p) not in sys.path:
        sys.path.append(str(ALG / _p))


def module(name: str):
    m = importlib.import_module(name)
    iv.prec = 256
    return m


def value(x):
    """A v2.4 scalar (Fraction, int, Decimal interval, rigorous.Interval, mpmath iv) as a kernel_spec value."""
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, Fraction)):
        return Fraction(x)
    if isinstance(x, IV):
        return x
    if hasattr(x, "lo") and hasattr(x, "hi"):
        lo, hi = x.lo, x.hi
        for e in (lo, hi):
            if isinstance(e, Decimal) and not e.is_finite():
                raise NotEstablished("v2.4 interval with a non-finite endpoint")
        return from_interval(Fraction(lo), Fraction(hi))
    raise TypeError(f"unknown v2.4 value type {type(x).__name__}")


def tree(x):
    if isinstance(x, (list, tuple)):
        return [tree(v) for v in x]
    return value(x)
