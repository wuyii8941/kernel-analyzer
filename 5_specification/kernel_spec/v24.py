"""Access to the v2.4 executable references (specification_v2/algorithm_specs) for the adapters.

The v2.4 code has three interval implementations (Decimal intervals in phase1 CE, rigorous.Interval in phase2, the
mpmath-based rigorous_mp). Adapters convert every result to the shared values of kernel_spec.num: exact endpoints are
read out exactly and re-enclosed outward. rigorous_mp sets the global mpmath interval precision to 200 bits on
import; it is set back to 256 here so that every enclosure in kernel_spec uses one precision.
"""
from __future__ import annotations

import importlib
import importlib.util
import sys
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

from mpmath import iv

from .num import IV, NotEstablished, from_interval

ALG = Path(__file__).resolve().parents[1] / "specification_v2" / "algorithm_specs"
_LOCATIONS = {"spec_cross_entropy": "phase1", "spec_optimizers": "phase2",
              "spec_training_program": "phase2", "spec_ema": "phase3",
              "rigorous": "phase2", "rigorous_mp": "phase2"}
_DEPENDENCIES = {"spec_optimizers": {"rigorous": ("Interval", "SpecInputError", "imax")},
                 "spec_training_program": {"rigorous_mp": ("I", "iv", "SpecInputError", "SpecNotEstablished")}}
for _p in ("phase1", "phase2", "phase3"):
    if str(ALG / _p) not in sys.path:
        sys.path.append(str(ALG / _p))


def module(name: str):
    """Resolve only declared references and their local numerical dependencies.

    Never execute a same-named module from an earlier sys.path entry or reuse
    one cached from another tree. Refuse conflicts rather than silently replace
    modules that other code may already be using. This is origin binding, not
    a claim that source contents have been reviewed or mathematically certified.
    """
    if name not in _LOCATIONS:
        raise NotEstablished(f"undeclared v2.4 reference module {name!r}")
    expected = (ALG / _LOCATIONS[name] / f"{name}.py").resolve()
    cached = sys.modules.get(name)
    if cached is not None:
        origin = getattr(cached, "__file__", None)
    else:
        try:
            spec = importlib.util.find_spec(name)
        except (ValueError, ImportError) as exc:
            raise NotEstablished(f"cannot resolve reference module {name}") from exc
        origin = spec.origin if spec else None
    if not origin or Path(origin).resolve() != expected or not expected.is_file():
        raise NotEstablished(f"reference module {name} is not bound to {expected}")
    dependencies = {dep: module(dep) for dep in _DEPENDENCIES.get(name, {})}
    m = importlib.import_module(name)
    for dep, backend in dependencies.items():
        for attribute in _DEPENDENCIES[name][dep]:
            if getattr(m, attribute, None) is not getattr(backend, attribute, None):
                raise NotEstablished(f"reference module {name} retains a different {dep}.{attribute}")
    if name == "spec_training_program" and m.R is not dependencies["rigorous_mp"]:
        raise NotEstablished("spec_training_program retains a different rigorous_mp backend")
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
