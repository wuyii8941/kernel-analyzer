from __future__ import annotations

import math
from collections.abc import Iterable


def clean_floats(values: Iterable[float]) -> list[float]:
    return [float(value) for value in values]


def mean(values: Iterable[float]) -> float:
    vals = clean_floats(values)
    return sum(vals) / len(vals) if vals else 0.0


def percentile(values: Iterable[float], q: float) -> float:
    vals = sorted(clean_floats(values))
    n = len(vals)
    if n == 0:
        return 0.0
    if n == 1:
        return vals[0]
    pos = (q / 100.0) * (n - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    frac = pos - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac

