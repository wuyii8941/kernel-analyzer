"""Three-factor bias-formation accounting.

This module keeps source variation, local F+B response, and trajectory
propagation separate.  It deliberately accepts summaries as well as vectors:
summaries can report what is already measured, while missing replay artifacts
remain unresolved rather than being inferred from a downstream verdict.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping, Sequence


def even_odd_response(
    plus: Sequence[float], minus: Sequence[float],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return the exact response-even and response-odd parts."""
    if len(plus) != len(minus) or not plus:
        raise ValueError("plus/minus responses must have the same non-empty shape")
    even = tuple((float(a) + float(b)) / 2.0 for a, b in zip(plus, minus))
    odd = tuple((float(a) - float(b)) / 2.0 for a, b in zip(plus, minus))
    if any(not math.isfinite(x) for x in (*even, *odd)):
        raise ValueError("response contains a nonfinite value")
    return even, odd


def vector_l2(values: Iterable[float]) -> float:
    return math.sqrt(math.fsum(float(x) * float(x) for x in values))


def vector_dot(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("vector shapes differ")
    return math.fsum(float(a) * float(b) for a, b in zip(left, right))


def parity_decomposition(
    plus_responses: Sequence[Sequence[float]],
    minus_responses: Sequence[Sequence[float]],
) -> dict[str, object]:
    """Compute empirical ``mu_even + mu_odd = mu`` over matched pairs.

    Each row is one fixed state/orbit.  The function does not call a row
    centered when vectors are absent; callers should record an unresolved
    certificate instead.
    """
    if len(plus_responses) != len(minus_responses) or not plus_responses:
        raise ValueError("matched response rows are required")
    pairs = [even_odd_response(p, m) for p, m in zip(plus_responses, minus_responses)]
    even_rows = [pair[0] for pair in pairs]
    odd_rows = [pair[1] for pair in pairs]

    def mean(rows: Sequence[Sequence[float]]) -> tuple[float, ...]:
        width = len(rows[0])
        if any(len(row) != width for row in rows):
            raise ValueError("response rows have inconsistent shapes")
        return tuple(math.fsum(row[i] for row in rows) / len(rows) for i in range(width))

    mu_even = mean(even_rows)
    mu_odd = mean(odd_rows)
    mu = tuple(a + b for a, b in zip(mu_even, mu_odd))
    return {
        "state_count": len(pairs),
        "mu_even": mu_even,
        "mu_odd": mu_odd,
        "mu": mu,
        "even_l2": vector_l2(mu_even),
        "odd_l2": vector_l2(mu_odd),
        "mu_l2": vector_l2(mu),
        "closure_l2": vector_l2(
            tuple(mu[i] - mu_even[i] - mu_odd[i] for i in range(len(mu)))
        ),
        "response_even_energy_fraction": (
            vector_l2(mu_even) ** 2
            / max(vector_l2(mu_even) ** 2 + vector_l2(mu_odd) ** 2, 1e-30)
        ),
    }


def prefix_resultant(rows: Sequence[Mapping[str, float]], value_key: str) -> dict[str, object]:
    """Summarize prefix resultant/persistence from scalar per-step records."""
    values = [float(row[value_key]) for row in rows if value_key in row]
    if not values:
        return {"status": "UNRESOLVED_MISSING_VECTOR_TRACE", "value_key": value_key}
    prefixes = {}
    for stop in (4, 8, 16, 32):
        if stop <= len(values):
            prefix = values[:stop]
            numerator = abs(math.fsum(prefix))
            denominator = math.sqrt(math.fsum(x * x for x in prefix))
            prefixes[str(stop)] = {
                "signed_sum": math.fsum(prefix),
                "l2": denominator,
                "scalar_resultant_ratio": numerator / max(denominator, 1e-30),
            }
    if values and str(len(values)) not in prefixes:
        denominator = math.sqrt(math.fsum(x * x for x in values))
        prefixes[str(len(values))] = {
            "signed_sum": math.fsum(values),
            "l2": denominator,
            "scalar_resultant_ratio": abs(math.fsum(values)) / max(denominator, 1e-30),
        }
    return {
        "status": "COMPLETE",
        "value_key": value_key,
        "step_count": len(values),
        "prefixes": prefixes,
    }


@dataclass(frozen=True)
class FactorStatus:
    source: str
    response: str
    propagation: str

    def as_dict(self) -> dict[str, str]:
        return {
            "source": self.source,
            "response": self.response,
            "propagation": self.propagation,
        }
