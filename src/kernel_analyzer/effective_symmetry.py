"""Basis-free geometry for effective antithetic-symmetry measurements.

The primary statistic removes the finite-population diagonal energy.  This is
important: ``||mean(w)||^2`` has a positive ``1/n`` floor even for centered
independent contributions, while the cross-event U-statistic does not.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Sequence


@dataclass(frozen=True)
class ContributionGeometry:
    events: int
    average_event_energy: float
    cross_event_inner_u: float
    cross_event_ratio: float
    resultant_l2: float
    path_l2: float
    resultant_path_ratio: float
    structure: str

    def to_dict(self) -> dict[str, float | int | str]:
        return asdict(self)


def contribution_geometry(gram: Sequence[Sequence[float]]) -> ContributionGeometry:
    """Summarize event vectors from their complete Gram matrix.

    ``cross_event_inner_u`` is the unbiased off-diagonal mean.  A positive
    value indicates directional structure; a negative value records systematic
    cancellation instead of silently relabeling it as zero.
    """

    n = len(gram)
    if n < 2 or any(len(row) != n for row in gram):
        raise ValueError("a complete Gram matrix with at least two events is required")
    values = [[float(value) for value in row] for row in gram]
    if any(not math.isfinite(value) for row in values for value in row):
        raise ValueError("Gram matrix must be finite")
    diagonal = [values[index][index] for index in range(n)]
    if any(value < -1e-12 for value in diagonal):
        raise ValueError("Gram diagonal cannot be negative")
    trace = sum(max(0.0, value) for value in diagonal)
    average = trace / n
    off_diagonal = sum(
        values[i][j] for i in range(n) for j in range(n) if i != j
    )
    cross = off_diagonal / (n * (n - 1))
    total = trace + off_diagonal
    resultant = math.sqrt(max(0.0, total))
    path = sum(math.sqrt(max(0.0, value)) for value in diagonal)
    tolerance = max(1e-30, average * 1e-12)
    # This is a descriptive sign, not a statistical verdict.  CENTERED versus
    # BIASED requires a separately frozen uncertainty interval/margin.
    structure = (
        "POSITIVE_CROSS_INNER" if cross > tolerance
        else "NEGATIVE_CROSS_INNER" if cross < -tolerance
        else "NUMERICAL_ZERO_CROSS_INNER"
    )
    return ContributionGeometry(
        events=n,
        average_event_energy=average,
        cross_event_inner_u=cross,
        cross_event_ratio=cross / average if average else 0.0,
        resultant_l2=resultant,
        path_l2=path,
        resultant_path_ratio=resultant / path if path else 0.0,
        structure=structure,
    )


@dataclass(frozen=True)
class PairingEffect:
    natural: ContributionGeometry
    disrupted: ContributionGeometry
    cross_event_ratio_change: float
    directional_energy_suppression: float | None

    def to_dict(self) -> dict[str, object]:
        return {
            "natural": self.natural.to_dict(),
            "disrupted": self.disrupted.to_dict(),
            "cross_event_ratio_change": self.cross_event_ratio_change,
            "directional_energy_suppression": self.directional_energy_suppression,
        }


def pairing_effect(
    natural_gram: Sequence[Sequence[float]],
    disrupted_gram: Sequence[Sequence[float]],
) -> PairingEffect:
    """Compare real and marginal-preserving disrupted event pairings."""

    natural = contribution_geometry(natural_gram)
    disrupted = contribution_geometry(disrupted_gram)
    if natural.events != disrupted.events:
        raise ValueError("natural and disrupted populations must have equal event counts")
    suppression = None
    if natural.cross_event_inner_u > 0.0:
        suppression = 1.0 - disrupted.cross_event_inner_u / natural.cross_event_inner_u
    return PairingEffect(
        natural=natural,
        disrupted=disrupted,
        cross_event_ratio_change=(
            natural.cross_event_ratio - disrupted.cross_event_ratio
        ),
        directional_energy_suppression=suppression,
    )


@dataclass(frozen=True)
class AntitheticResponseGeometry:
    """Odd/even decomposition of matched ``+delta`` and ``-delta`` responses.

    If ``plus = F(delta)`` and ``minus = F(-delta)``, then an odd downstream
    map has ``minus == -plus``.  ``response_even_l2`` is therefore the direct
    rectification term; it is intentionally not called error variance.
    """

    plus_l2: float
    minus_l2: float
    plus_minus_cosine: float
    response_even_l2: float
    response_odd_l2: float
    nonoddness_ratio: float
    even_energy_fraction: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def antithetic_response_geometry(
    plus: Sequence[float],
    minus: Sequence[float],
) -> AntitheticResponseGeometry:
    """Measure rectification without choosing a carrier direction.

    The returned components are

    ``F_even = (F(+delta) + F(-delta)) / 2`` and
    ``F_odd  = (F(+delta) - F(-delta)) / 2``.

    ``nonoddness_ratio`` is ``||F(+delta)+F(-delta)||`` divided by the sum of
    the two response norms.  It is zero exactly for a nonzero antithetic pair
    and one for a same-direction pair of equal magnitude.
    """

    if len(plus) != len(minus) or not plus:
        raise ValueError("plus and minus responses must be nonempty and aligned")
    left = [float(value) for value in plus]
    right = [float(value) for value in minus]
    if any(not math.isfinite(value) for value in left + right):
        raise ValueError("responses must be finite")
    plus_energy = sum(value * value for value in left)
    minus_energy = sum(value * value for value in right)
    cross = sum(a * b for a, b in zip(left, right))
    plus_l2 = math.sqrt(max(0.0, plus_energy))
    minus_l2 = math.sqrt(max(0.0, minus_energy))
    even_energy = max(0.0, (plus_energy + minus_energy + 2.0 * cross) / 4.0)
    odd_energy = max(0.0, (plus_energy + minus_energy - 2.0 * cross) / 4.0)
    response_even_l2 = math.sqrt(even_energy)
    response_odd_l2 = math.sqrt(odd_energy)
    norm_sum = plus_l2 + minus_l2
    component_energy = even_energy + odd_energy
    cosine = cross / (plus_l2 * minus_l2) if plus_l2 and minus_l2 else float("nan")
    return AntitheticResponseGeometry(
        plus_l2=plus_l2,
        minus_l2=minus_l2,
        plus_minus_cosine=cosine,
        response_even_l2=response_even_l2,
        response_odd_l2=response_odd_l2,
        nonoddness_ratio=(2.0 * response_even_l2 / norm_sum if norm_sum else 0.0),
        even_energy_fraction=(even_energy / component_energy if component_energy else 0.0),
    )


@dataclass(frozen=True)
class AdamCoordinateParity:
    """Exact one-coordinate Adam response around a repair gradient."""

    repair_update: float
    plus_residual_update: float
    minus_residual_update: float
    response_even: float
    response_odd: float
    nonoddness_ratio: float
    gradient_sign_crossing: bool

    def to_dict(self) -> dict[str, float | bool]:
        return asdict(self)


def adam_coordinate_parity(
    repair_gradient: float,
    gradient_residual: float,
    first_moment: float,
    second_moment: float,
    *,
    step: int,
    learning_rate: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
) -> AdamCoordinateParity:
    """Compute the exact Adam even/odd response to ``+delta_g/-delta_g``.

    Decoupled weight decay is absent because it is identical in the natural,
    antithetic, and repair arms and cancels from their update differences.
    """

    values = (
        repair_gradient, gradient_residual, first_moment, second_moment,
        learning_rate, beta1, beta2, epsilon,
    )
    if step < 1 or any(not math.isfinite(float(value)) for value in values):
        raise ValueError("Adam state and hyperparameters must be finite; step must be positive")
    if not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0:
        raise ValueError("Adam beta values must be in [0, 1)")
    if second_moment < 0.0 or learning_rate < 0.0 or epsilon <= 0.0:
        raise ValueError("Adam moments/rate must be nonnegative and epsilon positive")

    def update(gradient: float) -> float:
        moment = beta1 * first_moment + (1.0 - beta1) * gradient
        variance = beta2 * second_moment + (1.0 - beta2) * gradient * gradient
        corrected_moment = moment / (1.0 - beta1**step)
        corrected_variance = variance / (1.0 - beta2**step)
        return -learning_rate * corrected_moment / (math.sqrt(corrected_variance) + epsilon)

    repair = update(repair_gradient)
    plus = update(repair_gradient + gradient_residual) - repair
    minus = update(repair_gradient - gradient_residual) - repair
    response_even = (plus + minus) / 2.0
    response_odd = (plus - minus) / 2.0
    return AdamCoordinateParity(
        repair_update=repair,
        plus_residual_update=plus,
        minus_residual_update=minus,
        response_even=response_even,
        response_odd=response_odd,
        nonoddness_ratio=(
            abs(plus + minus) / (abs(plus) + abs(minus))
            if plus or minus else 0.0
        ),
        gradient_sign_crossing=(
            (repair_gradient + gradient_residual)
            * (repair_gradient - gradient_residual)
            <= 0.0
        ),
    )
