"""State-conditioned bias in the moving frame of the reference update.

Absolute parameter directions need not agree across unrelated training states.
For each state ``c`` we instead measure the dimensionless coefficient

    alpha_c = <candidate_update - repair_update, repair_update>
              / ||repair_update||^2.

A stable sign of ``alpha_c`` means the implementation repeatedly expands or
shrinks the same-state reference update.  The coordinate frame is determined
by the repair arm itself, not fitted from candidate errors or trajectory drift.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import random
from typing import Any, Sequence


@dataclass(frozen=True)
class ReferenceRelativeObservation:
    condition_id: str
    error_reference_dot: float
    error_energy: float
    reference_energy: float

    @property
    def coefficient(self) -> float:
        return self.error_reference_dot / self.reference_energy

    @property
    def cosine(self) -> float:
        return self.error_reference_dot / math.sqrt(
            self.error_energy * self.reference_energy
        )

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["coefficient"] = self.coefficient
        value["cosine"] = self.cosine
        return value


@dataclass(frozen=True)
class ReferenceRelativeCertificate:
    condition_count: int
    positive: int
    negative: int
    tied: int
    two_sided_sign_pvalue: float
    mean_coefficient: float
    bootstrap_interval: tuple[float, float]
    mean_cosine: float
    minimum_absolute_mean_coefficient: float
    status: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _two_sided_sign_pvalue(positive: int, negative: int) -> float:
    n = positive + negative
    if n == 0:
        return 1.0
    extreme = max(positive, negative)
    tail = math.fsum(math.comb(n, k) for k in range(extreme, n + 1)) / 2**n
    return min(1.0, 2.0 * tail)


def certify_reference_relative(
    observations: Sequence[ReferenceRelativeObservation],
    *,
    bootstrap_draws: int = 4000,
    seed: int = 20260820,
    minimum_absolute_mean_coefficient: float = 1e-5,
) -> ReferenceRelativeCertificate:
    rows = list(observations)
    if len(rows) < 4 or bootstrap_draws < 100:
        raise ValueError("reference-relative certification needs >=4 conditions")
    for row in rows:
        values = (row.error_reference_dot, row.error_energy, row.reference_energy)
        if any(not math.isfinite(value) for value in values):
            raise ValueError("reference-relative observation is nonfinite")
        if row.error_energy < 0.0 or row.reference_energy <= 0.0:
            raise ValueError("reference and error energies must be valid")
    coefficients = [row.coefficient for row in rows]
    cosines = [row.cosine if row.error_energy > 0.0 else 0.0 for row in rows]
    positive = sum(value > 0.0 for value in coefficients)
    negative = sum(value < 0.0 for value in coefficients)
    tied = len(rows) - positive - negative
    pvalue = _two_sided_sign_pvalue(positive, negative)
    mean = math.fsum(coefficients) / len(coefficients)
    rng = random.Random(seed)
    bootstrap = sorted(
        math.fsum(coefficients[rng.randrange(len(rows))] for _ in rows) / len(rows)
        for _ in range(bootstrap_draws)
    )
    lower = bootstrap[int(0.025 * bootstrap_draws)]
    upper = bootstrap[min(bootstrap_draws - 1, int(0.975 * bootstrap_draws))]
    # The scalar coefficient already maps rotating state-specific update
    # directions into one common, reference-defined coordinate.  Its
    # population mean—not unanimity of individual states—is the estimand.
    # The sign test remains a robustness diagnostic and is intentionally not a
    # hard gate (heterogeneous states can have a nonzero conditional mean).
    stable = lower > 0.0 or upper < 0.0
    material = abs(mean) >= minimum_absolute_mean_coefficient
    if stable and material:
        status = "REFERENCE_RELATIVE_DIRECTIONAL_RISK"
    elif lower <= 0.0 <= upper:
        status = "REFERENCE_RELATIVE_CENTERED_OR_SIGN_CHANGING"
    else:
        status = "REFERENCE_RELATIVE_UNRESOLVED"
    return ReferenceRelativeCertificate(
        condition_count=len(rows),
        positive=positive,
        negative=negative,
        tied=tied,
        two_sided_sign_pvalue=pvalue,
        mean_coefficient=mean,
        bootstrap_interval=(lower, upper),
        mean_cosine=math.fsum(cosines) / len(cosines),
        minimum_absolute_mean_coefficient=minimum_absolute_mean_coefficient,
        status=status,
    )
