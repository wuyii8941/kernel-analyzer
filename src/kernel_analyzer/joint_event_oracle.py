"""Direction-free coherence statistics for transported numerical events.

For one fixed semantic condition, let ``z_e`` be the parameter-space response
caused by numerical event ``e`` (chunk, token, reduction partition, ...).  The
complete event Gram is enough to test whether these responses reinforce one
another:

    C = sum_{e != f} <z_e, z_f> / ((n - 1) sum_e ||z_e||^2).

``C`` is one for identical events, approximately zero for unrelated/orthogonal
events, and negative for cancellation.  It uses no fitted carrier direction.
An exact random-sign test compares the natural all-positive event assembly with
the same event vectors under a sign-symmetric null.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import itertools
import math
import random
from typing import Any, Sequence


@dataclass(frozen=True)
class JointEventCertificate:
    event_count: int
    coordinate_count: int
    total_event_energy: float
    resultant_energy: float
    cross_event_energy: float
    normalized_cross_event_coherence: float
    resultant_over_independent_null: float
    random_sign_pvalue: float
    random_sign_draws: int
    status: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _validate_gram(gram: Sequence[Sequence[float]]) -> list[list[float]]:
    rows = [[float(value) for value in row] for row in gram]
    if len(rows) < 2 or any(len(row) != len(rows) for row in rows):
        raise ValueError("event Gram must be square with at least two events")
    if any(not math.isfinite(value) for row in rows for value in row):
        raise ValueError("event Gram must be finite")
    for i, row in enumerate(rows):
        if row[i] < 0.0:
            raise ValueError("event Gram has negative diagonal energy")
        for j in range(i):
            tolerance = 1e-8 * max(1.0, abs(row[j]), abs(rows[j][i]))
            if abs(row[j] - rows[j][i]) > tolerance:
                raise ValueError("event Gram is not symmetric")
    return rows


def certify_joint_event_gram(
    gram: Sequence[Sequence[float]],
    *,
    coordinate_count: int,
    random_sign_draws: int = 4000,
    seed: int = 20260820,
    minimum_coherence: float = 0.05,
    maximum_pvalue: float = 0.01,
) -> JointEventCertificate:
    rows = _validate_gram(gram)
    if coordinate_count < 1 or random_sign_draws < 100:
        raise ValueError("coordinate count and random-sign draws are insufficient")
    event_count = len(rows)
    trace = math.fsum(rows[index][index] for index in range(event_count))
    if trace <= 0.0:
        return JointEventCertificate(
            event_count, coordinate_count, trace, 0.0, 0.0, 0.0, 0.0,
            1.0, random_sign_draws, "NO_EVENT_ENERGY",
        )
    resultant = math.fsum(math.fsum(row) for row in rows)
    cross = resultant - trace
    coherence = cross / ((event_count - 1) * trace)
    if event_count <= 16:
        sign_population = itertools.product((-1.0, 1.0), repeat=event_count)
        effective_draws = 2**event_count
    else:
        rng = random.Random(seed)
        sign_population = (
            tuple(1.0 if rng.getrandbits(1) else -1.0 for _ in rows)
            for _ in range(random_sign_draws)
        )
        effective_draws = random_sign_draws
    exceed = 0
    for signs in sign_population:
        value = math.fsum(
            signs[i] * signs[j] * rows[i][j]
            for i in range(event_count)
            for j in range(event_count)
        )
        if value >= resultant - 1e-12 * max(trace, abs(resultant), 1.0):
            exceed += 1
    pvalue = (
        exceed / effective_draws
        if event_count <= 16 else (exceed + 1.0) / (effective_draws + 1.0)
    )
    if coherence >= minimum_coherence and pvalue <= maximum_pvalue:
        status = "COHERENT_JOINT_EVENT_RISK"
    elif coherence < 0.0:
        status = "CANCELING_EVENT_STRUCTURE"
    else:
        status = "NO_COHERENT_JOINT_EVENT_RISK"
    return JointEventCertificate(
        event_count=event_count,
        coordinate_count=coordinate_count,
        total_event_energy=trace,
        resultant_energy=resultant,
        cross_event_energy=cross,
        normalized_cross_event_coherence=coherence,
        resultant_over_independent_null=resultant / trace,
        random_sign_pvalue=pvalue,
        random_sign_draws=effective_draws,
        status=status,
    )


def gram_from_event_vectors(
    vectors: Sequence[Sequence[float]],
) -> list[list[float]]:
    rows = [[float(value) for value in vector] for vector in vectors]
    if len(rows) < 2 or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("event vectors must be a nonempty aligned matrix")
    if any(not math.isfinite(value) for row in rows for value in row):
        raise ValueError("event vectors must be finite")
    return [
        [math.fsum(a * b for a, b in zip(left, right)) for right in rows]
        for left in rows
    ]
