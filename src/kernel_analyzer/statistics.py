"""Candidate-independent directional statistics for complete carriers."""

from __future__ import annotations

import math
import random
from typing import Any, Dict, Sequence


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("carrier vectors have different dimensions")
    return math.fsum(float(a) * float(b) for a, b in zip(left, right))


def cross_inner_u(vectors: Sequence[Sequence[float]]) -> float:
    if len(vectors) < 2:
        raise ValueError("cross-state U statistic requires at least two states")
    values = [
        _dot(vectors[left], vectors[right])
        for left in range(len(vectors))
        for right in range(left + 1, len(vectors))
    ]
    return math.fsum(values) / len(values)


def coherence_certificate(
    vectors: Sequence[Sequence[float]], *, alpha: float = 0.05,
    bootstrap_samples: int = 2000, seed: int = 0,
) -> Dict[str, Any]:
    """Cluster bootstrap over states with a fixed complete coordinate set."""
    if len(vectors) < 4:
        return {
            "status": "UNRESOLVED_INSUFFICIENT_CONFIRMATION_STATES",
            "state_count": len(vectors),
            "required_states": 4,
        }
    dimension = len(vectors[0])
    if dimension == 0 or any(len(row) != dimension for row in vectors):
        return {"status": "UNRESOLVED_INVALID_COMPLETE_CARRIER"}
    if not all(math.isfinite(float(value)) for row in vectors for value in row):
        return {"status": "UNRESOLVED_NONFINITE_CARRIER"}
    observed = cross_inner_u(vectors)
    generator = random.Random(seed)
    estimates = []
    for _ in range(bootstrap_samples):
        indices = [generator.randrange(len(vectors)) for _ in vectors]
        pair_values = [
            _dot(vectors[indices[left]], vectors[indices[right]])
            for left in range(len(indices))
            for right in range(left + 1, len(indices))
            if indices[left] != indices[right]
        ]
        if pair_values:
            estimates.append(math.fsum(pair_values) / len(pair_values))
    if not estimates:
        return {"status": "UNRESOLVED_BOOTSTRAP_DEGENERATE"}
    estimates.sort()
    lower_index = max(0, min(len(estimates) - 1, int(alpha * len(estimates))))
    lower = estimates[lower_index]
    return {
        "status": "PASS" if lower > 0.0 else "FAIL_CAUSAL_NONCOHERENT",
        "state_count": len(vectors),
        "coordinate_count": dimension,
        "u_statistic": observed,
        "cluster_bootstrap_lower": lower,
        "alpha": alpha,
        "bootstrap_samples": bootstrap_samples,
        "fixed_complete_coordinates": True,
    }


def coherence_certificate_from_gram(
    gram: Sequence[Sequence[float]], *, coordinate_count: int,
    alpha: float = 0.05, bootstrap_samples: int = 2000, seed: int = 0,
) -> Dict[str, Any]:
    """Equivalent complete-carrier certificate using a streamed Gram matrix."""
    count = len(gram)
    if count < 4:
        return {"status": "UNRESOLVED_INSUFFICIENT_CONFIRMATION_STATES",
                "state_count": count, "required_states": 4}
    if coordinate_count < 1 or any(len(row) != count for row in gram):
        return {"status": "UNRESOLVED_INVALID_COMPLETE_CARRIER_GRAM"}
    if not all(math.isfinite(float(value)) for row in gram for value in row):
        return {"status": "UNRESOLVED_NONFINITE_CARRIER"}
    if any(abs(float(gram[i][j]) - float(gram[j][i])) > 1e-6 * max(
            1.0, abs(float(gram[i][j])), abs(float(gram[j][i])))
           for i in range(count) for j in range(count)):
        return {"status": "UNRESOLVED_NONSYMMETRIC_GRAM"}
    pairs = [float(gram[i][j]) for i in range(count) for j in range(i + 1, count)]
    observed = math.fsum(pairs) / len(pairs)
    generator = random.Random(seed)
    estimates = []
    for _ in range(bootstrap_samples):
        indices = [generator.randrange(count) for _ in range(count)]
        values = [
            float(gram[indices[i]][indices[j]])
            for i in range(count) for j in range(i + 1, count)
            if indices[i] != indices[j]
        ]
        if values:
            estimates.append(math.fsum(values) / len(values))
    if not estimates:
        return {"status": "UNRESOLVED_BOOTSTRAP_DEGENERATE"}
    estimates.sort()
    lower = estimates[max(0, min(len(estimates) - 1, int(alpha * len(estimates))))]
    return {
        "status": "PASS" if lower > 0.0 else "FAIL_CAUSAL_NONCOHERENT",
        "state_count": count,
        "coordinate_count": coordinate_count,
        "u_statistic": observed,
        "cluster_bootstrap_lower": lower,
        "alpha": alpha,
        "bootstrap_samples": bootstrap_samples,
        "fixed_complete_coordinates": True,
        "streamed_complete_gram": True,
    }
