"""Small, dependency-light helpers for matched antithetic endpoint probes."""

from __future__ import annotations

import torch


def reflected_endpoint(
    candidate: torch.Tensor, reference: torch.Tensor,
) -> tuple[torch.Tensor, bool, float]:
    """Construct ``reference - (candidate-reference)`` in endpoint dtype."""
    requested = reference.detach().double().mul(2.0) - candidate.detach().double()
    realized = requested.to(candidate.dtype)
    epsilon = candidate.detach().double() - reference.detach().double()
    relative_error = float(torch.linalg.vector_norm(realized.double() - requested)) / max(
        float(torch.linalg.vector_norm(epsilon)), 1e-30
    )
    return realized, bool(torch.equal(realized.double(), requested)), relative_error
