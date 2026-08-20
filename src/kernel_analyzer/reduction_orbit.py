"""Semantic-orbit probes for reduction implementations.

These probes never decide whether a training case is persistent.  They expose
whether mathematically equivalent reduction-axis orderings share a directional
implementation residual, which is a cheap candidate predictor for the later
trajectory measurement.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch


def frozen_permutations(length: int, count: int, seed: int) -> list[torch.Tensor]:
    if length < 2 or count < 2:
        raise ValueError("reduction length and orbit size must be at least two")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    values = [torch.arange(length, dtype=torch.int64)]
    values.extend(torch.randperm(length, generator=generator) for _ in range(count - 1))
    return values


def gemm_reduction_orbit(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    permutations: Sequence[torch.Tensor],
    candidate: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    reference: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
) -> dict[str, Any]:
    """Measure residuals over equivalent permutations of a GEMM K axis.

    ``left[..., K] @ right[K, ...]`` is invariant in real arithmetic when the
    same permutation is applied to both operands.  Candidate residuals are
    always measured against the unpermuted high-precision mathematical target,
    so variation across rows is implementation-orbit variation rather than a
    changed objective.
    """

    if left.ndim < 2 or right.ndim != 2 or left.shape[-1] != right.shape[0]:
        raise ValueError("incompatible GEMM operands")
    candidate = candidate or torch.matmul
    reference = reference or torch.matmul
    target = reference(left.float(), right.float()).detach().float()
    residuals: list[torch.Tensor] = []
    ids: list[str] = []
    for index, permutation in enumerate(permutations):
        if permutation.numel() != left.shape[-1]:
            raise ValueError("permutation length differs from reduction extent")
        permutation = permutation.to(device=left.device)
        if not torch.equal(torch.sort(permutation).values.cpu(), torch.arange(left.shape[-1])):
            raise ValueError("reduction orbit member is not a permutation")
        observed = candidate(
            left.index_select(-1, permutation), right.index_select(0, permutation)
        ).detach().float()
        if observed.shape != target.shape or not bool(torch.isfinite(observed).all()):
            raise ValueError("candidate orbit output is invalid")
        residuals.append((observed - target).reshape(-1).cpu())
        ids.append("identity" if index == 0 else f"perm_{index:02d}")
    matrix = torch.stack(residuals).double()
    gram = matrix @ matrix.T
    mean = matrix.mean(dim=0)
    centered = matrix - mean
    total_energy = float(matrix.square().sum().item())
    return {
        "schema": "kernel-analyzer-gemm-reduction-orbit-v1",
        "variant_ids": ids,
        "variants": len(ids),
        "reduction_extent": int(left.shape[-1]),
        "output_coordinates": int(target.numel()),
        "gram": gram.tolist(),
        "orbit_mean_l2": float(torch.linalg.vector_norm(mean).item()),
        "orbit_variance_energy": float(centered.square().sum().item()),
        "orbit_mean_energy_fraction": (
            len(ids) * float(mean.square().sum().item()) / max(total_energy, 1e-30)
        ),
        "default_residual_l2": float(torch.linalg.vector_norm(matrix[0]).item()),
        "mathematical_target_dtype": "FP32",
        "training_bias_or_persistence_verdict": False,
    }
