"""Gram-only statistics for persistence-property experiments.

The scientific objects in this module are vector sequences.  Production
runners may spool very large vectors temporarily, but only their complete
Gram matrices are needed here.  This keeps lag correlation, sign-flip nulls,
and semantic-orbit decomposition exact without retaining tensor values in a
result artifact.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .seup import _tree_dot


def _gram(value: Any) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("a square Gram matrix with at least two rows is required")
    if not np.isfinite(matrix).all():
        raise ValueError("Gram matrix is nonfinite")
    if not np.allclose(matrix, matrix.T, rtol=1e-8, atol=1e-10):
        raise ValueError("Gram matrix is not symmetric")
    if np.min(np.diag(matrix)) < -1e-10:
        raise ValueError("Gram matrix has negative diagonal energy")
    return (matrix + matrix.T) * 0.5


def _amplification(matrix: np.ndarray) -> float:
    energy = float(np.trace(matrix))
    if energy <= 0.0:
        return 0.0
    return math.sqrt(max(0.0, float(matrix.sum())) / energy)


def path_statistics_from_gram(
    gram: Any,
    *,
    state_ids: Sequence[str] | None = None,
    max_lag: int = 8,
    sign_flip_draws: int = 4000,
    seed: int = 20260820,
) -> dict[str, Any]:
    """Summarize one naturally ordered vector sequence.

    ``coherence_amplification`` equals ``||sum x_t|| / sqrt(sum ||x_t||^2)``.
    The lag curve is order-sensitive; the final amplification alone is not.
    A Rademacher sign-flip null preserves every vector norm and pairwise
    inner-product magnitude while destroying a persistent common sign.
    """

    matrix = _gram(gram)
    steps = matrix.shape[0]
    if state_ids is None:
        ids = [str(index) for index in range(steps)]
    else:
        ids = [str(value) for value in state_ids]
        if len(ids) != steps or len(set(ids)) != steps:
            raise ValueError("state IDs do not match Gram rows")
    if max_lag < 1 or sign_flip_draws < 100:
        raise ValueError("max_lag and sign_flip_draws are too small")

    energy = float(np.trace(matrix))
    resultant2 = max(0.0, float(matrix.sum()))
    path_l2 = float(np.sqrt(np.maximum(np.diag(matrix), 0.0)).sum())
    prefix: dict[str, dict[str, float]] = {}
    for stop in sorted(set([1, 2, 4, 8, 16, steps])):
        if stop > steps:
            continue
        block = matrix[:stop, :stop]
        block_energy = float(np.trace(block))
        block_resultant2 = max(0.0, float(block.sum()))
        prefix[str(stop)] = {
            "resultant_l2": math.sqrt(block_resultant2),
            "diffusive_scale_l2": math.sqrt(max(0.0, block_energy)),
            "coherence_amplification": (
                math.sqrt(block_resultant2 / block_energy) if block_energy > 0.0 else 0.0
            ),
        }

    lags: list[dict[str, float | int]] = []
    diagonal = np.maximum(np.diag(matrix), 0.0)
    for lag in range(1, min(max_lag, steps - 1) + 1):
        numerator = float(sum(matrix[index, index + lag] for index in range(steps - lag)))
        denominator = float(sum(
            math.sqrt(diagonal[index] * diagonal[index + lag])
            for index in range(steps - lag)
        ))
        lags.append({
            "lag": lag,
            "pairs": steps - lag,
            "normalized_correlation": numerator / max(denominator, 1e-30),
            "mean_inner_product": numerator / (steps - lag),
        })

    rng = np.random.default_rng(seed)
    null = np.empty(sign_flip_draws, dtype=np.float64)
    for draw in range(sign_flip_draws):
        signs = rng.choice(np.array([-1.0, 1.0]), size=steps)
        null_resultant2 = max(0.0, float(signs @ matrix @ signs))
        null[draw] = math.sqrt(null_resultant2 / energy) if energy > 0.0 else 0.0
    observed = math.sqrt(resultant2 / energy) if energy > 0.0 else 0.0
    upper95 = float(np.quantile(null, 0.95))
    return {
        "schema": "kernel-analyzer-path-statistics-v1",
        "state_ids": ids,
        "steps": steps,
        "energy": energy,
        "resultant_l2": math.sqrt(resultant2),
        "diffusive_scale_l2": math.sqrt(max(0.0, energy)),
        "path_l2": path_l2,
        "resultant_over_path": math.sqrt(resultant2) / max(path_l2, 1e-30),
        "coherence_amplification": observed,
        "prefix": prefix,
        "lag_correlation": lags,
        "sign_flip_null": {
            "draws": sign_flip_draws,
            "seed": seed,
            "median": float(np.quantile(null, 0.5)),
            "upper_95": upper95,
            "upper_99": float(np.quantile(null, 0.99)),
            "one_sided_p": float((1 + np.count_nonzero(null >= observed)) / (sign_flip_draws + 1)),
        },
        "above_sign_flip_95": bool(observed > upper95),
        "final_amplification_is_order_invariant": True,
        "lag_curve_and_prefix_are_order_sensitive": True,
    }


def semantic_orbit_statistics_from_gram(
    gram: Any,
    *,
    state_ids: Sequence[str],
    variant_ids: Sequence[str],
    default_variant: str,
    max_lag: int = 8,
    sign_flip_draws: int = 4000,
    seed: int = 20260820,
) -> dict[str, Any]:
    """Decompose state-major orbit vectors into mean and default residual.

    Rows must be ordered ``state_0/variant_0, ..., state_0/variant_K, ...``.
    The result is exact algebra on the complete Gram; no direction is fitted.
    """

    matrix = _gram(gram)
    states = len(state_ids)
    variants = len(variant_ids)
    if states < 2 or variants < 2 or matrix.shape[0] != states * variants:
        raise ValueError("semantic-orbit Gram shape is inconsistent")
    if len(set(state_ids)) != states or len(set(variant_ids)) != variants:
        raise ValueError("semantic-orbit IDs must be unique")
    if default_variant not in variant_ids:
        raise ValueError("default semantic-orbit variant is absent")
    default = list(variant_ids).index(default_variant)
    four = matrix.reshape(states, variants, states, variants)

    mean_gram = four.mean(axis=(1, 3))
    default_gram = four[:, default, :, default]
    default_to_mean = four[:, default, :, :].mean(axis=2)
    mean_to_default = four[:, :, :, default].mean(axis=1)
    residual_gram = default_gram - default_to_mean - mean_to_default + mean_gram

    per_state_total = np.array([
        np.mean([four[state, variant, state, variant] for variant in range(variants)])
        for state in range(states)
    ])
    mean_energy = np.maximum(np.diag(mean_gram), 0.0)
    orbit_variance = np.maximum(per_state_total - mean_energy, 0.0)
    total_energy = float(per_state_total.sum())

    kwargs = {
        "state_ids": state_ids,
        "max_lag": max_lag,
        "sign_flip_draws": sign_flip_draws,
        "seed": seed,
    }
    return {
        "schema": "kernel-analyzer-semantic-orbit-statistics-v1",
        "state_ids": list(state_ids),
        "variant_ids": list(variant_ids),
        "default_variant": default_variant,
        "row_order": "STATE_MAJOR_VARIANT_MINOR",
        "orbit_mean": path_statistics_from_gram(mean_gram, **kwargs),
        "default_schedule": path_statistics_from_gram(default_gram, **kwargs),
        "default_minus_orbit_mean": path_statistics_from_gram(residual_gram, **kwargs),
        "orbit_mean_energy_fraction": float(mean_energy.sum() / max(total_energy, 1e-30)),
        "per_state_expected_error_energy": per_state_total.tolist(),
        "per_state_orbit_mean_energy": mean_energy.tolist(),
        "per_state_orbit_variance": orbit_variance.tolist(),
        "claim_boundary": (
            "The orbit mean is a state-conditioned vector.  Its nonzero norm alone "
            "does not imply temporal persistence."
        ),
    }


def transported_orbit_certificate_from_gram(
    gram: Any,
    *,
    state_ids: Sequence[str],
    variant_ids: Sequence[str],
    reference_variant: str,
    sign_flip_draws: int = 4000,
    seed: int = 20260820,
) -> dict[str, Any]:
    """Certify persistence after every orbit residual traversed the same F+B map.

    Rows must be state-major effective-update errors, not endpoint residuals.
    The orbit mean therefore estimates ``M_t E_pi[epsilon_t,pi]`` directly.
    A null-like result is deliberately named ``NO_DETECTABLE_PERSISTENT_MEAN``;
    it is not a universal safety certificate.
    """

    statistics = semantic_orbit_statistics_from_gram(
        gram, state_ids=state_ids, variant_ids=variant_ids,
        default_variant=reference_variant, sign_flip_draws=sign_flip_draws,
        seed=seed,
    )
    persistent = statistics["orbit_mean"]["above_sign_flip_95"]
    return {
        "schema": "kernel-analyzer-transported-orbit-certificate-v1",
        "measurement": "M_t_EXPECTATION_OVER_SEMANTIC_ORBIT",
        "status": (
            "PERSISTENT_TRANSPORTED_CONDITIONAL_MEAN"
            if persistent else "NO_DETECTABLE_PERSISTENT_MEAN_UNDER_PROTOCOL"
        ),
        "statistics": statistics,
        "uses_candidate_orbit_measurements": True,
        "uses_trajectory_or_seup_verdict_as_label": False,
        "safe_verdict_emitted": False,
    }


def aligned_level_statistics_from_gram(
    gram: Any,
    *,
    state_ids: Sequence[str],
    level_ids: Sequence[str],
    max_lag: int = 8,
    sign_flip_draws: int = 4000,
    seed: int = 20260820,
) -> dict[str, Any]:
    """Summarize same-coordinate levels stored state-major in one Gram.

    This is used for the exact ``L/B/D`` recurrence signature.  Cross-level
    resultant cosines are derived from the same Gram and therefore never use
    a post-hoc projection direction.
    """

    matrix = _gram(gram)
    states = len(state_ids)
    levels = len(level_ids)
    if states < 2 or levels < 2 or matrix.shape[0] != states * levels:
        raise ValueError("aligned-level Gram shape is inconsistent")
    if len(set(level_ids)) != levels:
        raise ValueError("level IDs must be unique")
    four = matrix.reshape(states, levels, states, levels)
    result: dict[str, Any] = {
        "schema": "kernel-analyzer-aligned-level-statistics-v1",
        "state_ids": list(state_ids),
        "level_ids": list(level_ids),
        "levels": {},
        "resultant_cosines": {},
    }
    for index, level in enumerate(level_ids):
        result["levels"][level] = path_statistics_from_gram(
            four[:, index, :, index], state_ids=state_ids, max_lag=max_lag,
            sign_flip_draws=sign_flip_draws, seed=seed + index,
        )
    for left in range(levels):
        left_norm2 = max(0.0, float(four[:, left, :, left].sum()))
        for right in range(left + 1, levels):
            right_norm2 = max(0.0, float(four[:, right, :, right].sum()))
            cross = float(four[:, left, :, right].sum())
            cosine = cross / max(math.sqrt(left_norm2 * right_norm2), 1e-30)
            result["resultant_cosines"][f"{level_ids[left]}__{level_ids[right]}"] = cosine
    return result


def five_level_signature(
    *,
    independent_grams: Mapping[str, Any],
    state_ids: Sequence[str],
    aligned_lbd_gram: Any,
    sign_flip_draws: int = 4000,
    seed: int = 20260820,
) -> dict[str, Any]:
    """Build the frozen ``epsilon/g/L/B/D`` signature without mixing spaces."""

    required = {"epsilon", "gradient"}
    if not required.issubset(independent_grams):
        raise ValueError("epsilon and gradient Gram matrices are required")
    independent = {
        name: path_statistics_from_gram(
            value, state_ids=state_ids, sign_flip_draws=sign_flip_draws,
            seed=seed + index,
        )
        for index, (name, value) in enumerate(sorted(independent_grams.items()))
    }
    return {
        "schema": "kernel-analyzer-five-level-persistence-signature-v1",
        "state_ids": list(state_ids),
        "independent_coordinate_spaces": independent,
        "aligned_effective_update_space": aligned_level_statistics_from_gram(
            aligned_lbd_gram, state_ids=state_ids, level_ids=("local", "feedback", "actual"),
            sign_flip_draws=sign_flip_draws, seed=seed + 100,
        ),
        "uses_trajectory_or_seup_verdict_as_input": False,
    }


def _cpu_float_tree(value: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(item, torch.Tensor):
            raise TypeError("every trajectory-vector leaf must be a tensor")
        tensor = item.detach().to(device="cpu", dtype=torch.float32).contiguous()
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError("trajectory vector is nonfinite")
        result[str(key)] = tensor
    return result


def _tree_bytes(value: Mapping[str, Any]) -> int:
    return sum(item.numel() * item.element_size() for item in value.values())


class CompleteTreeGramPath:
    """Form an exact Gram for a short tree-vector trajectory in host RAM."""

    def __init__(self, *, total_steps: int, max_resident_bytes: int) -> None:
        if total_steps < 2 or max_resident_bytes <= 0:
            raise ValueError("invalid path size or memory bound")
        self.total_steps = int(total_steps)
        self.max_resident_bytes = int(max_resident_bytes)
        self.vectors: list[dict[str, Any]] = []
        self.bytes_per_vector: int | None = None
        self.gram = np.zeros((total_steps, total_steps), dtype=np.float64)

    def add(self, value: Mapping[str, Any]) -> None:
        if len(self.vectors) >= self.total_steps:
            raise RuntimeError("complete Gram path is already full")
        tree = _cpu_float_tree(value)
        size = _tree_bytes(tree)
        if self.bytes_per_vector is None:
            self.bytes_per_vector = size
            if size * self.total_steps > self.max_resident_bytes:
                raise MemoryError("complete Gram path exceeds the resident-memory bound")
        elif size != self.bytes_per_vector:
            raise ValueError("trajectory coordinate set changed")
        index = len(self.vectors)
        for previous, other in enumerate(self.vectors):
            dot = _tree_dot(tree, other)
            self.gram[index, previous] = self.gram[previous, index] = dot
        self.gram[index, index] = _tree_dot(tree, tree)
        self.vectors.append(tree)

    def finalize(self, **kwargs: Any) -> dict[str, Any]:
        if len(self.vectors) != self.total_steps:
            raise RuntimeError("complete Gram path is incomplete")
        result = path_statistics_from_gram(self.gram, **kwargs)
        result["resident_bytes"] = int((self.bytes_per_vector or 0) * self.total_steps)
        result["gram_kind"] = "EXACT_COMPLETE_TREE_GRAM"
        self.vectors.clear()
        return result
