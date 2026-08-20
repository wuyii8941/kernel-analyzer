"""Fail-closed infrastructure for held-out all-operator TCMP campaigns.

The module deliberately separates the invocation denominator from scientific
case counting.  Every executed invocation receives a disposition, while only
closed, non-overlapping F+B proof units may receive causal credit.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from pathlib import Path
import shutil
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


CAPABILITIES = {
    "TCMP_ORBIT_READY",
    "EXACT_REPAIR_ONLY",
    "FEEDBACK_ONLY",
    "NO_PARAMETER_REACH",
    "NONDIFFERENTIABLE_OR_DISCRETE",
    "UNRESOLVED_BOUNDARY",
}

DISPOSITIONS = {
    "NO_DETECTABLE_PERSISTENCE_UNDER_SCREEN",
    "SCREEN_HIT",
    "TCMP_SUPPORTED",
    "TCMP_COUNTEREXAMPLE",
    "FEEDBACK_SUSTAINED",
    "OPTIMIZER_RECTIFIED",
    "OUTSIDE_TCMP_MEASUREMENT_DOMAIN",
    "NO_PARAMETER_REACH",
    "NONDIFFERENTIABLE_OR_DISCRETE",
    "UNRESOLVED_PROOF",
    "UNRESOLVED_BOUNDARY",
    "RESOURCE_INELIGIBLE",
}

UNRESOLVED_DISPOSITIONS = {
    "UNRESOLVED_PROOF", "UNRESOLVED_BOUNDARY", "RESOURCE_INELIGIBLE"
}


@dataclass(frozen=True)
class ModelCellSpec:
    cell_id: str
    model_id: str
    modality: str
    sequence_length: int
    image_policy: str | None = None
    phase: str = "EXPANSION"

    def __post_init__(self) -> None:
        if not self.cell_id or not self.model_id:
            raise ValueError("model cell IDs must be non-empty")
        if self.modality not in {"TEXT", "IMAGE_TEXT"}:
            raise ValueError("unsupported modality")
        if self.sequence_length < 2:
            raise ValueError("sequence length is too small")
        if self.modality == "IMAGE_TEXT" and not self.image_policy:
            raise ValueError("image-text cells require a frozen image policy")
        if self.phase not in {"EXPANSION", "HELDOUT_CONFIRMATION"}:
            raise ValueError("unsupported campaign phase")


@dataclass(frozen=True)
class ModelCampaignSpec:
    campaign_id: str
    development_parent_commit: str
    cells: Sequence[ModelCellSpec]
    screening_steps: int = 8
    confirmation_steps: int = 16
    orbit_variants: int = 8
    screen_fdr: float = 0.10
    confirmation_fwer: float = 0.05
    max_vram_bytes: int = 44 * 1024**3
    min_free_disk_bytes: int = 500 * 1024**3
    temp_budget_bytes: int = 150 * 1024**3
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.campaign_id or not self.development_parent_commit:
            raise ValueError("campaign identity is incomplete")
        ids = [cell.cell_id for cell in self.cells]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("campaign cell IDs must be non-empty and unique")
        if (self.screening_steps, self.confirmation_steps, self.orbit_variants) != (8, 16, 8):
            raise ValueError("tcmp_allop_v1 freezes 8/16 states and eight orbit variants")
        if not (0.0 < self.screen_fdr < 1.0 and 0.0 < self.confirmation_fwer < 1.0):
            raise ValueError("invalid multiplicity thresholds")

    def as_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "cells": [asdict(cell) for cell in self.cells],
            "schema": "kernel-analyzer-tcmp-model-campaign-v1",
        }


@dataclass(frozen=True)
class TCMPDisposition:
    invocation_id: str
    proof_unit_id: str
    capability: str
    disposition: str
    causal_credit_unit_id: str | None = None
    evidence: Mapping[str, Any] = field(default_factory=dict)
    reason: str = ""

    def __post_init__(self) -> None:
        if not self.invocation_id or not self.proof_unit_id:
            raise ValueError("disposition lacks invocation/proof identity")
        if self.capability not in CAPABILITIES:
            raise ValueError("unsupported TCMP capability")
        if self.disposition not in DISPOSITIONS:
            raise ValueError("unsupported TCMP disposition")
        if self.disposition == "TCMP_SUPPORTED" and self.capability != "TCMP_ORBIT_READY":
            raise ValueError("TCMP support requires a valid semantic orbit")
        if self.disposition == "TCMP_COUNTEREXAMPLE" and self.capability != "TCMP_ORBIT_READY":
            raise ValueError("an inapplicable unit cannot falsify TCMP")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def resource_preflight(root: Path, *, min_free_bytes: int, temp_budget_bytes: int) -> dict[str, Any]:
    usage = shutil.disk_usage(root)
    status = "READY" if usage.free - temp_budget_bytes >= min_free_bytes else "BLOCKED_DISK_BUDGET"
    return {
        "status": status,
        "root": str(root.resolve()),
        "free_bytes": usage.free,
        "min_free_bytes": min_free_bytes,
        "temp_budget_bytes": temp_budget_bytes,
    }


def _gram(value: Any) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Gram matrix must be square")
    if not 2 <= matrix.shape[0] <= 16:
        raise ValueError("exact v1 sign-flip inference supports 2--16 states")
    if not np.isfinite(matrix).all() or not np.allclose(matrix, matrix.T, rtol=1e-8, atol=1e-10):
        raise ValueError("Gram matrix is nonfinite or asymmetric")
    matrix = (matrix + matrix.T) * 0.5
    if np.min(np.diag(matrix)) < -1e-10:
        raise ValueError("Gram matrix has negative diagonal energy")
    return matrix


def exact_sign_flip_statistics(gram: Any) -> dict[str, Any]:
    """Return exact Rademacher inference for the TCMP path amplification."""

    matrix = _gram(gram)
    states = matrix.shape[0]
    energy = float(np.trace(matrix))
    observed2 = max(0.0, float(matrix.sum()))
    observed = math.sqrt(observed2 / energy) if energy > 0.0 else 0.0
    values = np.empty(1 << states, dtype=np.float64)
    bit_positions = np.arange(states, dtype=np.uint64)
    cursor = 0
    for start in range(0, 1 << states, 4096):
        stop = min(1 << states, start + 4096)
        integers = np.arange(start, stop, dtype=np.uint64)[:, None]
        signs = 1.0 - 2.0 * ((integers >> bit_positions) & 1).astype(np.float64)
        quadratic = np.einsum("bi,ij,bj->b", signs, matrix, signs, optimize=True)
        block = np.sqrt(np.maximum(quadratic, 0.0) / energy) if energy > 0.0 else np.zeros(stop-start)
        values[cursor:cursor + len(block)] = block
        cursor += len(block)
    return {
        "schema": "kernel-analyzer-exact-sign-flip-path-v1",
        "states": states,
        "draws": int(len(values)),
        "coherence_amplification": observed,
        "null_median": float(np.quantile(values, 0.5)),
        "null_upper_95": float(np.quantile(values, 0.95)),
        "null_upper_99": float(np.quantile(values, 0.99)),
        "one_sided_p": float(np.count_nonzero(values >= observed - 1e-14) / len(values)),
        "above_null_95": bool(observed > float(np.quantile(values, 0.95))),
    }


def benjamini_hochberg(p_values: Mapping[str, float], q: float = 0.10) -> dict[str, bool]:
    if not 0.0 < q < 1.0:
        raise ValueError("invalid FDR")
    ordered = sorted((float(value), key) for key, value in p_values.items())
    if any(not 0.0 <= value <= 1.0 for value, _ in ordered):
        raise ValueError("p-values must lie in [0, 1]")
    cutoff = -1
    for rank, (value, _) in enumerate(ordered, start=1):
        if value <= q * rank / max(1, len(ordered)):
            cutoff = rank
    selected = {key for _, key in ordered[:cutoff]} if cutoff >= 0 else set()
    return {key: key in selected for key in p_values}


def holm_rejections(p_values: Mapping[str, float], alpha: float = 0.05) -> dict[str, bool]:
    if not 0.0 < alpha < 1.0:
        raise ValueError("invalid family-wise alpha")
    ordered = sorted((float(value), key) for key, value in p_values.items())
    rejected: set[str] = set()
    for index, (value, key) in enumerate(ordered):
        if not 0.0 <= value <= 1.0:
            raise ValueError("p-values must lie in [0, 1]")
        if value > alpha / (len(ordered) - index):
            break
        rejected.add(key)
    return {key: key in rejected for key in p_values}


def audit_denominator(
    invocation_ids: Sequence[str], dispositions: Iterable[TCMPDisposition]
) -> dict[str, Any]:
    if len(invocation_ids) != len(set(invocation_ids)):
        raise ValueError("invocation denominator contains duplicate IDs")
    rows = list(dispositions)
    by_id = {row.invocation_id: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("multiple dispositions exist for one invocation")
    expected, actual = set(invocation_ids), set(by_id)
    missing, extra = sorted(expected - actual), sorted(actual - expected)
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.disposition] = counts.get(row.disposition, 0) + 1
    unresolved = sum(counts.get(name, 0) for name in UNRESOLVED_DISPOSITIONS)
    complete = not missing and not extra
    return {
        "schema": "kernel-analyzer-tcmp-denominator-audit-v1",
        "status": "COMPLETE_DENOMINATOR_DISPOSITION" if complete else "INCOMPLETE_DENOMINATOR",
        "invocations": len(invocation_ids),
        "dispositions": len(rows),
        "missing_invocation_ids": missing,
        "extra_invocation_ids": extra,
        "counts": dict(sorted(counts.items())),
        "unresolved_count": unresolved,
        "universal_claim_eligible": bool(complete and unresolved == 0),
    }
