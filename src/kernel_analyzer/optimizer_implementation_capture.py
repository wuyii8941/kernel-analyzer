"""Shared capture helpers for optimizer implementations used in LLM training.

The caller chooses and reviews the two optimizers.  This module automates the
matched transition measurement after that choice: original-coordinate
statistics, deterministic replays, compact direction views, and the same
fixed-suite profile used by kernel captures.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .training_bias_profile import matched_training_bias_profile


STAGES = ("GRADIENT_INPUT", "MOMENT1_STATE", "MOMENT2_STATE", "PARAMETER_WRITE")
SKETCH_SEEDS = (5101, 5102, 5103)
SKETCH_DIMENSION = 4096


def tensor_digest(value: torch.Tensor) -> str:
    raw = value.detach().contiguous().cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def original_coordinate_row(effect: torch.Tensor, repair: torch.Tensor) -> dict[str, Any]:
    u = effect.detach().reshape(-1).to(device="cpu", dtype=torch.float64)
    r = repair.detach().reshape(-1).to(device="cpu", dtype=torch.float64)
    if u.shape != r.shape or not u.numel():
        raise ValueError("effect and repair must have the same nonempty coordinates")
    x = float(torch.dot(u, u)); b = float(torch.dot(r, r)); a = float(torch.dot(u, r))
    if not all(math.isfinite(value) for value in (x, b, a)):
        raise ValueError("nonfinite optimizer transition")
    return {
        "effect_energy": x,
        "repair_energy": b,
        "effect_repair_inner_product": a,
        "nonzero_effect_coordinates": int(torch.count_nonzero(u).item()),
        "accumulation_dtype": "float64",
        "coordinate_count": int(u.numel()),
    }


def _count_sketch(value: torch.Tensor, *, seed: int, dimension: int) -> np.ndarray:
    flat = value.detach().reshape(-1).to(device="cpu", dtype=torch.float64)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    buckets = torch.randint(dimension, (flat.numel(),), generator=generator)
    signs = torch.randint(2, (flat.numel(),), generator=generator, dtype=torch.int8)
    signs = signs.to(torch.float64).mul_(2).sub_(1)
    output = torch.zeros(dimension, dtype=torch.float64)
    output.scatter_add_(0, buckets, flat * signs)
    return output.numpy()


def compact_views(value: torch.Tensor) -> dict[str, np.ndarray]:
    flat = value.detach().reshape(-1)
    if flat.numel() <= SKETCH_DIMENSION:
        return {"EXACT": flat.to(device="cpu", dtype=torch.float64).numpy()}
    return {
        f"COUNT_SKETCH_V3_FLOAT64_SEED_{seed}": _count_sketch(
            flat, seed=seed, dimension=SKETCH_DIMENSION
        )
        for seed in SKETCH_SEEDS
    }


class OptimizerImplementationCapture:
    """Collect fixed-suite contrasts without keeping full vectors in JSON."""

    def __init__(self, state_ids: Sequence[str], calibration_count: int = 16) -> None:
        self.state_ids = [str(value) for value in state_ids]
        if len(self.state_ids) != 32 or calibration_count != 16:
            raise ValueError("optimizer implementation v1 requires frozen 16+16 states")
        if len(set(self.state_ids)) != len(self.state_ids):
            raise ValueError("state IDs must be unique")
        self.calibration_count = calibration_count
        self.rows = {stage: [] for stage in STAGES}
        self.views: dict[str, dict[str, dict[str, list[np.ndarray] | int]]] = {
            stage: {} for stage in STAGES
        }

    def append(self, values: Mapping[str, tuple[torch.Tensor, torch.Tensor]]) -> None:
        if set(values) != set(STAGES):
            raise ValueError("all optimizer stages are required")
        if len(self.rows[STAGES[0]]) >= len(self.state_ids):
            raise ValueError("too many transitions")
        for stage in STAGES:
            candidate, repair = values[stage]
            if candidate.shape != repair.shape:
                raise ValueError(f"{stage}: candidate and repair shapes differ")
            effect = candidate.detach().float() - repair.detach().float()
            self.rows[stage].append(original_coordinate_row(effect, repair))
            effect_views = compact_views(effect)
            repair_views = compact_views(repair)
            if set(effect_views) != set(repair_views):
                raise RuntimeError("candidate and repair compact views differ")
            for name in effect_views:
                slot = self.views[stage].setdefault(name, {
                    "effects": [], "repairs": [], "coordinate_count": int(effect.numel()),
                })
                if slot["coordinate_count"] != effect.numel():
                    raise RuntimeError(f"{stage}: coordinate count changed")
                slot["effects"].append(effect_views[name])
                slot["repairs"].append(repair_views[name])

    def finish(self) -> tuple[dict[str, Any], dict[str, Any]]:
        if any(len(self.rows[stage]) != len(self.state_ids) for stage in STAGES):
            raise ValueError("optimizer transition suite is incomplete")
        stages: dict[str, Any] = {}
        for stage_index, stage in enumerate(STAGES):
            stages[stage] = {}
            for view_index, (name, slot) in enumerate(sorted(self.views[stage].items())):
                effects = np.stack(slot["effects"])
                repairs = np.stack(slot["repairs"])
                stages[stage][name] = {
                    "coordinate_count": slot["coordinate_count"],
                    "profile": matched_training_bias_profile(
                        effects, repairs,
                        calibration_indices=range(self.calibration_count),
                        confirmation_indices=range(self.calibration_count, len(self.state_ids)),
                        inference_unit_ids=None,
                        include_joint_gram=True,
                        seed=7300 + 100 * stage_index + view_index,
                    ),
                }
        return self.rows, stages


def fixed_suite_total_rms(rows: Sequence[Mapping[str, float]], start: int = 16) -> float:
    effect = math.fsum(float(row["effect_energy"]) for row in rows[start:])
    repair = math.fsum(float(row["repair_energy"]) for row in rows[start:])
    if repair <= 0.0:
        raise ValueError("repair energy is zero")
    return math.sqrt(effect / repair)
