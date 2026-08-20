"""Ordered-vector persistence summaries for causal training trajectories.

The accumulator deliberately separates a basis-free ordered resultant from a
calibration-frozen rank-one projection.  The former asks whether realized
effective updates cancel along the measured trajectory; the latter is only an
interpretability view and never replaces the basis-free measurement.
"""

from __future__ import annotations

import math
from typing import Any

import torch


def _norm(value: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(value).item())


def cosine(left: torch.Tensor, right: torch.Tensor) -> float | None:
    left = left.detach().reshape(-1)
    right = right.detach().reshape(-1)
    if left.numel() != right.numel():
        raise ValueError("cosine coordinate counts differ")
    denominator = _norm(left) * _norm(right)
    if denominator == 0.0:
        return None
    return float(torch.sum(left.double() * right.double()).item() / denominator)


class OrderedVectorPath:
    """Stream one ordered vector path without retaining per-step vectors."""

    def __init__(self, *, total_steps: int, calibration_steps: int = 8) -> None:
        if total_steps < 2 or not 1 <= calibration_steps < total_steps:
            raise ValueError("require 1 <= calibration_steps < total_steps")
        self.total_steps = int(total_steps)
        self.calibration_steps = int(calibration_steps)
        self.count = 0
        self.coordinate_count: int | None = None
        self.total: torch.Tensor | None = None
        self.calibration_total: torch.Tensor | None = None
        self.calibration_odd: torch.Tensor | None = None
        self.calibration_even: torch.Tensor | None = None
        self.basis: torch.Tensor | None = None
        self.path_l2 = 0.0
        self.energy = 0.0
        self.eval_projection_sum = 0.0
        self.eval_projection_abs_sum = 0.0
        self.eval_projection_energy = 0.0
        self.eval_vector_energy = 0.0
        self.prefix: dict[str, dict[str, float]] = {}

    def add(self, value: torch.Tensor) -> dict[str, float | None]:
        if self.count >= self.total_steps:
            raise RuntimeError("ordered path is already complete")
        flat = value.detach().float().reshape(-1)
        if not bool(torch.isfinite(flat).all()):
            raise ValueError("trajectory vector is nonfinite")
        if self.coordinate_count is None:
            self.coordinate_count = flat.numel()
            self.total = torch.zeros_like(flat)
            self.calibration_total = torch.zeros_like(flat)
            self.calibration_odd = torch.zeros_like(flat)
            self.calibration_even = torch.zeros_like(flat)
        elif flat.numel() != self.coordinate_count:
            raise ValueError("trajectory coordinate count changed")

        step = self.count + 1
        norm = _norm(flat)
        assert self.total is not None
        self.total.add_(flat)
        self.path_l2 += norm
        self.energy += norm * norm

        projection: float | None = None
        if step <= self.calibration_steps:
            assert self.calibration_total is not None
            assert self.calibration_odd is not None and self.calibration_even is not None
            self.calibration_total.add_(flat)
            (self.calibration_odd if step % 2 else self.calibration_even).add_(flat)
            if step == self.calibration_steps:
                calibration_norm = _norm(self.calibration_total)
                if calibration_norm > 0.0:
                    self.basis = self.calibration_total / calibration_norm
        elif self.basis is not None:
            projection = float(torch.sum(flat.double() * self.basis.double()).item())
            self.eval_projection_sum += projection
            self.eval_projection_abs_sum += abs(projection)
            self.eval_projection_energy += projection * projection
            self.eval_vector_energy += norm * norm

        self.count = step
        if step in {1, self.calibration_steps, 16, 24, 32, self.total_steps}:
            resultant = _norm(self.total)
            self.prefix[str(step)] = {
                "resultant_l2": resultant,
                "path_l2": self.path_l2,
                "diffusive_scale_l2": math.sqrt(self.energy),
                "resultant_over_path": resultant / max(self.path_l2, 1e-30),
                "coherence_amplification": resultant / max(math.sqrt(self.energy), 1e-30),
            }
        return {"l2": norm, "frozen_carrier_projection": projection}

    def finalize(self) -> dict[str, Any]:
        if self.count != self.total_steps or self.total is None:
            raise RuntimeError("ordered path is incomplete")
        resultant = _norm(self.total)
        calibration_crossfit = None
        if self.calibration_odd is not None and self.calibration_even is not None:
            calibration_crossfit = cosine(self.calibration_odd, self.calibration_even)
        return {
            "steps": self.total_steps,
            "calibration_steps": self.calibration_steps,
            "coordinate_count": self.coordinate_count,
            "resultant_l2": resultant,
            "path_l2": self.path_l2,
            "diffusive_scale_l2": math.sqrt(self.energy),
            "resultant_over_path": resultant / max(self.path_l2, 1e-30),
            "coherence_amplification": resultant / max(math.sqrt(self.energy), 1e-30),
            "calibration_odd_even_cosine": calibration_crossfit,
            "calibration_carrier_available": self.basis is not None,
            "evaluation_signed_persistence": (
                abs(self.eval_projection_sum) / max(self.eval_projection_abs_sum, 1e-30)
                if self.basis is not None else None
            ),
            "evaluation_carrier_energy_capture": (
                self.eval_projection_energy / max(self.eval_vector_energy, 1e-30)
                if self.basis is not None else None
            ),
            "evaluation_projection_sum": (
                self.eval_projection_sum if self.basis is not None else None
            ),
            "prefix": self.prefix,
        }
