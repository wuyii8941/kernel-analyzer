"""Reusable fixed-suite capture for reviewed implementation comparisons."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .optimizer_implementation_capture import original_coordinate_row
from .short_persistence import count_sketch_mapping
from .training_bias_profile import matched_training_bias_profile


SKETCH_DIMENSION = 4096
SKETCH_SEEDS = (6201, 6202, 6203)
_PACKED: dict[tuple[int, int], np.ndarray] = {}


def compact_views(value: torch.Tensor) -> dict[str, np.ndarray]:
    flat = value.detach().float().reshape(-1).cpu().numpy()
    coordinates = int(flat.size)
    if coordinates <= SKETCH_DIMENSION:
        return {"EXACT": np.asarray(flat, dtype=np.float64).copy()}
    result = {}
    for seed in SKETCH_SEEDS:
        key = (coordinates, seed)
        packed = _PACKED.get(key)
        if packed is None:
            packed = np.empty(coordinates, dtype=np.uint16)
            for start in range(0, coordinates, 1_000_000):
                stop = min(coordinates, start + 1_000_000)
                indices = np.arange(start, stop, dtype=np.uint64)
                buckets, signs = count_sketch_mapping(
                    indices, projection_dim=SKETCH_DIMENSION, seed=seed
                )
                packed[start:stop] = buckets.astype(np.uint16) | (
                    (signs < 0).astype(np.uint16) << np.uint16(12)
                )
            _PACKED[key] = packed
        sketch = np.zeros(SKETCH_DIMENSION, dtype=np.float64)
        for start in range(0, coordinates, 1_000_000):
            stop = min(coordinates, start + 1_000_000)
            codes = packed[start:stop]
            buckets = (codes & np.uint16(SKETCH_DIMENSION - 1)).astype(np.int64)
            signs = np.where((codes & np.uint16(1 << 12)) == 0, 1.0, -1.0)
            sketch += np.bincount(
                buckets,
                weights=signs * np.asarray(flat[start:stop], dtype=np.float64),
                minlength=SKETCH_DIMENSION,
            )
        result[f"COUNT_SKETCH_V3_FLOAT64_SEED_{seed}"] = sketch
    return result


class FixedSuiteImplementationCapture:
    """Apply one statistical path to any reviewed list of matched stages."""

    def __init__(self, state_ids: Sequence[str], stages: Sequence[str]) -> None:
        self.state_ids = [str(value) for value in state_ids]
        self.stage_names = tuple(stages)
        if len(self.state_ids) != 32 or len(set(self.state_ids)) != 32:
            raise ValueError("fixed-suite v1 requires 32 unique states")
        if not self.stage_names or len(set(self.stage_names)) != len(self.stage_names):
            raise ValueError("unique stages are required")
        self.rows = {stage: [] for stage in self.stage_names}
        self.views: dict[str, dict[str, dict[str, Any]]] = {
            stage: {} for stage in self.stage_names
        }

    def append(self, values: Mapping[str, tuple[torch.Tensor, torch.Tensor]]) -> None:
        if set(values) != set(self.stage_names):
            raise ValueError("all declared stages are required")
        if len(self.rows[self.stage_names[0]]) >= 32:
            raise ValueError("too many states")
        for stage in self.stage_names:
            candidate, reference = values[stage]
            if candidate.shape != reference.shape:
                raise ValueError(f"{stage}: coordinate shapes differ")
            effect = candidate.detach().float() - reference.detach().float()
            self.rows[stage].append(original_coordinate_row(effect, reference))
            effects, repairs = compact_views(effect), compact_views(reference)
            for name in effects:
                slot = self.views[stage].setdefault(name, {
                    "effects": [], "repairs": [], "coordinate_count": effect.numel(),
                })
                if slot["coordinate_count"] != effect.numel():
                    raise ValueError(f"{stage}: coordinate count changed")
                slot["effects"].append(effects[name]); slot["repairs"].append(repairs[name])

    def finish(self) -> tuple[dict[str, Any], dict[str, Any]]:
        if any(len(rows) != 32 for rows in self.rows.values()):
            raise ValueError("fixed suite is incomplete")
        stages = {}
        for stage_index, stage in enumerate(self.stage_names):
            stages[stage] = {}
            for view_index, (name, slot) in enumerate(sorted(self.views[stage].items())):
                effects=np.stack(slot["effects"]); repairs=np.stack(slot["repairs"])
                stages[stage][name] = {
                    "coordinate_count": slot["coordinate_count"],
                    "profile": matched_training_bias_profile(
                        effects, repairs,
                        calibration_indices=range(16), confirmation_indices=range(16, 32),
                        inference_unit_ids=None, include_joint_gram=True,
                        seed=8100 + stage_index * 100 + view_index,
                    ),
                }
        return self.rows, stages
