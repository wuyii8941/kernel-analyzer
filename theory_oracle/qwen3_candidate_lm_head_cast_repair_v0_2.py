#!/usr/bin/env python
"""Corrected original-candidate repair for singleton lm-head cast/transpose."""

from __future__ import annotations
from typing import Any
from qwen3_candidate_kernel_repair_common_v0_1 import run_experiment


def repair(torch: Any, values: tuple[Any, ...]) -> None:
    source, destination = values[:2]
    transposed = destination.t()
    if transposed.shape != source.shape or not transposed.is_contiguous():
        raise RuntimeError("lm-head cast destination transpose does not match source layout")
    with torch.no_grad():
        transposed.copy_(source.to(destination.dtype))


if __name__ == "__main__":
    run_experiment({
        "description": __doc__,
        "schema_version": "forkcert.qwen3-candidate-lm-head-cast-repair.v0.2",
        "valid_status": "VALID_ORIGINAL_CANDIDATE_LM_HEAD_CAST_REPAIR",
        "kernel_family": "triton_poi_fused__to_copy_t_17",
        "expected_family_calls": 1,
        "selected_call_indices": [0],
        "claim_limits": ["singleton lm-head cast/transpose generated family only", "selected state only", "repair-only implementation-relative evidence", "no correctness claim"],
    }, repair)
