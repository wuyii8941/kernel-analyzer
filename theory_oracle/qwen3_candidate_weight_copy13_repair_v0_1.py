#!/usr/bin/env python
"""Repair representatives of Qwen3 generated MLP weight-copy family 13."""

from __future__ import annotations
from typing import Any
from qwen3_candidate_kernel_repair_common_v0_1 import run_experiment


def repair(torch: Any, values: tuple[Any, ...]) -> None:
    source, destination = values[:2]
    if source.numel() != destination.numel():
        raise RuntimeError("copy buffers do not have equal numel")
    with torch.no_grad():
        destination.copy_(source.reshape(destination.shape).to(destination.dtype))


if __name__ == "__main__":
    run_experiment({
        "description": __doc__,
        "schema_version": "forkcert.qwen3-candidate-weight-copy13-repair.v0.1",
        "valid_status": "VALID_ORIGINAL_CANDIDATE_WEIGHT_COPY13_REPAIR",
        "kernel_family": "triton_poi_fused__to_copy_13",
        "expected_family_calls": 84,
        "selected_call_indices": [0, 1, 2, 42, 43, 44, 81, 82, 83],
        "claim_limits": ["generated gate/up/down weight-copy family only", "selected state only", "repair-only implementation-relative evidence", "no correctness claim"],
    }, repair)
