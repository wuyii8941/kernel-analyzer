#!/usr/bin/env python
"""Repair representatives of Qwen3 generated q/o weight-copy family 1."""

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
        "schema_version": "forkcert.qwen3-candidate-weight-copy1-repair.v0.1",
        "valid_status": "VALID_ORIGINAL_CANDIDATE_WEIGHT_COPY1_REPAIR",
        "kernel_family": "triton_poi_fused__to_copy_1",
        "expected_family_calls": 56,
        "selected_call_indices": [0, 1, 28, 29, 54, 55],
        "claim_limits": ["generated q/o weight-copy family only", "selected state only", "repair-only implementation-relative evidence", "no correctness claim"],
    }, repair)
