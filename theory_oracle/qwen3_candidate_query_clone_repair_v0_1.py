#!/usr/bin/env python
"""Repair query-layout clone generated-kernel representatives."""

from __future__ import annotations
from typing import Any
from qwen3_candidate_kernel_repair_common_v0_1 import run_experiment


def repair(torch: Any, values: tuple[Any, ...]) -> None:
    source, destination = values[:2]
    if source.shape != destination.shape:
        raise RuntimeError("query clone source/destination shapes differ")
    with torch.no_grad():
        destination.copy_(source)


if __name__ == "__main__":
    run_experiment({"description": __doc__, "schema_version": "forkcert.qwen3-candidate-query-clone-repair.v0.1", "valid_status": "VALID_ORIGINAL_CANDIDATE_QUERY_CLONE_REPAIR", "kernel_family": "triton_poi_fused_clone_6", "expected_family_calls": 28, "selected_call_indices": [0, 14, 27], "claim_limits": ["query clone generated family only", "one state", "repair only", "no correctness claim"]}, repair)
