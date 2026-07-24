#!/usr/bin/env python
"""Repair attention-output transpose/cast/clone generated-kernel representatives."""

from __future__ import annotations
from typing import Any
from qwen3_candidate_kernel_repair_common_v0_1 import run_experiment


def repair(torch: Any, values: tuple[Any, ...]) -> None:
    source, destination = values[:2]
    if destination.ndim != 4 or destination.shape[-1] != 128:
        raise RuntimeError("unexpected attention-output destination layout")
    batch, sequence, heads, width = destination.shape
    if source.numel() != batch * sequence * heads * width:
        raise RuntimeError("attention-output source numel mismatch")
    with torch.no_grad():
        logical = source.reshape(batch, heads, sequence, width).to(destination.dtype)
        destination.copy_(logical.permute(0, 2, 1, 3).contiguous())


if __name__ == "__main__":
    run_experiment({"description": __doc__, "schema_version": "forkcert.qwen3-candidate-attention-output-layout-repair.v0.1", "valid_status": "VALID_ORIGINAL_CANDIDATE_ATTENTION_OUTPUT_LAYOUT_REPAIR", "kernel_family": "triton_poi_fused__to_copy_clone_transpose_view_11", "expected_family_calls": 28, "selected_call_indices": [0, 14, 27], "claim_limits": ["attention output layout/cast generated family only", "one state", "repair only", "no correctness claim"]}, repair)
