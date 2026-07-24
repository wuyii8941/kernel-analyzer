#!/usr/bin/env python
"""Revision 4: two source-graph repairs for remaining singleton backward families."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import theory_oracle.qwen3_backward_singleton_repair_v0_1 as v1


def repair_final_norm_backward(torch: Any, values: tuple[Any, ...]) -> None:
    tail_gradient, rms_weight, hidden, attention_gradient, mlp_gradient, rsqrt = values[:6]
    float_destination, half_destination = values[6:8]
    padded = torch.zeros_like(hidden)
    tail = tail_gradient.reshape(4, 129, 1024).float()
    padded[:, -tail.shape[1] :, :].copy_(tail)
    weighted = padded * rms_weight
    residual = (
        hidden
        + attention_gradient.reshape_as(hidden).float()
        + mlp_gradient.reshape_as(hidden).float()
    )
    inner = (weighted * residual).sum(dim=-1, keepdim=True)
    gradient = weighted * rsqrt + (
        (inner * -0.5) * rsqrt.pow(3) / 1024.0
    ) * (residual.pow(1.0) * 2.0)
    float_destination.copy_(gradient)
    half_destination.reshape_as(gradient).copy_(gradient.to(half_destination.dtype))


def repair_embedding_norm_backward_prep(torch: Any, values: tuple[Any, ...]) -> None:
    gradient_destination = values[0]
    mm0, mm1, mm2, rms_weight, embedding, embedding_square_sum, token_ids = values[1:8]
    index_destination = values[8]
    combined = (
        mm0.reshape_as(embedding).float()
        + mm1.reshape_as(embedding).float()
        + mm2.reshape_as(embedding).float()
    )
    weighted = combined * rms_weight
    rsqrt = torch.rsqrt(embedding_square_sum / 1024.0 + 1e-6)
    inner = (weighted * embedding).sum(dim=-1, keepdim=True)
    gradient = gradient_destination + weighted * rsqrt + (
        (inner * -0.5) * rsqrt.pow(3) / 1024.0
    ) * (embedding.pow(1.0) * 2.0)
    valid = (token_ids >= 0) & (token_ids < 151936) & (token_ids != 151643)
    gradient_destination.copy_(torch.where(valid.unsqueeze(-1), gradient, 0.0))
    index_destination.copy_(token_ids.clamp(min=-151936, max=151935))


TREATMENTS = {
    "final_norm_backward": {
        "family": "triton_per_fused__to_copy__unsafe_view_add_div_expand_mul_pow_slice_backward_sum_view_16",
        "repair": repair_final_norm_backward,
        "semantic_boundary": "tail slice-backward plus final RMSNorm derivative in eager FP32/FP16 stages",
    },
    "embedding_norm_backward_prep": {
        "family": "triton_per_fused__to_copy_add_div_embedding_dense_backward_expand_mean_mul_pow_rsqrt_sum_view_37",
        "repair": repair_embedding_norm_backward_prep,
        "semantic_boundary": "embedding RMSNorm derivative and masked embedding-gradient preparation before accumulation",
    },
}


def main() -> None:
    if "--treatment" not in sys.argv:
        raise ValueError("--treatment is required")
    treatment = sys.argv[sys.argv.index("--treatment") + 1]
    if treatment not in TREATMENTS:
        raise ValueError(f"unsupported revision-4 treatment {treatment!r}")
    if "--out-dir" not in sys.argv:
        raise ValueError("--out-dir is required")
    out_dir = Path(sys.argv[sys.argv.index("--out-dir") + 1]).resolve()
    v1.TREATMENTS.update(TREATMENTS)
    v1.main()
    result_path = out_dir / "result.json"
    result = json.loads(result_path.read_text())
    result["schema_version"] = (
        "forkcert.qwen3-natural-transition-with-backward-singleton-repair.v0.4"
    )
    result["backward_singleton_repair"]["repair_revision"] = "v0.4-source-graph"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
