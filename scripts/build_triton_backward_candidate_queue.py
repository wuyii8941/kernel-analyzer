#!/usr/bin/env python3
"""Select one finite, strongest backward Triton endpoint per model-shape cell."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def load(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> None:
    abi_audit_path = ROOT / "results/coverage/triton_reference_abi_audit.json"
    abi_audit = json.loads(abi_audit_path.read_text()) if abi_audit_path.exists() else None
    candidates = []
    for model in ("qwen", "mamba", "phi4", "deepseek8b"):
        for seq_len in (64, 128, 256):
            release = ROOT / f"results/coverage/runtime_releases/{model}_seq{seq_len}_r1"
            oracle = load(release / "triton_oracle.json.gz")
            campaign = load(release / "campaign.json.gz")
            eligible = [
                row for row in oracle["rows"]
                if row["verdict"] == "DIRECTIONAL_BIAS_SCREEN_POSITIVE"
                and row["phase"] == "BACKWARD"
                and row["sampled_coordinates"] == 64
                and math.isfinite(row["rms_mean_over_state_repeats"])
                and row["max_abs_over_state_repeats"] < 1e30
            ]
            if not eligible:
                continue
            screen = max(eligible, key=lambda row: row["cluster_bootstrap_95"]["lower_95"])
            bindings = [row for row in campaign["rows"] if row["region_id"] == screen["region_id"]]
            if len(bindings) != 1:
                raise RuntimeError(f"non-unique Triton binding: {model} seq{seq_len}")
            binding = bindings[0]
            candidates.append({
                "candidate_id": f"{model}_seq{seq_len}_{screen['region_id'].replace(':', '_')}_{screen['endpoint']}",
                "architecture": model,
                "sequence_length": seq_len,
                "semantic_status": "PENDING_EXACT_FB_BINDING_IF_CAST_AND_CAUSAL_GATES_PASS",
                "exact_generated_call": {
                    "implementation_kind": "TRITON",
                    "region_id": screen["region_id"],
                    "phase": screen["phase"],
                    "symbol": binding["symbol"],
                    "embedded_program_sha256": binding["embedded_program_sha256"],
                    "invocation_index": int(screen["runtime_identity"][1]),
                    "endpoint": screen["endpoint"],
                    "output_names": binding["output_names"],
                    "original_aten": binding["original_aten"],
                    "source_nodes": binding["source_nodes"],
                    "row_sha256": binding["row_sha256"],
                },
                "sampled_t1": {
                    "states": screen["states"],
                    "sampled_coordinates": 64,
                    "cross_state_inner_product_u": screen["cross_state_inner_product_u"],
                    "cluster_bootstrap_95": screen["cluster_bootstrap_95"],
                    "rms_mean": screen["rms_mean_over_state_repeats"],
                    "max_abs": screen["max_abs_over_state_repeats"],
                    "row_sha256": screen["row_sha256"],
                },
                "gates": {
                    "full_coordinate_t1": False,
                    "declared_dtype_cast_nonnull": False,
                    "causal_backward_carrier": False,
                    "exact_fb_binding": False,
                    "complete_fb": False,
                    "accumulation": False,
                },
                "claim": (
                    "INVALID_REFERENCE_ABI" if abi_audit else
                    "SAMPLED_BACKWARD_CANDIDATE_ONLY"
                ),
            })
    output = {
        "schema": "kernel-analyzer-triton-backward-candidate-queue-v1",
        "status": (
            "INVALID_REFERENCE_ABI_DO_NOT_USE_AS_CANDIDATES" if abi_audit else
            "FROZEN_STRONGEST_FINITE_BACKWARD_PER_CELL"
        ),
        "reference_abi_audit": None if not abi_audit else {
            "path": str(abi_audit_path.relative_to(ROOT)),
            "result_sha256": abi_audit["result_sha256"],
        },
        "candidate_count": len(candidates),
        "selection": (
            "Within each cell, maximum held-out lower bound among finite backward Triton "
            "screen positives after excluding FLT_MAX/sentinel-scale endpoints."
        ),
        "candidates": candidates,
        "claim_boundary": (
            "Sampled-coordinate candidates only. Exact declared-dtype intervention and full "
            "F+B binding are required before any case claim."
        ),
    }
    output["result_sha256"] = canonical_hash(output)
    path = ROOT / "results/coverage/triton_backward_candidate_queue.json"
    path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(path), "candidates": len(candidates)}))


if __name__ == "__main__":
    main()
