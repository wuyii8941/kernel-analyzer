#!/usr/bin/env python3
"""Freeze the full-coordinate scalar Triton candidates in all 12 cells."""

from __future__ import annotations

import gzip
import hashlib
import json
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
            full = [
                row for row in oracle["rows"]
                if row["verdict"] == "DIRECTIONAL_BIAS_SCREEN_POSITIVE"
                and row["sampled_coordinates"] == 1
            ]
            if len(full) != 1:
                raise RuntimeError(f"expected one scalar positive: {model} seq{seq_len}")
            screen = full[0]
            bindings = [
                row for row in campaign["rows"] if row["region_id"] == screen["region_id"]
            ]
            if len(bindings) != 1:
                raise RuntimeError(f"Triton campaign binding is non-unique: {model} seq{seq_len}")
            binding = bindings[0]
            if screen["endpoint"] not in binding["output_names"]:
                raise RuntimeError("screen endpoint absent from exact Triton ABI")
            candidates.append({
                "candidate_id": f"{model}_seq{seq_len}_fused_ce_loss_scalar",
                "architecture": model,
                "sequence_length": seq_len,
                "semantic_region": {
                    "forward": "logits -> log_softmax -> NLL mean loss",
                    "backward": "dlogits = (softmax(logits) - one_hot(labels)) / valid_tokens",
                    "selected_endpoint": "scalar loss only",
                    "carrier_warning": (
                        "The scalar loss is normally not saved for backward; a complete case "
                        "requires a changed backward carrier, not merely a changed reported loss."
                    ),
                },
                "exact_generated_call": {
                    "implementation_kind": "TRITON",
                    "region_id": screen["region_id"],
                    "phase": screen["phase"],
                    "symbol": binding["symbol"],
                    "embedded_program_sha256": binding["embedded_program_sha256"],
                    "invocation_index": int(screen["runtime_identity"][1]),
                    "endpoint": screen["endpoint"],
                    "output_names": binding["output_names"],
                    "row_sha256": binding["row_sha256"],
                },
                "full_coordinate_t1": {
                    "states": screen["states"],
                    "coordinates": 1,
                    "cross_state_inner_product_u": screen["cross_state_inner_product_u"],
                    "cluster_bootstrap_95": screen["cluster_bootstrap_95"],
                    "rms_mean": screen["rms_mean_over_state_repeats"],
                    "max_abs": screen["max_abs_over_state_repeats"],
                    "row_sha256": screen["row_sha256"],
                    "passed": True,
                },
                "gates": {
                    "full_coordinate_t1": True,
                    "declared_dtype_cast_nonnull": False,
                    "causal_backward_carrier": False,
                    "complete_fb": False,
                    "accumulation": False,
                },
                "claim": (
                    "INVALID_REFERENCE_ABI" if abi_audit else
                    "FULL_COORDINATE_SCALAR_PENDING_CAUSAL_T2"
                ),
            })
    output = {
        "schema": "kernel-analyzer-triton-scalar-candidate-queue-v1",
        "status": (
            "INVALID_REFERENCE_ABI_DO_NOT_USE_AS_CANDIDATES" if abi_audit else
            "FROZEN_12_CELL_SCALAR_CANDIDATES"
        ),
        "reference_abi_audit": None if not abi_audit else {
            "path": str(abi_audit_path.relative_to(ROOT)),
            "result_sha256": abi_audit["result_sha256"],
        },
        "candidate_count": len(candidates),
        "selection": (
            "Every Triton directional-screen-positive endpoint whose complete tensor has one coordinate."
        ),
        "candidates": candidates,
        "claim_boundary": (
            "A scalar loss difference is not a complete training-bias case unless exact repair "
            "changes the real backward parameter-gradient carrier and later passes coherence."
        ),
    }
    output["result_sha256"] = canonical_hash(output)
    path = ROOT / "results/coverage/triton_scalar_candidate_queue.json"
    path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(path), "candidates": len(candidates)}))


if __name__ == "__main__":
    main()
