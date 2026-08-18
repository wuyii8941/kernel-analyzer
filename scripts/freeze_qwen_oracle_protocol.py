#!/usr/bin/env python3
"""Freeze the candidate-blind 48-state calibration and 96-state heldout plan."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DESIGN = ROOT / "archive/round1_raw/training_semantic_oracle/qwen3_1p7b/state_design.json"
PROOF = ROOT / "results/coverage/qwen_inductor_identity_bridge.json.gz"
OUTPUT = ROOT / "results/coverage/qwen_oracle_protocol.json"


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def compact_states(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "state_id": row["sequence_id"],
            "cluster_id": row["cluster_id"],
            "stratum": row["length_bucket"],
            "length": row["length"],
            "record_sha256": row["record_sha256"],
        }
        for row in rows
    ]


def main() -> None:
    design = json.loads(DESIGN.read_text())
    calibration = [row for row in design["records"] if row["split"] == "calibration"]
    heldout = [row for row in design["records"] if row["split"] == "heldout"]
    if len(calibration) != 48 or len(heldout) != 96:
        raise RuntimeError("frozen state denominator changed")
    expected_calibration = Counter({"seq64": 16, "seq128": 16, "seq256": 16})
    expected_heldout = Counter({"seq64": 32, "seq128": 32, "seq256": 32})
    if Counter(row["length_bucket"] for row in calibration) != expected_calibration:
        raise RuntimeError("calibration strata changed")
    if Counter(row["length_bucket"] for row in heldout) != expected_heldout:
        raise RuntimeError("heldout strata changed")
    if {row["cluster_id"] for row in calibration} & {row["cluster_id"] for row in heldout}:
        raise RuntimeError("calibration and heldout clusters overlap")

    payload = {
        "schema": "kernel-analyzer-qwen-oracle-protocol-v1",
        "status": "FROZEN_CANDIDATE_BLIND",
        "candidate_values_used": False,
        "subject": "Qwen3-1.7B complete loss forward/backward step",
        "state_design_sha256": design["design_sha256"],
        "proof_bridge": str(PROOF.relative_to(ROOT)),
        "calibration_states": compact_states(calibration),
        "heldout_states": compact_states(heldout),
        "configurations": [
            {
                "id": "fp32_eager_strict",
                "dtype": "float32",
                "implementation": "eager",
                "repeats": 1,
                "role": "high_precision_reference",
                "tf32": False,
            },
            {
                "id": "bf16_eager",
                "dtype": "bfloat16",
                "implementation": "eager",
                "repeats": 2,
                "role": "same_precision_reference",
                "tf32": False,
            },
            {
                "id": "bf16_inductor_standard",
                "dtype": "bfloat16",
                "implementation": "inductor_full_step_standard",
                "repeats": 2,
                "role": "primary_candidate",
                "tf32": False,
            },
            {
                "id": "bf16_inductor_preserve_aot_aten",
                "dtype": "bfloat16",
                "implementation": "inductor_full_step",
                "repeats": 2,
                "role": "decomposition_control_candidate",
                "tf32": False,
            },
        ],
        "contrasts": {
            "precision": "bf16_eager - fp32_eager_strict",
            "optimization_standard": "bf16_inductor_standard - bf16_eager",
            "optimization_preserve_aot": "bf16_inductor_preserve_aot_aten - bf16_eager",
            "decomposition_fusion": "bf16_inductor_standard - bf16_inductor_preserve_aot_aten",
            "total_standard": "bf16_inductor_standard - fp32_eager_strict",
        },
        "observation_unit": (
            "one proof-bound semantic region endpoint; fused units inherit shared-region "
            "coverage but never independent causal credit"
        ),
        "endpoints": [
            "forward_or_region_tensor_outputs",
            "complete_local_input_vjp_outputs",
            "parameter_gradient_outputs",
            "loss",
        ],
        "primary_risk": "cross-state directional optimization bias",
        "secondary_fail_closed_risks": ["tail", "nonfinite", "runtime_instability"],
        "margin_rule": (
            "freeze per semantic-signature x shape-stratum x endpoint from calibration "
            "using reference-only BF16-FP32 and BF16-repeat deltas; candidate values forbidden"
        ),
        "heldout_verdicts": [
            "EQUIVALENT",
            "BIASED",
            "TAIL_RISK",
            "NONFINITE_RISK",
            "INVALID_BOUNDARY",
            "UNRESOLVED",
        ],
        "gates": {
            "calibration_candidate_blind": True,
            "heldout_clusters_disjoint": True,
            "all_unresolved_units_stay_in_denominator": True,
            "precision_and_optimization_errors_separated": True,
            "property_induction_allowed": False,
        },
    }
    payload["protocol_sha256"] = digest(payload)
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(OUTPUT.relative_to(ROOT)),
        "status": payload["status"],
        "calibration_states": len(calibration),
        "heldout_states": len(heldout),
        "protocol_sha256": payload["protocol_sha256"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
