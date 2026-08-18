#!/usr/bin/env python3
"""Freeze which models contribute to full-step and mechanism-only claims."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    payload = {
        "schema": "kernel-analyzer-model-scope-v1",
        "status": "FROZEN_BEFORE_PHI_DEEPSEEK_VALUES",
        "models": {
            "qwen3_1p7b": {"scope": "FULL_STEP", "status": "ACTIVE", "role": "PRIMARY"},
            "mamba_130m": {"scope": "FULL_STEP", "status": "ACTIVE", "role": "ARCHITECTURE_DELTA"},
            "phi4_mini_3p8b": {"scope": "FULL_STEP", "status": "PLANNED", "role": "PRIMARY"},
            "deepseek_r1_0528_qwen3_8b": {"scope": "FULL_STEP", "status": "PLANNED", "role": "SCALE_REPLICATION"},
            "deepseek_v4_flash": {"scope": "MECHANISM_REGION", "status": "PLANNED", "role": "LATEST_ARCHITECTURE"},
            "granite_3p1_1b_a400m": {"scope": "PAUSED_OUT_OF_SCOPE", "status": "PAUSED", "role": "RETAINED_EVIDENCE"},
        },
        "rules": {
            "full_step_claim_requires_actual_forward_backward_denominator": True,
            "mechanism_region_never_counts_toward_full_step_coverage": True,
            "paused_evidence_is_retained_but_not_in_active_denominator": True,
            "quantized_inference_cannot_substitute_for_training_reference": True,
        },
    }
    payload["scope_sha256"] = hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    output = ROOT / "results/coverage/model_scope.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "sha256": payload["scope_sha256"]}))


if __name__ == "__main__":
    main()
