#!/usr/bin/env python3
"""Freeze the held-out TCMP all-operator campaign before model measurements."""

from __future__ import annotations

import json
from pathlib import Path

from kernel_analyzer.tcmp_campaign import ModelCampaignSpec, ModelCellSpec, resource_preflight


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/property/tcmp_allop_v1"
PARENT = "50898c8321c1543c072ddd58b725bfc70026efa9"


def main() -> None:
    cells = [
        ModelCellSpec("gemma3_4b_text128", "google/gemma-3-4b-pt", "TEXT", 128),
        ModelCellSpec("gemma3_4b_text512", "google/gemma-3-4b-pt", "TEXT", 512),
        ModelCellSpec(
            "gemma3_4b_image_text128", "google/gemma-3-4b-pt", "IMAGE_TEXT", 128,
            image_policy="OFFICIAL_PROCESSOR_DEFAULT_RESOLUTION_FROZEN_AT_PREFLIGHT",
        ),
        ModelCellSpec("olmoe_1b7b_text128", "allenai/OLMoE-1B-7B-0125", "TEXT", 128),
        ModelCellSpec("olmoe_1b7b_text512", "allenai/OLMoE-1B-7B-0125", "TEXT", 512),
        ModelCellSpec(
            "llama32_3b_text128", "meta-llama/Llama-3.2-3B", "TEXT", 128,
            phase="HELDOUT_CONFIRMATION",
        ),
        ModelCellSpec(
            "llama32_3b_text512", "meta-llama/Llama-3.2-3B", "TEXT", 512,
            phase="HELDOUT_CONFIRMATION",
        ),
        ModelCellSpec(
            "ministral3_3b_text128", "mistralai/Ministral-3-3B-Base-2512", "TEXT", 128,
            phase="HELDOUT_CONFIRMATION",
        ),
        ModelCellSpec(
            "ministral3_3b_text512", "mistralai/Ministral-3-3B-Base-2512", "TEXT", 512,
            phase="HELDOUT_CONFIRMATION",
        ),
        ModelCellSpec(
            "ministral3_3b_image_text128", "mistralai/Ministral-3-3B-Base-2512",
            "IMAGE_TEXT", 128,
            image_policy="OFFICIAL_PROCESSOR_DEFAULT_RESOLUTION_FROZEN_AT_PREFLIGHT",
            phase="HELDOUT_CONFIRMATION",
        ),
    ]
    campaign = ModelCampaignSpec(
        campaign_id="tcmp_allop_v1",
        development_parent_commit=PARENT,
        cells=cells,
        metadata={
            "development_models_excluded_from_new_generalization_credit": [
                "Qwen3-1.7B", "Phi-4-mini-instruct", "DeepSeek-R1-0528-Qwen3-8B",
                "Mamba-130m", "Qwen3-VL-2B", "Liger-Qwen3-1.7B",
            ],
            "primary_candidate_contrast": "BF16_OPTIMIZED_MINUS_BF16_EAGER",
            "precision_decomposition": "BF16_EAGER_MINUS_FP32_EAGER",
            "long_shape_resource_fallback": "SEQ512_TO_SEQ256_BEFORE_SCIENTIFIC_VALUES_ONLY",
        },
    )
    protocol = {
        "schema": "kernel-analyzer-tcmp-allop-protocol-v1",
        "protocol_id": "tcmp_allop_v1",
        "status": "FROZEN_BEFORE_NEW_MODEL_MEASUREMENT",
        "hypothesis": (
            "Every source-persistent implementation bias in the declared TCMP domain "
            "contains a temporally non-canceling transported conditional mean M_t m_t."
        ),
        "primary_statistic": "norm(sum_t x_t)/sqrt(sum_t norm(x_t)^2)",
        "inference": {
            "screen": "exact_sign_flip_plus_BH_FDR_q_0.10",
            "confirmation": "exact_sign_flip_plus_Holm_FWER_0.05",
            "screening_states": 8,
            "confirmation_states": 16,
            "semantic_orbit_variants": 8,
        },
        "hard_rules": {
            "every_executed_invocation_enters_denominator": True,
            "causal_credit_only_for_nonoverlapping_closed_fb_units": True,
            "not_applicable_is_not_negative": True,
            "missing_orbit_cannot_falsify_tcmp": True,
            "trajectory_separation_is_not_persistence": True,
            "old_verdicts_are_calibration_only": True,
            "no_threshold_changes_after_reveal": True,
            "complete_vectors_are_deleted_after_float64_gram": True,
        },
        "final_outcomes": [
            "TCMP_SUPPORTED_WITHIN_DECLARED_DOMAIN",
            "MULTIPLE_PERSISTENCE_MECHANISMS_SUPPORTED",
            "TCMP_COUNTEREXAMPLE_FOUND",
            "INSUFFICIENT_POSITIVE_OR_UNRESOLVED_COVERAGE",
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    (OUT / "model_roster.json").write_text(json.dumps(campaign.as_dict(), indent=2, sort_keys=True) + "\n")
    resources = resource_preflight(
        Path("/data1/tzh"), min_free_bytes=campaign.min_free_disk_bytes,
        temp_budget_bytes=campaign.temp_budget_bytes,
    )
    resources.update({
        "gpu_memory_limit_bytes_per_process": campaign.max_vram_bytes,
        "temporary_vector_root": "/data1/tzh/tmp/tcmp_allop_v1",
        "home_writes_allowed": False,
    })
    (OUT / "resource_preflight.json").write_text(json.dumps(resources, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": resources["status"], "cells": len(cells), "output": str(OUT)}))


if __name__ == "__main__":
    main()
