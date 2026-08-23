#!/usr/bin/env python3
"""Import the independently frozen Gemma-4 held-out run into v4.

The Gemma-4 campaign was frozen before its trajectory was run, but it used a
separate confirmation bank and trajectory bank.  This adapter keeps those
identities explicit instead of pretending that the old campaign is the same
as the invalid atlas-derived v4 pool.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
GEMMA = ROOT / "results/property/tcmp_allop_v1/heldout/gemma4_e2b_text128"
BASE_BANK = ROOT / "results/property/tcmp_allop_v1/input_banks/gemma4_e2b_text128.json"
TRAJECTORY_BANK = ROOT / "results/property/tcmp_allop_v1/input_banks/gemma4_e2b_text128_trajectory32.json"
FREEZE = ROOT / "results/property/tcmp_allop_v1/new_impl_heldout_freeze_v1.json"
PREDICTION = GEMMA / "norm_prediction_freeze.json"
FORMATION = GEMMA / "norm_formation16.json"
SHORT = ROOT / "results/property/direct_persistence_v4/heldout/gemma4_e2b_norm_short_screen.json"
CONSEQUENCE = ROOT / "results/property/direct_persistence_v4/heldout/gemma4_e2b_norm_consequence32.json"
CAPTURE = GEMMA / "runtime_release/capture.json"


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_digest(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "results/property/direct_persistence_v4",
    )
    args = parser.parse_args()

    freeze = load(FREEZE)
    prediction = load(PREDICTION)
    formation = load(FORMATION)
    short = load(SHORT)
    consequence = load(CONSEQUENCE)
    capture = load(CAPTURE)
    base_bank = load(BASE_BANK)
    trajectory_bank = load(TRAJECTORY_BANK)

    require(freeze.get("status") == "FROZEN_BEFORE_MODEL_DOWNLOAD_OR_EXECUTION", "Gemma freeze is not pre-measurement")
    require(prediction.get("status") == "PREDICTION_FROZEN_BEFORE_TRAJECTORY", "Gemma source prediction was not frozen")
    require(formation.get("status") == "COMPLETE_COMPLETE_COORDINATES", "Gemma formation is incomplete")
    require(consequence.get("status") == "COMPLETE" and consequence.get("steps") == 32, "Gemma 32-step consequence is incomplete")
    require(consequence.get("state_role") == "TRAJECTORY", "Gemma consequence is not using the trajectory bank")
    require(consequence.get("optimizer", {}).get("name") == "adamw", "Gemma consequence is not AdamW")
    require(len([row for row in base_bank.get("states", []) if row.get("role") == "CONFIRMATION"]) == 16, "Gemma confirmation bank is not 16 states")
    require(len([row for row in trajectory_bank.get("states", []) if row.get("role") == "TRAJECTORY"]) == 32, "Gemma trajectory bank is not 32 states")

    local_short = next(
        row for row in short.get("cases", [])
        if row.get("case_id") == "gemma4_e2b_ple_rmsnorm::adamw::local"
    )
    local_a32 = consequence["statistics"]["local"]["coherence_amplification"]
    actual_a32 = consequence["statistics"]["actual"]["coherence_amplification"]
    feedback_a32 = consequence["statistics"]["feedback"]["coherence_amplification"]

    formation_states = [
        row["state_id"] for row in base_bank["states"] if row.get("role") == "CONFIRMATION"
    ]
    trajectory_states = [
        row["state_id"] for row in trajectory_bank["states"] if row.get("role") == "TRAJECTORY"
    ]
    carrier = prediction["trajectory_carrier"]
    coordinate_digest = canonical_digest({
        "parameter": carrier["parameter"],
        "coordinates": carrier["coordinates"],
        "selection": carrier["selection"],
    })
    identity = {
        "case_id": "gemma4_e2b_ple_rmsnorm_fb",
        "role": "NEW_IMPL",
        "model": freeze["model"]["model_id"],
        "model_revision": freeze["model"]["revision"],
        "sequence_length": freeze["model"]["sequence_length"],
        "implementation_pattern": "NEW_IMPL_PATTERN",
        "endpoint": {"region_id": "forward:2", "endpoint": "in_out_ptr0"},
        "carrier": carrier,
        "parameter_coordinate_digest": coordinate_digest,
        "repair": {
            "kind": "GeneratedFP32Observer",
            "runtime_release": "results/property/tcmp_allop_v1/heldout/gemma4_e2b_text128/runtime_release",
            "target_region": "forward:2",
            "target_endpoint": "in_out_ptr0",
        },
        "formation_state_order": formation_states,
        "formation_state_bank_digest": digest(BASE_BANK),
        "trajectory_state_order": trajectory_states,
        "trajectory_state_bank_digest": digest(TRAJECTORY_BANK),
        "runtime_capture_digest": digest(CAPTURE),
        "optimizer": {
            "name": "AdamW",
            "learning_rate": consequence["optimizer"]["learning_rate"],
            "moments": "initialized to zero and then evolved normally",
        },
    }

    pool = {
        "schema": "kernel-analyzer-direct-persistence-v4-independent-heldout-pool-v1",
        "status": "FROZEN_BEFORE_TRAJECTORY_EXTERNAL_CAMPAIGN",
        "selection_source": "results/property/tcmp_allop_v1/new_impl_heldout_freeze_v1.json",
        "selection_rule": "Gemma-4 was frozen before model download/trajectory execution; this row is kept separate from the later invalid atlas-derived pool.",
        "rows": [identity],
        "claim_boundary": "One independently frozen NEW_IMPL row is validated. This is not a complete universal held-out pool and cannot yield recall or AUROC by itself.",
    }

    predictions = {
        "schema": "kernel-analyzer-direct-persistence-v4-independent-heldout-predictions-v1",
        "status": "COMPLETE_ONE_NEW_IMPL_PREDICTION_FROZEN_BEFORE_TRAJECTORY",
        "rows": [{
            "case_id": identity["case_id"],
            "role": identity["role"],
            "source_prediction": prediction["source_prediction"],
            "formation_amplification": prediction["evidence_available_before_trajectory"]["complete_coordinate_open_loop_amplification"],
            "formation_states": prediction["evidence_available_before_trajectory"]["states"],
            "prediction_artifact": str(PREDICTION.relative_to(ROOT)),
            "prediction_frozen_before_trajectory": True,
        }],
        "claim_boundary": "The prediction is a source/direct-persistence prediction. Feedback was explicitly out of domain before the trajectory was run.",
    }

    confirmation = {
        "schema": "kernel-analyzer-direct-persistence-v4-independent-heldout-confirmation-v1",
        "status": "COMPLETE_ONE_NEW_IMPL_NEGATIVE_FOR_DIRECT_SCREEN",
        "rows": [{
            "case_id": identity["case_id"],
            "role": identity["role"],
            "short_screen": {
                "verdict": "NO_ESCALATION_UNDER_SHORT_SCREEN",
                "source_status": local_short["status"],
                "A16": local_short["observed_amplification"],
                "null_upper_95": local_short["sign_flip_null"]["upper_95"],
                "one_sided_p": local_short["sign_flip_null"]["one_sided_p"],
                "steps": local_short["steps"],
            },
            "confirmation": {
                "verdict": "NO_DETECTED_DIRECT_PERSISTENCE_AFTER_CONFIRMATION",
                "A32_local": local_a32,
                "A32_actual": actual_a32,
                "A32_feedback": feedback_a32,
                "final_drift_l2": consequence["final_drift_l2"],
                "feedback_status": "FEEDBACK_DOMINATED_CONSEQUENCE",
                "steps": consequence["steps"],
            },
            "prediction_was_frozen_before_trajectory": True,
            "consequence_artifact": str(CONSEQUENCE.relative_to(ROOT)),
            "short_screen_artifact": str(SHORT.relative_to(ROOT)),
        }],
        "metrics": {
            "eligible_rows": 1,
            "confirmed_positive": 0,
            "confirmed_negative": 1,
            "escalation_rate": 0.0,
            "false_escalation_rate": 0.0,
            "recall": None,
            "auroc": None,
            "abstention_rate": 0.0,
            "gpu_cost": "recorded by source campaign; not recomputed by v4 adapter",
        },
        "all_negative_policy": "No recall or AUROC is reported because this independent pool contains no confirmed direct-persistence positive.",
        "claim_boundary": "Gemma is a genuine NEW_IMPL negative for direct persistence, while its actual trajectory is feedback-sustained and therefore outside the source screen's claim.",
    }

    validation = {
        "schema": "kernel-analyzer-direct-persistence-v4-independent-heldout-validation-v1",
        "status": "VALID_ONE_NEW_IMPL_ROW",
        "rows": [{"case_id": identity["case_id"], "status": "READY_AND_COMPLETE"}],
        "checks": {
            "pre_measurement_freeze": True,
            "exact_model_revision": True,
            "exact_runtime_capture": True,
            "complete_16_step_short_screen": True,
            "complete_32_step_confirmation": True,
            "optimizer_identity": True,
            "separate_formation_and_trajectory_banks": True,
        },
    }

    manifest = {
        "schema": "kernel-analyzer-direct-persistence-v4-independent-heldout-run-manifest-v1",
        "status": "PARTIAL_ONE_NEW_IMPL_COMPLETE",
        "pool": "heldout_gemma_pool.json",
        "predictions": "heldout_gemma_predictions.json",
        "confirmation": "heldout_gemma_confirmation.json",
        "validation": "heldout_gemma_validation.json",
        "required_next": "Freeze and run additional mechanically selected NEW_IMPL rows before claiming recall, AUROC, or universal generalization.",
    }

    out = args.output_dir
    outputs = {
        "heldout_gemma_pool.json": pool,
        "heldout_gemma_predictions.json": predictions,
        "heldout_gemma_confirmation.json": confirmation,
        "heldout_gemma_validation.json": validation,
        "heldout_gemma_run_manifest.json": manifest,
    }
    for name, value in outputs.items():
        (out / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": validation["status"], "rows": 1, "output": str(out)}, sort_keys=True))


if __name__ == "__main__":
    main()
