#!/usr/bin/env python3
"""Consolidate the preregistered Gemma-4 held-out evidence without upgrading abstentions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--heldout-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    freeze = read(args.freeze)
    norm_prediction = read(args.heldout_dir / "norm_prediction_freeze.json")
    norm_result = read(args.heldout_dir / "norm_consequence32.json")
    recall_prediction = read(args.heldout_dir / "random_recall_prediction_freeze.json")
    attention_result = read(args.heldout_dir / "random_attention_consequence32.json")
    summary = read(args.heldout_dir / "gemma4_summary.json")

    if freeze.get("status") != "FROZEN_BEFORE_MODEL_DOWNLOAD_OR_EXECUTION":
        raise ValueError("held-out pool was not frozen before execution")
    for prediction in (norm_prediction, recall_prediction):
        if prediction.get("status") != "PREDICTION_FROZEN_BEFORE_TRAJECTORY":
            raise ValueError("prediction did not precede trajectory")
    for consequence in (norm_result, attention_result):
        if consequence.get("status") != "COMPLETE" or consequence.get("steps") != 32:
            raise ValueError("held-out consequence is not a complete 32-step result")

    result = {
        "schema": "kernel-analyzer-joint-heldout-confirmation-v1",
        "status": "PARTIAL_SOURCE_FACTOR_CONFIRMATION_FULL_JOINT_PREDICTOR_UNRESOLVED",
        "model": freeze["model"],
        "scientific_role": freeze["scientific_role"],
        "source_factor_confirmation": {
            "norm_new_impl": {
                "prediction": norm_prediction["source_prediction"],
                "local_amplification": norm_result["statistics"]["local"]["coherence_amplification"],
                "result": "CONFIRMED_NO_SOURCE_PERSISTENCE",
            },
            "attention_new_impl": {
                "prediction": recall_prediction["predictions"][0]["source_prediction"],
                "local_amplification": attention_result["statistics"]["local"]["coherence_amplification"],
                "result": "CONFIRMED_CANCELING_OR_ZERO_LOCAL_EFFECT_UNDER_PROTOCOL",
            },
        },
        "feedback_discovery": {
            "prediction": norm_prediction["feedback_prediction"],
            "observed_feedback_amplification": norm_result["statistics"]["feedback"]["coherence_amplification"],
            "observed_actual_amplification": norm_result["statistics"]["actual"]["coherence_amplification"],
            "optimizer_intervention": summary["feedback_intervention"],
            "result": "HELDOUT_NEW_IMPL_FEEDBACK_SUSTAINED_BUT_NOT_PREDICTED",
        },
        "unresolved": [
            {
                "case": "PLE/embedding backward:1401",
                "reason": "wrapper hashes differ from frozen runtime release",
            }
        ],
        "claim_boundary": (
            "Gemma-4 is a genuine frozen NEW_IMPL held-out population. It confirms "
            "two source-negative predictions and independently reveals an optimizer-"
            "moment feedback mechanism. Because the feedback predictor abstained, this "
            "is not confirmation of the full three-factor joint predictor."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name("." + args.output.name + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({"status": result["status"], "model": result["model"]["model_id"]}))


if __name__ == "__main__":
    main()
