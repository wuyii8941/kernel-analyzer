#!/usr/bin/env python
"""Build and validate one population-eligible calibration state record bundle."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from theory_oracle.bias_oracle_record_v0_2 import (
    SCHEMA_VERSION,
    attach_record_digest,
    endpoint,
    validate_record_bundle,
)
from theory_oracle.qwen3_grpo_natural_transition_v0_2 import json_sha256, sha256_file


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def mean_for_transition_repeat(
    rows: list[dict[str, Any]], repeat: int, field: str
) -> float:
    selected = [row for row in rows if int(row["transition_repeat"]) == repeat]
    if any("evaluator_repeat" not in row for row in selected):
        raise ValueError(f"evaluator repeat identity missing for transition repeat {repeat}")
    repeat_ids = [int(row["evaluator_repeat"]) for row in selected]
    if len(repeat_ids) != len(set(repeat_ids)) or sorted(repeat_ids) != list(
        range(1, len(repeat_ids) + 1)
    ):
        raise ValueError(f"invalid evaluator repeat identities for transition repeat {repeat}")
    values = [float(row[field]) for row in sorted(selected, key=lambda row: int(row["evaluator_repeat"]))]
    if not values:
        raise ValueError(f"no evaluator values for transition repeat {repeat}")
    return math.fsum(values) / len(values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", required=True)
    parser.add_argument("--transition-root", required=True)
    parser.add_argument("--transition-evaluation", required=True)
    parser.add_argument("--task-evaluation", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--validation-out", required=True)
    args = parser.parse_args()

    snapshot_dir = Path(args.snapshot_dir).resolve()
    transition_root = Path(args.transition_root).resolve()
    transition_evaluation_path = Path(args.transition_evaluation).resolve()
    task_evaluation_path = Path(args.task_evaluation).resolve()
    transition_evaluation = load_json(transition_evaluation_path)
    task_evaluation = load_json(task_evaluation_path)
    metadata = load_json(snapshot_dir / "forkcert_transition_snapshot.json")
    if not transition_evaluation.get("construction_valid"):
        raise ValueError("transition evaluation is invalid")
    if not task_evaluation.get("valid"):
        raise ValueError("task evaluation is invalid")
    task_randomness = task_evaluation.get("randomness_decomposition")
    if not isinstance(task_randomness, dict) or not task_randomness:
        raise ValueError("task evaluation lacks an explicit randomness decomposition")

    target = metadata["capture_target_identity"]
    identity_base = {
        "query_id": target["query_id"],
        "trajectory_id": target["trajectory_id"],
        "trajectory_anchor": target["trajectory_anchor"],
        "trajectory_seed": target["trajectory_seed"],
        "data_slice_id": target["data_slice_id"],
        "phase": target["phase"],
        "eligible_step_population": target["eligible_step_population"],
        "state_selection_prng_seed": target["state_selection_prng_seed"],
        "state_id": metadata["state_id"],
        "optimizer_step": int(metadata["optimizer_step"]),
    }
    t1a_status = task_evaluation["T1a"]["status"]
    t1b_status = task_evaluation["T1b"]["status"]
    if t1b_status != "MEASURED":
        raise ValueError("T1b must be measured")

    arm_records: list[dict[str, Any]] = []
    arm_lookup: dict[tuple[int, str], dict[str, Any]] = {}
    raw_results: dict[tuple[int, str], dict[str, Any]] = {}
    for repeat in (1, 2):
        for arm_label, directory_prefix, execution_arm in (
            ("reference", "eager", "eager"),
            ("candidate", "compiled", "compiled"),
        ):
            result_path = transition_root / f"{directory_prefix}_{repeat}" / "result.json"
            result = load_json(result_path)
            if not result.get("valid") or result.get("arm") != execution_arm:
                raise ValueError(f"invalid transition arm: {result_path}")
            raw_results[(repeat, arm_label)] = result
            pre = result["pre_state"]
            update = result["vector_artifacts"]["parameter_updates"]
            t1a_value = (
                mean_for_transition_repeat(
                    task_evaluation["T1a"]["arm_results"][arm_label], repeat, "loss"
                )
                if t1a_status == "MEASURED"
                else None
            )
            t1b_value = mean_for_transition_repeat(
                task_evaluation["T1b"]["arm_results"][arm_label], repeat, "mean_nll"
            )
            record = attach_record_digest(
                {
                    "identity": {
                        **identity_base,
                        "repeat_id": repeat,
                        "arm": arm_label,
                    },
                    "pre_state": {
                        "model_digest": pre["parameter_digest"],
                        "buffer_digest": pre["buffer_digest"],
                        "optimizer_digest": pre["optimizer_digest"],
                        "scheduler_digest": pre["scheduler_digest"],
                        "scaler_digest": pre["scaler_digest"],
                        "rng_digest": json_sha256(pre["rng"]),
                        "minibatch_digest": result["snapshot"]["target_minibatch_sha256"],
                    },
                    "realization": result["realization"],
                    "outcomes": {
                        "parameter_update_artifact": {
                            "path": update["path"],
                            "sha256": update["sha256"],
                        },
                        "T1a_arm_loss": endpoint("MEASURED", t1a_value)
                        if t1a_value is not None
                        else endpoint("UNINSTANTIATED", reason=t1a_status),
                        "T1b_arm_nll": endpoint("MEASURED", t1b_value),
                        "propagation_ledgers": {
                            "training_loss": result["continuous"]["loss"],
                            "pre_clip_gradient_norm": result["continuous"]["pre_clip_gradient_norm"],
                            "scaled_gradient_l2": result["continuous"]["scaled_gradient"]["l2"],
                            "unscaled_gradient_l2": result["continuous"]["unscaled_gradient"]["l2"],
                            "clipped_gradient_l2": result["continuous"]["clipped_gradient"]["l2"],
                            "parameter_update_l2": result["continuous"]["parameter_update"]["l2"],
                        },
                        "semantic_events": result["semantic"],
                        "next_state_digests": {
                            key: value
                            for key, value in result["post_state"].items()
                            if key.endswith("_digest")
                        },
                    },
                    "provenance": {
                        "transition_result": {
                            "path": str(result_path),
                            "sha256": sha256_file(result_path),
                        },
                        "task_evaluation": {
                            "path": str(task_evaluation_path),
                            "sha256": sha256_file(task_evaluation_path),
                        },
                    },
                }
            )
            arm_records.append(record)
            arm_lookup[(repeat, arm_label)] = record

    update_profile = transition_evaluation["profiles"]["parameter_update"]
    u1_values = update_profile["paired_U1_repeats"]
    u2_l2_values = update_profile["paired_effect_l2_repeats"]
    u2_artifacts = update_profile["paired_effect_vector_artifacts"]
    if len(u1_values) != 2 or len(u2_l2_values) != 2 or len(u2_artifacts) != 2:
        raise ValueError("transition evaluation did not materialize two paired U1/U2 effects")

    pair_records = []
    for repeat in (1, 2):
        reference = arm_lookup[(repeat, "reference")]
        candidate = arm_lookup[(repeat, "candidate")]
        u2_artifact = next(row for row in u2_artifacts if int(row["repeat"]) == repeat)
        t1a_ref = reference["outcomes"]["T1a_arm_loss"]
        t1a_cand = candidate["outcomes"]["T1a_arm_loss"]
        t1b_ref = reference["outcomes"]["T1b_arm_nll"]
        t1b_cand = candidate["outcomes"]["T1b_arm_nll"]
        coupling = {
            "paired_direction": "candidate minus reference",
            "realization_contract_sha256": raw_results[(repeat, "reference")]["anchors"]["realization_contract_sha256"],
            "shared_pre_state": reference["pre_state"],
            "transition_repeat": repeat,
            "task_evaluator_nesting": task_evaluation["nesting"],
            "task_endpoint_randomness_decomposition": task_randomness,
        }
        ref_events = reference["outcomes"]["semantic_events"]
        cand_events = candidate["outcomes"]["semantic_events"]
        pair = attach_record_digest(
            {
                "identity": {**identity_base, "repeat_id": repeat},
                "links": {
                    "reference_arm_record_digest": reference["record_digest"],
                    "candidate_arm_record_digest": candidate["record_digest"],
                    "coupling_protocol_digest": json_sha256(coupling),
                },
                "effects": {
                    "U1": endpoint("MEASURED", float(u1_values[repeat - 1]))
                    if u1_values[repeat - 1] is not None
                    else endpoint("UNDEFINED", reason="zero reference update norm"),
                    "U2_delta": endpoint(
                        "MEASURED",
                        float(u2_l2_values[repeat - 1]),
                        value_role="L2 norm of this repeat's paired full-vector artifact; direction is retained only by the artifact",
                        artifact={"path": u2_artifact["path"], "sha256": u2_artifact["sha256"]},
                    ),
                    "T1a_shift": endpoint(
                        "MEASURED", float(t1a_cand["value"] - t1a_ref["value"])
                    )
                    if t1a_ref["status"] == t1a_cand["status"] == "MEASURED"
                    else endpoint("UNINSTANTIATED", reason=t1a_status),
                    "T1b_shift": endpoint(
                        "MEASURED", float(t1b_cand["value"] - t1b_ref["value"])
                    ),
                    "paired_semantic_events": {
                        "reference": ref_events,
                        "candidate": cand_events,
                        "clip_count_difference": cand_events["clip_count"] - ref_events["clip_count"],
                        "gradient_clip_trigger_difference": int(cand_events["gradient_clip_triggered"]) - int(ref_events["gradient_clip_triggered"]),
                        "optimizer_skip_difference": int(cand_events["optimizer_step_skipped"]) - int(ref_events["optimizer_step_skipped"]),
                    },
                    "paired_next_state_digests": {
                        "reference": reference["outcomes"]["next_state_digests"],
                        "candidate": candidate["outcomes"]["next_state_digests"],
                    },
                },
                "coupling": coupling,
            }
        )
        pair_records.append(pair)

    bundle = {
        "schema_version": SCHEMA_VERSION,
        "scope": {
            "population_eligible": True,
            "population_query": "C0 Q-R calibration distribution",
            "population_verdict_allowed": False,
            "reason": "state was selected prospectively by the frozen trajectory/phase design; one trajectory is still insufficient for a verdict",
            "paired_effect_direction": "candidate minus reference",
            "task_endpoint_randomness_decomposition": task_randomness,
        },
        "arm_records": arm_records,
        "paired_effect_records": pair_records,
        "nonclaims": [
            "record eligibility is not a population B estimate",
            "one state does not identify H or trajectory uncertainty",
            "implementation-relative effects are not correctness errors",
        ],
    }
    validation = validate_record_bundle(bundle, verify_artifacts=True)
    out = Path(args.out).resolve()
    validation_out = Path(args.validation_out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    validation_out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    validation["bundle"] = {"path": str(out), "sha256": sha256_file(out)}
    validation_out.write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"validation": validation["verdict"], "state_id": metadata["state_id"]}, indent=2))
    if not validation["valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
