#!/usr/bin/env python
"""Build a real two-level arm/pair record bundle from the Qwen3 B-state smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from bias_oracle_record_v0_2 import SCHEMA_VERSION, attach_record_digest, endpoint
from qwen3_grpo_natural_transition_v0_2 import json_sha256, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transition-root", required=True)
    parser.add_argument("--u1-evaluation", required=True)
    parser.add_argument("--t1a-evaluation", required=True)
    parser.add_argument("--t1b-evaluation", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def arm_identity(repeat: int, arm: str, optimizer_step: int) -> dict[str, Any]:
    return {
        "query_id": "Q-R",
        "trajectory_id": "legacy-B",
        "trajectory_anchor": "EAGER_TRAJECTORY",
        "trajectory_seed": 781084057,
        "data_slice_id": "forkcert_builtin_arithmetic[448:512]",
        "phase": "legacy_selected",
        "eligible_step_population": "LEGACY_30_STEP_RESTART_NOT_C0_POPULATION",
        "state_selection_prng_seed": "NOT_RANDOMLY_SELECTED_LEGACY_B",
        "state_id": "B-step29-natural-transition",
        "optimizer_step": optimizer_step,
        "repeat_id": repeat,
        "arm": arm,
    }


def paired_identity(identity: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in identity.items() if key != "arm"}


def raw_endpoint(value: float, source: Path, name: str) -> dict[str, Any]:
    return endpoint(
        "MEASURED",
        value,
        endpoint=name,
        source={"path": str(source.resolve()), "sha256": sha256_file(source)},
    )


def main() -> None:
    args = parse_args()
    transition_root = Path(args.transition_root).resolve()
    u1_path = Path(args.u1_evaluation).resolve()
    t1a_path = Path(args.t1a_evaluation).resolve()
    t1b_path = Path(args.t1b_evaluation).resolve()
    out_path = Path(args.out).resolve()
    u1 = load_json(u1_path)
    t1a = load_json(t1a_path)
    t1b = load_json(t1b_path)
    if not t1a["valid"] or not t1b["valid"]:
        raise ValueError("T1a/T1b smoke evaluation is invalid")
    u1_rows = [row for row in u1["records"] if row["state_id"] == "B"]
    if len(u1_rows) != 1:
        raise ValueError("expected exactly one retrospective U1 B record")
    u1_row = u1_rows[0]

    arm_records: list[dict[str, Any]] = []
    arm_lookup: dict[tuple[int, str], dict[str, Any]] = {}
    for repeat in (1, 2):
        for arm_name, directory_name in (
            ("reference", f"eager_{repeat}"),
            ("candidate", f"compiled_{repeat}"),
        ):
            result_path = transition_root / directory_name / "result.json"
            result = load_json(result_path)
            if not result["valid"] or result["repeat"] != repeat:
                raise ValueError(f"invalid or mismatched transition result: {result_path}")
            if (arm_name == "reference") != (result["arm"] == "eager"):
                raise ValueError(f"arm label mismatch: {result_path}")
            pre = result["pre_state"]
            compiler = result["compiler"]
            graph_digest = (
                "NOT_APPLICABLE_EAGER"
                if arm_name == "reference"
                else json_sha256(
                    {
                        "graph_code_sha256": compiler["graph_code_sha256"],
                        "graph_node_counts": compiler["graph_node_counts"],
                        "history_identity_scope": compiler["history_identity_scope"],
                    }
                )
            )
            t1a_arm = next(
                row for row in t1a["endpoint"][arm_name] if row["repeat"] == repeat
            )
            t1b_arm = next(
                row for row in t1b["endpoint"][arm_name] if row["repeat"] == repeat
            )
            update = result["vector_artifacts"]["parameter_updates"]
            record = attach_record_digest(
                {
                    "identity": arm_identity(repeat, arm_name, int(result["snapshot"]["optimizer_step"])),
                    "pre_state": {
                        "model_digest": pre["parameter_digest"],
                        "buffer_digest": pre["buffer_digest"],
                        "optimizer_digest": pre["optimizer_digest"],
                        "scheduler_digest": pre["scheduler_digest"],
                        "scaler_digest": pre["scaler_digest"],
                        "rng_digest": json_sha256(pre["rng"]),
                        "minibatch_digest": result["snapshot"]["target_minibatch_sha256"],
                    },
                    "realization": {
                        "compiler_config_digest": json_sha256(
                            {
                                "arm": result["arm"],
                                "environment": result["environment"],
                                "candidate_identity_valid": compiler["candidate_identity_valid"],
                            }
                        ),
                        "graph_family_digest": graph_digest,
                    },
                    "outcomes": {
                        "parameter_update_artifact": {
                            "path": update["path"],
                            "sha256": update["sha256"],
                        },
                        "T1a_arm_loss": raw_endpoint(
                            float(t1a_arm["loss"]), t1a_path, "heldout_grpo_surrogate_loss"
                        ),
                        "T1b_arm_nll": raw_endpoint(
                            float(t1b_arm["mean_nll"]), t1b_path, "heldout_correct_answer_nll"
                        ),
                        "propagation_ledgers": {
                            "training_loss": result["continuous"]["loss"],
                            "pre_clip_gradient_norm": result["continuous"]["pre_clip_gradient_norm"],
                            "scaled_gradient_l2": result["continuous"]["scaled_gradient"]["l2"],
                            "unscaled_gradient_l2": result["continuous"]["unscaled_gradient"]["l2"],
                            "clipped_gradient_l2": result["continuous"]["clipped_gradient"]["l2"],
                            "parameter_update_l2": result["continuous"]["parameter_update"]["l2"],
                        },
                        "semantic_events": {
                            key: value
                            for key, value in result["semantic"].items()
                            if key != "clip_decisions"
                        },
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
                        "transition_schema_version": result["schema_version"],
                    },
                }
            )
            arm_records.append(record)
            arm_lookup[(repeat, arm_name)] = record

    paired_records: list[dict[str, Any]] = []
    for repeat in (1, 2):
        reference = arm_lookup[(repeat, "reference")]
        candidate = arm_lookup[(repeat, "candidate")]
        t1a_shift = (
            candidate["outcomes"]["T1a_arm_loss"]["value"]
            - reference["outcomes"]["T1a_arm_loss"]["value"]
        )
        t1b_shift = (
            candidate["outcomes"]["T1b_arm_nll"]["value"]
            - reference["outcomes"]["T1b_arm_nll"]["value"]
        )
        coupling = {
            "shared_pre_state": reference["pre_state"],
            "reference_update_sha256": reference["outcomes"]["parameter_update_artifact"]["sha256"],
            "candidate_update_sha256": candidate["outcomes"]["parameter_update_artifact"]["sha256"],
            "t1a_bank_content_sha256": t1a["artifacts"]["bank"]["content_sha256"],
            "t1b_bank_sha256": t1b["artifacts"]["bank_sha256"],
            "randomness_protocol": "same frozen transition state and RNG; common deterministic post evaluator",
        }
        ref_events = reference["outcomes"]["semantic_events"]
        cand_events = candidate["outcomes"]["semantic_events"]
        record = attach_record_digest(
            {
                "identity": paired_identity(reference["identity"]),
                "links": {
                    "reference_arm_record_digest": reference["record_digest"],
                    "candidate_arm_record_digest": candidate["record_digest"],
                    "coupling_protocol_digest": json_sha256(coupling),
                },
                "effects": {
                    "U1": endpoint(
                        "MEASURED",
                        float(u1_row["aligned_shift"]),
                        source={"path": str(u1_path), "sha256": sha256_file(u1_path)},
                        reference_update_sha256=reference["outcomes"]["parameter_update_artifact"]["sha256"],
                        candidate_update_sha256=candidate["outcomes"]["parameter_update_artifact"]["sha256"],
                    ),
                    "U2_delta": endpoint(
                        "UNINSTANTIATED",
                        reason="selected-state smoke retained arm updates but did not materialize a separate delta artifact",
                    ),
                    "T1a_shift": endpoint(
                        "MEASURED",
                        t1a_shift,
                        source={"path": str(t1a_path), "sha256": sha256_file(t1a_path)},
                    ),
                    "T1b_shift": endpoint(
                        "MEASURED",
                        t1b_shift,
                        source={"path": str(t1b_path), "sha256": sha256_file(t1b_path)},
                    ),
                    "paired_semantic_events": {
                        "reference": ref_events,
                        "candidate": cand_events,
                        "clip_count_difference": cand_events["clip_count"] - ref_events["clip_count"],
                        "gradient_clip_trigger_difference": (
                            int(cand_events["gradient_clip_triggered"])
                            - int(ref_events["gradient_clip_triggered"])
                        ),
                        "optimizer_skip_difference": (
                            int(cand_events["optimizer_step_skipped"])
                            - int(ref_events["optimizer_step_skipped"])
                        ),
                    },
                    "paired_next_state_digests": {
                        "reference": reference["outcomes"]["next_state_digests"],
                        "candidate": candidate["outcomes"]["next_state_digests"],
                    },
                },
                "coupling": coupling,
            }
        )
        paired_records.append(record)

    bundle = {
        "schema_version": SCHEMA_VERSION,
        "scope": {
            "population_eligible": False,
            "reason": "legacy selected B state is a construction smoke, not a draw from the frozen C0 calibration sampling design",
            "paired_effect_direction": "candidate minus reference",
        },
        "arm_records": arm_records,
        "paired_effect_records": paired_records,
        "nonclaims": [
            "internal record validity does not imply population eligibility",
            "selected-state paired effects are not B",
            "shared eager evaluation is not correctness authority",
            "U2 remains uninstantiated in this selected-state bundle",
        ],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "schema_version": bundle["schema_version"],
                "population_eligible": bundle["scope"]["population_eligible"],
                "arm_records": len(arm_records),
                "paired_effect_records": len(paired_records),
                "paired_endpoints": [
                    {
                        "repeat_id": row["identity"]["repeat_id"],
                        "U1": row["effects"]["U1"]["value"],
                        "T1a_shift": row["effects"]["T1a_shift"]["value"],
                        "T1b_shift": row["effects"]["T1b_shift"]["value"],
                    }
                    for row in paired_records
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
