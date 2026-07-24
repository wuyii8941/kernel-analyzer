#!/usr/bin/env python
"""Evaluate A/B/C transport of the singleton final-norm backward repair."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
from pathlib import Path
from typing import Any

from theory_oracle.evaluate_qwen3_backward_singleton_repairs_v0_1 import (
    sha256_file,
    vector_metrics,
)
from theory_oracle.evaluate_qwen3_operator_attribution_transport_v0_1 import (
    ENDPOINTS,
    base_valid,
    cross_state_effect_metrics,
    repair_valid,
    result,
    result_signature,
)


def singleton_valid(value: dict[str, Any]) -> bool:
    repair = value.get("backward_singleton_repair", {})
    return (
        base_valid(value, compiled=True)
        and repair.get("status") == "VALID_BACKWARD_SINGLETON_REPAIR"
        and repair.get("treatment") == "final_norm_backward"
        and bool(repair.get("gates"))
        and all(repair["gates"].values())
    )


def add_projection(profile: dict[str, Any]) -> dict[str, Any]:
    output = dict(profile)
    target = float(profile["candidate_to_eager_l2"])
    effect = float(profile["candidate_to_repair_l2"])
    cosine = profile["cosine_repair_with_candidate_to_eager"]
    if target and cosine is not None:
        output["normalized_target_projection"] = float(cosine) * effect / target
    elif not target and bool(profile["repair_exactly_null"]):
        output["normalized_target_projection"] = 0.0
        output["fractional_eager_distance_reduction"] = 0.0
    else:
        output["normalized_target_projection"] = None
    return output


def scalar_summary(values: list[float]) -> dict[str, float | None]:
    return {
        "mean": statistics.mean(values) if values else None,
        "sample_variance_across_declared_states": (
            statistics.variance(values) if len(values) > 1 else None
        ),
        "minimum": min(values) if values else None,
        "maximum": max(values) if values else None,
    }


def endpoint_verdict(profiles: dict[str, dict[str, Any]]) -> str:
    nulls = [bool(row["repair_exactly_null"]) for row in profiles.values()]
    if any(nulls):
        return "STATE_CONDITIONAL_DIRECTION"
    projections = [row["normalized_target_projection"] for row in profiles.values()]
    if any(value is None or value == 0.0 for value in projections):
        return "TRANSPORTED_EXISTENCE_ONLY"
    signs = {1 if float(value) > 0 else -1 for value in projections}
    return "TRANSPORTED_DIRECTION" if len(signs) == 1 else "STATE_CONDITIONAL_DIRECTION"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest["status"] != "FROZEN_PRE_EXECUTION":
        raise ValueError("manifest is not frozen")
    artifact_gates = {
        name: sha256_file(Path(row["path"]).resolve()) == row["sha256"]
        for name, row in manifest["artifacts"].items()
    }

    states: dict[str, Any] = {}
    vector_paths: dict[str, dict[str, Path]] = {}
    all_state_gates = True
    runtime_exact = True
    cast_all_null = True
    for state_name, spec in manifest["states"].items():
        dirs = {name: Path(path).resolve() for name, path in spec["arms"].items()}
        values = {name: result(directory) for name, directory in dirs.items()}
        gates = {
            "snapshot_verification_valid": json.loads(
                Path(spec["snapshot_verification"]).read_text()
            ).get("status")
            == "VALID_CROSS_STATE_TRANSITION_CAPTURE",
            "eager_results_valid": base_valid(values["eager_1"], False)
            and base_valid(values["eager_2"], False),
            "compiled_results_valid": base_valid(values["compiled_1"], True)
            and base_valid(values["compiled_2"], True),
            "final_norm_results_valid": singleton_valid(values["final_norm_1"])
            and singleton_valid(values["final_norm_2"]),
            "cast_results_valid": repair_valid(
                values["cast_1"],
                "backward_multirole_cast_repair",
                "VALID_BACKWARD_MULTIROLE_CAST_REPAIR",
            )
            and repair_valid(
                values["cast_2"],
                "backward_multirole_cast_repair",
                "VALID_BACKWARD_MULTIROLE_CAST_REPAIR",
            ),
            "all_arms_same_pre_state": all(
                value["pre_state"] == values["eager_1"]["pre_state"]
                for value in values.values()
            ),
            "primary_vectors_present": all(
                (dirs[arm] / filename).is_file()
                for arm in ("eager_1", "compiled_1", "final_norm_1")
                for _, filename in ENDPOINTS.values()
            ),
        }
        repeats = {
            "eager": result_signature(values["eager_1"])
            == result_signature(values["eager_2"]),
            "compiled": result_signature(values["compiled_1"])
            == result_signature(values["compiled_2"]),
            "final_norm": result_signature(values["final_norm_1"])
            == result_signature(values["final_norm_2"]),
            "cast": result_signature(values["cast_1"])
            == result_signature(values["cast_2"]),
        }
        profiles = {}
        for endpoint, (key, filename) in ENDPOINTS.items():
            profiles[endpoint] = add_projection(
                vector_metrics(
                    dirs["eager_1"] / filename,
                    dirs["compiled_1"] / filename,
                    dirs["final_norm_1"] / filename,
                )
            )
            cast_null = (
                values["cast_1"]["continuous"][key]["tensor_hashes_sha256"]
                == values["compiled_1"]["continuous"][key]["tensor_hashes_sha256"]
            )
            cast_all_null = cast_all_null and cast_null
        vector_paths[state_name] = {
            "compiled_gradient": dirs["compiled_1"] / ENDPOINTS["clipped_gradient"][1],
            "repair_gradient": dirs["final_norm_1"] / ENDPOINTS["clipped_gradient"][1],
            "compiled_update": dirs["compiled_1"] / ENDPOINTS["parameter_update"][1],
            "repair_update": dirs["final_norm_1"] / ENDPOINTS["parameter_update"][1],
        }
        states[state_name] = {"gates": gates, "repeat_exact": repeats, "profiles": profiles}
        all_state_gates = all_state_gates and all(gates.values())
        runtime_exact = runtime_exact and all(repeats.values())

    by_endpoint = {
        endpoint: {state: states[state]["profiles"][endpoint] for state in states}
        for endpoint in ENDPOINTS
    }
    endpoint_verdicts = {
        endpoint: endpoint_verdict(profiles) for endpoint, profiles in by_endpoint.items()
    }
    summaries = {
        endpoint: {
            "normalized_target_projection": scalar_summary(
                [
                    float(row["normalized_target_projection"])
                    for row in profiles.values()
                    if row["normalized_target_projection"] is not None
                ]
            ),
            "fractional_eager_distance_reduction": scalar_summary(
                [
                    float(row["fractional_eager_distance_reduction"])
                    for row in profiles.values()
                    if row["fractional_eager_distance_reduction"] is not None
                ]
            ),
        }
        for endpoint, profiles in by_endpoint.items()
    }
    pairwise = {}
    for left, right in itertools.combinations(states, 2):
        pairwise[f"{left}_vs_{right}"] = {
            "clipped_gradient": cross_state_effect_metrics(
                vector_paths[left]["compiled_gradient"],
                vector_paths[left]["repair_gradient"],
                vector_paths[right]["compiled_gradient"],
                vector_paths[right]["repair_gradient"],
            ),
            "parameter_update": cross_state_effect_metrics(
                vector_paths[left]["compiled_update"],
                vector_paths[left]["repair_update"],
                vector_paths[right]["compiled_update"],
                vector_paths[right]["repair_update"],
            ),
        }

    gates = {
        "manifest_artifacts_exact": all(artifact_gates.values()),
        "all_three_states_present": set(states) == {"a_replay", "b_original", "c_replay"},
        "all_state_gates_valid": all_state_gates,
        "all_runtime_repeats_exact": runtime_exact,
        "cast_control_exact_null_all_states": cast_all_null,
    }
    if not all(gates[name] for name in ("manifest_artifacts_exact", "all_three_states_present", "all_state_gates_valid")):
        verdict = "INVALID"
    elif not runtime_exact:
        verdict = "INDETERMINATE_RUNTIME"
    elif not cast_all_null:
        verdict = "INDETERMINATE_INTERVENTION_CONTROL"
    elif all(value == "TRANSPORTED_DIRECTION" for value in endpoint_verdicts.values()):
        verdict = "TRANSPORTED_DIRECTION"
    elif any(value == "STATE_CONDITIONAL_DIRECTION" for value in endpoint_verdicts.values()):
        verdict = "STATE_CONDITIONAL_DIRECTION"
    else:
        verdict = "TRANSPORTED_EXISTENCE_ONLY"

    payload = {
        "schema_version": "forkcert.qwen3-final-norm-attribution-transport-evaluation.v0.1",
        "status": "VALID_FINAL_NORM_ATTRIBUTION_TRANSPORT_EVALUATION"
        if verdict != "INVALID"
        else "INVALID_EVALUATION",
        "verdict": verdict,
        "manifest": str(manifest_path),
        "artifact_gates": artifact_gates,
        "gates": gates,
        "endpoint_verdicts": endpoint_verdicts,
        "states": states,
        "cross_state_summaries": summaries,
        "pairwise_effect_alignment": pairwise,
        "claim_limits": [
            "three selected states do not estimate population prevalence",
            "fused-region repair is not source-operator decomposition",
            "repair is intervention-dependent attribution rather than root cause",
            "no injection, correctness or long-run claim",
        ],
    }
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError(out)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": payload["status"], "verdict": verdict, "gates": gates}, indent=2))
    if payload["status"] != "VALID_FINAL_NORM_ATTRIBUTION_TRANSPORT_EVALUATION":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
