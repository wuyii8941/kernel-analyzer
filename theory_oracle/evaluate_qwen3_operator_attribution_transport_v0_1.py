#!/usr/bin/env python
"""Evaluate three-state transport of one non-null repair and one null control."""

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


ENDPOINTS = {
    "clipped_gradient": ("clipped_gradient", "clipped_gradients.safetensors"),
    "parameter_update": ("parameter_update", "parameter_updates.safetensors"),
}


def result(directory: Path) -> dict[str, Any]:
    return json.loads((directory / "result.json").read_text())


def result_signature(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "continuous": value["continuous"],
        "semantic": value["semantic"],
        "post_state": value["post_state"],
    }


def base_valid(value: dict[str, Any], compiled: bool) -> bool:
    return (
        value.get("valid") is True
        and value.get("verdict") == "VALID"
        and value.get("anchors", {}).get("scorer_anchor_exact") is True
        and (
            not compiled
            or value.get("compiler", {}).get("candidate_identity_valid") is True
        )
    )


def repair_valid(value: dict[str, Any], field: str, expected_status: str) -> bool:
    repair = value.get(field, {})
    return (
        base_valid(value, compiled=True)
        and repair.get("status") == expected_status
        and bool(repair.get("gates"))
        and all(repair["gates"].values())
    )


def exact_null_profile(
    eager_result: dict[str, Any], endpoint: str
) -> dict[str, Any]:
    key, _ = ENDPOINTS[endpoint]
    baseline = eager_result["continuous"][key]
    return {
        "coordinates": sum(
            math.prod(row.get("shape", []))
            for row in baseline["per_parameter"]
            if row.get("present")
        ),
        "candidate_to_repair_l2": 0.0,
        "candidate_to_repair_max_abs": 0.0,
        "changed_tensors": 0,
        "repair_exactly_null": True,
        "normalized_target_projection": 0.0,
        "fractional_eager_distance_reduction": 0.0,
        "cosine_repair_with_candidate_to_eager": None,
    }


def add_projection(profile: dict[str, Any]) -> dict[str, Any]:
    output = dict(profile)
    target = float(profile["candidate_to_eager_l2"])
    effect = float(profile["candidate_to_repair_l2"])
    cosine = profile["cosine_repair_with_candidate_to_eager"]
    output["normalized_target_projection"] = (
        float(cosine) * effect / target if cosine is not None and target else None
    )
    return output


def cross_state_effect_metrics(
    left_candidate_path: Path,
    left_repair_path: Path,
    right_candidate_path: Path,
    right_repair_path: Path,
) -> dict[str, Any]:
    import torch
    from safetensors import safe_open

    left_square = 0.0
    right_square = 0.0
    difference_square = 0.0
    dot = 0.0
    coordinates = 0
    with safe_open(left_candidate_path, framework="pt", device="cpu") as lc, safe_open(
        left_repair_path, framework="pt", device="cpu"
    ) as lr, safe_open(right_candidate_path, framework="pt", device="cpu") as rc, safe_open(
        right_repair_path, framework="pt", device="cpu"
    ) as rr:
        keys = sorted(lc.keys())
        if keys != sorted(lr.keys()) or keys != sorted(rc.keys()) or keys != sorted(rr.keys()):
            raise ValueError("cross-state vector tensor-key sets differ")
        for key in keys:
            left = lr.get_tensor(key).float() - lc.get_tensor(key).float()
            right = rr.get_tensor(key).float() - rc.get_tensor(key).float()
            if left.shape != right.shape:
                raise ValueError(f"cross-state shape mismatch for {key}")
            difference = left - right
            left_square += float(torch.sum(left * left, dtype=torch.float64).item())
            right_square += float(torch.sum(right * right, dtype=torch.float64).item())
            difference_square += float(
                torch.sum(difference * difference, dtype=torch.float64).item()
            )
            dot += float(torch.sum(left * right, dtype=torch.float64).item())
            coordinates += int(left.numel())
            del left, right, difference
    left_norm = math.sqrt(left_square)
    right_norm = math.sqrt(right_square)
    denominator = left_norm * right_norm
    return {
        "coordinates": coordinates,
        "left_effect_l2": left_norm,
        "right_effect_l2": right_norm,
        "effect_difference_l2": math.sqrt(difference_square),
        "effect_cosine": dot / denominator if denominator else None,
    }


def scalar_summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {
            "mean": None,
            "sample_variance_across_declared_states": None,
            "minimum": None,
            "maximum": None,
        }
    return {
        "mean": statistics.mean(values),
        "sample_variance_across_declared_states": statistics.variance(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def endpoint_verdict(profiles: dict[str, dict[str, Any]]) -> str:
    nulls = [bool(row["repair_exactly_null"]) for row in profiles.values()]
    if all(nulls):
        return "NO_EFFECT_IN_DECLARED_STATES"
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
    vector_paths: dict[str, dict[str, dict[str, Path]]] = {}
    runtime_exact = True
    all_state_gates = True
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
            "silu_results_valid": repair_valid(
                values["silu_middle_1"],
                "backward_repeated_family_repair",
                "VALID_BACKWARD_REPEATED_FAMILY_REPAIR",
            )
            and repair_valid(
                values["silu_middle_2"],
                "backward_repeated_family_repair",
                "VALID_BACKWARD_REPEATED_FAMILY_REPAIR",
            ),
            "cast_results_valid": repair_valid(
                values["cast_up_control_1"],
                "backward_multirole_cast_repair",
                "VALID_BACKWARD_MULTIROLE_CAST_REPAIR",
            )
            and repair_valid(
                values["cast_up_control_2"],
                "backward_multirole_cast_repair",
                "VALID_BACKWARD_MULTIROLE_CAST_REPAIR",
            ),
            "all_arms_same_pre_state": all(
                value["pre_state"] == values["eager_1"]["pre_state"]
                for value in values.values()
            ),
            "primary_vectors_present": all(
                (dirs[arm] / filename).is_file()
                for arm in ("eager_1", "compiled_1", "silu_middle_1", "cast_up_control_1")
                for _, filename in ENDPOINTS.values()
            ),
        }
        repeat_exact = {
            "eager": result_signature(values["eager_1"])
            == result_signature(values["eager_2"]),
            "compiled": result_signature(values["compiled_1"])
            == result_signature(values["compiled_2"]),
            "silu_middle": result_signature(values["silu_middle_1"])
            == result_signature(values["silu_middle_2"]),
            "cast_up_control": result_signature(values["cast_up_control_1"])
            == result_signature(values["cast_up_control_2"]),
        }
        runtime_exact = runtime_exact and all(repeat_exact.values())
        all_state_gates = all_state_gates and all(gates.values())
        profiles: dict[str, dict[str, Any]] = {"silu_middle": {}, "cast_up_control": {}}
        for endpoint, (summary_key, filename) in ENDPOINTS.items():
            eager_path = dirs["eager_1"] / filename
            candidate_path = dirs["compiled_1"] / filename
            for treatment, arm in (
                ("silu_middle", "silu_middle_1"),
                ("cast_up_control", "cast_up_control_1"),
            ):
                repaired_result = values[arm]
                null = (
                    repaired_result["continuous"][summary_key]["tensor_hashes_sha256"]
                    == values["compiled_1"]["continuous"][summary_key][
                        "tensor_hashes_sha256"
                    ]
                )
                if null:
                    profile = exact_null_profile(values["eager_1"], endpoint)
                    if treatment == "cast_up_control":
                        profile["candidate_to_eager_l2"] = profiles["silu_middle"][
                            endpoint
                        ]["candidate_to_eager_l2"]
                    else:
                        profile["candidate_to_eager_l2"] = vector_metrics(
                            eager_path, candidate_path, candidate_path
                        )["candidate_to_eager_l2"]
                else:
                    profile = add_projection(
                        vector_metrics(eager_path, candidate_path, dirs[arm] / filename)
                    )
                profiles[treatment][endpoint] = profile
            cast_all_null = cast_all_null and profiles["cast_up_control"][endpoint][
                "repair_exactly_null"
            ]
        vector_paths[state_name] = {
            "candidate": {endpoint: dirs["compiled_1"] / filename for endpoint, (_, filename) in ENDPOINTS.items()},
            "silu_middle": {endpoint: dirs["silu_middle_1"] / filename for endpoint, (_, filename) in ENDPOINTS.items()},
        }
        states[state_name] = {
            "gates": gates,
            "repeat_exact": repeat_exact,
            "profiles": profiles,
            "semantic": {
                arm: values[arm]["semantic"]
                for arm in ("eager_1", "compiled_1", "silu_middle_1", "cast_up_control_1")
            },
        }

    silu_by_endpoint = {
        endpoint: {
            state: states[state]["profiles"]["silu_middle"][endpoint]
            for state in states
        }
        for endpoint in ENDPOINTS
    }
    endpoint_verdicts = {
        endpoint: endpoint_verdict(profiles)
        for endpoint, profiles in silu_by_endpoint.items()
    }
    summaries = {}
    for endpoint, profiles in silu_by_endpoint.items():
        projections = [
            float(row["normalized_target_projection"])
            for row in profiles.values()
            if row["normalized_target_projection"] is not None
        ]
        reductions = [float(row["fractional_eager_distance_reduction"]) for row in profiles.values()]
        summaries[endpoint] = {
            "normalized_target_projection": scalar_summary(projections),
            "fractional_eager_distance_reduction": scalar_summary(reductions),
        }

    pairwise = {}
    for left, right in itertools.combinations(states, 2):
        pairwise[f"{left}_vs_{right}"] = {
            endpoint: cross_state_effect_metrics(
                vector_paths[left]["candidate"][endpoint],
                vector_paths[left]["silu_middle"][endpoint],
                vector_paths[right]["candidate"][endpoint],
                vector_paths[right]["silu_middle"][endpoint],
            )
            for endpoint in ENDPOINTS
        }

    gates = {
        "manifest_artifacts_exact": all(artifact_gates.values()),
        "all_three_states_present": set(states) == {"a_replay", "b_original", "c_replay"},
        "all_state_gates_valid": all_state_gates,
        "all_runtime_repeats_exact": runtime_exact,
        "cast_control_exact_null_all_states": cast_all_null,
    }
    if not gates["manifest_artifacts_exact"] or not gates["all_three_states_present"] or not gates[
        "all_state_gates_valid"
    ]:
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
        "schema_version": "forkcert.qwen3-operator-attribution-transport-evaluation.v0.1",
        "status": "VALID_OPERATOR_ATTRIBUTION_TRANSPORT_EVALUATION"
        if verdict != "INVALID"
        else "INVALID_EVALUATION",
        "verdict": verdict,
        "manifest": str(manifest_path),
        "artifact_gates": artifact_gates,
        "gates": gates,
        "endpoint_verdicts": endpoint_verdicts,
        "states": states,
        "cross_state_summaries": summaries,
        "pairwise_silu_effect_alignment": pairwise,
        "claim_limits": [
            "three deliberately selected states do not estimate population prevalence",
            "state sample variance is descriptive rather than an identified population variance",
            "repair is intervention-dependent attribution rather than arithmetic root cause",
            "no injection, necessity, sufficiency, long-run or correctness claim",
        ],
    }
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError(out)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": payload["status"], "verdict": verdict, "gates": gates}, indent=2))
    if payload["status"] != "VALID_OPERATOR_ATTRIBUTION_TRANSPORT_EVALUATION":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
