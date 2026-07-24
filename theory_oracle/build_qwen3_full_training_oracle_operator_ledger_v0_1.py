#!/usr/bin/env python
"""Build the fail-closed Qwen3 Oracle/operator coverage integration ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


INPUTS = {
    "forward_coverage": "results/operator_oracle/qwen3_operator_coverage_v0_1/coverage.json",
    "forward_equivalence": "results/operator_oracle/qwen3_operator_coverage_v0_1/equivalence_classes.json",
    "backward_coverage": "results/operator_oracle/qwen3_backward_runtime_census_v0_1/backward_coverage_ledger_v0_3.json",
    "backward_delta": "results/operator_oracle/qwen3_backward_runtime_census_v0_1/backward_coverage_delta_v0_4.json",
    "event_oracle": "results/training_step_oracle/qwen3_grpo_grad_event_bank_v0_4/evaluation.json",
    "transition_oracle": "results/training_step_oracle/qwen3_grpo_natural_transition_v0_2/evaluation.json",
    "transport_attribution": "results/operator_oracle/qwen3_operator_attribution_transport_v0_1/evaluation.json",
}


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=".")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    repo = Path(args.repo).resolve()
    paths = {name: repo / value for name, value in INPUTS.items()}
    values = {name: load(path) for name, path in paths.items()}

    forward = values["forward_coverage"]
    forward_eq = values["forward_equivalence"]
    backward = values["backward_coverage"]
    backward_delta = values["backward_delta"]
    events = values["event_oracle"]
    transition = values["transition_oracle"]
    transport = values["transport_attribution"]

    backward_constituents: dict[str, list[str]] = defaultdict(list)
    for family in backward["treatment_families"]:
        for operator in family.get("constituent_aten", []):
            backward_constituents[operator].append(family["treatment_id"])

    gates = {
        "forward_inventory_valid": forward["source_inventory_status"]
        == "VALID_ORIGINAL_CANDIDATE_KERNEL_INVENTORY",
        "forward_descriptive_complete": forward["coverage_verdicts"][
            "frozen_forward_descriptive"
        ]
        == "COMPLETE",
        "backward_denominator_exact": backward["denominator"]["treatment_families"]
        == backward_delta["updated_metrics"]["runtime_observed_families"]
        == 41,
        "event_mechanics_valid": events["mechanics_valid"] is True,
        "event_repeats_stable": events["repeat_unstable_event_count"] == 0,
        "transition_valid": transition["construction_valid"] is True
        and all(transition["validity"].values()),
        "transport_valid": transport["status"]
        == "VALID_OPERATOR_ATTRIBUTION_TRANSPORT_EVALUATION"
        and all(transport["gates"].values()),
    }

    measurement_complete = all(gates.values())
    forward_families = forward_eq["metrics"]["generated_treatment_families"]
    backward_families = backward["denominator"]["treatment_families"]
    forward_fully_causal = forward_eq["metrics"]["fully_covered_treatment_families"]
    backward_fully_causal = backward_delta["updated_metrics"]["fully_covered_families"]
    full_causal = (
        forward_fully_causal == forward_families
        and backward_fully_causal == backward_families
        and backward_delta["updated_metrics"]["valid_injections"] > 0
    )

    payload = {
        "schema_version": "forkcert.qwen3-full-training-oracle-operator-ledger.v0.1",
        "status": "VALID_INTEGRATED_LEDGER" if measurement_complete else "INVALID_LEDGER",
        "scope": "Qwen3-0.6B GRPO declared FP16/SDPA-math/Inductor subjects and frozen matched states",
        "gates": gates,
        "unit_boundaries": {
            "source_operator": (
                "ATen/prims graph target; several source operators may be fused into one region, "
                "so region repair cannot be assigned to each constituent operator"
            ),
            "generated_region_family": (
                "named generated Triton callable; repeated calls may have different source roles "
                "and state-conditioned effects"
            ),
            "runtime_kernel_family": (
                "generated Triton family or external mm/bmm family observed during execution"
            ),
        },
        "descriptive_denominators": {
            "forward_atomic_target_types": forward["metrics"][
                "forward_atomic_target_types_observed"
            ],
            "forward_atomic_calls": forward["metrics"]["forward_atomic_calls_observed"],
            "forward_generated_or_external_families": forward_families,
            "forward_semantic_invocations": forward_eq["metrics"]["semantic_invocations"],
            "backward_atomic_target_types": forward["metrics"][
                "backward_atomic_target_types_observed"
            ],
            "backward_atomic_calls": forward["metrics"]["backward_atomic_calls_observed"],
            "backward_runtime_families": backward_families,
            "backward_runtime_calls": backward["denominator"]["runtime_calls"],
            "backward_constituent_aten_types_in_generated_families": len(
                backward_constituents
            ),
            "event_bank_states": events["total_rollout_states"],
            "event_bank_applicable_tokens": events["applicable_tokens"],
        },
        "oracle_application": {
            "numerical_event_profile": {
                "mechanics_valid": events["mechanics_valid"],
                "average_shift_by_trajectory": {
                    row["trajectory"]: row["average_implementation_relative_shift"]
                    for row in events["trajectories"]
                },
                "state_conditioned_heterogeneity_by_trajectory": {
                    row["trajectory"]: row["state_conditioned_heterogeneity"]
                    for row in events["trajectories"]
                },
                "within_state_runtime_variability_by_trajectory": {
                    row["trajectory"]: row["within_state_runtime_variability"]
                    for row in events["trajectories"]
                },
                "directional_semantic_shift": events["directional_semantic_shift"],
                "semantic_disagreement": events["semantic_disagreement"],
                "population_inference": events["population_inference"],
            },
            "one_step_transition": {
                "selected_state_verdict": transition["next_state"]["verdict"],
                "self_stable": transition["next_state"]["self_stable"],
                "cross_arm_semantic_disagreement": transition["semantic_events"][
                    "any_disagreement"
                ],
                "population_prevalence": transition["verdict_ledgers"][
                    "population_prevalence"
                ],
            },
            "operator_attribution_transport": {
                "verdict": transport["verdict"],
                "endpoint_verdicts": transport["endpoint_verdicts"],
                "runtime_repeats_exact": transport["gates"][
                    "all_runtime_repeats_exact"
                ],
                "negative_control_exact_null": transport["gates"][
                    "cast_control_exact_null_all_states"
                ],
                "tested_states": sorted(transport["states"]),
            },
        },
        "causal_coverage": {
            "forward": {
                "families_in_denominator": forward_families,
                "families_with_partial_original_candidate_evidence": (
                    forward_eq["metrics"][
                        "original_candidate_generated_kernel_families_partially_covered"
                    ]
                    + forward_eq["metrics"][
                        "original_candidate_external_families_partially_covered"
                    ]
                ),
                "fully_covered_families": forward_fully_causal,
            },
            "backward": {
                "families_in_denominator": backward_families,
                "selected_state_repair_families": backward_delta["updated_metrics"][
                    "selected_state_repair_families"
                ],
                "valid_repair_invocations": backward_delta["updated_metrics"][
                    "valid_repair_invocations"
                ],
                "transport_tested_nonnull_candidates": 1,
                "transport_tested_exact_null_controls": 1,
                "valid_injections": backward_delta["updated_metrics"][
                    "valid_injections"
                ],
                "fully_covered_families": backward_fully_causal,
            },
            "full_operator_causal_analysis_complete": full_causal,
        },
        "verdicts": {
            "declared_measurement_and_denominator_audit": (
                "COMPLETE" if measurement_complete else "INVALID"
            ),
            "oracle_application": (
                "VALID_FOR_DECLARED_MATCHED_STATES" if measurement_complete else "INVALID"
            ),
            "selected_operator_attribution": transport["verdict"],
            "full_source_operator_causal_attribution": (
                "COMPLETE" if full_causal else "INCOMPLETE"
            ),
            "compiler_correctness": "UNINSTANTIATED_NO_INDEPENDENT_AUTHORITY",
        },
        "blocking_gaps": [
            "source-operator attribution is not identified by a fused-region repair",
            "31/41 backward runtime families lack any selected-state repair evidence",
            "no valid operator injection exists",
            "three selected transport states do not estimate a target-state population",
            "no independent correctness authority exists",
        ],
        "input_artifacts": {
            name: {"path": str(path.relative_to(repo)), "sha256": sha256(path)}
            for name, path in paths.items()
        },
    }

    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError(out)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": payload["status"], "verdicts": payload["verdicts"]}, indent=2))
    if payload["status"] != "VALID_INTEGRATED_LEDGER":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
