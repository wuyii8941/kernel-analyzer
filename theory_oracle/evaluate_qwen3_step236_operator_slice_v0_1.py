#!/usr/bin/env python
"""Join forward evidence with four fresh-process transition intervention arms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from forkcert.operator_evidence import (
    EvidenceGates,
    allowed_claim_level,
    canonical_json_sha256,
    compare_non_target_context,
    sha256_file,
)


def load(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def transition_context(row: dict[str, Any]) -> dict[str, Any]:
    audit = row["compile_context"]["after_transition"]
    graphs = []
    artifacts = []
    for region_id, record in audit.items():
        for graph in record["graphs"]:
            item = {"region_id": region_id, **graph}
            graphs.append(item)
            artifacts.append(
                {"target_id": region_id, "sha256": graph["sha256"], "node_count": graph["node_count"]}
            )
    return {
        "compiler_config_digest": None,
        "graph_count": sum(record["backend_compiles"] for record in audit.values()),
        "graphs": graphs,
        "artifacts": artifacts,
        "shape_layout_contracts": [],
        "autotuning": row["compile_context"]["autotuning"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forward-report", required=True)
    parser.add_argument("--candidate-1", required=True)
    parser.add_argument("--candidate-2", required=True)
    parser.add_argument("--repair-1", required=True)
    parser.add_argument("--repair-2", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    report = load(args.forward_report)
    candidate = [load(args.candidate_1), load(args.candidate_2)]
    repair = [load(args.repair_1), load(args.repair_2)]
    errors = []
    if not all(row.get("valid") for row in candidate + repair):
        errors.append("one or more transition arms are invalid")
    if not all(row["anchors"]["candidate_control_anchor_exact"] for row in candidate + repair):
        errors.append("partitioned candidate control did not preserve scorer anchor")
    if not all(row["anchors"]["split_candidate_transition_exact"] for row in candidate):
        errors.append("partitioned candidate did not preserve original gradient/update transition")
    pre_states = [row["pre_state"] for row in candidate + repair]
    if any(row != pre_states[0] for row in pre_states[1:]):
        errors.append("transition pre-states differ")

    candidate_repeat_exact = all(
        candidate[0][key] == candidate[1][key]
        for key in ("continuous", "semantic", "post_state", "vector_artifacts")
        if key != "vector_artifacts"
    ) and all(
        candidate[0]["vector_artifacts"][name]["sha256"]
        == candidate[1]["vector_artifacts"][name]["sha256"]
        for name in ("clipped_gradients", "parameter_updates")
    )
    repair_repeat_exact = all(
        repair[0][key] == repair[1][key]
        for key in ("continuous", "semantic", "post_state")
    ) and all(
        repair[0]["vector_artifacts"][name]["sha256"]
        == repair[1]["vector_artifacts"][name]["sha256"]
        for name in ("clipped_gradients", "parameter_updates")
    )
    if not candidate_repeat_exact or not repair_repeat_exact:
        errors.append("fresh-process transition repeats are not exact")

    context_checks = [
        compare_non_target_context(
            transition_context(candidate[index]),
            transition_context(repair[index]),
            ["final_rmsnorm"],
        )
        for index in range(2)
    ]
    observed_context_exact = all(row["exact"] for row in context_checks)
    if not observed_context_exact:
        errors.append("observed non-target graph context changed under repair")

    # Compute the one-step vector contrasts without treating coordinates as samples.
    import torch
    from safetensors.torch import load_file

    manifest = load(report["case_identity"]["manifest"])
    eager_baseline = load(manifest["baseline_records"]["eager_1"])
    eager_update = load_file(eager_baseline["vector_artifacts"]["parameter_updates"]["path"])
    candidate_update = load_file(candidate[0]["vector_artifacts"]["parameter_updates"]["path"])
    repair_update = load_file(repair[0]["vector_artifacts"]["parameter_updates"]["path"])

    def vector_contrast(left: dict[str, Any], right: dict[str, Any]) -> dict[str, float]:
        square = 0.0
        dot_ref = 0.0
        ref_square = 0.0
        for name in sorted(left):
            delta = right[name].double() - left[name].double()
            square += float((delta * delta).sum().item())
            reference = left[name].double()
            dot_ref += float((delta * reference).sum().item())
            ref_square += float((reference * reference).sum().item())
        return {
            "delta_l2": square ** 0.5,
            "reference_aligned_dot": dot_ref,
            "reference_l2": ref_square ** 0.5,
        }

    update_oracle = {
        "reference_to_candidate": vector_contrast(eager_update, candidate_update),
        "reference_to_repair": vector_contrast(eager_update, repair_update),
        "candidate_to_repair": vector_contrast(candidate_update, repair_update),
        "candidate_update_artifact_sha256": candidate[0]["vector_artifacts"]["parameter_updates"]["sha256"],
        "repair_update_artifact_sha256": repair[0]["vector_artifacts"]["parameter_updates"]["sha256"],
    }

    def event_contrast(left: list[list[bool]], right: list[list[bool]]) -> dict[str, Any]:
        left_tensor = torch.tensor(left, dtype=torch.bool)
        right_tensor = torch.tensor(right, dtype=torch.bool)
        upward = (~left_tensor) & right_tensor
        downward = left_tensor & (~right_tensor)
        count = left_tensor.numel()
        return {
            "off_to_on": int(upward.sum().item()),
            "on_to_off": int(downward.sum().item()),
            "signed_semantic_effect": float((upward.sum() - downward.sum()).item() / count),
            "semantic_disagreement": float((upward.sum() + downward.sum()).item() / count),
        }

    transition_oracle = {
        "candidate_repeat_exact": candidate_repeat_exact,
        "repair_repeat_exact": repair_repeat_exact,
        "candidate_clip_count": candidate[0]["semantic"]["clip_count"],
        "repair_clip_count": repair[0]["semantic"]["clip_count"],
        "reference_to_candidate": event_contrast(
            eager_baseline["semantic"]["clip_decisions"], candidate[0]["semantic"]["clip_decisions"]
        ),
        "reference_to_repair": event_contrast(
            eager_baseline["semantic"]["clip_decisions"], repair[0]["semantic"]["clip_decisions"]
        ),
        "candidate_to_repair": event_contrast(
            candidate[0]["semantic"]["clip_decisions"], repair[0]["semantic"]["clip_decisions"]
        ),
        "update": update_oracle,
    }
    report["oracle"]["one_step_transition"] = transition_oracle
    report["intervention"]["one_step_transition_status"] = (
        "VALID" if not errors else "INVALID_OR_COARSE"
    )
    report["intervention"]["transition_arms"] = {
        "candidate": [
            {"path": path, "sha256": sha256_file(path)}
            for path in (args.candidate_1, args.candidate_2)
        ],
        "repair": [
            {"path": path, "sha256": sha256_file(path)}
            for path in (args.repair_1, args.repair_2)
        ],
    }
    report["intervention"]["transition_non_target_context"] = context_checks
    old_gates = report["gates"]
    gates = EvidenceGates(
        complete_witness=old_gates["complete_witness"],
        same_input_local_replay=old_gates["same_input_local_replay"],
        local_discrepancy_reproducible=old_gates["local_discrepancy_reproducible"],
        provenance_complete=old_gates["provenance_complete"],
        candidate_realization_preserved=(
            old_gates["candidate_realization_preserved"]
            and all(row["anchors"]["split_candidate_transition_exact"] for row in candidate)
        ),
        intervention_executed=True,
        oracle_recomputed=True,
        non_target_context_invariant=observed_context_exact,
        null_controls_valid=(candidate_repeat_exact and repair_repeat_exact),
    )
    report["gates"] = gates.__dict__
    report["allowed_claim_level"] = allowed_claim_level(gates)
    report["limitations"] = [
        item for item in report["limitations"] if item != "forward repair is not yet a one-step update intervention"
    ]
    report["limitations"].extend(
        [
            "autotuning variant identity remains unobserved",
            "observed graph-hash invariance does not prove every layout/fusion runtime choice invariant",
            "transition intervention is selected-state and intervention-dependent",
        ]
    )
    report["evaluation_errors"] = errors
    report["content_sha256"] = canonical_json_sha256(
        {key: value for key, value in report.items() if key != "content_sha256"}
    )
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "allowed_claim_level": report["allowed_claim_level"],
                "errors": errors,
                "candidate_repeat_exact": candidate_repeat_exact,
                "repair_repeat_exact": repair_repeat_exact,
                "observed_context_exact": observed_context_exact,
                "transition_oracle": transition_oracle,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
