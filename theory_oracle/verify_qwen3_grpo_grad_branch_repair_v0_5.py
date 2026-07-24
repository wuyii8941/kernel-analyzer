#!/usr/bin/env python
"""Independent audit for the frozen Qwen grad-context branch repair v0.5."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from theory_oracle.verify_qwen3_grpo_branch_repair_v0_3 import independent_l2


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Audit:
    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []

    def add(self, name: str, passed: bool, detail: Any = None) -> None:
        self.checks.append({"name": name, "passed": bool(passed), "detail": detail})

    @property
    def valid(self) -> bool:
        return all(row["passed"] for row in self.checks)


def preflight(manifest: dict[str, Any], audit: Audit) -> None:
    audit.add(
        "manifest_frozen_before_execution",
        manifest.get("status") == "FROZEN_PRE_EXECUTION",
        manifest.get("status"),
    )
    root = Path(manifest["workspace_root"])
    for name, row in manifest["artifacts"].items():
        path = root / row["path"]
        audit.add(f"artifact_exists:{name}", path.is_file(), str(path))
        if path.is_file():
            actual = sha256_file(path)
            audit.add(
                f"artifact_hash:{name}", actual == row["sha256"],
                {"expected": row["sha256"], "actual": actual},
            )
    evaluation = json.loads((root / manifest["artifacts"]["evaluation"]["path"]).read_text())
    audit.add(
        "selected_event_identity",
        evaluation.get("first_stable_event_for_one_step_followup") == manifest["selected_event"],
    )


def result_audit(
    manifest: dict[str, Any], result: dict[str, Any], audit: Audit
) -> dict[str, Any] | None:
    audit.add(
        "result_schema",
        result.get("schema_version") == "forkcert.qwen3-grpo-grad-branch-repair.v0.5",
    )
    audit.add(
        "executor_acceptance",
        result.get("status") == "MECHANICALLY_VALID_PENDING_INDEPENDENT_DISTANCE_AUDIT",
        result.get("status"),
    )
    if result.get("status") == "INVALID":
        return None
    audit.add("event_identity", result.get("event") == manifest["selected_event"])
    audit.add("intervention_integrity", result.get("intervention_integrity_valid") is True)
    audit.add("correctness_withheld", result.get("compiler_correctness") == "NO CLAIM")
    audit.add(
        "natural_update_withheld", result.get("natural_training_update_effect") == "NOT CLAIMED"
    )
    arms = {row["arm"]: row for row in result.get("arms", [])}
    audit.add(
        "three_unique_arms",
        set(arms) == {"A_reference", "B_candidate", "C_branch_repair"}
        and len(result.get("arms", [])) == 3,
        sorted(arms),
    )
    if set(arms) != {"A_reference", "B_candidate", "C_branch_repair"}:
        return None
    event = manifest["selected_event"]
    expected = manifest["expected"]
    a, b, c = arms["A_reference"], arms["B_candidate"], arms["C_branch_repair"]
    audit.add(
        "arm_treatments",
        a["compiled"] is False
        and a["forced_reference_clip"] is None
        and b["compiled"] is True
        and b["forced_reference_clip"] is None
        and c["compiled"] is True
        and c["forced_reference_clip"] is bool(event["ref_clip"]),
    )
    audit.add(
        "exact_scalar_anchors",
        a["target_logp"] == event["logp_ref"]
        and b["target_logp"] == event["logp_alt"]
        and c["target_logp"] == event["logp_alt"]
        and len({row["target_old_logp"] for row in arms.values()}) == 1
        and a["target_old_logp"] == event["old_logp"],
    )
    audit.add(
        "complete_tensor_anchors",
        set(a["scorer_call_sha256"]) == {expected["reference_scorer_sha256"]}
        and set(b["scorer_call_sha256"]) == {expected["candidate_scorer_sha256"]}
        and b["scorer_call_sha256"] == c["scorer_call_sha256"],
    )
    audit.add(
        "compiled_graph_identity",
        b["compile_audit"]["graph_hashes"] == c["compile_audit"]["graph_hashes"]
        and b["compile_audit"]["graph_nodes"] == c["compile_audit"]["graph_nodes"],
    )
    audit.add(
        "compiled_runtime_identity",
        b["candidate_identity_valid"] is True
        and c["candidate_identity_valid"] is True
        and all(value > 0 for value in b["compile_audit"]["per_call_runtime_invocations"])
        and all(value > 0 for value in c["compile_audit"]["per_call_runtime_invocations"]),
    )
    audit.add(
        "branch_gradient_realization",
        a["target_logp_loss_gradient"] != 0.0
        and b["target_logp_loss_gradient"] == 0.0
        and c["target_logp_loss_gradient"] != 0.0,
        {
            "A": a["target_logp_loss_gradient"],
            "B": b["target_logp_loss_gradient"],
            "C": c["target_logp_loss_gradient"],
        },
    )
    root = Path(manifest["workspace_root"])
    weight_root = root / manifest["outputs"]["weights_dir"]
    distances = {
        "A_B": independent_l2(weight_root / "A_reference", weight_root / "B_candidate"),
        "A_C": independent_l2(weight_root / "A_reference", weight_root / "C_branch_repair"),
        "B_C": independent_l2(weight_root / "B_candidate", weight_root / "C_branch_repair"),
    }
    audit.add("finite_distances", all(value >= 0.0 for row in distances.values() for value in row.values()))
    total = distances["A_B"]["l2"]
    repair = distances["B_C"]["l2"]
    repaired_residual = distances["A_C"]["l2"]
    return {
        "distances": distances,
        "total_effect_l2": total,
        "repair_effect_l2": repair,
        "reference_directed_repair_effect_l2": total - repaired_residual,
        "repair_residual_ratio": repaired_residual / total if total else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--result")
    parser.add_argument("--out-audit")
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    audit = Audit()
    preflight(manifest, audit)
    metrics = None
    mode = "preflight"
    if args.result:
        mode = "result"
        path = Path(args.result)
        audit.add("result_exists", path.is_file(), str(path))
        if path.is_file():
            metrics = result_audit(manifest, json.loads(path.read_text()), audit)
    payload = {
        "schema_version": "forkcert.qwen3-grpo-grad-branch-repair-audit.v0.5",
        "mode": mode,
        "verdict": "VALID" if audit.valid else "INVALID",
        "checks": audit.checks,
        "metrics": metrics,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.out_audit:
        Path(args.out_audit).write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    raise SystemExit(0 if audit.valid else 1)


if __name__ == "__main__":
    main()
