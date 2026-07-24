#!/usr/bin/env python
"""Independent preflight and result audit for Qwen GRPO branch repair v0.3."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def model_files(directory: Path) -> list[Path]:
    index = directory / "model.safetensors.index.json"
    if index.exists():
        payload = json.loads(index.read_text(encoding="utf-8"))
        return [directory / name for name in sorted(set(payload["weight_map"].values()))]
    files = sorted(directory.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no safetensors model weights in {directory}")
    return files


def tensor_index(directory: Path) -> dict[str, Path]:
    from safetensors import safe_open

    result: dict[str, Path] = {}
    for path in model_files(directory):
        with safe_open(path, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key in result:
                    raise ValueError(f"duplicate tensor key {key}")
                result[key] = path
    return result


def independent_l2(left: Path, right: Path) -> dict[str, float]:
    import torch
    from safetensors import safe_open

    left_index, right_index = tensor_index(left), tensor_index(right)
    if left_index.keys() != right_index.keys():
        raise ValueError("model state keys differ")
    difference_square = 0.0
    left_square = 0.0
    for key in sorted(left_index):
        with safe_open(left_index[key], framework="pt", device="cpu") as left_handle:
            left_tensor = left_handle.get_tensor(key).float()
        with safe_open(right_index[key], framework="pt", device="cpu") as right_handle:
            right_tensor = right_handle.get_tensor(key).float()
        difference_square += float(
            torch.sum((left_tensor - right_tensor).double().square()).item()
        )
        left_square += float(torch.sum(left_tensor.double().square()).item())
    distance, norm = math.sqrt(difference_square), math.sqrt(left_square)
    return {"l2": distance, "relative_l2": distance / norm if norm else 0.0}


class Audit:
    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []

    def add(self, name: str, passed: bool, detail: Any = None) -> None:
        self.checks.append({"name": name, "passed": bool(passed), "detail": detail})

    @property
    def valid(self) -> bool:
        return all(item["passed"] for item in self.checks)


def artifact_path(manifest: dict[str, Any], name: str) -> Path:
    root = Path(manifest["workspace_root"])
    return root / manifest["artifacts"][name]["path"]


def audit_preflight(manifest: dict[str, Any], audit: Audit) -> None:
    audit.add(
        "manifest_frozen_before_execution",
        manifest.get("status") == "FROZEN_PRE_EXECUTION",
        manifest.get("status"),
    )
    for name, item in manifest["artifacts"].items():
        path = Path(manifest["workspace_root"]) / item["path"]
        exists = path.is_file()
        audit.add(f"artifact_exists:{name}", exists, str(path))
        if exists:
            actual = sha256_file(path)
            audit.add(
                f"artifact_hash:{name}",
                actual == item["sha256"],
                {"expected": item["sha256"], "actual": actual},
            )

    evaluation = json.loads(artifact_path(manifest, "evaluation").read_text(encoding="utf-8"))
    event = evaluation.get("first_event_for_one_step_followup")
    audit.add("parent_mechanics_valid", evaluation.get("mechanics_valid") is True)
    audit.add("selected_event_frozen", event == manifest["selected_event"])

    rows = read_jsonl(artifact_path(manifest, "reconstruction_online"))
    matches = [
        row
        for row in rows
        if int(row.get("optimizer_step", -1)) == int(event["optimizer_step"])
        and str(row.get("case_id")) == str(event["case_id"])
        and int(row.get("token_index", -1)) == int(event["token_index"])
    ]
    audit.add("unique_reconstruction_anchor", len(matches) == 1, len(matches))
    if len(matches) == 1:
        row = matches[0]
        exact = (
            float(row["old_logp"]) == float(event["old_logp"])
            and float(row["logp_ref"]) == float(event["logp_ref"])
            and float(row["logp_alt"]) == float(event["logp_alt"])
            and int(row["advantage_sign"]) == int(event["advantage_sign"])
            and bool(row["candidate_identity_valid"])
            and float(row["delta_self_ref"]) == 0.0
            and float(row["delta_self_alt"]) == 0.0
        )
        audit.add("reconstruction_anchor_exact", exact)

    states = read_jsonl(artifact_path(manifest, "states"))
    samples = read_jsonl(artifact_path(manifest, "samples"))
    selected_states = [
        row
        for row in states
        if int(row.get("optimizer_step", -1)) == int(event["optimizer_step"])
        and int(row.get("rollout_batch", -1)) == int(event["rollout_batch"])
        and str(row.get("state")) == "pre_minibatch"
    ]
    cases = {str(row["case_id"]) for row in selected_states}
    selected_samples = [row for row in samples if str(row["case_id"]) in cases]
    state_map = {
        (str(row["case_id"]), int(row["token_index"])): row for row in selected_states
    }
    aligned = [
        state_map[(str(sample["case_id"]), index)]
        for sample in selected_samples
        for index in range(len(sample["response_ids"]))
    ]
    targets = [
        index
        for index, row in enumerate(aligned)
        if str(row["case_id"]) == str(event["case_id"])
        and int(row["token_index"]) == int(event["token_index"])
    ]
    audit.add(
        "frozen_batch_shape",
        len(selected_samples) == 4
        and len(aligned) == 512
        and all(len(sample["response_ids"]) == 128 for sample in selected_samples),
        {"samples": len(selected_samples), "tokens": len(aligned)},
    )
    audit.add(
        "target_flat_index",
        targets == [int(manifest["expected"]["target_flat_index"])],
        targets,
    )


def close(left: float, right: float, *, absolute: float = 1e-9) -> bool:
    return math.isclose(float(left), float(right), rel_tol=1e-7, abs_tol=absolute)


def audit_result(
    manifest: dict[str, Any], result: dict[str, Any], audit: Audit, recompute_weights: bool
) -> None:
    event = manifest["selected_event"]
    expected = manifest["expected"]
    audit.add(
        "result_schema",
        result.get("schema_version") == "forkcert.qwen3-grpo-one-step-branch-repair.v0.3",
        result.get("schema_version"),
    )
    audit.add(
        "executor_did_not_refuse",
        result.get("status") == "MECHANICALLY_VALID_PENDING_INDEPENDENT_AUDIT",
        result.get("status"),
    )
    if result.get("status") == "INVALID":
        return
    audit.add("result_event_identity", result.get("event") == event)
    audit.add("reconstruction_exact", result.get("reconstruction_exact") is True)
    audit.add("endpoint_anchor_parity", result.get("trainer_realization_anchor_parity") is True)
    audit.add("full_batch_bc_identity", result.get("full_batch_bc_scoring_identity") is True)
    audit.add("branch_functional_gate", result.get("reference_branch_functional_gate") is True)
    audit.add("intervention_integrity", result.get("intervention_integrity_valid") is True)
    audit.add("correctness_claim_withheld", result.get("numerical_correctness") == "UNINSTANTIATED")

    arms = {arm["arm"]: arm for arm in result.get("arms", [])}
    audit.add(
        "three_unique_arms",
        set(arms) == {"A_reference", "B_candidate", "C_branch_repair"}
        and len(result.get("arms", [])) == 3,
        sorted(arms),
    )
    if set(arms) != {"A_reference", "B_candidate", "C_branch_repair"}:
        return
    a_arm, b_arm, c_arm = arms["A_reference"], arms["B_candidate"], arms["C_branch_repair"]

    audit.add("A_eager", a_arm["compiled"] is False and a_arm["forced_target_clip"] is None)
    audit.add(
        "B_compiled_ordinary",
        b_arm["compiled"] is True
        and b_arm["forced_target_clip"] is None
        and b_arm["candidate_identity_valid"] is True
        and int(b_arm["compile_audit"]["scored_invocations"]) > 0,
    )
    audit.add(
        "C_compiled_reference_branch",
        c_arm["compiled"] is True
        and c_arm["forced_target_clip"] is bool(event["ref_clip"])
        and c_arm["candidate_identity_valid"] is True
        and int(c_arm["compile_audit"]["scored_invocations"]) > 0,
    )
    audit.add(
        "exact_selected_anchors",
        a_arm["target_logp"] == event["logp_ref"]
        and b_arm["target_logp"] == event["logp_alt"]
        and c_arm["target_logp"] == event["logp_alt"],
    )
    audit.add(
        "shared_target_inputs",
        len({arm["target_flat_index"] for arm in arms.values()}) == 1
        and len({arm["target_old_logp"] for arm in arms.values()}) == 1
        and len({arm["target_advantage"] for arm in arms.values()}) == 1,
    )
    audit.add(
        "complete_BC_score_hash_equal",
        b_arm["measured_logps_sha256"] == c_arm["measured_logps_sha256"],
    )
    audit.add(
        "BC_graph_identity_equal",
        b_arm["compile_audit"]["graph_hashes"] == c_arm["compile_audit"]["graph_hashes"]
        and b_arm["compile_audit"]["graph_nodes"] == c_arm["compile_audit"]["graph_nodes"],
    )
    audit.add(
        "all_scoring_calls_self_stable",
        all(arm["scoring_self_stable"] for arm in arms.values()),
    )

    epsilon = float(expected["epsilon"])
    lower, upper = 1.0 - epsilon, 1.0 + epsilon
    advantage = float(b_arm["target_advantage"])
    a_ratio = math.exp(float(a_arm["target_logp"]) - float(a_arm["target_old_logp"]))
    b_ratio = math.exp(float(b_arm["target_logp"]) - float(b_arm["target_old_logp"]))
    reference_clips = (advantage > 0.0 and a_ratio > upper) or (
        advantage < 0.0 and a_ratio < lower
    )
    candidate_clips = (advantage > 0.0 and b_ratio > upper) or (
        advantage < 0.0 and b_ratio < lower
    )
    audit.add(
        "independent_event_decisions",
        reference_clips is bool(event["ref_clip"])
        and candidate_clips is bool(event["alt_clip"])
        and reference_clips != candidate_clips,
        {"reference": reference_clips, "candidate": candidate_clips},
    )
    batch_tokens = int(expected["batch_tokens"])
    expected_b_gradient = -advantage * b_ratio / batch_tokens
    audit.add("A_flat_gradient_zero", a_arm["target_logp_loss_gradient"] == 0.0)
    audit.add(
        "B_ordinary_gradient",
        close(b_arm["target_logp_loss_gradient"], expected_b_gradient, absolute=1e-8)
        and b_arm["target_logp_loss_gradient"] != 0.0,
        {"expected": expected_b_gradient, "actual": b_arm["target_logp_loss_gradient"]},
    )
    audit.add("C_repaired_flat_gradient_zero", c_arm["target_logp_loss_gradient"] == 0.0)

    distances = result["distances"]
    finite = all(
        math.isfinite(float(distances[pair][field])) and float(distances[pair][field]) >= 0.0
        for pair in ("A_B", "A_C", "B_C")
        for field in ("l2", "relative_l2")
    )
    audit.add("finite_nonnegative_distances", finite)
    signed = float(distances["A_B"]["l2"]) - float(distances["A_C"]["l2"])
    audit.add(
        "signed_repair_effect_recomputed",
        close(result["reference_directed_repair_effect_l2"], signed, absolute=1e-15),
    )
    denominator = float(distances["A_B"]["l2"])
    expected_residual = float(distances["A_C"]["l2"]) / denominator if denominator else None
    audit.add(
        "residual_ratio_recomputed",
        (expected_residual is None and result["residual_ratio_A_C_over_A_B"] is None)
        or (
            expected_residual is not None
            and close(result["residual_ratio_A_C_over_A_B"], expected_residual, absolute=1e-15)
        ),
    )

    if recompute_weights:
        root = Path(manifest["workspace_root"])
        weights = root / manifest["weights_dir"]
        independent = {
            "A_B": independent_l2(weights / "A_reference", weights / "B_candidate"),
            "A_C": independent_l2(weights / "A_reference", weights / "C_branch_repair"),
            "B_C": independent_l2(weights / "B_candidate", weights / "C_branch_repair"),
        }
        for pair in independent:
            audit.add(
                f"independent_weight_distance:{pair}",
                close(independent[pair]["l2"], distances[pair]["l2"], absolute=1e-15)
                and close(
                    independent[pair]["relative_l2"],
                    distances[pair]["relative_l2"],
                    absolute=1e-18,
                ),
                {"reported": distances[pair], "independent": independent[pair]},
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--result")
    parser.add_argument("--out-audit")
    parser.add_argument("--skip-weight-recompute", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    audit = Audit()
    try:
        audit_preflight(manifest, audit)
        if args.result:
            result = json.loads(Path(args.result).read_text(encoding="utf-8"))
            audit_result(manifest, result, audit, not args.skip_weight_recompute)
    except Exception as error:
        audit.add(
            "verifier_exception",
            False,
            {"type": type(error).__name__, "message": str(error)},
        )

    payload = {
        "schema_version": "forkcert.qwen3-grpo-one-step-branch-repair-audit.v0.3",
        "mode": "result" if args.result else "preflight",
        "verdict": "VALID" if audit.valid else "INVALID",
        "checks": audit.checks,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.out_audit:
        Path(args.out_audit).write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    raise SystemExit(0 if audit.valid else 1)


if __name__ == "__main__":
    main()
