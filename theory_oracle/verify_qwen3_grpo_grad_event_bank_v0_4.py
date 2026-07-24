#!/usr/bin/env python
"""Independent preflight and result audit for the Qwen GRPO grad event bank."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import sys
from pathlib import Path
from typing import Any

from forkcert.detector import clip_active


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class Audit:
    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []

    def add(self, name: str, passed: bool, detail: Any = None) -> None:
        self.checks.append({"name": name, "passed": bool(passed), "detail": detail})

    @property
    def valid(self) -> bool:
        return all(row["passed"] for row in self.checks)


def artifact_path(manifest: dict[str, Any], name: str) -> Path:
    return Path(manifest["workspace_root"]) / manifest["artifacts"][name]["path"]


def output_path(manifest: dict[str, Any], trajectory: str, name: str) -> Path:
    return Path(manifest["workspace_root"]) / manifest["outputs"][trajectory][name]


def preflight(manifest: dict[str, Any], audit: Audit) -> None:
    audit.add(
        "manifest_frozen_before_execution",
        manifest.get("status") == "FROZEN_PRE_EXECUTION",
        manifest.get("status"),
    )
    for name, row in manifest.get("artifacts", {}).items():
        path = Path(manifest["workspace_root"]) / row["path"]
        audit.add(f"artifact_exists:{name}", path.is_file(), str(path))
        if path.is_file():
            actual = sha256_file(path)
            audit.add(
                f"artifact_hash:{name}",
                actual == row["sha256"],
                {"expected": row["sha256"], "actual": actual},
            )
    for trajectory in ("A", "B"):
        config = json.loads(json.dumps(manifest["expected"]["trajectories"][trajectory]))
        import yaml

        actual = yaml.safe_load(artifact_path(manifest, f"config_{trajectory.lower()}").read_text())
        audit.add(f"config_identity:{trajectory}", actual == config, actual)
    audit.add(
        "selection_rule_frozen",
        manifest.get("expected", {}).get("witness_selection_rule")
        == "first stable disagreement ordered by trajectory, optimizer_step, rollout_batch, flat_index",
    )
    runtime = manifest.get("expected", {}).get("runtime", {})
    import torch

    actual_runtime = {
        "python_executable": sys.executable,
        "torch": torch.__version__,
        "trl": importlib.metadata.version("trl"),
        "transformers": importlib.metadata.version("transformers"),
        "accelerate": importlib.metadata.version("accelerate"),
    }
    audit.add("runtime_identity", actual_runtime == runtime, actual_runtime)


def event_direction(row: dict[str, Any], repeat: str) -> str:
    sign = int(row["advantage_sign"])
    old = float(row["old_logp"])
    reference = clip_active(float(row[f"logp_ref_{repeat}"]), old, sign, 0.2)
    candidate = clip_active(float(row[f"logp_alt_{repeat}"]), old, sign, 0.2)
    if not reference and candidate:
        return "0->1"
    if reference and not candidate:
        return "1->0"
    return "same"


def audit_trajectory(
    manifest: dict[str, Any], trajectory: str, result_row: dict[str, Any], audit: Audit
) -> tuple[list[dict[str, Any]], int]:
    tokens_path = output_path(manifest, trajectory, "tokens")
    states_path = output_path(manifest, trajectory, "states")
    metadata_path = output_path(manifest, trajectory, "metadata")
    margin_path = output_path(manifest, trajectory, "margin")
    samples_path = output_path(manifest, trajectory, "samples")
    rollout_path = output_path(manifest, trajectory, "final_rollout")
    final_model_path = output_path(manifest, trajectory, "final_model")
    for name, path in (
        ("tokens", tokens_path),
        ("states", states_path),
        ("metadata", metadata_path),
        ("margin", margin_path),
        ("samples", samples_path),
        ("final_rollout", rollout_path),
        ("final_model", final_model_path),
    ):
        audit.add(f"output_exists:{trajectory}:{name}", path.is_file(), str(path))
    if not all(
        path.is_file()
        for path in (
            tokens_path,
            states_path,
            metadata_path,
            margin_path,
            samples_path,
            rollout_path,
            final_model_path,
        )
    ):
        return [], 0

    baseline_prefix = f"baseline_{trajectory.lower()}"
    parity = {
        "margin": sha256_file(margin_path)
        == sha256_file(artifact_path(manifest, baseline_prefix + "_margin")),
        "samples": sha256_file(samples_path)
        == sha256_file(artifact_path(manifest, baseline_prefix + "_samples")),
        "final_rollout": sha256_file(rollout_path)
        == sha256_file(artifact_path(manifest, baseline_prefix + "_final_rollout")),
        "final_model": sha256_file(final_model_path)
        == sha256_file(artifact_path(manifest, baseline_prefix + "_final_model")),
    }
    audit.add(f"trajectory_noninterference:{trajectory}", all(parity.values()), parity)

    rows, states = read_jsonl(tokens_path), read_jsonl(states_path)
    audit.add(f"token_rows:{trajectory}", len(rows) == 5120, len(rows))
    audit.add(f"state_rows:{trajectory}", len(states) == 10, len(states))
    state_ids = {str(row["state_id"]) for row in states}
    audit.add(
        f"state_token_coverage:{trajectory}",
        {str(row["state_id"]) for row in rows} == state_ids
        and all(sum(str(row["state_id"]) == state_id for row in rows) == 512 for state_id in state_ids),
    )
    state_integrity = all(
        bool(row.get(field))
        for row in states
        for field in (
            "autograd_enabled",
            "all_outputs_require_grad",
            "accelerate_native_amp",
            "accelerate_forward_wrapped",
            "candidate_identity_valid",
            "gradients_preserved",
            "tensor_versions_preserved",
            "trainer_steps_preserved",
            "rng_restored_exactly",
        )
    )
    audit.add(f"state_integrity:{trajectory}", state_integrity)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    compile_audit = metadata.get("grad_compile_audit") or {}
    audit.add(
        f"compiled_realization:{trajectory}",
        int(compile_audit.get("backend_compiles", 0)) > 0
        and int(compile_audit.get("runtime_invocations", 0)) >= 30
        and bool(compile_audit.get("graph_code_sha256")),
        compile_audit,
    )
    environment = metadata.get("environment") or {}
    audit.add(
        f"gpu_environment:{trajectory}",
        environment.get("gpu_name") == "Tesla T4"
        and list(environment.get("gpu_capability") or []) == [7, 5]
        and environment.get("deterministic_algorithms") is True
        and environment.get("cudnn_benchmark") is False,
        environment,
    )
    audit.add(
        f"result_input_hashes:{trajectory}",
        result_row.get("token_sha256") == sha256_file(tokens_path)
        and result_row.get("state_sha256") == sha256_file(states_path)
        and result_row.get("metadata_sha256") == sha256_file(metadata_path),
    )

    stable_events: list[dict[str, Any]] = []
    unstable = 0
    for row in rows:
        if int(row["advantage_sign"]) == 0:
            continue
        first, second = event_direction(row, "first"), event_direction(row, "second")
        old, sign = float(row["old_logp"]), int(row["advantage_sign"])
        ref_first = clip_active(float(row["logp_ref_first"]), old, sign, 0.2)
        ref_second = clip_active(float(row["logp_ref_second"]), old, sign, 0.2)
        alt_first = clip_active(float(row["logp_alt_first"]), old, sign, 0.2)
        alt_second = clip_active(float(row["logp_alt_second"]), old, sign, 0.2)
        stable = first == second and ref_first == ref_second and alt_first == alt_second
        if not stable:
            unstable += 1
        elif first != "same":
            stable_events.append(
                {
                    "trajectory": trajectory,
                    "optimizer_step": int(row["optimizer_step"]),
                    "rollout_batch": int(row["rollout_batch"]),
                    "flat_index": int(row["flat_index"]),
                    "direction": first,
                    "case_id": str(row["case_id"]),
                    "token_index": int(row["token_index"]),
                }
            )
    audit.add(
        f"event_counts:{trajectory}",
        len(stable_events) == int(result_row["stable_event_count"])
        and unstable == int(result_row["repeat_unstable_event_count"]),
        {"stable": len(stable_events), "unstable": unstable},
    )
    return stable_events, unstable


def audit_result(manifest: dict[str, Any], result: dict[str, Any], audit: Audit) -> None:
    audit.add(
        "result_schema",
        result.get("schema_version") == "forkcert.qwen3-grpo-grad-event-bank.v0.4",
        result.get("schema_version"),
    )
    audit.add("mechanics_valid", result.get("mechanics_valid") is True)
    audit.add("correctness_withheld", result.get("compiler_correctness") == "NO CLAIM")
    audit.add("update_effect_withheld", result.get("update_effect") == "NOT IN SCOPE")
    by_name = {row.get("trajectory"): row for row in result.get("trajectories", [])}
    audit.add("two_trajectories", set(by_name) == {"A", "B"})
    if set(by_name) != {"A", "B"}:
        return
    events: list[dict[str, Any]] = []
    unstable = 0
    for trajectory in ("A", "B"):
        found, count = audit_trajectory(manifest, trajectory, by_name[trajectory], audit)
        events.extend(found)
        unstable += count
    events.sort(
        key=lambda row: (
            row["trajectory"], row["optimizer_step"], row["rollout_batch"], row["flat_index"]
        )
    )
    reported = result.get("events", [])
    reported_keys = [
        (
            row["trajectory"],
            int(row["optimizer_step"]),
            int(row["rollout_batch"]),
            int(row["flat_index"]),
            row["direction"],
            row["case_id"],
            int(row["token_index"]),
        )
        for row in reported
    ]
    independent_keys = [
        (
            row["trajectory"],
            row["optimizer_step"],
            row["rollout_batch"],
            row["flat_index"],
            row["direction"],
            row["case_id"],
            row["token_index"],
        )
        for row in events
    ]
    audit.add("independent_event_identity", reported_keys == independent_keys)
    audit.add("independent_unstable_count", result.get("repeat_unstable_event_count") == unstable)
    expected_first = reported[0] if reported else None
    audit.add(
        "frozen_witness_selection",
        result.get("first_stable_event_for_one_step_followup") == expected_first,
    )
    verdict = result.get("finite_bank_grad_context_compatibility_verdict")
    expected_verdict = "INDETERMINATE" if unstable else "REJECT" if events else "ACCEPT"
    audit.add("finite_bank_verdict", verdict == expected_verdict, {"expected": expected_verdict, "actual": verdict})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--result")
    parser.add_argument("--out-audit")
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    audit = Audit()
    preflight(manifest, audit)
    mode = "preflight"
    if args.result:
        mode = "result"
        result_path = Path(args.result)
        audit.add("result_exists", result_path.is_file(), str(result_path))
        if result_path.is_file():
            audit_result(
                manifest, json.loads(result_path.read_text(encoding="utf-8")), audit
            )
    payload = {
        "schema_version": "forkcert.qwen3-grpo-grad-event-bank-audit.v0.4",
        "mode": mode,
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
