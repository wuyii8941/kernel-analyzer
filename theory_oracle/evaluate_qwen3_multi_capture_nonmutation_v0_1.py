#!/usr/bin/env python
"""Evaluate whether multi-state capture changed a matched eager source trajectory."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "forkcert.qwen3-multi-capture-nonmutation.v0.1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def trainer_state_without_timing(value: dict[str, Any]) -> dict[str, Any]:
    result = json.loads(json.dumps(value))
    for row in result.get("log_history", []):
        row.pop("step_time", None)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-model-dir", required=True)
    parser.add_argument("--capture-model-dir", required=True)
    parser.add_argument("--baseline-prefix", required=True)
    parser.add_argument("--capture-prefix", required=True)
    parser.add_argument("--snapshot-audit", action="append", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    baseline_model = Path(args.baseline_model_dir).resolve()
    capture_model = Path(args.capture_model_dir).resolve()
    baseline_prefix = Path(args.baseline_prefix).resolve()
    capture_prefix = Path(args.capture_prefix).resolve()
    state_files = [
        "model.safetensors",
        "optimizer.pt",
        "rng_state.pth",
        "scaler.pt",
        "scheduler.pt",
    ]
    file_checks: dict[str, bool] = {}
    file_identities: dict[str, dict[str, str]] = {}
    for relative in state_files:
        left = baseline_model / "checkpoint-6" / relative
        right = capture_model / "checkpoint-6" / relative
        left_sha = sha256_file(left)
        right_sha = sha256_file(right)
        file_checks[f"checkpoint_{relative}_exact"] = left_sha == right_sha
        file_identities[relative] = {"baseline": left_sha, "capture": right_sha}
    top_left = sha256_file(baseline_model / "model.safetensors")
    top_right = sha256_file(capture_model / "model.safetensors")
    file_checks["final_exported_model_exact"] = top_left == top_right
    file_identities["final_exported_model"] = {"baseline": top_left, "capture": top_right}

    stream_checks: dict[str, bool] = {}
    stream_identities: dict[str, dict[str, str]] = {}
    for suffix in ("dump.jsonl", "samples.jsonl", "final_rollout.jsonl"):
        left = Path(f"{baseline_prefix}_{suffix}")
        right = Path(f"{capture_prefix}_{suffix}")
        left_sha = sha256_file(left)
        right_sha = sha256_file(right)
        stream_checks[f"{suffix}_exact"] = left_sha == right_sha
        stream_identities[suffix] = {"baseline": left_sha, "capture": right_sha}

    baseline_state = load_json(baseline_model / "checkpoint-6" / "trainer_state.json")
    capture_state = load_json(capture_model / "checkpoint-6" / "trainer_state.json")
    trainer_state_exact_except_timing = trainer_state_without_timing(
        baseline_state
    ) == trainer_state_without_timing(capture_state)
    timing_fields_differ = baseline_state != capture_state

    baseline_metadata = load_json(Path(f"{baseline_prefix}_dump.metadata.json"))
    capture_metadata = load_json(Path(f"{capture_prefix}_dump.metadata.json"))
    metadata_fields = [
        "training_kind",
        "advantage_source",
        "old_logp_source",
        "dataset_source",
        "config",
        "training_compute_dtype",
        "model_parameter_dtype",
        "trl_version",
        "torch_version",
        "resume_step",
        "resumed_from_checkpoint",
        "deterministic_warn_messages",
    ]
    core_metadata_equal = all(
        baseline_metadata.get(field) == capture_metadata.get(field) for field in metadata_fields
    )
    capture_plan_declared = (
        baseline_metadata.get("transition_capture_plan") is None
        and isinstance(capture_metadata.get("transition_capture_targets"), list)
        and len(capture_metadata["transition_capture_targets"]) == 2
    )

    audit_rows = []
    for raw in args.snapshot_audit:
        path = Path(raw).resolve()
        audit = load_json(path)
        metadata = load_json(Path(audit["snapshot_dir"]) / "forkcert_transition_snapshot.json")
        audit_rows.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "valid": bool(audit["valid"]),
                "capture_preservation_valid": bool(audit["capture_preservation_valid"]),
                "history_steps": audit["history_steps"],
                "expected_history_steps": audit["expected_history_steps"],
                "state_id": metadata.get("state_id"),
                "optimizer_step": metadata.get("optimizer_step"),
                "training_horizon_optimizer_steps": metadata.get(
                    "training_horizon_optimizer_steps"
                ),
                "plan_digest": metadata.get("transition_capture_plan_digest"),
            }
        )
    snapshot_checks = {
        "all_snapshot_audits_valid": all(row["valid"] for row in audit_rows),
        "all_capture_preservation_valid": all(
            row["capture_preservation_valid"] for row in audit_rows
        ),
        "all_history_sequences_exact": all(
            row["history_steps"] == row["expected_history_steps"] for row in audit_rows
        ),
        "training_horizon_recorded_as_six": all(
            row["training_horizon_optimizer_steps"] == 6 for row in audit_rows
        ),
        "one_shared_capture_plan": len({row["plan_digest"] for row in audit_rows}) == 1,
        "two_distinct_states": len({row["state_id"] for row in audit_rows}) == 2,
    }
    checks = {
        **file_checks,
        **stream_checks,
        "trainer_state_exact_except_wall_clock_step_time": trainer_state_exact_except_timing,
        "wall_clock_step_time_changed_as_expected": timing_fields_differ,
        "core_run_metadata_equal": core_metadata_equal,
        "capture_plan_only_on_treatment_run": capture_plan_declared,
        **snapshot_checks,
    }
    valid = all(checks.values())
    payload = {
        "schema_version": SCHEMA_VERSION,
        "valid": valid,
        "verdict": "MULTI_CAPTURE_NONMUTATING_SMOKE_PASS" if valid else "INVALID",
        "checks": checks,
        "file_identities": file_identities,
        "stream_identities": stream_identities,
        "snapshot_audits": audit_rows,
        "interpretation": (
            "Two selected-state captures changed wall-clock time but left model, optimizer, scheduler, scaler, RNG, token streams, and non-timing trainer state byte-identical over this six-step smoke."
            if valid
            else "The capture run differs from its no-capture source control; population collection must not start."
        ),
        "nonclaims": [
            "six steps do not validate 300-step resource reliability",
            "two capture targets do not validate all 96 calibration states",
            "non-mutation does not estimate eager-compiled bias",
            "snapshot validity does not prove compiled replay identity",
        ],
    }
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": payload["verdict"], "checks": checks}, indent=2, sort_keys=True))
    if not valid:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
