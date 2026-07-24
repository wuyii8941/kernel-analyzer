#!/usr/bin/env python
"""Fail-closed audit of every state in one frozen calibration capture plan."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from theory_oracle.verify_qwen3_grpo_transition_snapshot_v0_1 import audit


SCHEMA_VERSION = "forkcert.qwen3-calibration-capture-batch-audit.v0.1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_snapshot_audit(snapshot_dir_raw: str, audit_path_raw: str) -> dict[str, Any]:
    snapshot_dir = Path(snapshot_dir_raw)
    audit_path = Path(audit_path_raw)
    payload = audit(snapshot_dir)
    audit_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def validate_source_binding(
    plan: dict[str, Any],
    plan_digest: str,
    source_config: dict[str, Any],
    source_metadata: dict[str, Any],
) -> dict[str, bool]:
    """Bind declared trajectory identity to the config actually recorded by source."""
    identity = plan.get("identity", {})
    dataset = source_config.get("dataset", {})
    training = source_config.get("training", {})
    offset = dataset.get("offset")
    count = dataset.get("max_prompts")
    expected_slice = None
    if isinstance(offset, int) and isinstance(count, int):
        expected_slice = f"{dataset.get('name')}[{offset}:{offset + count}]"
    target_digests = {
        row.get("plan_digest")
        for row in source_metadata.get("transition_capture_targets", [])
        if isinstance(row, dict)
    }
    return {
        "source_metadata_config_exact": source_metadata.get("config") == source_config,
        "source_seed_matches_plan": training.get("seed")
        == identity.get("trajectory_seed"),
        "source_data_slice_matches_plan": expected_slice
        == identity.get("data_slice_id"),
        "source_horizon_is_300": training.get("max_steps") == 300,
        "source_target_count_is_24": len(
            source_metadata.get("transition_capture_targets", [])
        )
        == 24,
        "source_target_plan_digests_exact": target_digests == {plan_digest},
        "source_is_eager_anchor": source_metadata.get("compile_audit", {}).get(
            "backend_compiles"
        )
        == 0
        and source_metadata.get("compile_audit", {}).get("runtime_invocations")
        == 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--audit-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--source-config")
    parser.add_argument("--source-metadata")
    args = parser.parse_args()

    if bool(args.source_config) != bool(args.source_metadata):
        raise ValueError("source-config and source-metadata must be provided together")

    plan_path = Path(args.plan).resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("schema_version") != "forkcert.multi-transition-capture-plan.v0.1":
        raise ValueError("unsupported capture plan schema")
    plan_digest = sha256_file(plan_path)
    capture_root = Path(plan["capture_root"]).resolve()
    identity = plan.get("identity") or {}
    targets = plan.get("targets") or []
    audit_dir = Path(args.audit_dir).resolve()
    audit_dir.mkdir(parents=True, exist_ok=True)

    if args.workers <= 0 or args.workers > 4:
        raise ValueError("workers must be between 1 and 4")
    reusable: dict[str, dict[str, Any]] = {}
    jobs: dict[str, tuple[Path, Path]] = {}
    for target in targets:
        state_id = str(target["state_id"])
        snapshot_dir = capture_root / target["relative_dir"]
        audit_path = audit_dir / f"{state_id}.json"
        if audit_path.is_file():
            existing = json.loads(audit_path.read_text(encoding="utf-8"))
            if existing.get("valid") is True and existing.get("snapshot_dir") == str(
                snapshot_dir.resolve()
            ):
                reusable[state_id] = existing
                continue
        jobs[state_id] = (snapshot_dir, audit_path)

    futures: dict[str, concurrent.futures.Future[dict[str, Any]]] = {}
    rows: list[dict[str, Any]] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        for state_id, (snapshot_dir, audit_path) in jobs.items():
            futures[state_id] = pool.submit(
                run_snapshot_audit, str(snapshot_dir), str(audit_path)
            )
        for target in targets:
            snapshot_dir = capture_root / target["relative_dir"]
            state_id = str(target["state_id"])
            audit_path = audit_dir / f"{state_id}.json"
            reused = state_id in reusable
            payload = reusable[state_id] if reused else futures[state_id].result()
            metadata_path = snapshot_dir / "forkcert_transition_snapshot.json"
            metadata = (
                json.loads(metadata_path.read_text(encoding="utf-8"))
                if metadata_path.is_file()
                else {}
            )
            expected_target_identity = {
                **identity,
                **{
                    key: target[key]
                    for key in ("phase", "eligible_step_population")
                    if key in target
                },
                "state_id": state_id,
            }
            identity_exact = all(
                metadata.get("capture_target_identity", {}).get(key) == value
                for key, value in expected_target_identity.items()
            )
            target_exact = all(
                (
                    int(metadata.get("optimizer_step", -1))
                    == int(target["optimizer_step"]),
                    metadata.get("state_id") == state_id,
                    metadata.get("history_selection")
                    == target.get("history_selection", "FINAL_POLICY_ITERATION_ONLY"),
                    metadata.get("transition_capture_plan_digest") == plan_digest,
                    int(metadata.get("training_horizon_optimizer_steps", -1)) == 300,
                    identity_exact,
                )
            )
            rows.append(
                {
                    "state_id": state_id,
                    "optimizer_step": int(target["optimizer_step"]),
                    "phase": target.get("phase"),
                    "snapshot_dir": str(snapshot_dir),
                    "audit_path": str(audit_path),
                    "audit_sha256": sha256_file(audit_path),
                    "audit_reused_from_same_completed_source": reused,
                    "snapshot_valid": bool(payload.get("valid")),
                    "target_identity_exact": target_exact,
                    "history_length": len(payload.get("history_steps", [])),
                    "history_exact": payload.get("history_steps")
                    == payload.get("expected_history_steps"),
                }
            )

    planned_dirs = {(capture_root / row["relative_dir"]).resolve() for row in targets}
    observed_dirs = {
        path.parent.resolve()
        for path in capture_root.rglob("forkcert_transition_snapshot.json")
    }
    phase_counts = Counter(str(row["phase"]) for row in rows)
    checks = {
        "plan_has_24_unique_targets": len(targets) == 24
        and len({row["state_id"] for row in targets}) == 24
        and len({int(row["optimizer_step"]) for row in targets}) == 24,
        "eight_states_per_phase": phase_counts
        == Counter({"early": 8, "middle": 8, "late": 8}),
        "all_snapshots_valid": len(rows) == 24
        and all(row["snapshot_valid"] for row in rows),
        "all_target_identities_exact": len(rows) == 24
        and all(row["target_identity_exact"] for row in rows),
        "all_histories_exact": len(rows) == 24
        and all(row["history_exact"] for row in rows),
        "no_unplanned_completed_snapshots": observed_dirs == planned_dirs,
    }
    source_evidence: dict[str, Any] | None = None
    if args.source_config and args.source_metadata:
        import yaml

        source_config_path = Path(args.source_config).resolve()
        source_metadata_path = Path(args.source_metadata).resolve()
        source_config = yaml.safe_load(source_config_path.read_text(encoding="utf-8"))
        source_metadata = json.loads(
            source_metadata_path.read_text(encoding="utf-8")
        )
        source_checks = validate_source_binding(
            plan, plan_digest, source_config, source_metadata
        )
        checks.update(source_checks)
        source_evidence = {
            "config_path": str(source_config_path),
            "config_sha256": sha256_file(source_config_path),
            "metadata_path": str(source_metadata_path),
            "metadata_sha256": sha256_file(source_metadata_path),
            "checks": source_checks,
        }
    valid = all(checks.values())
    result = {
        "schema_version": SCHEMA_VERSION,
        "valid": valid,
        "verdict": "VALID" if valid else "INVALID",
        "plan_path": str(plan_path),
        "plan_sha256": plan_digest,
        "capture_root": str(capture_root),
        "audit_workers": args.workers,
        "reused_valid_audits": len(reusable),
        "checks": checks,
        "phase_counts": dict(phase_counts),
        "states": rows,
        "source_evidence": source_evidence,
        "scope": {
            "trajectory_count": 1,
            "population_eligible": False,
            "interpretation": "construction and scale calibration only",
        },
        "nonclaims": [
            "valid snapshots do not establish eager-compiled bias",
            "one trajectory does not identify population uncertainty",
            "snapshot validity does not instantiate a compiled realization",
        ],
    }
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": result["verdict"], "checks": checks}, indent=2))
    if not valid:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
