#!/usr/bin/env python
"""Run one fully frozen Qwen3 confirmation campaign sequentially and fail closed."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from theory_oracle.evaluate_qwen3_bias_oracle_confirmation_v0_1 import (  # noqa: E402
    load_json,
    sha256_file,
    validate_confirmation_manifest,
)
from theory_oracle.run_qwen3_calibration_campaign_v0_1 import (  # noqa: E402
    DEFAULT_MIN_FREE_BYTES,
    TrajectorySpec,
    audit_command,
    capture_audit_valid,
    count_valid_records,
    partial_source_paths,
    process_environment,
    remaining_command,
    require_storage,
    run_logged,
    source_command,
    wait_for_gpu_idle,
)


SCHEMA_VERSION = "forkcert.qwen3-confirmation-campaign-ledger.v0.1"


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def row_to_spec(index: int, row: dict[str, Any]) -> TrajectorySpec:
    return TrajectorySpec(
        index=index,
        trajectory_id=str(row["trajectory_id"]),
        config=Path(row["source_config_path"]).resolve(),
        plan=Path(row["capture_plan_path"]).resolve(),
        results_root=Path(row["results_root"]).resolve(),
        data_root=Path(row["data_root"]).resolve(),
    )


def preflight_manifest(
    manifest_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    manifest = load_json(manifest_path)
    _, rows, errors = validate_confirmation_manifest(manifest, manifest_path)
    if not errors:
        planned = manifest["precision_plan"]["planned_confirmation_trajectories"]
        if len(rows) != planned:
            errors.append("validated trajectory count differs from frozen precision J")
    return manifest, rows, errors


def evaluator_command(manifest_path: Path, out: Path) -> list[str]:
    return [
        sys.executable,
        "theory_oracle/evaluate_qwen3_bias_oracle_confirmation_v0_1.py",
        "--manifest",
        str(manifest_path),
        "--out",
        str(out),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--min-free-bytes", type=int, default=DEFAULT_MIN_FREE_BYTES)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.poll_seconds <= 0 or args.poll_seconds > 60:
        raise ValueError("poll-seconds must be in 1..60")
    if args.min_free_bytes < 0:
        raise ValueError("min-free-bytes must be non-negative")

    manifest_path = Path(args.manifest).resolve()
    ledger_path = Path(args.ledger).resolve()
    result_path = Path(args.result).resolve()
    manifest, rows, errors = preflight_manifest(manifest_path)
    if errors:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "status": "INVALID_MANIFEST_DO_NOT_COLLECT_CONFIRMATION",
            "errors": errors,
            "automatic_launch": False,
        }
        if not args.dry_run:
            atomic_json(ledger_path, payload)
        print(json.dumps(payload, indent=2))
        raise SystemExit(2)

    specs = [row_to_spec(index, row) for index, row in enumerate(rows)]
    commands = [
        {
            "trajectory_id": spec.trajectory_id,
            "source": source_command(spec),
            "audit": audit_command(spec),
            "remaining": remaining_command(spec),
        }
        for spec in specs
    ]
    if args.dry_run:
        print(
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "manifest": str(manifest_path),
                    "manifest_sha256": sha256_file(manifest_path),
                    "commands": commands,
                    "evaluator": evaluator_command(manifest_path, result_path),
                    "automatic_deletion": False,
                },
                indent=2,
            )
        )
        return

    environment = process_environment()
    campaign_root = ledger_path.parent
    log_dir = campaign_root / "logs"
    ledger: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "RUNNING",
        "manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
        },
        "planned_trajectories": len(specs),
        "trajectories": {},
        "automatic_deletion": False,
        "population_B_claim_allowed_before_final_evaluator": False,
    }
    atomic_json(ledger_path, ledger)
    try:
        for spec in specs:
            row: dict[str, Any] = {"status": "RUNNING", "stages": []}
            ledger["trajectories"][spec.trajectory_id] = row
            atomic_json(ledger_path, ledger)
            if not capture_audit_valid(spec):
                partial = partial_source_paths(spec)
                if partial:
                    raise RuntimeError(
                        f"partial source blocks fail-closed resume for {spec.trajectory_id}: "
                        + ", ".join(str(path) for path in partial)
                    )
                require_storage(args.min_free_bytes)
                wait_for_gpu_idle(args.poll_seconds)
                spec.results_root.mkdir(parents=True, exist_ok=True)
                spec.data_root.mkdir(parents=True, exist_ok=True)
                run_logged(
                    source_command(spec),
                    f"{spec.trajectory_id}_source",
                    log_dir,
                    environment,
                )
                row["stages"].append("SOURCE_COMPLETED")
                run_logged(
                    audit_command(spec),
                    f"{spec.trajectory_id}_capture_audit",
                    log_dir,
                    environment,
                )
                if not capture_audit_valid(spec):
                    raise RuntimeError(
                        f"{spec.trajectory_id} source-binding capture audit failed"
                    )
                row["stages"].append("CAPTURE_AUDIT_VALID")
                atomic_json(ledger_path, ledger)
            else:
                row["stages"].append("CAPTURE_AUDIT_REUSED_VALID")

            plan = load_json(spec.plan)
            if count_valid_records(plan, spec.results_root) != 24:
                require_storage(args.min_free_bytes)
                wait_for_gpu_idle(args.poll_seconds)
                run_logged(
                    remaining_command(spec),
                    f"{spec.trajectory_id}_remaining",
                    log_dir,
                    environment,
                )
            if count_valid_records(plan, spec.results_root) != 24:
                raise RuntimeError(f"{spec.trajectory_id} does not have 24 valid records")
            row["stages"].append("TWENTY_FOUR_RECORDS_PRESENT")
            row["status"] = "COMPLETE_VALID_CONFIRMATION_INPUT"
            atomic_json(ledger_path, ledger)

        run_logged(
            evaluator_command(manifest_path, result_path),
            "final_confirmation_evaluator",
            log_dir,
            environment,
        )
        result = load_json(result_path)
        if (
            result.get("valid") is not True
            or result.get("verdict") != "VALID_CONFIRMATION_CONSTRUCTION"
        ):
            raise RuntimeError("final confirmation evaluator did not validate")
        ledger["status"] = "COMPLETE_VALID_CONFIRMATION_EVALUATED"
        ledger["result"] = {
            "path": str(result_path),
            "sha256": sha256_file(result_path),
        }
        ledger["operator_ready_endpoints"] = result.get(
            "operator_attribution_gate", {}
        ).get("ready_endpoints", [])
        ledger["automatic_operator_launch"] = False
        atomic_json(ledger_path, ledger)
    except Exception as error:
        ledger["status"] = "FAILED_CLOSED"
        ledger["error"] = str(error)
        ledger["automatic_operator_launch"] = False
        atomic_json(ledger_path, ledger)
        raise


if __name__ == "__main__":
    main()
