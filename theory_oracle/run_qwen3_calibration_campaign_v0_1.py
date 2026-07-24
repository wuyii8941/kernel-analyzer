#!/usr/bin/env python
"""Run the frozen four-trajectory Qwen3 calibration campaign fail closed.

This driver is deliberately limited to calibration-0..calibration-3.  It does
not create a population-B verdict and it never launches confirmation data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "forkcert.qwen3-calibration-campaign-ledger.v0.1"
DEFAULT_MIN_FREE_BYTES = 2 * 1024**4


@dataclass(frozen=True)
class TrajectorySpec:
    index: int
    trajectory_id: str
    config: Path
    plan: Path
    results_root: Path
    data_root: Path

    @property
    def capture_audit(self) -> Path:
        return self.results_root / "capture_batch_audit.json"

    @property
    def scalar_summary(self) -> Path:
        return self.results_root / "trajectory_scalar_summary.json"

    @property
    def null_control_summary(self) -> Path:
        return self.results_root / "trajectory_null_control_summary.json"

    @property
    def boundary_summary(self) -> Path:
        return self.results_root / "trajectory_boundary_conditioned_summary.json"

    @property
    def vector_summary(self) -> Path:
        return self.results_root / "trajectory_u2_vector_summary.json"

    @property
    def vector_artifact_dir(self) -> Path:
        return self.results_root / "trajectory_u2_vector_artifacts"


def frozen_specs() -> list[TrajectorySpec]:
    rows: list[TrajectorySpec] = []
    for index in range(4):
        stem = f"qwen3_bias_oracle_calibration_{index}_v0_1"
        rows.append(
            TrajectorySpec(
                index=index,
                trajectory_id=f"calibration-{index}",
                config=ROOT / "configs" / f"{stem}.yaml",
                plan=(
                    ROOT
                    / "theory_oracle"
                    / f"QWEN3_BIAS_ORACLE_CALIBRATION_{index}_CAPTURE_PLAN_V0_1.json"
                ),
                results_root=ROOT / "results" / "oracle_calibration" / stem,
                data_root=ROOT / "data" / "oracle_calibration" / stem,
            )
        )
    return rows


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def validate_frozen_spec(spec: TrajectorySpec) -> list[str]:
    errors: list[str] = []
    if not spec.config.is_file():
        errors.append(f"missing config: {spec.config}")
    if not spec.plan.is_file():
        errors.append(f"missing plan: {spec.plan}")
        return errors
    plan = load_json(spec.plan)
    identity = plan.get("identity", {})
    if plan.get("schema_version") != "forkcert.multi-transition-capture-plan.v0.1":
        errors.append(f"{spec.trajectory_id}: unsupported plan schema")
    if identity.get("trajectory_id") != spec.trajectory_id:
        errors.append(f"{spec.trajectory_id}: trajectory identity mismatch")
    if identity.get("query_id") != "Q-R":
        errors.append(f"{spec.trajectory_id}: expected query Q-R")
    if identity.get("trajectory_anchor") != "EAGER_TRAJECTORY":
        errors.append(f"{spec.trajectory_id}: expected eager trajectory anchor")
    if Path(plan.get("capture_root", "")).resolve() != (spec.data_root / "captures"):
        errors.append(f"{spec.trajectory_id}: capture root mismatch")
    targets = plan.get("targets", [])
    if len(targets) != 24:
        errors.append(f"{spec.trajectory_id}: expected 24 targets")
    return errors


def capture_audit_valid(spec: TrajectorySpec) -> bool:
    if not spec.capture_audit.is_file():
        return False
    try:
        audit = load_json(spec.capture_audit)
    except (OSError, json.JSONDecodeError):
        return False
    source_evidence = audit.get("source_evidence") or {}
    source_metadata = spec.results_root / "source_dump.metadata.json"
    return all(
        (
            audit.get("valid") is True,
            audit.get("verdict") == "VALID",
            audit.get("plan_sha256") == sha256_file(spec.plan),
            Path(audit.get("capture_root", "")).resolve()
            == (spec.data_root / "captures"),
            source_evidence.get("config_sha256") == sha256_file(spec.config),
            source_metadata.is_file(),
            source_evidence.get("metadata_sha256")
            == (sha256_file(source_metadata) if source_metadata.is_file() else None),
            all((source_evidence.get("checks") or {}).values()),
        )
    )


def count_valid_records(plan: dict[str, Any], results_root: Path) -> int:
    count = 0
    for target in plan.get("targets", []):
        state_root = results_root / f"step{int(target['optimizer_step']):03d}"
        path = state_root / "record_validation.json"
        if not path.is_file():
            continue
        try:
            row = load_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        if row.get("valid") is True and row.get("population_eligible") is True:
            count += 1
    return count


def remaining_ledger_status(spec: TrajectorySpec) -> str | None:
    path = spec.results_root / "remaining_chain_ledger.json"
    if not path.is_file():
        return None
    try:
        return str(load_json(path).get("status"))
    except (OSError, json.JSONDecodeError):
        return "UNREADABLE"


def source_paths(spec: TrajectorySpec) -> list[Path]:
    return [
        spec.results_root / "source_dump.jsonl",
        spec.results_root / "source_dump.metadata.json",
        spec.results_root / "source_samples.jsonl",
        spec.results_root / "source_final_rollout.jsonl",
        spec.data_root / "source_final_model",
        spec.data_root / "captures",
    ]


def partial_source_paths(spec: TrajectorySpec) -> list[Path]:
    if capture_audit_valid(spec):
        return []
    return [path for path in source_paths(spec) if path.exists()]


def source_command(spec: TrajectorySpec) -> list[str]:
    return [
        sys.executable,
        "scripts/phase0_grpo_train.py",
        "--config",
        str(spec.config),
        "--out-jsonl",
        str(spec.results_root / "source_dump.jsonl"),
        "--samples-jsonl",
        str(spec.results_root / "source_samples.jsonl"),
        "--final-rollout-jsonl",
        str(spec.results_root / "source_final_rollout.jsonl"),
        "--output-dir",
        str(spec.data_root / "source_final_model"),
        "--transition-capture-plan",
        str(spec.plan),
    ]


def audit_command(spec: TrajectorySpec) -> list[str]:
    return [
        sys.executable,
        "theory_oracle/audit_qwen3_calibration_capture_batch_v0_1.py",
        "--plan",
        str(spec.plan),
        "--audit-dir",
        str(spec.results_root / "snapshot_audits"),
        "--out",
        str(spec.capture_audit),
        "--workers",
        "4",
        "--source-config",
        str(spec.config),
        "--source-metadata",
        str(spec.results_root / "source_dump.metadata.json"),
    ]


def remaining_command(spec: TrajectorySpec) -> list[str]:
    return [
        sys.executable,
        "theory_oracle/run_qwen3_calibration_remaining_v0_1.py",
        "--plan",
        str(spec.plan),
        "--results-root",
        str(spec.results_root),
    ]


def scalar_command(spec: TrajectorySpec) -> list[str]:
    return [
        sys.executable,
        "theory_oracle/aggregate_qwen3_calibration_records_v0_1.py",
        "--plan",
        str(spec.plan),
        "--results-root",
        str(spec.results_root),
        "--out",
        str(spec.scalar_summary),
    ]


def null_control_command(spec: TrajectorySpec) -> list[str]:
    return [
        sys.executable,
        "theory_oracle/aggregate_qwen3_calibration_null_controls_v0_1.py",
        "--plan",
        str(spec.plan),
        "--results-root",
        str(spec.results_root),
        "--out",
        str(spec.null_control_summary),
    ]


def boundary_command(spec: TrajectorySpec) -> list[str]:
    return [
        sys.executable,
        "theory_oracle/aggregate_qwen3_boundary_conditioned_calibration_v0_1.py",
        "--plan",
        str(spec.plan),
        "--results-root",
        str(spec.results_root),
        "--taus",
        "0.0001",
        "0.0005",
        "0.001",
        "0.005",
        "0.01",
        "0.05",
        "--out",
        str(spec.boundary_summary),
    ]


def vector_command(spec: TrajectorySpec) -> list[str]:
    return [
        sys.executable,
        "theory_oracle/aggregate_qwen3_calibration_u2_vector_v0_1.py",
        "--plan",
        str(spec.plan),
        "--results-root",
        str(spec.results_root),
        "--artifact-dir",
        str(spec.vector_artifact_dir),
        "--out",
        str(spec.vector_summary),
    ]


def process_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "0",
            "PYTHONPATH": f"{ROOT / 'src'}:{ROOT}",
            "TOKENIZERS_PARALLELISM": "false",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "PYTHONHASHSEED": "0",
        }
    )
    return environment


def gpu_compute_processes() -> list[str]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_memory,process_name",
            "--format=csv,noheader",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed: {completed.stderr.strip()}")
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def wait_for_gpu_idle(poll_seconds: int, timeout_seconds: int = 900) -> None:
    deadline = time.monotonic() + timeout_seconds
    while True:
        processes = gpu_compute_processes()
        if not processes:
            return
        if time.monotonic() >= deadline:
            raise RuntimeError(f"GPU did not become idle: {processes}")
        time.sleep(poll_seconds)


def require_storage(min_free_bytes: int) -> None:
    free = shutil.disk_usage(ROOT).free
    if free < min_free_bytes:
        raise RuntimeError(
            f"free storage {free} is below frozen safety floor {min_free_bytes}"
        )


def run_logged(
    command: list[str],
    stage: str,
    log_dir: Path,
    environment: dict[str, str],
) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{stage}.log"
    with log_path.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"stage {stage} failed with return code {completed.returncode}; log={log_path}"
        )


def wait_for_external_trajectory(
    spec: TrajectorySpec,
    ledger: dict[str, Any],
    ledger_path: Path,
    poll_seconds: int,
    wait_timeout_hours: float,
) -> None:
    deadline = time.monotonic() + wait_timeout_hours * 3600
    plan = load_json(spec.plan)
    last_count = -1
    while True:
        count = count_valid_records(plan, spec.results_root)
        status = remaining_ledger_status(spec)
        if count != last_count:
            ledger["active_wait"] = {
                "trajectory_id": spec.trajectory_id,
                "valid_records": count,
                "expected_records": 24,
                "remaining_ledger_status": status,
            }
            atomic_json(ledger_path, ledger)
            last_count = count
        if status in {"FAILED_CLOSED", "UNREADABLE"}:
            raise RuntimeError(
                f"external {spec.trajectory_id} chain status is {status}"
            )
        if count == 24 and status == "COMPLETE_VALID":
            ledger.pop("active_wait", None)
            atomic_json(ledger_path, ledger)
            return
        if time.monotonic() >= deadline:
            raise RuntimeError(f"timed out waiting for {spec.trajectory_id}")
        time.sleep(poll_seconds)


def summary_valid(path: Path, expected_verdict: str) -> bool:
    if not path.is_file():
        return False
    try:
        row = load_json(path)
    except (OSError, json.JSONDecodeError):
        return False
    return row.get("valid") is True and row.get("verdict") == expected_verdict


def multi_scalar_command(specs: list[TrajectorySpec], out: Path) -> list[str]:
    command = [
        sys.executable,
        "theory_oracle/aggregate_qwen3_calibration_multi_trajectory_v0_1.py",
    ]
    for spec in specs:
        command.extend(["--trajectory", str(spec.plan), str(spec.results_root)])
    command.extend(["--out", str(out)])
    return command


def multi_null_control_command(specs: list[TrajectorySpec], out: Path) -> list[str]:
    command = [
        sys.executable,
        "theory_oracle/aggregate_qwen3_calibration_null_controls_multi_v0_1.py",
    ]
    for spec in specs:
        command.extend(["--trajectory-summary", str(spec.null_control_summary)])
    command.extend(["--out", str(out)])
    return command


def multi_boundary_command(specs: list[TrajectorySpec], out: Path) -> list[str]:
    command = [
        sys.executable,
        "theory_oracle/aggregate_qwen3_boundary_conditioned_multi_v0_1.py",
    ]
    for spec in specs:
        command.extend(["--trajectory-summary", str(spec.boundary_summary)])
    command.extend(["--out", str(out)])
    return command


def multi_vector_command(
    specs: list[TrajectorySpec], artifact_dir: Path, out: Path
) -> list[str]:
    command = [
        sys.executable,
        "theory_oracle/aggregate_qwen3_calibration_u2_multi_trajectory_v0_1.py",
    ]
    for spec in specs:
        command.extend(["--trajectory-summary", str(spec.vector_summary)])
    command.extend(["--artifact-dir", str(artifact_dir), "--out", str(out)])
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--wait-timeout-hours", type=float, default=336.0)
    parser.add_argument("--min-free-bytes", type=int, default=DEFAULT_MIN_FREE_BYTES)
    parser.add_argument(
        "--external-wait-trajectory",
        action="append",
        default=[],
        help="Wait for an already-running trajectory instead of launching its state chain.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.poll_seconds <= 0 or args.poll_seconds > 60:
        raise ValueError("poll-seconds must be in 1..60")
    if args.wait_timeout_hours <= 0 or args.min_free_bytes < 0:
        raise ValueError("invalid wait/storage safety argument")

    specs = frozen_specs()
    errors = [error for spec in specs for error in validate_frozen_spec(spec)]
    external = set(args.external_wait_trajectory)
    known = {spec.trajectory_id for spec in specs}
    if not external.issubset(known):
        errors.append(f"unknown external-wait trajectory: {sorted(external - known)}")
    if errors:
        raise ValueError("; ".join(errors))

    campaign_root = (
        ROOT
        / "results"
        / "oracle_calibration"
        / "qwen3_bias_oracle_calibration_campaign_v0_1"
    )
    ledger_path = campaign_root / "campaign_ledger.json"
    log_dir = campaign_root / "logs"
    multi_scalar = campaign_root / "four_trajectory_scalar_calibration.json"
    multi_null_controls = campaign_root / "four_trajectory_null_controls.json"
    multi_boundary = campaign_root / "four_trajectory_boundary_conditioned_calibration.json"
    multi_vector = campaign_root / "four_trajectory_u2_vector_calibration.json"
    multi_vector_artifacts = campaign_root / "four_trajectory_u2_vector_artifacts"

    if args.dry_run:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "external_wait_trajectories": sorted(external),
            "trajectories": [
                {
                    "trajectory_id": spec.trajectory_id,
                    "source": source_command(spec),
                    "audit": audit_command(spec),
                    "remaining": remaining_command(spec),
                    "scalar": scalar_command(spec),
                    "null_controls": null_control_command(spec),
                    "boundary": boundary_command(spec),
                    "vector": vector_command(spec),
                }
                for spec in specs
            ],
            "multi_scalar": multi_scalar_command(specs, multi_scalar),
            "multi_null_controls": multi_null_control_command(
                specs, multi_null_controls
            ),
            "multi_boundary": multi_boundary_command(specs, multi_boundary),
            "multi_vector": multi_vector_command(
                specs, multi_vector_artifacts, multi_vector
            ),
        }
        print(json.dumps(payload, indent=2))
        return

    environment = process_environment()
    ledger: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "RUNNING",
        "scope": "CALIBRATION_ONLY_NO_POPULATION_B_VERDICT",
        "external_wait_trajectories": sorted(external),
        "trajectories": {},
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
                if spec.trajectory_id in external:
                    raise RuntimeError(
                        f"external trajectory {spec.trajectory_id} lacks a valid capture audit"
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
                        f"{spec.trajectory_id} capture audit did not validate"
                    )
                row["stages"].append("CAPTURE_AUDIT_VALID")
                atomic_json(ledger_path, ledger)
            else:
                row["stages"].append("CAPTURE_AUDIT_REUSED_VALID")

            plan = load_json(spec.plan)
            if count_valid_records(plan, spec.results_root) != 24:
                if spec.trajectory_id in external:
                    wait_for_external_trajectory(
                        spec,
                        ledger,
                        ledger_path,
                        args.poll_seconds,
                        args.wait_timeout_hours,
                    )
                else:
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

            run_logged(
                scalar_command(spec),
                f"{spec.trajectory_id}_scalar",
                log_dir,
                environment,
            )
            if not summary_valid(
                spec.scalar_summary, "VALID_COMPLETE_TRAJECTORY_DESCRIPTION"
            ):
                raise RuntimeError(f"{spec.trajectory_id} scalar summary invalid")
            row["stages"].append("SCALAR_SUMMARY_VALID")

            run_logged(
                null_control_command(spec),
                f"{spec.trajectory_id}_null_controls",
                log_dir,
                environment,
            )
            if not summary_valid(
                spec.null_control_summary,
                "VALID_COMPLETE_NULL_CONTROL_DESCRIPTION",
            ):
                raise RuntimeError(
                    f"{spec.trajectory_id} null-control summary invalid"
                )
            row["stages"].append("NULL_CONTROL_SUMMARY_VALID")

            run_logged(
                boundary_command(spec),
                f"{spec.trajectory_id}_boundary",
                log_dir,
                environment,
            )
            if not summary_valid(
                spec.boundary_summary,
                "COMPLETE_CALIBRATION_BOUNDARY_DESCRIPTION",
            ):
                raise RuntimeError(f"{spec.trajectory_id} boundary summary invalid")
            row["stages"].append("BOUNDARY_SUMMARY_VALID_CALIBRATION_ONLY")

            require_storage(args.min_free_bytes)
            run_logged(
                vector_command(spec),
                f"{spec.trajectory_id}_vector",
                log_dir,
                environment,
            )
            if not summary_valid(
                spec.vector_summary, "VALID_COMPLETE_ONE_TRAJECTORY_VECTOR_DESCRIPTION"
            ):
                raise RuntimeError(f"{spec.trajectory_id} vector summary invalid")
            row["stages"].append("VECTOR_SUMMARY_VALID")
            row["status"] = "COMPLETE_VALID_CALIBRATION_DESCRIPTION"
            atomic_json(ledger_path, ledger)

        run_logged(
            multi_scalar_command(specs, multi_scalar),
            "four_trajectory_scalar",
            log_dir,
            environment,
        )
        if not summary_valid(
            multi_scalar, "VALID_COMPLETE_FOUR_TRAJECTORY_CALIBRATION"
        ):
            raise RuntimeError("four-trajectory scalar calibration invalid")
        run_logged(
            multi_null_control_command(specs, multi_null_controls),
            "four_trajectory_null_controls",
            log_dir,
            environment,
        )
        if not summary_valid(
            multi_null_controls,
            "VALID_COMPLETE_FOUR_TRAJECTORY_NULL_CONTROL_DESCRIPTION",
        ):
            raise RuntimeError("four-trajectory null-control calibration invalid")
        run_logged(
            multi_boundary_command(specs, multi_boundary),
            "four_trajectory_boundary",
            log_dir,
            environment,
        )
        if not summary_valid(
            multi_boundary,
            "VALID_COMPLETE_FOUR_TRAJECTORY_BOUNDARY_CALIBRATION_DESCRIPTION",
        ):
            raise RuntimeError("four-trajectory boundary calibration invalid")
        require_storage(args.min_free_bytes)
        run_logged(
            multi_vector_command(specs, multi_vector_artifacts, multi_vector),
            "four_trajectory_vector",
            log_dir,
            environment,
        )
        if not summary_valid(
            multi_vector, "VALID_COMPLETE_FOUR_TRAJECTORY_VECTOR_CALIBRATION"
        ):
            raise RuntimeError("four-trajectory vector calibration invalid")
        ledger["status"] = "COMPLETE_VALID_CALIBRATION_ONLY"
        ledger["outputs"] = {
            "scalar": str(multi_scalar),
            "scalar_sha256": sha256_file(multi_scalar),
            "null_controls": str(multi_null_controls),
            "null_controls_sha256": sha256_file(multi_null_controls),
            "boundary_conditioned": str(multi_boundary),
            "boundary_conditioned_sha256": sha256_file(multi_boundary),
            "u2_vector": str(multi_vector),
            "u2_vector_sha256": sha256_file(multi_vector),
        }
        ledger["population_B_claim_allowed"] = False
        atomic_json(ledger_path, ledger)
    except Exception as error:
        ledger["status"] = "FAILED_CLOSED"
        ledger["error"] = str(error)
        ledger["population_B_claim_allowed"] = False
        atomic_json(ledger_path, ledger)
        raise


if __name__ == "__main__":
    main()
