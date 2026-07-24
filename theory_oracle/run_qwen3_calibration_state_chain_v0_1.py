#!/usr/bin/env python
"""Run one frozen Qwen3 calibration state chain sequentially and fail closed."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def stage_valid(name: str, path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        value = load_json(path)
    except (OSError, json.JSONDecodeError):
        return False
    checks: dict[str, Callable[[dict[str, Any]], bool]] = {
        "contract": lambda row: row.get("status") == "FROZEN_BEFORE_TRANSITION_ENDPOINT_EXECUTION",
        "arm": lambda row: row.get("valid") is True and row.get("verdict") == "VALID",
        "transition_evaluation": lambda row: row.get("construction_valid") is True,
        "endpoint_manifest": lambda row: row.get("schema_version") == "forkcert.qwen3-calibration-state-endpoint-manifest.v0.1",
        "bank": lambda row: row.get("valid") is True and row.get("verdict") == "VALID_FROZEN_T1A_BANK",
        "task_evaluation": lambda row: row.get("valid") is True and row.get("verdict") == "VALID_CALIBRATION_STATE_ENDPOINTS",
        "record_validation": lambda row: row.get("valid") is True and row.get("population_eligible") is True,
    }
    return checks[name](value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--state-id", required=True)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    plan_path = Path(args.plan).resolve()
    plan = load_json(plan_path)
    target_rows = [row for row in plan["targets"] if row["state_id"] == args.state_id]
    if len(target_rows) != 1:
        raise ValueError(f"expected one target for {args.state_id}, found {len(target_rows)}")
    target = target_rows[0]
    capture_root = Path(plan["capture_root"]).resolve()
    snapshot_dir = capture_root / target["relative_dir"]
    state_root = Path(args.results_root).resolve() / f"step{int(target['optimizer_step']):03d}"
    state_root.mkdir(parents=True, exist_ok=True)
    logs = state_root / "logs"
    logs.mkdir(exist_ok=True)
    ledger_path = state_root / "chain_ledger.json"
    ledger: dict[str, Any] = {
        "schema_version": "forkcert.qwen3-calibration-state-chain-ledger.v0.1",
        "state_id": args.state_id,
        "optimizer_step": int(target["optimizer_step"]),
        "snapshot_dir": str(snapshot_dir),
        "status": "RUNNING",
        "stages": [],
    }
    atomic_json(ledger_path, ledger)

    python = sys.executable
    contract = state_root / "realization_contract.json"
    transitions = state_root / "transitions"
    transition_evaluation = state_root / "transition_evaluation.json"
    endpoint_manifest = state_root / "endpoint_manifest.json"
    bank_1 = state_root / "t1a_bank_1.json"
    bank_2 = state_root / "t1a_bank_2.json"
    task_evaluation = state_root / "task_endpoint_evaluation.json"
    record_bundle = state_root / "record_bundle.json"
    record_validation = state_root / "record_validation.json"

    commands: list[tuple[str, list[str], str, Path]] = [
        (
            "contract",
            [
                python,
                "theory_oracle/qwen3_grpo_natural_transition_v0_2.py",
                "--snapshot-dir", str(snapshot_dir),
                "--arm", "compiled", "--repeat", "0",
                "--out-dir", str(state_root / "realization_preflight"),
                "--instantiate-realization-contract", str(contract),
            ],
            "contract",
            contract,
        )
    ]
    for repeat in (1, 2):
        for arm, directory in (("eager", f"eager_{repeat}"), ("compiled", f"compiled_{repeat}")):
            result = transitions / directory / "result.json"
            commands.append(
                (
                    f"{arm}_{repeat}",
                    [
                        python,
                        "theory_oracle/qwen3_grpo_natural_transition_v0_2.py",
                        "--snapshot-dir", str(snapshot_dir),
                        "--realization-contract", str(contract),
                        "--arm", arm, "--repeat", str(repeat),
                        "--out-dir", str(transitions / directory),
                        "--save-vectors",
                    ],
                    "arm",
                    result,
                )
            )
    commands.extend(
        [
            (
                "transition_evaluation",
                [
                    python,
                    "theory_oracle/evaluate_qwen3_grpo_natural_transition_v0_2.py",
                    "--eager-1", str(transitions / "eager_1" / "result.json"),
                    "--eager-2", str(transitions / "eager_2" / "result.json"),
                    "--compiled-1", str(transitions / "compiled_1" / "result.json"),
                    "--compiled-2", str(transitions / "compiled_2" / "result.json"),
                    "--effect-vector-dir", str(state_root / "effects"),
                    "--out", str(transition_evaluation),
                ],
                "transition_evaluation",
                transition_evaluation,
            ),
            (
                "endpoint_manifest",
                [
                    python,
                    "theory_oracle/build_qwen3_calibration_state_endpoint_manifest_v0_1.py",
                    "--snapshot-dir", str(snapshot_dir),
                    "--transition-root", str(transitions),
                    "--out", str(endpoint_manifest),
                ],
                "endpoint_manifest",
                endpoint_manifest,
            ),
            (
                "t1a_bank_1",
                [python, "theory_oracle/generate_qwen3_t1a_bank_v0_1.py", "--manifest", str(endpoint_manifest), "--repeat", "1", "--out", str(bank_1)],
                "bank",
                bank_1,
            ),
            (
                "t1a_bank_2",
                [python, "theory_oracle/generate_qwen3_t1a_bank_v0_1.py", "--manifest", str(endpoint_manifest), "--repeat", "2", "--out", str(bank_2)],
                "bank",
                bank_2,
            ),
            (
                "task_evaluation",
                [
                    python,
                    "theory_oracle/evaluate_qwen3_calibration_state_endpoints_v0_1.py",
                    "--manifest", str(endpoint_manifest),
                    "--t1a-bank", str(bank_1),
                    "--t1a-bank-repeat", str(bank_2),
                    "--out", str(task_evaluation),
                ],
                "task_evaluation",
                task_evaluation,
            ),
            (
                "record_bundle",
                [
                    python,
                    "theory_oracle/build_qwen3_calibration_state_record_bundle_v0_1.py",
                    "--snapshot-dir", str(snapshot_dir),
                    "--transition-root", str(transitions),
                    "--transition-evaluation", str(transition_evaluation),
                    "--task-evaluation", str(task_evaluation),
                    "--out", str(record_bundle),
                    "--validation-out", str(record_validation),
                ],
                "record_validation",
                record_validation,
            ),
        ]
    )

    if args.dry_run:
        print(json.dumps({"state_id": args.state_id, "commands": commands}, indent=2, default=str))
        return

    environment = os.environ.copy()
    environment["PYTHONPATH"] = "src:."
    for stage, command, validity_kind, expected in commands:
        if stage_valid(validity_kind, expected):
            status = "REUSED_VALID"
        else:
            if expected.parent.exists() and stage in {"contract", "eager_1", "eager_2", "compiled_1", "compiled_2"}:
                partial = (
                    state_root / "realization_preflight"
                    if stage == "contract"
                    else expected.parent
                )
                if partial.exists() and any(partial.iterdir()):
                    raise RuntimeError(
                        f"partial output blocks fail-closed resume for {stage}: {partial}"
                    )
            log_path = logs / f"{stage}.log"
            with log_path.open("w", encoding="utf-8") as handle:
                completed = subprocess.run(
                    command,
                    cwd=Path(__file__).resolve().parents[1],
                    env=environment,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
            if completed.returncode != 0 or not stage_valid(validity_kind, expected):
                ledger["status"] = "FAILED_CLOSED"
                ledger["failed_stage"] = stage
                ledger["stages"].append(
                    {"stage": stage, "status": "FAILED", "returncode": completed.returncode, "log": str(log_path)}
                )
                atomic_json(ledger_path, ledger)
                raise SystemExit(2)
            status = "COMPLETED_VALID"
        ledger["stages"].append(
            {"stage": stage, "status": status, "evidence": str(expected)}
        )
        atomic_json(ledger_path, ledger)

    ledger["status"] = "COMPLETE_VALID"
    atomic_json(ledger_path, ledger)
    print(json.dumps({"state_id": args.state_id, "status": ledger["status"]}, indent=2))


if __name__ == "__main__":
    main()
