#!/usr/bin/env python3
"""Check that the frozen sample-completion campaign can be started safely.

This is only a readiness check.  It never runs a model and never creates a
scientific label.  In particular, a missing CUDA driver is reported as a
blocker rather than being replaced with a CPU or synthetic result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "results/property/sample_completion_v1/execution_manifest.json"
OUT = ROOT / "results/property/sample_completion_v1/preflight_report.json"


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def cuda_ready() -> tuple[bool, str]:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - environment-specific
        return False, f"torch_import_failed: {exc}"
    if not torch.cuda.is_available():
        return False, "torch.cuda.is_available=false"
    try:
        count = int(torch.cuda.device_count())
    except Exception as exc:  # pragma: no cover - environment-specific
        return False, f"cuda_device_count_failed: {exc}"
    if count < 1:
        return False, "cuda_device_count=0"
    return True, f"cuda_devices={count}"


def check_group(name: str, group: dict[str, Any]) -> dict[str, Any]:
    required = {
        "model": Path(group["model"]),
        "input_bank": ROOT / group["input_bank"],
        "release_dir": ROOT / group["release_dir"],
        "case_plan": ROOT / group["case_plan"],
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    plan = load(required["case_plan"]) if not missing else {}
    cases = plan.get("cases", [])
    errors: list[str] = []
    if len(cases) != int(group.get("case_count", -1)):
        errors.append("case_plan_count_mismatch")
    for row in cases:
        if not row.get("case_id") or not row.get("task_id"):
            errors.append("case_missing_case_id_or_task_id")
        if not row.get("carrier"):
            errors.append(f"{row.get('case_id', '<unknown>')}:missing_carrier")
    input_count = None
    if not missing:
        bank = load(required["input_bank"])
        input_count = len(bank.get("states", bank.get("records", [])))
        if input_count < 32:
            errors.append(f"input_bank_has_only_{input_count}_states")
    return {
        "group": name,
        "model_exists": required["model"].exists(),
        "input_bank_exists": required["input_bank"].exists(),
        "release_dir_exists": required["release_dir"].exists(),
        "case_plan_exists": required["case_plan"].exists(),
        "input_state_count": input_count,
        "case_count": len(cases),
        "missing_paths": missing,
        "errors": sorted(set(errors)),
        "ready_without_gpu": not missing and not errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    manifest = load(args.manifest)
    gpu_ok, gpu_detail = cuda_ready()
    groups = [check_group(name, group) for name, group in manifest["groups"].items()]
    all_files_ready = all(row["ready_without_gpu"] for row in groups)
    status = "READY_TO_START" if gpu_ok and all_files_ready else "BLOCKED_PRECAMPAIGN"
    result = {
        "schema": "kernel-analyzer-sample-completion-preflight-v1",
        "status": status,
        "scientific_results_written": False,
        "gpu": {"available": gpu_ok, "detail": gpu_detail},
        "groups": groups,
        "claim_boundary": (
            "This report checks only files, bindings, state-bank size, and device readiness. "
            "It is not evidence that any 2/8/16/32-step measurement ran."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "gpu": gpu_detail, "groups": len(groups)}, sort_keys=True))


if __name__ == "__main__":
    main()
