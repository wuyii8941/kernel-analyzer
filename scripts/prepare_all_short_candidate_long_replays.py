#!/usr/bin/env python3
"""Prepare long-replay plans for every frozen short-screen bias candidate.

The short candidate map is not a negative set.  This script turns each row
with a usable confirmation and trajectory bank into an exact one-case plan;
rows without a legal bank/release are written to the same manifest as
``RECAPTURE_REQUIRED``.  No candidate is silently dropped.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAP = ROOT / "results/property/bias_formation/hotspot_search/bias_mechanism_candidate_map.json"
OUT = ROOT / "results/property/declared_persistent_4096/all_short_candidate_long_replay_manifest.json"
PLAN_DIR = ROOT / "results/property/declared_persistent_4096/all_candidate_plans"

MODEL_INFO = {
    "deepseek8b": {
        "architecture": "deepseek8b",
        "model": "/data1/tzh/models/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
        "release_template": "deepseek8b_seq{seq}_r1",
        "bank_template": "results/property/tcmp_allop_v1/input_banks/deepseek8b_seq{seq}_trajectory4096.json",
    },
    "phi4": {
        "architecture": "phi",
        "model": "/data1/tzh/models/microsoft/Phi-4-mini-instruct",
        "release_template": "phi4_seq{seq}_r1",
        "bank_template": "results/property/declared_persistent_4096/expanded_controls/input_banks/phi4_seq{seq}_cycled_4224.json",
    },
    "qwen": {
        "architecture": "qwen",
        "model": "/data1/tzh/models/Qwen/Qwen3-1.7B",
        "release_template": "qwen_seq{seq}_r1",
        "bank_template": "results/property/tcmp_allop_v1/input_banks/qwen_seq{seq}_trajectory4096.json",
    },
    "mamba": {
        "architecture": "mamba",
        "model": "/data1/tzh/models/state-spaces/mamba-130m-hf",
        "release_template": "mamba_seq{seq}_r1",
        "bank_template": "results/property/tcmp_allop_v1/input_banks/mamba_seq{seq}_trajectory4096.json",
    },
}


def read(path: Path):
    return json.loads(path.read_text())


_TASK_CACHE: dict[Path, set[str]] = {}


def release_has_task(path: Path, task_id: str) -> bool:
    if path in _TASK_CACHE:
        return task_id in _TASK_CACHE[path]
    task_path = path / "same_dtype_tasks.json.gz"
    if not task_path.exists():
        return False
    with gzip.open(task_path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    _TASK_CACHE[path] = {str(row.get("task_id")) for row in payload.get("rows", [])}
    return task_id in _TASK_CACHE[path]


def main() -> None:
    candidates = read(MAP).get("candidates", [])
    PLAN_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for candidate in candidates:
        model_key = str(candidate.get("model"))
        seq = int(candidate.get("sequence_length"))
        info = MODEL_INFO[model_key]
        release = ROOT / "results/coverage/runtime_releases" / info["release_template"].format(seq=seq)
        bank = ROOT / info["bank_template"].format(seq=seq)
        # Some frozen confirmation banks use the explicit 4224-state naming
        # even though they implement the same 4096-step trajectory protocol.
        if model_key == "qwen" and seq == 128:
            bank = ROOT / "results/property/declared_persistent_4096/qwen_seq128_bank_4224.json"
        if model_key == "phi4" and seq == 64:
            bank = ROOT / "results/property/declared_persistent_4096/phi_seq64_bank_4224.json"
        if model_key == "mamba" and seq == 256:
            bank = ROOT / "results/property/declared_persistent_4096/mamba_seq256_bank_cycled_4224.json"
        reasons = []
        warnings = []
        if not candidate.get("confirmation_available"):
            warnings.append("SHORT_CONFIRMATION_NOT_AVAILABLE_LONG_REPLAY_IS_FIRST_EXACT_CONFIRMATION")
        if not release.exists():
            reasons.append("RUNTIME_RELEASE_MISSING")
        elif not release_has_task(release, str(candidate.get("task_id"))):
            reasons.append("TASK_NOT_IN_RUNTIME_RELEASE")
        if not bank.exists():
            reasons.append("TRAJECTORY_BANK_MISSING")
        plan_path = PLAN_DIR / f"{candidate['case_id']}.json"
        runnable = not reasons
        if runnable:
            payload = {
                "schema": "kernel-analyzer-single-candidate-long-plan-v1",
                "model": model_key,
                "sequence_length": seq,
                "claim_boundary": "This plan is generated from the frozen short-screen candidate map; only the completed 4096-step replay may assign a long-horizon label.",
                "cases": [{
                    "case_id": candidate["case_id"],
                    "task_id": candidate["task_id"],
                    "carrier": candidate["carrier"],
                    "family": candidate.get("family"),
                    "member_count": 1,
                }],
            }
            plan_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        rows.append({
            "case_id": candidate["case_id"],
            "model_key": model_key,
            "architecture": info["architecture"],
            "model_path": info["model"],
            "sequence_length": seq,
            "task_id": candidate["task_id"],
            "carrier": candidate["carrier"],
            "family": candidate.get("family"),
            "short_screen_signature": candidate.get("screen_signature"),
            "short_screen_gradient_ratio": candidate.get("screen_gradient_ratio"),
            "confirmation_available": bool(candidate.get("confirmation_available")),
            "confirmation_outcome": candidate.get("confirmation_outcome"),
            "runnable": runnable,
            "status": "READY_FOR_LONG_REPLAY" if runnable else "RECAPTURE_REQUIRED",
        "reasons": reasons,
            "warnings": warnings,
            "release_dir": str(release.relative_to(ROOT)) if release.exists() else None,
            "input_bank": str(bank.relative_to(ROOT)) if bank.exists() else None,
            "case_plan": str(plan_path.relative_to(ROOT)) if runnable else None,
            "output": f"results/property/declared_persistent_4096/all_candidates/{candidate['case_id']}_4096.json",
        })
    payload = {
        "schema": "all-short-screen-candidates-long-replay-manifest-v1",
        "source": str(MAP.relative_to(ROOT)),
        "selection": "All 69 frozen short-screen candidates; no outcome or long-horizon result was used for selection.",
        "long_horizon_steps": 4096,
        "rows": rows,
        "summary": {
            "candidate_count": len(rows),
            "ready_for_long_replay": sum(row["runnable"] for row in rows),
        "recapture_required": sum(not row["runnable"] for row in rows),
            "confirmation_available": sum(row["confirmation_available"] for row in rows),
        },
        "claim_boundary": "READY rows must be run; RECAPTURE_REQUIRED rows are unresolved and cannot be called negative.",
    }
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
