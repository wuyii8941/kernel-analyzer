#!/usr/bin/env python3
"""Prepare 4096-step replays for every legacy coherent F+B endpoint.

The original endpoint census contains 57 concrete rows labelled
``COHERENT_F_B_BIAS``.  They are disjoint from the later 69-row backward
candidate map.  This script keeps all 57 in the long-run denominator and
builds an executable one-case plan whenever the frozen runtime release,
trajectory bank, task, and carrier evidence are available.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/property/hypothesis_matrix.json"
OUT = ROOT / "results/property/declared_persistent_4096/legacy_coherent_long_replay_manifest.json"
PLAN_DIR = ROOT / "results/property/declared_persistent_4096/legacy_coherent_plans"

MODEL_INFO = {
    "deepseek8b": {
        "architecture": "deepseek8b",
        "model": "/data1/tzh/models/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
        "release": "deepseek8b_seq{seq}_r1",
        "bank": "results/property/tcmp_allop_v1/input_banks/deepseek8b_seq{seq}_trajectory4096.json",
    },
    "phi4": {
        "architecture": "phi",
        "model": "/data1/tzh/models/microsoft/Phi-4-mini-instruct",
        "release": "phi4_seq{seq}_r1",
        "bank": "results/property/declared_persistent_4096/expanded_controls/input_banks/phi4_seq{seq}_cycled_4224.json",
    },
    "qwen": {
        "architecture": "qwen",
        "model": "/data1/tzh/models/Qwen/Qwen3-1.7B",
        "release": "qwen_seq{seq}_r1",
        "bank": "results/property/tcmp_allop_v1/input_banks/qwen_seq{seq}_trajectory4096.json",
    },
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def read_gzip_json(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def safe_case_id(candidate_id: str) -> str:
    return "legacy-coherent-" + candidate_id.replace(":", "-")


def main() -> None:
    source = read_json(SOURCE)
    candidates = [
        row for row in source.get("rows", [])
        if row.get("observed_label", {}).get("role") == "COHERENT_F_B_BIAS"
    ]
    PLAN_DIR.mkdir(parents=True, exist_ok=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for candidate in candidates:
        metadata = candidate["grouping_metadata"]
        model_key = str(metadata["model"])
        seq = int(metadata["sequence_length"])
        info = MODEL_INFO[model_key]
        candidate_id = str(candidate["candidate_id"])
        case_id = safe_case_id(candidate_id)
        task_id = ":".join(candidate_id.split(":")[2:])
        release = ROOT / "results/coverage/runtime_releases" / info["release"].format(seq=seq)
        bank = ROOT / info["bank"].format(seq=seq)
        if model_key == "qwen" and seq == 128:
            bank = ROOT / "results/property/declared_persistent_4096/qwen_seq128_bank_4224.json"

        evidence_paths = [ROOT / item["artifact"] for item in candidate.get("label_evidence", [])]
        evidence = next((path for path in evidence_paths if path.exists()), None)
        reasons: list[str] = []
        carrier = None
        if evidence is None:
            reasons.append("LEGACY_CARRIER_EVIDENCE_MISSING")
        else:
            payload = read_gzip_json(evidence) if evidence.suffix == ".gz" else read_json(evidence)
            carrier = payload.get("carrier_parameter")
            if not carrier:
                reasons.append("CARRIER_PARAMETER_MISSING")
            if str(payload.get("task_id")) != task_id:
                reasons.append("EVIDENCE_TASK_MISMATCH")
        if not release.exists():
            reasons.append("RUNTIME_RELEASE_MISSING")
        if not bank.exists():
            reasons.append("TRAJECTORY_BANK_MISSING")

        plan_path = PLAN_DIR / f"{case_id}.json"
        runnable = not reasons
        if runnable:
            plan = {
                "schema": "kernel-analyzer-legacy-coherent-long-plan-v1",
                "model": model_key,
                "sequence_length": seq,
                "selection": "Every concrete endpoint labelled COHERENT_F_B_BIAS in the frozen hypothesis matrix.",
                "claim_boundary": "The legacy 32-state label is candidate evidence only; only the completed 4096-step replay may assign a long-horizon result.",
                "cases": [{
                    "case_id": case_id,
                    "task_id": task_id,
                    "carrier": carrier,
                    "family": f"LEGACY_{metadata.get('phase')}_{metadata.get('mathematical_operation')}",
                    "member_count": 1,
                }],
            }
            plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")

        rows.append({
            "case_id": case_id,
            "candidate_id": candidate_id,
            "model_key": model_key,
            "architecture": info["architecture"],
            "model_path": info["model"],
            "sequence_length": seq,
            "phase": metadata.get("phase"),
            "mathematical_operation": metadata.get("mathematical_operation"),
            "implementation_kind": metadata.get("implementation_kind"),
            "exact_aot_endpoint_id": candidate.get("exact_aot_endpoint_id"),
            "task_id": task_id,
            "carrier": carrier,
            "legacy_evidence": str(evidence.relative_to(ROOT)) if evidence else None,
            "release_dir": str(release.relative_to(ROOT)) if release.exists() else None,
            "input_bank": str(bank.relative_to(ROOT)) if bank.exists() else None,
            "case_plan": str(plan_path.relative_to(ROOT)) if runnable else None,
            "output": f"results/property/declared_persistent_4096/legacy_coherent_candidates/{case_id}_4096.json",
            "runnable": runnable,
            "status": "READY_FOR_LONG_REPLAY" if runnable else "RECAPTURE_REQUIRED",
            "reasons": reasons,
        })

    manifest = {
        "schema": "legacy-coherent-endpoints-long-replay-manifest-v1",
        "source": str(SOURCE.relative_to(ROOT)),
        "selection": "All concrete endpoints with frozen observed role COHERENT_F_B_BIAS; no 4096-step outcome was used for selection.",
        "long_horizon_steps": 4096,
        "summary": {
            "candidate_count": len(rows),
            "ready_for_long_replay": sum(row["runnable"] for row in rows),
            "recapture_required": sum(not row["runnable"] for row in rows),
        },
        "claim_boundary": "These are legacy bias candidates, not confirmed persistent cases. Failed or missing replays remain unresolved.",
        "rows": rows,
    }
    OUT.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
