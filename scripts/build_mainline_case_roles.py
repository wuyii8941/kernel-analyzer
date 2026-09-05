#!/usr/bin/env python3
"""Build the traceable mainline case-role registry from retained artifacts."""

from __future__ import annotations

import hashlib
import gzip
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/mainline_case_roles.json"

DECLARATIONS = (
    {
        "case_id": "gemma4_text128_scan_0037",
        "semantic_family": "normalization/reduction",
        "candidate_role": "TRITON_CANDIDATE",
        "purpose": "FIRST_TRITON_FAMILY",
        "artifact": "results/property/three_mechanism_profiles_v1/runs/gemma4_text128_scan_0037/prediction.json",
        "formation_artifact": "results/property/three_mechanism_profiles_v1/runs/gemma4_text128_scan_0037/formation.json",
    },
    {
        "case_id": "llama32_text128_scan_0000",
        "semantic_family": "softmax backward",
        "candidate_role": "TRITON_CANDIDATE",
        "purpose": "SECOND_TRITON_FAMILY",
        "artifact": "results/property/three_mechanism_profiles_v1/runs/llama32_text128_scan_0000_profile_carrier/prediction.json",
        "formation_artifact": "results/property/three_mechanism_profiles_v1/runs/llama32_text128_scan_0000_profile_carrier/formation.json",
    },
    {
        "case_id": "phi4_seq64_lmhead_dx",
        "semantic_family": "lm_head backward matrix multiplication",
        "candidate_role": "CONVENTIONAL_EXTERNAL_MM_CANDIDATE",
        "purpose": "CONVENTIONAL_IMPLEMENTATION_FAMILY",
        "artifact": "results/property/training_bias_profile_v2/five_case_raw/phi.json",
    },
    {
        "case_id": "liger_fused_ce_t128",
        "semantic_family": "fused cross entropy and weight-gradient accumulation",
        "candidate_role": "MIXED_TRAINING_COMPUTATION",
        "purpose": "MECHANISM_AND_CONSEQUENCE_SUPPORT",
        "artifact": "results/property/training_bias_profile_v2/five_case_raw/liger.json",
    },
    {
        "case_id": "deepseek8b_seq256_backward_1714_in_out_ptr0",
        "semantic_family": "normalization backward",
        "candidate_role": "COMPILED_TRAINING_CANDIDATE",
        "purpose": "STATE_DEPENDENCE_PIPELINE_VALIDATION",
        "artifact": "results/property/training_bias_profile_v2/prospective_batch_1/raw/deepseek8b_seq256_backward_1714_in_out_ptr0.json",
    },
    {
        "case_id": "deepseek8b_seq128_backward_1256_out_ptr0",
        "semantic_family": "attention projection backward",
        "candidate_role": "COMPILED_TRAINING_CANDIDATE",
        "purpose": "STATE_DEPENDENCE_PIPELINE_VALIDATION",
        "artifact": "results/property/training_bias_profile_v2/prospective_batch_2/raw/deepseek8b_seq128_backward_1256_out_ptr0.json",
    },
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _measurement_geometry(payload: dict) -> list[str]:
    geometries = set()
    for stage in payload.get("stages", {}).values():
        for name in stage:
            geometries.add("FULL_VECTOR" if name == "EXACT" else "COUNT_SKETCH")
    return sorted(geometries) or ["LEGACY_CASE_SPECIFIC"]


def _backend_evidence(runtime: dict) -> dict:
    evidence = {
        key: runtime.get(key) for key in (
            "target_kind", "target_region_id", "target_symbol", "function",
            "endpoint", "source_sha256", "exact_aot_endpoint_id",
            "frozen_release_id", "task_id",
        ) if runtime.get(key) is not None
    }
    identities = runtime.get("identities")
    if identities:
        unique = {
            json.dumps(row, sort_keys=True): row for row in identities
        }
        evidence["runtime_identity_count"] = len(identities)
        evidence["unique_runtime_identities"] = list(unique.values())
    return evidence


def _release_backend_evidence(runtime: dict) -> dict | None:
    release = runtime.get("frozen_release_id")
    task_id = runtime.get("task_id")
    if not release or not task_id:
        return None
    plan = ROOT / "results/coverage/runtime_releases" / release / "same_dtype_tasks.json.gz"
    if not plan.is_file():
        return None
    with gzip.open(plan, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    for row in payload.get("rows", []):
        if str(row.get("task_id", "")).removeprefix("same-dtype:") == task_id:
            return {
                "runtime_plan": str(plan.relative_to(ROOT)),
                "runtime_plan_sha256": _sha(plan),
                "implementation_kind": row.get("implementation_kind"),
                "symbol": row.get("symbol"),
                "phase": row.get("phase"),
                "binding_status": row.get("status"),
                "exact_aot_endpoint_id": row.get("exact_aot_endpoint_id"),
            }
    return None


def main() -> None:
    rows = []
    for declaration in DECLARATIONS:
        path = ROOT / declaration["artifact"]
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text())
        runtime = payload.get("runtime_boundary", payload)
        row = dict(declaration)
        row.update({
            "artifact_sha256": _sha(path),
            "artifact_status": payload.get("status", "UNKNOWN"),
            "carrier": payload.get("carrier"),
            "claim_boundary": payload.get("claim_boundary"),
            "measurement_geometry": _measurement_geometry(payload),
            "actual_backend_evidence": _backend_evidence(runtime),
            "data_availability": {
                "joint_gram": bool(payload.get("stages")),
                "original_coordinate_statistics": bool(
                    payload.get("original_coordinate_statistics")
                ),
                "actual_parameter_write": payload.get("primary_update_endpoint") == "PARAMETER_WRITE",
            },
        })
        release_evidence = _release_backend_evidence(runtime)
        if release_evidence:
            row["actual_backend_evidence"].update(release_evidence)
        recapture = (
            ROOT / "results/property/training_numerical_analysis_v1/recapture"
            / f"{declaration['case_id']}.json"
        )
        if declaration["case_id"] == "liger_fused_ce_t128":
            recapture = recapture.with_name("liger.json")
        if declaration["case_id"] == "phi4_seq64_lmhead_dx":
            recapture = recapture.with_name("phi.json")
        if recapture.is_file():
            recaptured = json.loads(recapture.read_text())
            if recaptured.get("schema") == "kernel-analyzer-training-bias-profile-v2-raw-case":
                row["new_measurement"] = {
                    "artifact": str(recapture.relative_to(ROOT)),
                    "artifact_sha256": _sha(recapture),
                    "status": recaptured.get("status"),
                    "primary_update_endpoint": recaptured.get("primary_update_endpoint"),
                    "original_coordinate_statistics": bool(
                        recaptured.get("original_coordinate_statistics")
                    ),
                }
        formation = declaration.get("formation_artifact")
        if formation:
            formation_path = ROOT / formation
            row["formation_status"] = json.loads(formation_path.read_text()).get("status")
            row["formation_artifact_sha256"] = _sha(formation_path)
        rows.append(row)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps({
        "schema": "kernel-analyzer-mainline-case-roles-v1",
        "source_commit_baseline": "c726911af41efc3d8cccb62dc4cdbf1d3f99341b",
        "generation_rule": "Declared roles plus fields extracted from retained machine artifacts",
        "cases": rows,
    }, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
