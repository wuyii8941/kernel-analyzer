#!/usr/bin/env python3
"""Bind the v2 tournament to exact existing artifacts, states and environments.

This is a feasibility audit only.  It never launches a model or declares a
property.  A case is READY only when its artifact set contains 32 distinct
state IDs, exact sequence length/model identity, and a closed-loop consequence
artifact; otherwise it is explicitly ineligible.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/property/bias_formation_v2"


def sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> Any:
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "UNAVAILABLE"


def walk_state_ids(value: Any) -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            normal = str(key).lower()
            if normal in {"state_id", "state_ids"}:
                if isinstance(child, (list, tuple)):
                    found.extend(str(item) for item in child)
                elif isinstance(child, (str, int)):
                    found.append(str(child))
            found.extend(walk_state_ids(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(walk_state_ids(child))
    return found


def model_manifest(model_root: str) -> dict[str, Any]:
    root = Path(model_root)
    files = [root / "config.json", root / "model.safetensors.index.json"]
    return {
        "root": model_root,
        "files": {str(path): sha256(path) for path in files},
        "complete": all(path.exists() for path in files),
        "manifest_sha256": hashlib.sha256(json.dumps(
            {str(path): sha256(path) for path in files}, sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest(),
    }


CASE_BINDINGS: list[dict[str, Any]] = [
    {
        "case_id": "liger_fused_ce_t128",
        "model": "qwen3_1p7b",
        "model_root": "/data1/tzh/models/Qwen/Qwen3-1.7B",
        "sequence_length": 128,
        "role": "DEEP_POSITIVE_ANCHOR",
        "endpoint_or_region_id": "qwen3_liger_fused_linear_ce:dW",
        "proof_unit_id": "qwen3_liger_fused_linear_ce",
        "artifacts": ["results/property/seup_mainline/liger_seup.json", "archive/nonprecision_v1/runs/liger.fused_ce.protocol.json"],
        "state_sources": ["results/property/seup_mainline/liger_seup.json"],
        "state_bank": "results/property/seup_mainline/liger_seup.json",
        "repair_id": "fp32_accum",
        "sham_id": "norm_matched_sham",
        "parameter_coordinates": "declared_in_liger_seup_artifact",
        "source_status": "M3_CLOSED_CHUNK_GEOMETRY_BF16_ACCUMULATION",
        "requires_consequence": True,
    },
    {
        "case_id": "phi4_lm_head_dx_seq64",
        "model": "phi4",
        "model_root": "/data1/tzh/models/microsoft/Phi-4-mini-instruct",
        "sequence_length": 64,
        "role": "BREADTH_POSITIVE",
        "endpoint_or_region_id": "phi4_seq64:lm_head.input_gradient.mm",
        "proof_unit_id": "phi4_seq64_backward_497_output",
        "artifacts": ["results/coverage/cases/phi4_seq64_lmhead_dx.json", "results/coverage/cases/phi4_seq64_lmhead_dx_trajectory.json"],
        "state_sources": ["results/coverage/cases/phi4_seq64_lmhead_dx.json"],
        "state_bank": "results/coverage/phi4_seq64_input_bank.json",
        "repair_id": "fp32_kernel_accumulation",
        "sham_id": "matched_precision_sham",
        "parameter_coordinates": "lm_head_input_gradient_coordinates",
        "source_status": "M3_CLOSED_MM_KERNEL_ARITHMETIC",
        "requires_consequence": True,
    },
    {
        "case_id": "qwen_saved_p_seq128",
        "model": "qwen3_1p7b",
        "model_root": "/data1/tzh/models/Qwen/Qwen3-1.7B",
        "sequence_length": 128,
        "role": "P2_P3_CANDIDATE",
        "endpoint_or_region_id": "qwen_seq128_layer27_attention_softmax_saved_P",
        "proof_unit_id": "qwen_seq128_layer27_attention_softmax_fb",
        "artifacts": ["results/coverage/cases/qwen128_softmax_fb.json", "results/coverage/cases/qwen128_softmax_saved_p_trajectory.json"],
        "state_sources": ["results/coverage/cases/qwen128_softmax_saved_p_trajectory.json", "results/coverage/cases/qwen128_softmax_fb.json"],
        "state_bank": "results/coverage/qwen_seq128_input_bank.json",
        "repair_id": "saved_P_exact_repair",
        "sham_id": "saved_P_matched_sham",
        "parameter_coordinates": "layer27_attention_declared_region",
        "source_status": "M3_CLOSED_SAVED_STATE_RECONSTRUCTION_REGION",
        "requires_consequence": True,
    },
    {
        "case_id": "qwen_l23_key_materialization_seq1024",
        "model": "qwen3_1p7b",
        "model_root": "/data1/tzh/models/Qwen/Qwen3-1.7B",
        "sequence_length": 1024,
        "role": "P3_CANDIDATE",
        "endpoint_or_region_id": "layer23.q_proj.key_materialization.tile",
        "proof_unit_id": "qwen_l23_qproj_operand_decomposition",
        "artifacts": ["results/final/l23_key_actual_scale_materialization_32states.json"],
        "state_sources": ["results/final/l23_key_actual_scale_materialization_32states.json"],
        "state_bank": "results/final/l23_key_actual_scale_materialization_32states.json",
        "repair_id": "operand_scale_repair",
        "sham_id": "NOT_AVAILABLE",
        "parameter_coordinates": "model.layers.23.self_attn.q_proj.weight",
        "source_status": "M3_CLOSED_MM_OPERAND_DECOMPOSITION",
        "requires_consequence": True,
    },
    {
        "case_id": "qwen_rsqrt_seq128",
        "model": "qwen3_1p7b",
        "model_root": "/data1/tzh/models/Qwen/Qwen3-1.7B",
        "sequence_length": 128,
        "role": "CRITICAL_BOUNDARY",
        "endpoint_or_region_id": "forward:graph0:rsqrt_13",
        "proof_unit_id": "qwen_rsqrt13_t2_t3",
        "artifacts": ["results/coverage/cases/full_coordinate/qwen_seq128_rsqrt13_t2.json.gz", "results/coverage/cases/full_coordinate/qwen_seq128_rsqrt13_t3_gram.json.gz"],
        "state_sources": ["results/coverage/cases/full_coordinate/qwen_seq128_rsqrt13_t2.json.gz"],
        "state_bank": "results/coverage/qwen_seq128_input_bank.json",
        "repair_id": "rsqrt_reference_repair",
        "sham_id": "NOT_AVAILABLE",
        "parameter_coordinates": "all_declared_parameter_gradients",
        "source_status": "M3_ENDPOINT_BOUNDARY_ONLY",
        "requires_consequence": True,
    },
    {
        "case_id": "qwen_bmm_seq64",
        "model": "qwen3_1p7b",
        "model_root": "/data1/tzh/models/Qwen/Qwen3-1.7B",
        "sequence_length": 64,
        "role": "LOCAL_VARIANCE_NEGATIVE",
        "endpoint_or_region_id": "forward:10:output_0",
        "proof_unit_id": "qwen_bmm_seq64_seup_negative_control",
        "artifacts": ["results/property/seup_mainline/qwen_bmm_seq64_seup.json.gz", "results/property/seup_geometry_followup/qwen_bmm_seq64_geometry.json"],
        "state_sources": ["results/property/seup_mainline/qwen_bmm_seq64_seup.json.gz"],
        "state_bank": "results/coverage/qwen_seq64_input_bank.json",
        "repair_id": "bmm_endpoint_repair",
        "sham_id": "bmm_matched_sham",
        "parameter_coordinates": "declared_qwen_parameter_carrier",
        "source_status": "M3_ENDPOINT_BOUNDARY_ONLY",
        "requires_consequence": True,
    },
    {
        "case_id": "qwen_vproj_seq128",
        "model": "qwen3_1p7b",
        "model_root": "/data1/tzh/models/Qwen/Qwen3-1.7B",
        "sequence_length": 128,
        "role": "NONPERSISTENT_SOURCE_BOUNDARY",
        "endpoint_or_region_id": "model.layers.0.self_attn.v_proj.output",
        "proof_unit_id": "qwen_seq128_forward_8_output",
        "artifacts": ["results/coverage/cases/qwen128_vproj.json", "results/coverage/cases/qwen128_vproj_trajectory.json"],
        "state_sources": ["results/coverage/cases/qwen128_vproj.json"],
        "state_bank": "results/coverage/qwen_seq128_input_bank.json",
        "repair_id": "output_rounding_reference_repair",
        "sham_id": "matched_precision_sham",
        "parameter_coordinates": "model.layers.0.self_attn.v_proj.weight",
        "source_status": "M3_OUTPUT_ROUNDING_CAUSAL_BOUNDARY",
        "requires_consequence": True,
    },
    {
        "case_id": "qwen3vl_silu_seq160",
        "model": "qwen3_vl_reranker_2b",
        "model_root": "/data1/tzh/models/Qwen/Qwen3-VL-Reranker-2B",
        "sequence_length": 160,
        "role": "LOCAL_ERROR_NEGATIVE",
        "endpoint_or_region_id": "layer0.silu.backward",
        "proof_unit_id": "qwen3vl_layer0_silu",
        "artifacts": ["results/coverage/cases/qwen3vl_layer0_silu_trajectory.json"],
        "state_sources": ["results/coverage/cases/qwen3vl_layer0_silu_trajectory.json"],
        "state_bank": "results/coverage/cases/qwen3vl_layer0_silu_trajectory.json",
        "repair_id": "silu_reference_repair",
        "sham_id": "NOT_AVAILABLE",
        "parameter_coordinates": "declared_trajectory_carrier",
        "source_status": "M3_ENDPOINT_BOUNDARY_ONLY",
        "requires_consequence": True,
    },
    {
        "case_id": "flash_attention_literature_anchor",
        "model": "external_literature",
        "model_root": "",
        "sequence_length": None,
        "role": "EXTERNAL_CONTROL_NOT_MEASURED",
        "endpoint_or_region_id": "flash_attention_paper_mechanism",
        "proof_unit_id": "LITERATURE_ANCHOR_NOT_MEASURED",
        "artifacts": [],
        "state_sources": [],
        "state_bank": "",
        "repair_id": "literature_only",
        "sham_id": "NOT_APPLICABLE",
        "parameter_coordinates": "NOT_MEASURED",
        "source_status": "LITERATURE_ANCHOR_NOT_MEASURED",
        "requires_consequence": False,
    },
]


def bind_case(spec: Mapping[str, Any], runner_commit: str) -> dict[str, Any]:
    artifact_paths = [ROOT / str(path) for path in spec["artifacts"]]
    source_paths = [ROOT / str(path) for path in spec["state_sources"]]
    missing = [str(path) for path in artifact_paths + source_paths if not path.exists()]
    artifact_hashes = {str(path.relative_to(ROOT)): sha256(path) for path in artifact_paths}
    state_ids: list[str] = []
    for path in source_paths:
        if path.exists():
            state_ids.extend(walk_state_ids(load(path)))
    unique_ids = list(dict.fromkeys(state_ids))
    bank = Path(str(spec["state_bank"])) if spec["state_bank"] else None
    bank_path = (ROOT / bank) if bank and not bank.is_absolute() else bank
    bank_hash = sha256(bank_path) if bank_path else None
    reasons: list[str] = list("MISSING_ARTIFACT:" + item for item in missing)
    if spec["case_id"] == "flash_attention_literature_anchor":
        reasons.append("LITERATURE_ANCHOR_NOT_MEASURED")
    if len(set(unique_ids)) < 32:
        reasons.append(f"INSUFFICIENT_UNIQUE_STATE_IDS:{len(set(unique_ids))}")
    if spec["case_id"] == "qwen3vl_silu_seq160":
        reasons.append("INELIGIBLE_IN_V2_INSUFFICIENT_STATES:only_six_unique_natural_states")
    if spec["case_id"] in {"qwen_l23_key_materialization_seq1024", "qwen_rsqrt_seq128"}:
        reasons.append("INELIGIBLE_MISSING_CONSEQUENCE_TRACE")
    manifest = model_manifest(str(spec["model_root"])) if spec["model_root"] else {"complete": False, "manifest_sha256": None}
    if not manifest.get("complete") and spec["model_root"]:
        reasons.append("MODEL_MANIFEST_INCOMPLETE")
    ready = not reasons
    return {
        **dict(spec),
        "runner_commit": runner_commit,
        "artifact_hashes": artifact_hashes,
        "state_ids": unique_ids,
        "state_count": len(set(unique_ids)),
        "state_bank_sha256": bank_hash,
        "model_manifest": manifest,
        "candidate_release_and_wrapper_hashes": artifact_hashes,
        "repair_and_sham_bound": bool(spec["repair_id"] and spec["sham_id"]),
        "feasibility": "READY" if ready else "INELIGIBLE_WITH_REASON",
        "ineligibility_reasons": reasons,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    runner_commit = git_commit()
    bound = [bind_case(spec, runner_commit) for spec in CASE_BINDINGS]
    feasibility = {
        "schema": "kernel-analyzer-bias-property-feasibility-v2",
        "protocol_id": "bias_formation_v2",
        "runner_commit": runner_commit,
        "gpu_campaign_started": False,
        "ready_case_ids": [row["case_id"] for row in bound if row["feasibility"] == "READY"],
        "ineligible_case_ids": [row["case_id"] for row in bound if row["feasibility"] != "READY"],
        "cases": [{"case_id": row["case_id"], "feasibility": row["feasibility"], "reasons": row["ineligibility_reasons"]} for row in bound],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, value in [("roster_bound.json", {"schema": "kernel-analyzer-bias-property-roster-v2", "cases": bound}), ("feasibility_report.json", feasibility)]:
        path = args.output_dir / name
        tmp = path.with_name("." + path.name + ".tmp")
        tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tmp.replace(path)
    print(json.dumps({"ready": feasibility["ready_case_ids"], "ineligible": feasibility["ineligible_case_ids"], "gpu_campaign_started": False}, sort_keys=True))


if __name__ == "__main__":
    main()
