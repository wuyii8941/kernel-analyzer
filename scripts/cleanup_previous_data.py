#!/usr/bin/env python3
"""Remove rebuildable campaign data while retaining canonical derivations."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CACHE = Path("/data1/tzh/cache/kernel_analyzer")
RAW = ROOT / "archive/round1_raw/training_semantic_oracle/qwen3_1p7b"
FULL = RAW / "full_step_inventory"
RUNS = ROOT / "archive/nonprecision_v1/runs"
REPORT = ROOT / "results/final/cleanup.json"

RAW_CAMPAIGN_DIRS = (
    "analytic_local_vjp_campaign_v1",
    "analytic_local_vjp_heldout_replication_v1",
    "candidates",
    "non_bf16_all_op_v1",
    "supplementary_all_op_bias_v1",
)

RUN_DIRS = (
    "fp32_inductor_strict",
    "liger_fused_ce",
    "liger_fused_ce_chunk",
    "liger_fused_ce_propagation",
    "liger_rmsnorm",
    "liger_rope",
    "liger_swiglu",
    "linear_layout",
    "lmhead_extent",
    "lmhead_layout",
    "lmhead_propagation",
    "sdpa",
    "silu_domain",
    "silu_propagation",
)

RUN_RAW_FILES = (
    "generated.seq64.triton.json",
    "generated.seq128.triton.json",
    "generated.seq256.triton.json",
    "generated.seq128.external.json",
    "generated.seq256.external.json",
    "generated.seq64.confirmation.json",
    "generated.seq128.confirmation.json",
    "generated.seq256.confirmation.json",
)

# These files are the canonical full F+B derivations and the structural
# witnesses required to interpret them. Superseded discovery/replay files are
# deliberately excluded.
FULL_KEEP = {
    "all_op_forward_backward_proof_ledger.json",
    "all_op_reference_protocol_v2.json",
    "all_representation_math_completion.json",
    "aot_derivation_witness_census_v3.json",
    "aot_v4_lowered_cut_interface_audit.json",
    "aot_v4_to_inductor_source_alignment.json",
    "aot_v4_to_lowered_structural_alignment.json",
    "atomic_forward_vjp_mathematical_dossier_v4.json",
    "candidate_region_mathematical_atlas_v3.json",
    "cross_precision_forward_backward_region_alignment_v1.json",
    "eager_bf16_forward_backward_proof_units_v3.json",
    "eager_bf16_invocation_derivation_witness_v3.json",
    "eager_bf16_operator_mathematical_atlas_seq_v6.json",
    "eager_fp32_forward_backward_proof_units_v3.json",
    "eager_fp32_invocation_derivation_witness_v3.json",
    "eager_fp32_operator_mathematical_atlas_seq_v6.json",
    "eager_fp32_sequence_forward_backward_proof_units.json",
    "eager_sequence_forward_backward_proof_units.json",
    "eager_to_aot_structural_alignment.json",
    "forward_backward_proof_unit_atlas.json",
    "full_step_candidate_semantic_mapping.json",
    "full_step_forward_backward_derivation_completion_v1.json",
    "generated_compute_dataflow_audit_v1.json",
    "generated_implementation_correctness_gap_ledger_v3.json",
    "generated_implementation_mechanism_census_v1.json",
    "generated_runtime_call_completeness_audit_v1.json",
    "gradient_accumulation_complete_proof.json",
    "inductor_generated_region_inventory.json",
    "inductor_lowered_graph_inventory.json",
    "invocation_mechanistic_correctness_protocol_v1.json",
    "joint_forward_backward_candidate_registry_v2.json",
    "lowered_abi_runtime_identity_v1.json",
    "lowered_to_generated_region_alignment.json",
    "operand_bound_repeated_layer_proof.json",
    "reference_all_op_math_completion.json",
    "reference_cut_execution_certificate_cross_phase_v2.json",
    "repeated_transformer_layers_1_27_complete_proof.json",
    "rigorous_forward_backward_derivation_certificate_v1.json",
    "silu_invocation_0_numerical_derivation.json",
    "strict_all_op_operand_bound_proof.json",
    "triton_online_reference_campaign_v1.json",
}


def canonical_derivation(name: str) -> bool:
    return (
        name in FULL_KEEP
        or name.endswith("_complete_proof.json")
        or name.endswith("_exact_binding_v1.json")
    )


def tree_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if ROOT.resolve() != Path("/data1/tzh/kernel-analyzer"):
        raise SystemExit(f"unexpected repository root: {ROOT}")

    targets = [CACHE]
    targets.extend(RAW / name for name in RAW_CAMPAIGN_DIRS)
    targets.extend(RUNS / name for name in RUN_DIRS)
    targets.extend(RUNS / name for name in RUN_RAW_FILES)
    targets.extend(
        path for path in FULL.iterdir()
        if path.is_file() and not canonical_derivation(path.name)
    )
    targets = [path for path in targets if path.exists()]
    before = sum(tree_bytes(path) for path in targets)
    preview = {
        "mode": "apply" if args.apply else "dry-run",
        "target_count": len(targets),
        "reclaimable_bytes": before,
        "canonical_full_step_files_retained": sum(
            path.is_file() and canonical_derivation(path.name)
            for path in FULL.iterdir()
        ),
    }
    if not args.apply:
        print(json.dumps(preview, indent=2, sort_keys=True))
        return

    removed = []
    for path in targets:
        size = tree_bytes(path)
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        removed.append({"path": str(path), "bytes": size})

    result = {
        "schema": "kernel-analyzer-cleanup-v1",
        "policy": (
            "Retain canonical mathematical F+B derivations, compact final evidence, "
            "tracked case certificates, and implementation mappings; remove only "
            "rebuildable checkpoints, worker replay, discovery runs, and superseded files."
        ),
        "removed_bytes": sum(row["bytes"] for row in removed),
        "removed_targets": removed,
        "retained": {
            "full_step_derivation_directory": str(FULL),
            "canonical_full_step_files": sorted(
                path.name for path in FULL.iterdir()
                if path.is_file() and canonical_derivation(path.name)
            ),
            "compact_results": str(ROOT / "results/final"),
            "case_document": str(ROOT / "case.md"),
        },
    }
    result["result_sha256"] = hashlib.sha256(
        json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    REPORT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "report": str(REPORT),
        "removed_bytes": result["removed_bytes"],
        "removed_targets": len(removed),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
