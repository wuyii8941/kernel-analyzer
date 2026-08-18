#!/usr/bin/env python3
"""Build the fail-closed 4-model x 3-shape full-operator status matrix."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/coverage"
MODELS = {
    "qwen3_1p7b": "qwen",
    "mamba_130m": "mamba",
    "phi4_mini_3p8b": "phi4",
    "deepseek_r1_0528_qwen3_8b": "deepseek8b",
}
SHAPES = (64, 128, 256)


def load(path: Path) -> dict[str, Any]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def artifact(path: Path, expected_status: str) -> dict[str, Any]:
    relative = str(path.relative_to(ROOT))
    if not path.exists():
        return {"path": relative, "status": "MISSING", "gate": False}
    stem = path.name[:-len(".json.gz")] if path.name.endswith(".json.gz") else path.stem
    summary_path = path.with_name(stem + ".summary.json")
    data = load(summary_path) if summary_path.exists() else load(path)
    return {
        "path": relative, "status": data.get("status"),
        "result_sha256": data.get("result_sha256"),
        "denominator": data.get("denominator"),
        "schema": data.get("schema"),
        "gates": data.get("gates"),
        "gate": data.get("status") == expected_status,
    }


def inventory_path(prefix: str, shape: int) -> Path:
    runtime = RESULTS / "runtime_releases" / f"{prefix}_seq{shape}_r1" / "inventory.json.gz"
    if runtime.exists():
        return runtime
    if prefix == "qwen" and shape == 64:
        return RESULTS / "qwen_seq64_executed_v5_inventory.json.gz"
    return RESULTS / f"{prefix}_seq{shape}_executed_v1_inventory.json.gz"


def triton_oracle_path(prefix: str, shape: int) -> Path:
    runtime = RESULTS / "runtime_releases" / f"{prefix}_seq{shape}_r1" / "triton_oracle.json.gz"
    typed = runtime.with_name("typed_triton_oracle.json.gz")
    if typed.exists():
        return typed
    if runtime.exists():
        return runtime
    if prefix == "qwen":
        version = "v5" if shape == 64 else "v1"
        return RESULTS / f"qwen_seq{shape}_executed_{version}_triton_oracle.json.gz"
    return RESULTS / f"{prefix}_seq{shape}_fp32_oracle.json.gz"


def default_aot_math_path(prefix: str, shape: int) -> Path:
    """Locate only mathematics derived from Inductor's actual AOT partition."""

    runtime = (
        RESULTS / "runtime_releases" / f"{prefix}_seq{shape}_r1"
        / "default_aot_math.json.gz"
    )
    if runtime.exists():
        return runtime
    standard = RESULTS / "standard_aot" / f"{prefix}_seq{shape}_math.json.gz"
    versioned = standard.with_name(f"{prefix}_seq{shape}_math_v2.json.gz")
    return versioned if versioned.exists() else standard


def default_aot_capture_path(prefix: str, shape: int) -> Path:
    """Locate the capture that proves the math ledger is Inductor's partition."""

    runtime = (
        RESULTS / "runtime_releases" / f"{prefix}_seq{shape}_r1"
        / "default_aot_capture.json.gz"
    )
    if runtime.exists():
        return runtime
    # Segmented runtime-identity captures are intentionally retained under the
    # ``_raw`` name because the candidate bridge also binds to that exact
    # source artifact.  It is still the authoritative actual-Inductor capture;
    # do not make a second copy merely to satisfy the coverage matrix.
    runtime_raw = runtime.with_name("default_aot_capture_raw.json.gz")
    if runtime_raw.exists():
        return runtime_raw
    standard = RESULTS / "standard_aot" / f"{prefix}_seq{shape}_capture.json.gz"
    versioned = standard.with_name(f"{prefix}_seq{shape}_capture_v2.json.gz")
    return versioned if versioned.exists() else standard


def default_aot_evidence(prefix: str, shape: int) -> dict[str, Any]:
    """Fail closed unless a complete ledger binds an actual Inductor AOT graph."""

    math_path = default_aot_math_path(prefix, shape)
    capture_path = default_aot_capture_path(prefix, shape)
    math_row = artifact(math_path, "COMPLETE_AOT_FORWARD_BACKWARD_DERIVATION")
    capture_row = {
        "path": str(capture_path.relative_to(ROOT)),
        "status": "MISSING",
        "gate": False,
    }
    binding_gates = {
        "math_ledger_complete": bool(math_row["gate"]),
        "capture_present": capture_path.exists(),
        "math_capture_sha256_exact": False,
        "actual_inductor_default_partition": False,
        "segmented_boundary_accounting_valid": False,
    }
    if not math_path.exists() or not capture_path.exists():
        math_row["capture"] = capture_row
        math_row["binding_gates"] = binding_gates
        math_row["gate"] = False
        return math_row

    math_data = load(math_path)
    capture_data = load(capture_path)
    capture_payload = capture_data.get("capture", {})
    capture_sha = capture_payload.get("capture_sha256")
    binding_gates["math_capture_sha256_exact"] = (
        math_data.get("capture_sha256") == capture_sha
    )

    # New generic captures explicitly attest that only Inductor's lower
    # compiler was replaced.  The original Qwen seq64 release predates that
    # field, but is cryptographically bound to its proof-tagged real Inductor
    # compilation and can be validated without trusting a filename.
    if capture_data.get("aot_partition") == (
        "EXACT_INDUCTOR_DEFAULT_DECOMPOSITION_AND_PARTITION"
    ):
        normalized = capture_data.get("status") == (
            "COMPLETE_SEGMENT_LOCAL_CAPTURE_WITH_EXPLICIT_CROSS_SEGMENT_BOUNDARIES"
        )
        segment_pairing = capture_payload.get("segment_pairing", {})
        pair_rows = segment_pairing.get("pairs", [])
        forward_indices = {
            int(node["segmented_origin"]["graph_index"])
            for graph in capture_payload.get("graphs", [])
            for node in graph.get("nodes", [])
            if node.get("segmented_origin", {}).get("phase") == "FORWARD"
        }
        backward_indices = {
            int(node["segmented_origin"]["graph_index"])
            for graph in capture_payload.get("graphs", [])
            for node in graph.get("nodes", [])
            if node.get("segmented_origin", {}).get("phase") == "BACKWARD"
        }
        declared_forward_indices = {
            int(row["forward_graph_index"]) for row in pair_rows
        } | {
            int(value) for value in segment_pairing.get(
                "unpaired_forward_graph_indices", []
            )
        }
        declared_backward_indices = {
            int(row["backward_graph_index"]) for row in pair_rows
        } | {
            int(value) for value in segment_pairing.get(
                "unpaired_backward_graph_indices", []
            )
        }
        segmented_boundary_gate = (
            normalized
            and segment_pairing.get("pairing_uses_runtime_identity_only") is True
            and "cross_segment_unresolved_edges_remain_explicit" in segment_pairing
            and not segment_pairing.get("unpaired_backward_graph_indices", [None])
            and len(pair_rows)
            == int(segment_pairing.get("unique_compiled_runtime_pairs", -1))
            and int(segment_pairing.get("raw_runtime_pair_observations", 0))
            >= len(pair_rows) > 0
            and all(
                row.get("runtime_identity_gates", {}).get(
                    "runtime_identity_only"
                ) is True
                and row.get("runtime_identity_gates", {}).get(
                    "name_shape_or_ordinal_pairing_used"
                ) is False
                for row in pair_rows
            )
            and forward_indices == declared_forward_indices
            and backward_indices == declared_backward_indices
        )
        normalized_complete = (
            normalized
            and bool(capture_data.get("source_result_sha256"))
            and segmented_boundary_gate
        )
        capture_gate = (
            (
                capture_data.get("status") == "COMPLETE_AOT_FB_CAPTURE"
                or normalized_complete
            )
            and capture_data.get("gates", {}).get(
                "actual_inductor_default_aot_partition"
            ) is True
            and all(capture_data.get("observation_stability", {}).values())
        )
        binding_gates["actual_inductor_default_partition"] = capture_gate
        binding_gates["segmented_boundary_accounting_valid"] = (
            segmented_boundary_gate if normalized else True
        )
    elif capture_data.get("status") == (
        "COMPLETE_STANDARD_AOT_FORWARD_BACKWARD_CAPTURE"
    ):
        proof_name = capture_path.name.replace("_capture", "_proof_capture")
        proof_path = capture_path.with_name(proof_name)
        if proof_path.exists():
            proof = load(proof_path)
            binding_gates["actual_inductor_default_partition"] = (
                proof.get("status") == "COMPLETE_PROOF_ID_PROPAGATION_CAPTURE"
                and proof.get("result_sha256")
                == capture_data.get("proof_capture_result_sha256")
                and proof.get("standard_aot_capture", {}).get("capture_sha256")
                == capture_sha
            )
            binding_gates["segmented_boundary_accounting_valid"] = True

    capture_row = {
        "path": str(capture_path.relative_to(ROOT)),
        "status": capture_data.get("status"),
        "result_sha256": capture_data.get("result_sha256"),
        "capture_sha256": capture_sha,
        "gate": binding_gates["actual_inductor_default_partition"],
    }
    math_row["capture"] = capture_row
    math_row["binding_gates"] = binding_gates
    math_row["gate"] = all(binding_gates.values())
    if not math_row["gate"] and math_row["status"] == (
        "COMPLETE_AOT_FORWARD_BACKWARD_DERIVATION"
    ):
        math_row["status"] = "UNBOUND_TO_ACTUAL_INDUCTOR_DEFAULT_AOT"
    return math_row


def triton_oracle_expected_status(path: Path) -> str:
    """Use the unified status for strict runtime releases.

    The older Qwen-only screen predates the four-model runtime release schema
    and used a different completion label.  Do not apply that legacy label to
    newly frozen Qwen releases.
    """
    runtime_root = RESULTS / "runtime_releases"
    if runtime_root in path.parents:
        return "COMPLETE_PRECISION_ONLY_RUNTIME_DENOMINATOR_ORACLE"
    if path.name.startswith("qwen_"):
        return "COMPLETE_TRITON_DENOMINATOR_HELDOUT_SCREEN"
    return "COMPLETE_PRECISION_ONLY_RUNTIME_DENOMINATOR_ORACLE"


def candidate_binding(model: str, shape: int) -> dict[str, Any]:
    """Return an explicit candidate-region to F+B binding gate.

    A generated-call inventory is not a semantic binding.  Keep the two
    artifacts separate so a warmed source census cannot certify an op case.
    """

    prefix = MODELS[model]
    runtime = (
        RESULTS / "runtime_releases" / f"{prefix}_seq{shape}_r1"
        / "candidate_fb_bridge.json.gz"
    )
    if runtime.exists():
        return artifact(
            runtime,
            "COMPLETE_ALL_EXECUTED_REGIONS_BOUND_TO_PROVED_FB_MATHEMATICS",
        )
    if model == "qwen3_1p7b" and shape == 64:
        return artifact(
            RESULTS / "qwen_inductor_identity_bridge.json.gz",
            "COMPLETE_CANONICAL_AOT_TO_PROOF_TAGGED_INDUCTOR_ACCOUNTING",
        )
    if model == "qwen3_1p7b" and shape in {128, 256}:
        return artifact(
            RESULTS / f"qwen_seq{shape}_candidate_fb_bridge.json.gz",
            "COMPLETE_ALL_EXECUTED_REGIONS_BOUND_TO_FROZEN_FB_REGISTRY",
        )
    if model == "mamba_130m":
        return artifact(
            RESULTS / "mamba_aot_inductor_bridge.json.gz", "COMPLETE"
        )
    return {
        "path": None,
        "status": "MISSING_COMPLETE_CANDIDATE_FB_BRIDGE",
        "result_sha256": None,
        "denominator": None,
        "gate": False,
    }


def triton_abi_cells() -> dict[tuple[str, int], dict[str, Any]]:
    path = RESULTS / "triton_reference_abi_audit.json"
    if not path.exists():
        return {}
    audit = load(path)
    return {
        (str(row["model"]), int(row["sequence_length"])): row
        for row in audit.get("cells", [])
    }


def main() -> None:
    math_path = RESULTS / "fb_multishape_ledger.json.gz"
    math_summary = RESULTS / "fb_multishape_ledger.summary.json"
    if math_summary.exists():
        math = load(math_summary)
    else:
        full_math = load(math_path)
        math = {
            "schema": "kernel-analyzer-multishape-fb-ledger-summary-v1",
            "status": full_math["status"],
            "result_sha256": full_math["result_sha256"],
            "denominator_cells": full_math["denominator_cells"],
            "denominators": full_math["denominators"],
            "source_ledger": str(math_path.relative_to(ROOT)),
            "source_ledger_bytes": math_path.stat().st_size,
        }
        math_summary.write_text(json.dumps(math, sort_keys=True, indent=2) + "\n")
    if not str(math["status"]).startswith("COMPLETE_EXECUTION_AND_ORIGIN_ACCOUNTING"):
        raise RuntimeError("four-model eager F+B origin ledger is not complete")
    abi_cells = triton_abi_cells()
    cells = []
    for model, prefix in MODELS.items():
        for shape in SHAPES:
            math_row = math["denominator_cells"][model][f"batch1_seq{shape}"]
            candidate = artifact(
                inventory_path(prefix, shape),
                "COMPLETE_GENERATED_SCHEDULE_AND_POINTER_DATAFLOW",
            )
            triton_path = triton_oracle_path(prefix, shape)
            triton = artifact(
                triton_path,
                triton_oracle_expected_status(triton_path),
            )
            triton_execution_gate = bool(triton["gate"])
            abi_row = abi_cells.get((prefix, shape))
            typed_reference = triton_path.name == "typed_triton_oracle.json.gz"
            abi_valid = (
                triton.get("gates", {}).get("typed_triton_pointer_abi_valid") is True
                if typed_reference else
                not abi_row or int(abi_row.get("invalid_reference_abi_campaign_rows", 0)) == 0
            )
            triton["execution_census_gate"] = triton_execution_gate
            triton["numeric_reference_gate"] = triton_execution_gate and abi_valid
            triton["abi_audit"] = (
                {
                    "status": "VALID_INDEPENDENT_RECOMPILED_FP32_POINTER_ABI",
                    "historical_invalid_audit_not_applied": True,
                }
                if typed_reference else abi_row
            )
            triton["gate"] = triton["numeric_reference_gate"]
            if not abi_valid:
                triton["status"] = "INVALID_REFERENCE_ABI"
            nontriton = artifact(
                (
                    RESULTS / "runtime_releases" / f"{prefix}_seq{shape}_r1"
                    / "nontriton_oracle.json.gz"
                    if (
                        RESULTS / "runtime_releases" / f"{prefix}_seq{shape}_r1"
                        / "nontriton_oracle.json.gz"
                    ).exists()
                    else RESULTS / f"{prefix}_seq{shape}_nontriton_fp32_oracle.json.gz"
                ),
                "COMPLETE_PRECISION_ONLY_RUNTIME_DENOMINATOR_ORACLE",
            )
            same_dtype = artifact(
                RESULTS / "runtime_releases" / f"{prefix}_seq{shape}_r1"
                / "same_dtype_oracle.json.gz",
                "COMPLETE_SAME_DTYPE_OPTIMIZATION_ORACLE",
            )
            default_aot_math = default_aot_evidence(prefix, shape)
            binding = candidate_binding(model, shape)
            analytic_complete = (
                math_row.get("analytic_fb_proof_units", 0)
                == math_row.get("primary_fb_proof_units", -1)
            )
            row = {
                "model": model, "shape": f"batch1_seq{shape}",
                "canonical_eager_fb_math": math_row,
                "default_aot_fb_math": default_aot_math,
                "candidate_inventory": candidate,
                "candidate_fb_binding": binding,
                "triton_precision_oracle": triton,
                "nontriton_precision_oracle": nontriton,
                "same_dtype_optimization_oracle": same_dtype,
                "gates": {
                    "execution_census": True,
                    "fb_origin_bound": True,
                    "canonical_eager_fb_analytically_proved": analytic_complete,
                    "default_aot_fb_analytically_proved": bool(default_aot_math["gate"]),
                    "candidate_inventory": bool(candidate["gate"]),
                    "candidate_fb_binding": bool(binding["gate"]),
                    "triton_execution_census": triton_execution_gate,
                    "triton_numeric_reference_valid": bool(triton["gate"]),
                    "nontriton_precision_reference_valid": bool(nontriton["gate"]),
                    "same_dtype_optimization_reference_valid": bool(same_dtype["gate"]),
                },
            }
            row["status"] = (
                "COMPLETE_ALL_DECLARED_GATES"
                if all(row["gates"].values())
                else "PENDING_FAIL_CLOSED"
            )
            cells.append(row)
    counts = {
        "declared_cells": len(cells),
        "canonical_eager_fb_math_closed": sum(
            row["gates"]["canonical_eager_fb_analytically_proved"]
            for row in cells
        ),
        "default_aot_fb_math_closed": sum(
            row["gates"]["default_aot_fb_analytically_proved"]
            for row in cells
        ),
        "fb_origin_bound": sum(row["gates"]["fb_origin_bound"] for row in cells),
        "candidate_inventories_closed": sum(row["candidate_inventory"]["gate"] for row in cells),
        "candidate_fb_bindings_closed": sum(row["candidate_fb_binding"]["gate"] for row in cells),
        "triton_execution_censuses_closed": sum(
            row["gates"]["triton_execution_census"] for row in cells
        ),
        "triton_precision_oracles_closed": sum(row["triton_precision_oracle"]["gate"] for row in cells),
        "nontriton_precision_oracles_closed": sum(row["nontriton_precision_oracle"]["gate"] for row in cells),
        "same_dtype_optimization_oracles_closed": sum(
            row["gates"]["same_dtype_optimization_reference_valid"] for row in cells
        ),
        "fully_closed_cells": sum(
            row["status"] == "COMPLETE_ALL_DECLARED_GATES"
            for row in cells
        ),
    }
    payload = {
        "schema": "kernel-analyzer-four-model-full-operator-status-v1",
        "status": "COMPLETE" if counts["fully_closed_cells"] == 12 else "PARTIAL_FAIL_CLOSED",
        "scope": {
            "models": list(MODELS), "shapes": [f"batch1_seq{x}" for x in SHAPES],
            "deduplication_across_models_shapes_or_families": False,
        },
        "counts": counts, "cells": cells,
        "claim_boundary": (
            "Completion requires each of 12 independent model-shape cells to close execution, "
            "actual F+B origins, concrete analytic F+B proofs, candidate-to-F+B bindings, valid "
            "typed Triton and non-Triton references, and a same-dtype optimization contrast. "
            "Formula registration, source inventory, or an invalidated replay cannot substitute "
            "for those gates. Bias-mechanism promotion remains a separate later stage."
        ),
    }
    payload["result_sha256"] = digest(payload)
    output = RESULTS / "four_model_full_operator_status.json"
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
    temporary.replace(output)
    print(json.dumps({"output": str(output.relative_to(ROOT)), **counts}, sort_keys=True))


if __name__ == "__main__":
    main()
