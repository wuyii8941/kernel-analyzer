#!/usr/bin/env python3
"""Build fail-closed concrete F+B proof tasks for every pending candidate."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
COVERAGE = ROOT / "results/coverage"


def load(path: Path) -> dict[str, Any]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as handle:
            return json.load(handle)
    return json.loads(path.read_text())


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


def derivation(function: str) -> dict[str, Any]:
    if function.endswith(".mm"):
        row = {"forward": "Y=A@B", "vjp": "dA=q@B^T; dB=A^T@q",
               "saved": ["A", "B"], "cotangent": "q=dL/dY"}
    elif function.endswith(".bmm"):
        row = {"forward": "Y_b=A_b@B_b", "vjp": "dA_b=q_b@B_b^T; dB_b=A_b^T@q_b",
               "saved": ["A", "B"], "cotangent": "q=dL/dY"}
    elif function == "torch.ops.aten.convolution_backward.default":
        row = {"forward": "Y=convolution(X,W,b; stride,padding,dilation,groups)",
               "vjp": "(dX,dW,db)=convolution_backward(q,X,W,geometry,output_mask)",
               "saved": ["X", "W", "geometry", "output_mask"], "cotangent": "q=dL/dY"}
    else:
        row = {"forward": "UNRESOLVED", "vjp": "UNRESOLVED", "saved": [], "cotangent": "UNRESOLVED"}
    row["function"] = function
    row["analytic_derivation_sha256"] = digest(row)
    return row


def main() -> None:
    queue = load(COVERAGE / "bias_candidate_queue.json")
    with gzip.open(COVERAGE / "fb_proof_unit_ledger.json.gz", "rt") as handle:
        ledger = json.load(handle)
    qwen_region_units: dict[str, list[str]] = {}
    for unit in ledger["units"]:
        if unit["model"] != "qwen3_1p7b" or unit["denominator_role"] != "PRIMARY_FB_PROOF":
            continue
        cell = unit.get("candidate_cells", {}).get("bf16_inductor_full_step", {})
        for region in cell.get("candidate_region_ids", {}).get("ids", ()) or ():
            qwen_region_units.setdefault(str(region), []).append(unit["unit_id"])
    bridges = {}
    for seq in (128, 256):
        bridge = load(COVERAGE / f"qwen_seq{seq}_candidate_fb_bridge.json.gz")
        bridges[seq] = {str(row["current_region_id"]): row for row in bridge["rows"]}

    rows = []
    for candidate in queue["candidates"]:
        if candidate["claim"] != "PENDING_EXHAUSTIVE_FULL_COORDINATE_AND_FB_BINDING":
            continue
        call = candidate["exact_generated_call"]
        regions = [str(call["region_id"])]
        method = "UNRESOLVED_NO_CANDIDATE_AOT_EAGER_BRIDGE"
        fingerprint = None
        if candidate["architecture"] == "qwen3_1p7b":
            seq = int(candidate["sequence_length"])
            if seq in bridges:
                bridge_row = bridges[seq].get(str(call["region_id"]))
                if bridge_row is not None:
                    regions = list(bridge_row["source_seq64_region_ids"])
                    method = bridge_row["method"]
                    fingerprint = bridge_row["proof_binding_fingerprint"]
            else:
                method = "EXACT_SEQ64_RUNTIME_REGION_TO_FB_LEDGER"
        unit_ids = sorted({unit for region in regions for unit in qwen_region_units.get(region, [])})
        exact_binding = bool(unit_ids) and candidate["architecture"] == "qwen3_1p7b"
        math = derivation(str(call["function"]))
        row = {
            "candidate_id": candidate["candidate_id"],
            "architecture": candidate["architecture"],
            "sequence_length": candidate["sequence_length"],
            "generated_region_id": call["region_id"],
            "generated_source_line_sha256": call["source_line_sha256"],
            "analytic_derivation": math,
            "candidate_to_fb_binding": {
                "status": "EXACT" if exact_binding else "UNRESOLVED",
                "method": method, "proof_binding_fingerprint": fingerprint,
                "source_region_ids": regions, "fb_unit_ids": unit_ids,
            },
            "concrete_program_proof": None,
            "gates": {
                "analytic_formula_registered": math["forward"] != "UNRESOLVED",
                "candidate_to_fb_binding_exact": exact_binding,
                "saved_tensor_origins_exact": False,
                "cotangent_edge_exact": False,
                "backward_program_matches_analytic_vjp": False,
                "non_tensor_arguments_exact": False,
                "output_edges_exact": False,
                "concrete_fb_proof_complete": False,
            },
            "status": "PENDING_CONCRETE_FB_PROOF",
        }
        row["row_sha256"] = digest(row)
        rows.append(row)
    payload = {
        "schema": "kernel-analyzer-candidate-fb-proof-tasks-v1",
        "status": "PENDING_CONCRETE_FB_PROOFS",
        "queue_sha256": queue["result_sha256"], "task_count": len(rows),
        "counts": {
            "exact_candidate_to_fb_binding": sum(row["gates"]["candidate_to_fb_binding_exact"] for row in rows),
            "formula_registered": sum(row["gates"]["analytic_formula_registered"] for row in rows),
            "concrete_fb_proof_complete": 0,
        },
        "rows": rows,
        "claim_boundary": "A formula or generated-region bridge is not a concrete invocation F+B proof.",
    }
    payload["result_sha256"] = digest(payload)
    output = COVERAGE / "candidate_fb_proof_tasks.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output.relative_to(ROOT)), **payload["counts"]}, sort_keys=True))


if __name__ == "__main__":
    main()
