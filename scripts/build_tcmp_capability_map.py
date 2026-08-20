#!/usr/bin/env python3
"""Classify every proved AOT F+B unit by predeclared TCMP capability."""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any


ORBIT_BY_PROOF_KIND = {
    "TWO_DIMENSIONAL_LINEAR_MM_ADJOINT": "JOINT_CONTRACTION_AXIS_PERMUTATION",
    "STANDARD_DECOMPOSED_LINEAR_MM_ADJOINT": "JOINT_CONTRACTION_AXIS_PERMUTATION",
    "STANDARD_DECOMPOSED_BATCHED_MATMUL_ADJOINT": "JOINT_CONTRACTION_AXIS_PERMUTATION",
    "MEAN_ADJOINT": "REDUCTION_AXIS_PERMUTATION",
    "STANDARD_DECOMPOSED_FP32_SOFTMAX_ADJOINT": "ROW_CONSTANT_SHIFT_AND_AXIS_PERMUTATION",
    "STANDARD_DECOMPOSED_CROSS_ENTROPY_MEAN_ADJOINT": "JOINT_CLASS_PERMUTATION",
}


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def load(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell-id", required=True)
    parser.add_argument("--aot-math", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    source = load(args.aot_math)
    if source["status"] != "COMPLETE_AOT_FORWARD_BACKWARD_DERIVATION":
        raise RuntimeError("capability classification requires complete F+B derivation")

    rows = []
    for unit in source["units"]:
        proof = unit["composite_vjp_proof"]
        if not proof or not proof.get("passed"):
            raise RuntimeError("incomplete unit entered capability classification")
        kind = str(proof["proof_kind"])
        if kind == "NO_REQUESTED_TRAINABLE_INPUT_VJP":
            capability = "NO_PARAMETER_REACH"
            orbit = None
            disposition = "NO_PARAMETER_REACH"
        elif kind in ORBIT_BY_PROOF_KIND:
            capability = "TCMP_ORBIT_READY"
            orbit = ORBIT_BY_PROOF_KIND[kind]
            disposition = "PENDING_SCREEN"
        else:
            capability = "EXACT_REPAIR_ONLY"
            orbit = None
            disposition = "PENDING_EXACT_REPAIR_SCREEN"
        row = {
            "unit_id": unit["unit_id"],
            "proof_kind": kind,
            "forward_program": unit["forward_program"],
            "actual_backward_program": unit["actual_backward_program"],
            "capability": capability,
            "semantic_orbit": orbit,
            "disposition": disposition,
            "proof_sha256": digest(proof),
        }
        row["row_sha256"] = digest(row)
        rows.append(row)
    counts = Counter(row["capability"] for row in rows)
    payload = {
        "schema": "kernel-analyzer-tcmp-capability-map-v1",
        "status": "COMPLETE_PROVED_UNIT_CAPABILITY_MAP",
        "cell_id": args.cell_id,
        "aot_math_result_sha256": source.get(
            "result_sha256", source["ledger_sha256"]
        ),
        "counts": dict(sorted(counts.items())),
        "rows": rows,
        "claim_boundary": (
            "Capability is assigned from the independently proved concrete F+B program. "
            "It is not a numerical verdict. Units without a declared semantics-preserving "
            "orbit remain exact-repair-only and cannot falsify TCMP."
        ),
    }
    payload["result_sha256"] = digest(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.output, "wt", encoding="utf-8", compresslevel=6) as handle:
        json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
    print(json.dumps({"output": str(args.output), "counts": payload["counts"]}, sort_keys=True))


if __name__ == "__main__":
    main()
