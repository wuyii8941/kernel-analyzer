#!/usr/bin/env python3
"""Verify that the evolving candidate matrix has no silent gaps or duplicates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "results" / "final"
STEPS = [0, 1, 2, 4, 8, 16, 32, 64]
SHAPES = [64, 128, 256]


def read(name: str) -> dict:
    return json.loads((FINAL / name).read_text())


def main() -> None:
    matrix = read("implementation_matrix.json")
    schedule = read("source_replay_schedule.json")
    replay = read("source_replay_matrix.json") if (FINAL / "source_replay_matrix.json").exists() else {}
    source_matrix = read("source_matrix_static.json")
    inductor = matrix["evolving_full_step_inductor"]
    observed_inductor = [
        (str(row["dtype"]), bool(row["tf32"]), int(row["seq_len"]))
        for row in inductor["measured_cells"]
    ]
    expected_inductor = [
        (dtype, tf32, seq_len)
        for dtype, tf32 in (("bf16", False), ("fp16", False), ("fp32", False), ("fp32", True))
        for seq_len in SHAPES
    ]
    source_rows = [
        (str(row["dtype"]), bool(row["tf32"]), int(row["seq_len"]), int(row["step"]))
        for row in schedule["rows"]
    ]
    expected_source = [
        (str(cell["dtype"]), bool(cell["tf32"]), int(cell["seq_len"]), step)
        for cell in source_matrix["cells"]
        for step in STEPS
    ]
    def set_audit(observed: list, expected: list) -> dict:
        observed_set, expected_set = set(observed), set(expected)
        return {
            "expected_count": len(expected),
            "observed_count": len(observed),
            "unique_observed_count": len(observed_set),
            "duplicate_rows": len(observed) != len(observed_set),
            "missing": [list(row) for row in sorted(expected_set - observed_set)],
            "unexpected": [list(row) for row in sorted(observed_set - expected_set)],
            "complete": observed_set == expected_set and len(observed) == len(observed_set),
        }
    output = {
        "schema": "kernel-analyzer-matrix-cross-product-audit-v1",
        "subject": "Qwen3-1.7B evolving checkpoint x real implementation matrix",
        "candidate_values_used_to_select_or_classify": False,
        "checkpoint_steps": STEPS,
        "shape_strata": SHAPES,
        "full_step_inductor": set_audit(observed_inductor, expected_inductor),
        "source_replay": set_audit(source_rows, expected_source),
        "source_static_cell_count": len(source_matrix["cells"]),
        "fused_configurations": matrix.get("fused_configurations", {}),
        "source_static_all_mapped": all(
            int(cell["mapped_invocations"]) == int(cell["runtime_invocations"])
            and int(cell["unresolved_invocations"]) == 0
            for cell in source_matrix["cells"]
        ),
        "numeric_source_replay": replay.get("numeric_replay", "PENDING_GPU_REMEASUREMENT"),
        "natural_bias_case_added": False,
        "property_claim": False,
        "boundary": "Cross-product completeness does not imply a bias verdict: the corrected real kernel is observed twice in every source row, but carrier-level causal adjudication remains separate.",
    }
    output["matrix_has_no_static_gap"] = (
        output["full_step_inductor"]["complete"]
        and output["source_replay"]["complete"]
        and output["source_static_all_mapped"]
        and all(value.get("status") == "COMPLETE" for value in output["fused_configurations"].values())
    )
    output["result_sha256"] = hashlib.sha256(json.dumps(output, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    path = FINAL / "matrix_cross_product_audit.json"
    path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(path), "matrix_has_no_static_gap": output["matrix_has_no_static_gap"]}, sort_keys=True))


if __name__ == "__main__":
    main()
