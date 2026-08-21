#!/usr/bin/env python3
"""Freeze a mechanically enumerated, non-duplicate semantic-family pool.

This script is deliberately pre-measurement: it reads graph/closure metadata,
never candidate residuals, T4 labels, SEUP labels, or trajectory results.  A
row is only a candidate; dynamic parameter reach, finite residuals, and a
semantics-preserving orbit remain explicit gates for later experiments.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EQUIVALENCE = ROOT / "results/property/bias_formation/hotspot_search/multishape_backward_equivalence.json"
CLOSURE = ROOT / "results/property/bias_formation/hotspot_search/semantic_region_closure_coverage.json"
OUT = ROOT / "results/property/tcmp_allop_v1/semantic_family_heldout_pool_v1.json"

# These are deep-measurement representatives already used for development or
# prior consequence claims.  The list is committed in the generated artifact;
# adding a row here after reading outcomes would invalidate the freeze.
EXISTING_TASKS = {
    "backward:497:output_0", "backward:517:output_0",  # Phi/Qwen lm-head anchors
    "backward:663:output_0", "backward:676:output_0",  # Qwen/DeepSeek attention dV
    "backward:665:out_ptr0", "backward:666:in_out_ptr0",  # Qwen saved/softmax
    "backward:1880:in_out_ptr0", "backward:1401:out_ptr3",  # Gemma controls
    "backward:15107:out_ptr0",  # Mamba closure attempt, tracked separately
    "backward:1957:out_ptr0",  # DeepSeek normalization closure control
}


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> None:
    equivalence = json.loads(EQUIVALENCE.read_text())
    closure = json.loads(CLOSURE.read_text())
    closure_by_task = {str(row["task_id"]): row for row in closure["rows"]}

    # One deterministic representative per implementation pattern.  We retain
    # topology/depth in the key so distinct fused regions are not collapsed.
    grouped: dict[tuple[str, int, str, str, str, str], dict[str, Any]] = {}
    denominator = 0
    for cell in equivalence["cells"]:
        denominator += 1
        rep = cell["representative"]
        task_id = str(rep["task_id"])
        key = (
            str(cell["model"]),
            int(cell["sequence_length"]),
            str(cell["family"]),
            str(cell.get("implementation_kind", rep.get("implementation_kind", "UNKNOWN"))),
            str(rep.get("region_symbol", "UNKNOWN")),
            str(cell.get("depth_stratum", "UNKNOWN")),
        )
        candidate = {
            "cell_id": str(cell["cell_id"]),
            "task_id": task_id,
            "model": str(cell["model"]),
            "sequence_length": int(cell["sequence_length"]),
            "family": str(cell["family"]),
            "capture_boundary": str(cell.get("capture_boundary", "UNKNOWN")),
            "implementation_kind": str(cell.get("implementation_kind", rep.get("implementation_kind", "UNKNOWN"))),
            "region_symbol": str(rep.get("region_symbol", "UNKNOWN")),
            "depth_stratum": str(cell.get("depth_stratum", "UNKNOWN")),
            "member_count": int(cell.get("member_count", 1)),
            "exact_endpoint_executable": bool(rep.get("exact_endpoint_executable", False)),
            "semantic_region_executable": bool(rep.get("semantic_region_executable", False)),
            "source_internal_region": task_id if not rep.get("exact_endpoint_executable", False) else None,
            "existing_deep_measurement": task_id in EXISTING_TASKS,
            "selection_key": list(key),
        }
        if key not in grouped or (candidate["cell_id"], candidate["task_id"]) < (
            grouped[key]["cell_id"], grouped[key]["task_id"]
        ):
            grouped[key] = candidate

    rows = []
    for key in sorted(grouped):
        row = grouped[key]
        closure_row = closure_by_task.get(row["task_id"])
        if closure_row is not None:
            row["has_exact_downstream_closure"] = bool(closure_row["has_exact_closure"])
            row["screened_downstream_closure"] = bool(closure_row["at_least_one_closure_screened"])
            row["closure_task_ids"] = [str(x["closure_task_id"]) for x in closure_row["closures"]]
        else:
            row["has_exact_downstream_closure"] = row["exact_endpoint_executable"]
            row["screened_downstream_closure"] = False
            row["closure_task_ids"] = []
        row["pool_status"] = (
            "EXCLUDE_EXISTING_DEEP_MEASUREMENT"
            if row["existing_deep_measurement"]
            else "PRE_MEASUREMENT_CANDIDATE"
        )
        rows.append(row)

    # A fixed order plus seed is part of the artifact, even though the current
    # release retains all rows.  Future bounded sampling must use this order.
    seed = 20260821
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    for rank, index in enumerate(order):
        rows[index]["frozen_pool_rank"] = rank

    eligible = [r for r in rows if r["pool_status"] == "PRE_MEASUREMENT_CANDIDATE"]
    result = {
        "schema": "kernel-analyzer-semantic-family-heldout-pool-v1",
        "status": "FROZEN_METADATA_ONLY",
        "selection_seed": seed,
        "source_equivalence_sha256": digest(equivalence),
        "source_closure_sha256": digest(closure),
        "denominator_semantic_cells": denominator,
        "deduplicated_implementation_patterns": len(rows),
        "existing_deep_measurements_excluded": len(rows) - len(eligible),
        "pre_measurement_candidates": len(eligible),
        "eligibility_gates": [
            "exact F+B boundary or exact downstream semantic closure",
            "parameter reach verified by a two-state engineering run",
            "finite nonzero candidate-reference residual",
            "semantics-preserving orbit or an explicit abstention reason",
            "predictor frozen before any live-weight consequence",
        ],
        "claim_boundary": (
            "This pool is an exhaustive metadata-derived candidate list, not a bias verdict. "
            "NOT_APPLICABLE, UNRESOLVED, and centered controls remain in the denominator."
        ),
        "rows": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(OUT),
        "denominator": denominator,
        "patterns": len(rows),
        "candidates": len(eligible),
    }))


if __name__ == "__main__":
    main()
