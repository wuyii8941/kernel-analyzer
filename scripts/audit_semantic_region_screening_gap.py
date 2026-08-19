#!/usr/bin/env python3
"""Report backward F+B regions lost by leaf-endpoint-only screening."""

from __future__ import annotations

import collections
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/property/bias_formation/hotspot_search/multishape_backward_equivalence.json"
OUT = ROOT / "results/property/bias_formation/hotspot_search/semantic_region_gap.json"


def main() -> None:
    source = json.loads(SOURCE.read_text())
    region_cells = [c for c in source["cells"]
                    if c["capture_boundary"] == "COMPILER_BOUND_SEMANTIC_REGION"]
    counts = collections.Counter((c["model"], c["sequence_length"], c["family"])
                                 for c in region_cells)
    anchor = next(c for c in region_cells
                  if c["representative"]["model"] == "phi4"
                  and c["representative"]["sequence_length"] == 64
                  and c["representative"]["task_id"] == "backward:497:output_0")
    result = {
        "schema": "kernel-analyzer-semantic-region-screening-gap-v1",
        "status": "LEAF_ONLY_SCREEN_WAS_NOT_SENSITIVE_TO_KNOWN_POSITIVE",
        "complete_fb_unit_remains_required": True,
        "backward_is_never_tested_without_its_bound_forward": True,
        "old_leaf_only_cells": 724,
        "corrected_semantic_cells": source["equivalence_cell_count"],
        "compiler_bound_region_representatives": len(region_cells),
        "counts": {f"{m}:seq{s}:{f}": n for (m, s, f), n in sorted(counts.items())},
        "known_sensitivity_anchor": {
            "case": "phi4_seq64_lm_head_input_gradient_mm",
            "task_id": anchor["representative"]["task_id"],
            "cell_id": anchor["cell_id"],
            "capture_boundary": anchor["capture_boundary"],
            "internal_aot_node": "backward:graph0:backward_g0__mm_2",
            "old_screen_included": False,
            "corrected_denominator_includes": True,
        },
        "scientific_consequence": (
            "Existing negative leaf-screen results cover exact leaf endpoints only. They do not "
            "exclude bias formation in compiler-bound backward semantic regions, especially loss, "
            "normalization, and recurrent bottlenecks."
        ),
        "provenance_policy": (
            "Retain release/input/common-state identity only; do not hash transient vectors or "
            "duplicate derived reports."
        ),
    }
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(OUT), "region_cells": len(region_cells)}))


if __name__ == "__main__":
    main()
