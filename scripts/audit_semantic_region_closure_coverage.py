#!/usr/bin/env python3
"""Audit whether unresolved internal ports were screened at exact downstream closures."""

from __future__ import annotations

import glob
import gzip
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/property/bias_formation/hotspot_search"
OUT = BASE / "semantic_region_closure_coverage.json"


def load(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        return json.load(handle)


def main() -> None:
    equivalence = load(BASE / "multishape_backward_equivalence.json")
    cells = {row["cell_id"]: row for row in equivalence["cells"]}
    membership = {row["member_id"]: row["cell_id"] for row in equivalence["membership"]}
    atlas_path = BASE / "backward_rescreen_atlas.json"
    atlas = load(atlas_path) if atlas_path.exists() else {"rows": []}
    screened = {
        (str(row["model"]), int(row["sequence_length"]), str(row["task_id"]))
        for row in atlas["rows"]
    }
    rows = []
    for plan_path_string in sorted(glob.glob(str(BASE / "semantic_region_capture_plans/*.json"))):
        plan = load(Path(plan_path_string))
        model = str(plan["model"]); sequence_length = int(plan["sequence_length"])
        release = ROOT / f"results/coverage/runtime_releases/{model}_seq{sequence_length}_r1"
        tasks = {str(row["task_id"]): row for row in load(release / "same_dtype_tasks.json.gz")["rows"]}
        for blocked in plan["blocked"]:
            task = tasks[str(blocked["task_id"])]
            closures = []
            for closure_task in task.get("closed_by_semantic_endpoint_tasks", []):
                member_id = f"{model}:seq{sequence_length}:{closure_task}"
                cell_id = membership.get(member_id)
                representative = cells[cell_id]["representative"]["task_id"] if cell_id else None
                closures.append({
                    "closure_task_id": closure_task,
                    "equivalence_cell_id": cell_id,
                    "representative_task_id": representative,
                    "representative_screened": (
                        representative is not None
                        and (model, sequence_length, representative) in screened
                    ),
                })
            rows.append({
                "model": model,
                "sequence_length": sequence_length,
                "case_id": blocked["case_id"],
                "task_id": blocked["task_id"],
                "family": blocked["family"],
                "closures": closures,
                "has_exact_closure": bool(closures),
                "at_least_one_closure_screened": any(
                    row["representative_screened"] for row in closures
                ),
            })
    result = {
        "schema": "kernel-analyzer-semantic-region-closure-coverage-v1",
        "status": "PARTIAL",
        "blocked_internal_region_count": len(rows),
        "with_exact_downstream_closure": sum(row["has_exact_closure"] for row in rows),
        "with_screened_downstream_closure": sum(
            row["at_least_one_closure_screened"] for row in rows
        ),
        "claim_boundary": (
            "A screened downstream closure can detect whether an internal implementation "
            "difference survives into the closed semantic output. It does not localize the "
            "arithmetic source inside a fused region."
        ),
        "rows": rows,
    }
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(OUT), "blocked": len(rows),
        "closed": result["with_exact_downstream_closure"],
        "screened": result["with_screened_downstream_closure"],
    }))


if __name__ == "__main__":
    main()
