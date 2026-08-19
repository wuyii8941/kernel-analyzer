#!/usr/bin/env python3
"""Audit the four-model F+B finding denominator and explicit screen gaps."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/property/bias_formation/hotspot_search"
OUT_JSON = BASE / "bias_finding_coverage_audit.json"
OUT_MD = BASE / "bias_finding_coverage_audit.md"


def main() -> None:
    equivalence = json.loads((BASE / "multishape_backward_equivalence.json").read_text())
    binding = json.loads((BASE / "multishape_backward_carriers.json").read_text())
    atlas = json.loads((BASE / "backward_rescreen_atlas.json").read_text())
    screened = {
        (str(row["model"]), int(row["sequence_length"]), str(row["task_id"]))
        for row in atlas["rows"]
    }
    new_screened = set()
    # Direct leaf screens and exact semantic-region downstream screens both
    # represent the same F+B unit.  Keep their task coordinates in one set so
    # a later semantic capture cannot remain falsely pending merely because it
    # was not a leaf endpoint.
    screen_paths = list((BASE / "multishape_screen").glob("*/screening_gram.json"))
    screen_paths += list((BASE / "semantic_region_screen").glob("*/screening_gram.json"))
    for path in screen_paths:
        model, length = path.parent.name.split("_seq", 1)[0], None
        # Directory names include the original mamba_seq{N} screens and the
        # supplemental *_newly_bound/*_pending screens.
        import re
        match = re.match(r"^(.+)_seq(\d+)(?:_(?:newly_bound|pending))?$", path.parent.name)
        if match is None:
            continue
        model, length = match.group(1), int(match.group(2))
        payload = json.loads(path.read_text())
        new_screened.update((model, length, str(case["task_id"])) for case in payload["cases"])

    # Formal semantic-region formation captures are authoritative for their
    # exact task coordinates.  They are intentionally added after the
    # screening paths; the set deduplicates the two records.
    import re
    for path in (BASE / "semantic_region_formation").glob("*_seq*/**/*.json"):
        match = re.match(r"^(qwen|phi4|mamba|deepseek8b)_seq(\d+)", path.parent.name)
        if match is None:
            continue
        payload = json.loads(path.read_text())
        task_id = payload.get("binding", {}).get("task_id")
        if task_id is not None:
            new_screened.add((match.group(1), int(match.group(2)), str(task_id)))
    screened |= new_screened

    closure = json.loads((BASE / "semantic_region_closure_coverage.json").read_text())
    closure_screened = set()
    membership = {
        str(row["member_id"]): str(row["cell_id"])
        for row in equivalence["membership"]
    }
    for row in closure["rows"]:
        if not row["at_least_one_closure_screened"]:
            continue
        closure_screened.add(str(row["case_id"]))

    bound_by_cell = {str(row["cell_id"]): row for row in binding["cells"]}
    rows = []
    for cell in equivalence["cells"]:
        model = str(cell["model"]); length = int(cell["sequence_length"])
        task = str(cell["representative"]["task_id"])
        key = (model, length, task)
        bound = bound_by_cell[str(cell["cell_id"])]
        if key in screened:
            status = "SCREENED"
        elif str(cell["cell_id"]) in closure_screened:
            status = "SCREENED_VIA_EXACT_DOWNSTREAM_CLOSURE"
        elif bound.get("nearest_carrier") is None:
            status = "BOUNDARY_CARRIER_PENDING"
        elif cell["capture_boundary"] == "COMPILER_BOUND_SEMANTIC_REGION":
            status = "SEMANTIC_REGION_PENDING"
        else:
            status = "EXACT_ENDPOINT_PENDING"
        rows.append({
            "cell_id": str(cell["cell_id"]), "model": model,
            "sequence_length": length, "task_id": task,
            "family": cell["family"], "status": status,
            "carrier": None if bound.get("nearest_carrier") is None else bound["nearest_carrier"]["name"],
        })

    counts = Counter(row["status"] for row in rows)
    result = {
        "schema": "kernel-analyzer-bias-finding-coverage-audit-v1",
        "status": "IN_PROGRESS",
        "complete_forward_backward_unit": True,
        "denominator_cells": len(rows),
        "counts": dict(sorted(counts.items())),
        "screened_task_coordinates": len(screened),
        "closure_rows_screened": len(closure_screened),
        "rows": rows,
        "claim_boundary": (
            "A screened representative or exact downstream closure is evidence for its "
            "equivalence cell. It does not claim that a downstream closure localizes the "
            "arithmetic source inside a fused region."
        ),
    }
    OUT_JSON.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    lines = [
        "# Four-model F+B bias-finding coverage audit", "",
        "The denominator is the compiler-bound semantic equivalence inventory; every",
        "cell remains in the denominator even when its carrier or closure is unresolved.", "",
        f"- F+B semantic cells: **{len(rows)}**.",
        f"- Screened task coordinates (including newly bound screens): **{len(screened)}**.",
        "",
        "| status | cells |", "|---|---:|",
    ]
    for status, count in sorted(counts.items()):
        lines.append(f"| {status} | {count} |")
    lines += [
        "", "## Interpretation", "",
        "The deepest module-stack binding recovered Mamba `dt_proj.bias` and",
        "`conv1d.bias` paths that were previously mislabeled ambiguous. Short-screen",
        "ratios never become cases without disjoint confirmation. Current formal case",
        "count is recorded in the mechanism candidate map, not inferred from this audit.", "",
    ]
    OUT_MD.write_text("\n".join(lines))
    print(json.dumps({"output": str(OUT_JSON), "cells": len(rows), "counts": dict(counts)}))


if __name__ == "__main__":
    main()
