#!/usr/bin/env python3
"""Build minimal 4-state plans for exact F+B cells not yet screened.

The four-model semantic denominator remains the authority.  This helper only
selects exact endpoint representatives whose stable (model, shape, task) key
is absent from both the static atlas and completed newly-bound screens.  It
does not promote a short screen to a scientific verdict.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re


def completed_keys(base: Path) -> set[tuple[str, int, str]]:
    atlas = json.loads((base / "backward_rescreen_atlas.json").read_text())
    keys = {
        (str(row["model"]), int(row["sequence_length"]), str(row["task_id"]))
        for row in atlas["rows"]
    }
    for path in (base / "multishape_screen").glob(
        "*/screening_gram.json"
    ):
        match = re.match(r"^(.+)_seq(\d+)(?:_(?:newly_bound|pending))?$", path.parent.name)
        if match is None:
            continue
        model = match.group(1)
        if model == "deepseek8":
            model = "deepseek8b"
        shape = int(match.group(2))
        payload = json.loads(path.read_text())
        keys.update((model, shape, str(case["task_id"]))
                    for case in payload.get("cases", []))
    return keys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path(
        "results/property/bias_formation/hotspot_search"
    ))
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    base = args.base
    output = args.output_dir or (base / "pending_screen_plans")
    output.mkdir(parents=True, exist_ok=True)
    done = completed_keys(base)
    equivalence = json.loads((base / "multishape_backward_equivalence.json").read_text())
    binding = json.loads((base / "multishape_backward_carriers.json").read_text())
    bindings = {str(row["cell_id"]): row for row in binding["cells"]}
    rows = []
    for cell in equivalence["cells"]:
        if cell["capture_boundary"] != "EXACT_AOT_ENDPOINT":
            continue
        rep = cell["representative"]
        key = (str(cell["model"]), int(cell["sequence_length"]), str(rep["task_id"]))
        if key in done:
            continue
        carrier = bindings.get(str(cell["cell_id"]), {}).get("nearest_carrier")
        if not carrier:
            continue
        rows.append({
            "case_id": cell["cell_id"],
            "task_id": rep["task_id"],
            "carrier": carrier["name"],
            "family": cell["family"],
            "depth_stratum": cell["depth_stratum"],
            "member_count": cell["member_count"],
        })
    groups = {}
    for row in rows:
        cell = next(c for c in equivalence["cells"] if c["cell_id"] == row["case_id"])
        key = (cell["model"], int(cell["sequence_length"]))
        groups.setdefault(key, []).append(row)
    outputs = []
    for (model, shape), cases in sorted(groups.items()):
        payload = {
            "schema": "kernel-analyzer-pending-backward-screen-plan-v1",
            "model": model, "sequence_length": shape,
            "selection_uses_candidate_values_or_historical_verdict": False,
            "cases": sorted(cases, key=lambda x: str(x["task_id"])),
        }
        target = output / f"{model}_seq{shape}.json"
        target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        outputs.append({"output": str(target), "cases": len(cases)})
    print(json.dumps({"pending_cases": len(rows), "plans": outputs}, sort_keys=True))


if __name__ == "__main__":
    main()
