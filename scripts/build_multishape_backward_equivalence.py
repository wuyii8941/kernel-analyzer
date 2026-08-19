#!/usr/bin/env python3
"""Reduce the 4×3 backward denominator into shape-preserving semantic cells."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/property/bias_formation/hotspot_search/multishape_backward_hotspots.json"
OUTPUT = ROOT / "results/property/bias_formation/hotspot_search/multishape_backward_equivalence.json"


def stratum(row: dict[str, Any], maximum: int) -> str:
    values = row.get("layer_indices") or []
    if not values or maximum <= 0:
        return "GLOBAL_OR_UNLAYERED"
    ratio = (sum(values) / len(values)) / maximum
    return "EARLY" if ratio < 1 / 3 else "MIDDLE" if ratio < 2 / 3 else "LATE"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    source = json.loads(args.source.read_text())
    rows = [row for row in source["rows"] if row.get("semantic_region_executable")]
    maxima = {}
    for model in {row["model"] for row in rows}:
        for shape in {row["sequence_length"] for row in rows if row["model"] == model}:
            values = [index for row in rows if row["model"] == model
                      and row["sequence_length"] == shape for index in row.get("layer_indices", [])]
            maxima[(model, shape)] = max(values) if values else 0
    cells = collections.defaultdict(list)
    for row in rows:
        depth = stratum(row, maxima[(row["model"], row["sequence_length"])])
        key = (row["model"], row["sequence_length"], row["family"],
               row["implementation_kind"], tuple(row["semantic_operations"]), depth)
        cells[key].append(row)
    output_cells = []
    membership = []
    for index, (key, members) in enumerate(sorted(cells.items(), key=lambda item: str(item[0]))):
        members.sort(key=lambda row: str(row["task_id"]))
        layer_values = [sum(row["layer_indices"]) / len(row["layer_indices"])
                        for row in members if row.get("layer_indices")]
        target = sum(layer_values) / len(layer_values) if layer_values else 0.0
        representative = min(members, key=lambda row: (
            not bool(row["exact_endpoint_executable"]),
            abs(sum(row["layer_indices"]) / len(row["layer_indices"]) - target)
            if row.get("layer_indices") else 0.0,
            str(row["task_id"]),
        ))
        cell_id = f"multishape-backward-cell-{index:04d}"
        output_cells.append({
            "cell_id": cell_id, "model": key[0], "sequence_length": key[1],
            "family": key[2], "implementation_kind": key[3],
            "semantic_operations": list(key[4]), "depth_stratum": key[5],
            "member_count": len(members), "representative": representative,
            "capture_boundary": (
                "EXACT_AOT_ENDPOINT" if representative["exact_endpoint_executable"]
                else "COMPILER_BOUND_SEMANTIC_REGION"
            ),
        })
        membership.extend({
            "member_id": f'{row["model"]}:seq{row["sequence_length"]}:{row["task_id"]}',
            "cell_id": cell_id,
        } for row in members)
    counts = collections.Counter((row["model"], row["sequence_length"], row["family"])
                                 for row in output_cells)
    result = {
        "schema": "kernel-analyzer-multishape-backward-equivalence-v1",
        "status": "COMPLETE_DENOMINATOR_REDUCTION",
        "semantic_region_hotspot_denominator": len(rows),
        "exact_leaf_representative_count": sum(
            cell["capture_boundary"] == "EXACT_AOT_ENDPOINT" for cell in output_cells),
        "semantic_region_representative_count": sum(
            cell["capture_boundary"] == "COMPILER_BOUND_SEMANTIC_REGION" for cell in output_cells),
        "equivalence_cell_count": len(output_cells),
        "every_executable_hotspot_assigned_once": (
            len(membership) == len(rows)
            and len({row["member_id"] for row in membership}) == len(rows)
        ),
        "equivalence_key": ["model", "sequence_length", "family",
                            "implementation_kind", "semantic_operations", "depth_stratum"],
        "excluded_selection_inputs": ["T1", "T2", "T3", "T4", "SEUP", "ERROR_MAGNITUDE"],
        "counts": {f"{m}:seq{s}:{f}": n for (m, s, f), n in sorted(counts.items())},
        "cells": output_cells, "membership": membership,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "denominator": len(rows),
                      "cells": len(output_cells)}))


if __name__ == "__main__":
    main()
