#!/usr/bin/env python3
"""Cover every backward hotspot through semantic-signature/layer representatives.

This is a denominator-preserving reduction, not candidate selection by old
numerical verdict.  Every executable hotspot is assigned to exactly one
equivalence cell.  Each cell retains representatives at all observed depth
strata so repeated transformer layers do not dominate GPU work while early,
middle, and late training paths remain represented.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/property/bias_formation/hotspot_search/cross_model_backward_hotspots.json"
OUTPUT = ROOT / "results/property/bias_formation/hotspot_search/backward_equivalence_rescreen.json"


def depth_stratum(row: dict[str, Any], maximum: int | None) -> str:
    indices = row.get("layer_indices") or []
    if not indices or maximum is None or maximum <= 0:
        return "GLOBAL_OR_UNLAYERED"
    position = sum(indices) / len(indices)
    fraction = position / maximum
    if fraction < 1 / 3:
        return "EARLY"
    if fraction < 2 / 3:
        return "MIDDLE"
    return "LATE"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    source = json.loads(args.source.read_text())
    executable = [row for row in source["rows"] if row["exact_endpoint_executable"]]
    maxima: dict[str, int] = {}
    for model in source["models"]:
        values = [index for row in executable if row["model"] == model
                  for index in row.get("layer_indices", [])]
        maxima[model] = max(values) if values else 0

    cells: dict[tuple[Any, ...], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in executable:
        stratum = depth_stratum(row, maxima.get(row["model"]))
        key = (
            row["model"], row["family"], row["implementation_kind"],
            tuple(row["semantic_operations"]), stratum,
        )
        cells[key].append(row)

    representatives = []
    membership = []
    for cell_index, (key, rows) in enumerate(sorted(cells.items(), key=lambda item: str(item[0]))):
        rows.sort(key=lambda row: str(row["task_id"]))
        # Prefer a row whose layer is closest to the centre of its stratum;
        # lexical task ID breaks ties deterministically without using values.
        layer_values = [sum(row.get("layer_indices", [])) / len(row["layer_indices"])
                        for row in rows if row.get("layer_indices")]
        target = sum(layer_values) / len(layer_values) if layer_values else 0.0
        representative = min(
            rows,
            key=lambda row: (
                abs((sum(row.get("layer_indices", [])) / len(row["layer_indices"])) - target)
                if row.get("layer_indices") else 0.0,
                str(row["task_id"]),
            ),
        )
        cell_id = f"backward-cell-{cell_index:04d}"
        representatives.append({
            "cell_id": cell_id,
            "model": key[0], "family": key[1], "implementation_kind": key[2],
            "semantic_operations": list(key[3]), "depth_stratum": key[4],
            "member_count": len(rows), "representative": representative,
            "selection_uses_error_or_historical_verdict": False,
        })
        membership.extend({"task_id": row["task_id"], "cell_id": cell_id} for row in rows)

    counts = collections.Counter((row["model"], row["family"])
                                 for row in representatives)
    result = {
        "schema": "kernel-analyzer-backward-equivalence-rescreen-v1",
        "status": "REPRESENTATIVE_DENOMINATOR_READY",
        "source_hotspot_count": len(source["rows"]),
        "executable_hotspot_denominator": len(executable),
        "equivalence_cell_count": len(representatives),
        "every_executable_hotspot_assigned_once": (
            len(membership) == len(executable)
            and len({row["task_id"] for row in membership}) == len(executable)
        ),
        "equivalence_key": [
            "model", "semantic_bottleneck_family", "implementation_kind",
            "AOT_semantic_operations", "layer_depth_stratum",
        ],
        "excluded_selection_inputs": ["T1", "T2", "T3", "T4", "SEUP", "ERROR_MAGNITUDE"],
        "representative_counts": {
            f"{model}:{family}": count for (model, family), count in sorted(counts.items())
        },
        "representatives": representatives,
        "membership": membership,
        "next_step": (
            "Bind exact AOT-reachable parameter carriers, run a two-state engineering reach "
            "screen, then run 16+16 formation only for nonzero carrier effects."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "denominator": len(executable), "cells": len(representatives),
        "counts": result["representative_counts"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
