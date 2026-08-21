#!/usr/bin/env python3
"""Run the shared short screen on an existing effective-update geometry spool.

The spool is the runtime boundary used by the paired trajectory runners.  It
contains real tensors only temporarily; this command keeps projected paths and
provenance, never raw vectors.  Scalar-only trajectory JSON is intentionally
rejected rather than silently treated as a vector.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from kernel_analyzer.short_persistence import SharedShortPersistenceScreen


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    if path.suffix not in {".pt", ".pth"}:
        raise ValueError(f"{path}: expected a torch geometry spool (.pt/.pth)")
    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, dict) or "rows" not in value:
        raise ValueError(f"{path}: malformed geometry spool")
    return value


def _iter_tree_chunks(tree: Any, parameter_order: Iterable[str]) -> tuple[Iterable[np.ndarray], int]:
    """Stream a declared tensor tree in deterministic parameter order."""

    if not isinstance(tree, dict):
        raise ValueError("spool field must be a parameter->tensor mapping")
    arrays: list[np.ndarray] = []
    coordinate_count = 0
    for name in parameter_order:
        if name not in tree:
            raise ValueError(f"spool field is missing declared parameter {name}")
        value = tree[name]
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"spool field {name} is not a tensor")
        value = value.detach().to(dtype=torch.float64, device="cpu").reshape(-1)
        array = value.numpy()
        if not np.isfinite(array).all():
            raise ValueError(f"spool field {name} is nonfinite")
        arrays.append(array)
        coordinate_count += int(array.size)
    if not arrays or coordinate_count == 0:
        raise ValueError("spool field has zero declared coordinates")
    return iter(arrays), coordinate_count


def _rows_for_phase(rows: list[dict[str, Any]], phase: str, steps: int) -> list[dict[str, Any]]:
    selected = [row for row in rows if phase == "all" or row.get("phase") == phase]
    selected.sort(key=lambda row: (int(row.get("step", -1)), str(row.get("state_id", ""))))
    if len(selected) < steps:
        raise ValueError(f"phase {phase} has {len(selected)} rows, needs {steps}")
    chosen = selected[:steps]
    expected = list(range(1, steps + 1))
    observed = [int(row.get("step", -1)) for row in chosen]
    # Calibration/evaluation rows in existing runners use absolute step numbers
    # for evaluation.  Require strict ordering, but allow either 0/1-based or
    # the natural 17..32 evaluation numbering.
    if any(right <= left for left, right in zip(observed, observed[1:])):
        raise ValueError(f"phase {phase} steps are not strictly ordered: {observed}")
    return chosen


def screen_spools(
    paths: list[Path], *, field: str, phase: str, steps: int,
    projection_dim: int, projection_seed: int, null_draws: int,
) -> dict[str, Any]:
    screen = SharedShortPersistenceScreen(
        projection_dim=projection_dim,
        projection_seed=projection_seed,
        expected_steps=steps,
        null_draws=null_draws,
    )
    source_rows: list[dict[str, Any]] = []
    for path in paths:
        spool = _load(path)
        parameters = tuple(str(value) for value in spool.get("carrier_parameters", ()))
        if not parameters:
            raise ValueError(f"{path}: carrier_parameters are required")
        rows = _rows_for_phase(list(spool["rows"]), phase, steps)
        case_id = str(spool.get("case_id") or path.stem)
        screen_case = f"{case_id}::{phase}::{field}"
        for row in rows:
            if field not in row:
                raise ValueError(f"{path}: row is missing field {field}")
            chunks, coordinate_count = _iter_tree_chunks(row[field], parameters)
            screen.add_chunks(screen_case, chunks)
        source_rows.append({
            "path": str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path),
            "sha256": _sha256(path),
            "case_id": case_id,
            "parameter_order": list(parameters),
            "coordinate_count": coordinate_count,
            "state_ids": [str(row.get("state_id", row.get("step"))) for row in rows],
        })
    result = screen.finalize()
    result["status"] = "COMPLETE"
    result["input"] = {
        "kind": "KERNEL_ANALYZER_GEOMETRY_SPOOL",
        "field": field,
        "phase": phase,
        "sources": source_rows,
        "raw_vectors_retained": False,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spool", type=Path, action="append", required=True)
    parser.add_argument("--field", default="effective_update",
                        choices=("effective_update", "local", "feedback", "actual", "gradient_delta"))
    parser.add_argument("--phase", default="evaluation", choices=("calibration", "evaluation", "all"))
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--projection-dim", type=int, default=64)
    parser.add_argument("--projection-seed", type=int, default=20260822)
    parser.add_argument("--null-draws", type=int, default=2000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        payload = screen_spools(
            args.spool, field=args.field, phase=args.phase, steps=args.steps,
            projection_dim=args.projection_dim, projection_seed=args.projection_seed,
            null_draws=args.null_draws,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        payload = {
            "schema": "kernel-analyzer-shared-short-persistence-screen-v1",
            "status": "INVALID_INPUT",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "raw_vectors_retained": False,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        raise SystemExit(2) from exc
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "cases": len(payload["cases"]), "output": str(args.output)}))


if __name__ == "__main__":
    main()
