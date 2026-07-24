#!/usr/bin/env python
"""Evaluate a predeclared reference-update-aligned discrepancy without loading a flat model vector."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "forkcert.update-aligned-shift.v0.1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--state",
        action="append",
        nargs=3,
        metavar=("STATE_ID", "REFERENCE_SAFETENSORS", "CANDIDATE_SAFETENSORS"),
        required=True,
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--chunk-elements", type=int, default=1_048_576)
    return parser.parse_args()


def finite(value: float, label: str) -> float:
    if not math.isfinite(value):
        raise ValueError(f"non-finite {label}: {value}")
    return value


def evaluate_pair(
    state_id: str,
    reference_path: str,
    candidate_path: str,
    chunk_elements: int,
) -> dict[str, Any]:
    if chunk_elements <= 0:
        raise ValueError("chunk-elements must be positive")

    import torch
    from safetensors import safe_open

    reference_file = Path(reference_path).resolve()
    candidate_file = Path(candidate_path).resolve()
    if not reference_file.is_file() or not candidate_file.is_file():
        raise FileNotFoundError(f"missing vector artifact for {state_id}")

    reference = safe_open(reference_file, framework="pt", device="cpu")
    candidate = safe_open(candidate_file, framework="pt", device="cpu")
    reference_keys = sorted(reference.keys())
    candidate_keys = sorted(candidate.keys())
    if not reference_keys or reference_keys != candidate_keys:
        raise ValueError(f"parameter key mismatch for {state_id}")

    reference_square = 0.0
    candidate_square = 0.0
    discrepancy_square = 0.0
    discrepancy_dot_reference = 0.0
    coordinates = 0
    per_parameter: list[dict[str, Any]] = []

    for name in reference_keys:
        reference_tensor = reference.get_tensor(name)
        candidate_tensor = candidate.get_tensor(name)
        if reference_tensor.shape != candidate_tensor.shape:
            raise ValueError(f"shape mismatch for {state_id}/{name}")
        reference_flat = reference_tensor.reshape(-1)
        candidate_flat = candidate_tensor.reshape(-1)
        current_reference_square = 0.0
        current_discrepancy_square = 0.0
        current_dot = 0.0
        for start in range(0, reference_flat.numel(), chunk_elements):
            stop = min(start + chunk_elements, reference_flat.numel())
            ref = reference_flat[start:stop].to(dtype=torch.float64)
            cand = candidate_flat[start:stop].to(dtype=torch.float64)
            if not bool(torch.isfinite(ref).all()) or not bool(torch.isfinite(cand).all()):
                raise ValueError(f"non-finite vector value for {state_id}/{name}")
            discrepancy = cand - ref
            current_reference_square += float(torch.dot(ref, ref).item())
            candidate_square += float(torch.dot(cand, cand).item())
            current_discrepancy_square += float(torch.dot(discrepancy, discrepancy).item())
            current_dot += float(torch.dot(discrepancy, ref).item())

        reference_square += current_reference_square
        discrepancy_square += current_discrepancy_square
        discrepancy_dot_reference += current_dot
        coordinates += reference_flat.numel()
        per_parameter.append(
            {
                "name": name,
                "coordinates": reference_flat.numel(),
                "reference_l2": math.sqrt(current_reference_square),
                "discrepancy_l2": math.sqrt(current_discrepancy_square),
                "aligned_shift": current_dot / current_reference_square
                if current_reference_square
                else None,
            }
        )

    if reference_square == 0.0:
        status = "UNDEFINED_ZERO_REFERENCE_UPDATE"
        aligned_shift = None
        parallel_relative_l2 = None
        orthogonal_relative_l2 = None
        cosine = None
    else:
        status = "VALID"
        aligned_shift = discrepancy_dot_reference / reference_square
        parallel_square = discrepancy_dot_reference**2 / reference_square
        orthogonal_square = max(0.0, discrepancy_square - parallel_square)
        parallel_relative_l2 = math.sqrt(parallel_square / reference_square)
        orthogonal_relative_l2 = math.sqrt(orthogonal_square / reference_square)
        cosine_denominator = math.sqrt(discrepancy_square * reference_square)
        cosine = discrepancy_dot_reference / cosine_denominator if cosine_denominator else None

    return {
        "state_id": state_id,
        "status": status,
        "reference_path": str(reference_file),
        "candidate_path": str(candidate_file),
        "parameter_keys": len(reference_keys),
        "coordinates": coordinates,
        "reference_update_l2": finite(math.sqrt(reference_square), "reference norm"),
        "candidate_update_l2": finite(math.sqrt(candidate_square), "candidate norm"),
        "discrepancy_l2": finite(math.sqrt(discrepancy_square), "discrepancy norm"),
        "relative_discrepancy_l2": finite(
            math.sqrt(discrepancy_square / reference_square), "relative discrepancy"
        )
        if reference_square
        else None,
        "aligned_shift": finite(aligned_shift, "aligned shift") if aligned_shift is not None else None,
        "parallel_relative_l2": finite(parallel_relative_l2, "parallel relative norm")
        if parallel_relative_l2 is not None
        else None,
        "orthogonal_relative_l2": finite(orthogonal_relative_l2, "orthogonal relative norm")
        if orthogonal_relative_l2 is not None
        else None,
        "discrepancy_reference_cosine": finite(cosine, "cosine") if cosine is not None else None,
        "per_parameter": per_parameter,
    }


def main() -> None:
    args = parse_args()
    state_ids = [item[0] for item in args.state]
    if len(state_ids) != len(set(state_ids)):
        raise ValueError("state ids must be unique")
    records = [evaluate_pair(*item, args.chunk_elements) for item in args.state]
    valid_shifts = [item["aligned_shift"] for item in records if item["aligned_shift"] is not None]
    output = {
        "schema_version": SCHEMA_VERSION,
        "scope": "retrospective selected-state endpoint sanity check; no population inference",
        "estimand": "dot(candidate_update-reference_update, reference_update) / ||reference_update||^2",
        "records": records,
        "descriptive_selected_state_summary": {
            "states": len(records),
            "valid_states": len(valid_shifts),
            "aligned_shift_mean": sum(valid_shifts) / len(valid_shifts) if valid_shifts else None,
            "aligned_shift_min": min(valid_shifts) if valid_shifts else None,
            "aligned_shift_max": max(valid_shifts) if valid_shifts else None,
            "population_inference": "NOT_ESTIMATED_SELECTED_STATES",
        },
        "nonclaims": [
            "selected states do not estimate a target training-state distribution",
            "aligned shift is baseline-relative, not correctness or harm",
            "one-step shift does not establish long-run contribution",
        ],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()

