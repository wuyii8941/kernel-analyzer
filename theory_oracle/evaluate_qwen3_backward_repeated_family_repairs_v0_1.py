#!/usr/bin/env python
"""Audit and evaluate early/middle/late repairs of one repeated backward family."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Any

from theory_oracle.evaluate_qwen3_backward_singleton_repairs_v0_1 import (
    sha256_file,
    vector_metrics,
)


def pairwise_effect_metrics(
    candidate_path: Path, left_path: Path, right_path: Path
) -> dict[str, Any]:
    import torch
    from safetensors import safe_open

    square_left = 0.0
    square_right = 0.0
    square_difference = 0.0
    dot = 0.0
    coordinates = 0
    with safe_open(candidate_path, framework="pt", device="cpu") as candidate, safe_open(
        left_path, framework="pt", device="cpu"
    ) as left, safe_open(right_path, framework="pt", device="cpu") as right:
        keys = sorted(candidate.keys())
        if keys != sorted(left.keys()) or keys != sorted(right.keys()):
            raise ValueError("vector tensor-key sets differ")
        for key in keys:
            c = candidate.get_tensor(key).float()
            l = left.get_tensor(key).float()
            r = right.get_tensor(key).float()
            if c.shape != l.shape or c.shape != r.shape:
                raise ValueError(f"shape mismatch for {key}")
            le = l - c
            re = r - c
            difference = le - re
            square_left += float(torch.sum(le * le, dtype=torch.float64).item())
            square_right += float(torch.sum(re * re, dtype=torch.float64).item())
            square_difference += float(
                torch.sum(difference * difference, dtype=torch.float64).item()
            )
            dot += float(torch.sum(le * re, dtype=torch.float64).item())
            coordinates += int(c.numel())
            del c, l, r, le, re, difference
    left_norm = math.sqrt(square_left)
    right_norm = math.sqrt(square_right)
    denominator = left_norm * right_norm
    return {
        "coordinates": coordinates,
        "left_effect_l2": left_norm,
        "right_effect_l2": right_norm,
        "effect_difference_l2": math.sqrt(square_difference),
        "effect_cosine": dot / denominator if denominator else None,
        "either_effect_exactly_null": square_left == 0.0 or square_right == 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest["status"] != "FROZEN_PRE_EXECUTION":
        raise ValueError("manifest is not frozen")
    artifact_gates = {
        name: sha256_file(Path(row["path"]).resolve()) == row["sha256"]
        for name, row in manifest["artifacts"].items()
    }

    eager_root = Path(
        "results/training_step_oracle/qwen3_grpo_natural_transition_v0_2/eager_1"
    ).resolve()
    candidate_root = Path(
        "results/training_step_oracle/qwen3_grpo_natural_transition_v0_2/compiled_1"
    ).resolve()
    positions = manifest["selection"]["positions_zero_based"]
    directories = {
        name: Path(manifest["output_directories"][name]).resolve() for name in positions
    }
    results: dict[str, Any] = {}
    for name, directory in directories.items():
        result = json.loads((directory / "result.json").read_text())
        repair = result.get("backward_repeated_family_repair", {})
        result_gates = {
            "repair_status_valid": repair.get("status")
            == "VALID_BACKWARD_REPEATED_FAMILY_REPAIR",
            "all_embedded_repair_gates_true": bool(repair.get("gates"))
            and all(repair["gates"].values()),
            "base_transition_valid": result.get("valid") is True
            and result.get("verdict") == "VALID",
            "preselected_call_exact": repair.get("selected_call_index_zero_based")
            == positions[name],
            "vector_artifacts_present": all(
                (directory / filename).is_file()
                for filename in (
                    "clipped_gradients.safetensors",
                    "parameter_updates.safetensors",
                )
            ),
        }
        if not all(result_gates.values()):
            raise ValueError(f"invalid repeated-family repair {name}: {result_gates}")
        profiles = {}
        for endpoint, filename in (
            ("clipped_gradient", "clipped_gradients.safetensors"),
            ("parameter_update", "parameter_updates.safetensors"),
        ):
            profiles[endpoint] = vector_metrics(
                eager_root / filename, candidate_root / filename, directory / filename
            )
        results[name] = {
            "selected_call_index_zero_based": repair["selected_call_index_zero_based"],
            "result_gates": result_gates,
            "semantic": result["semantic"],
            "profiles": profiles,
        }

    pairwise: dict[str, Any] = {}
    for left, right in itertools.combinations(positions, 2):
        key = f"{left}_vs_{right}"
        pairwise[key] = {}
        for endpoint, filename in (
            ("clipped_gradient", "clipped_gradients.safetensors"),
            ("parameter_update", "parameter_updates.safetensors"),
        ):
            pairwise[key][endpoint] = pairwise_effect_metrics(
                candidate_root / filename,
                directories[left] / filename,
                directories[right] / filename,
            )

    gates = {
        "manifest_artifacts_exact": all(artifact_gates.values()),
        "all_predeclared_positions_present": set(results) == set(positions),
        "all_results_valid": all(
            all(row["result_gates"].values()) for row in results.values()
        ),
        "all_pairwise_comparisons_present": len(pairwise) == 3,
    }
    payload = {
        "schema_version": "forkcert.qwen3-backward-repeated-family-repair-evaluation.v0.1",
        "status": "VALID_BACKWARD_REPEATED_FAMILY_REPAIR_EVALUATION"
        if all(gates.values())
        else "INVALID_EVALUATION",
        "manifest": str(manifest_path),
        "artifact_gates": artifact_gates,
        "gates": gates,
        "positions": results,
        "pairwise_effect_alignment": pairwise,
        "claim_limits": [
            "selected-state and selected-call intervention impact only",
            "equal signatures do not establish semantic interchangeability",
            "eager is a baseline rather than correctness authority",
            "no injection, necessity, sufficiency, population, long-run or correctness claim",
        ],
    }
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError(out)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": payload["status"], "gates": gates}, indent=2))
    if payload["status"] != "VALID_BACKWARD_REPEATED_FAMILY_REPAIR_EVALUATION":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
