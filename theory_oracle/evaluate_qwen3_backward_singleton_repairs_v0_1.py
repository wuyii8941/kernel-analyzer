#!/usr/bin/env python
"""Audit and stream-evaluate completed Qwen3 backward singleton repairs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def vector_metrics(eager_path: Path, candidate_path: Path, repaired_path: Path) -> dict[str, Any]:
    import torch
    from safetensors import safe_open

    square_ce = 0.0
    square_rc = 0.0
    square_re = 0.0
    dot_effect_target = 0.0
    max_rc = 0.0
    max_re = 0.0
    coordinates = 0
    changed_tensors = 0
    with safe_open(eager_path, framework="pt", device="cpu") as eager, safe_open(
        candidate_path, framework="pt", device="cpu"
    ) as candidate, safe_open(repaired_path, framework="pt", device="cpu") as repaired:
        keys = sorted(eager.keys())
        if keys != sorted(candidate.keys()) or keys != sorted(repaired.keys()):
            raise ValueError("vector tensor-key sets differ")
        for key in keys:
            e = eager.get_tensor(key).float()
            c = candidate.get_tensor(key).float()
            r = repaired.get_tensor(key).float()
            if e.shape != c.shape or e.shape != r.shape:
                raise ValueError(f"shape mismatch for {key}")
            ce = e - c
            rc = r - c
            re = r - e
            square_ce += float(torch.sum(ce * ce, dtype=torch.float64).item())
            square_rc += float(torch.sum(rc * rc, dtype=torch.float64).item())
            square_re += float(torch.sum(re * re, dtype=torch.float64).item())
            dot_effect_target += float(torch.sum(rc * ce, dtype=torch.float64).item())
            current_rc = float(rc.abs().max().item())
            current_re = float(re.abs().max().item())
            max_rc = max(max_rc, current_rc)
            max_re = max(max_re, current_re)
            coordinates += int(r.numel())
            changed_tensors += int(current_rc != 0.0)
            del e, c, r, ce, rc, re
    l2_ce = math.sqrt(square_ce)
    l2_rc = math.sqrt(square_rc)
    l2_re = math.sqrt(square_re)
    denominator = l2_rc * l2_ce
    return {
        "coordinates": coordinates,
        "changed_tensors": changed_tensors,
        "candidate_to_eager_l2": l2_ce,
        "candidate_to_repair_l2": l2_rc,
        "eager_to_repair_l2": l2_re,
        "candidate_to_repair_max_abs": max_rc,
        "eager_to_repair_max_abs": max_re,
        "fractional_eager_distance_reduction": (l2_ce - l2_re) / l2_ce if l2_ce else None,
        "cosine_repair_with_candidate_to_eager": dot_effect_target / denominator if denominator else None,
        "repair_exactly_null": square_rc == 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--revision-manifest", required=True)
    parser.add_argument("--repair-root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest["status"] != "FROZEN_PRE_EXECUTION":
        raise ValueError("manifest is not frozen")
    revision_manifest_path = Path(args.revision_manifest).resolve()
    revision_manifest = json.loads(revision_manifest_path.read_text())
    if revision_manifest["status"] != "FROZEN_PRE_EXECUTION":
        raise ValueError("revision manifest is not frozen")
    artifact_gates = {
        name: sha256_file(Path(row["path"]).resolve()) == row["sha256"]
        for name, row in manifest["artifacts"].items()
    }
    revision_artifact_gates = {
        name: sha256_file(Path(row["path"]).resolve()) == row["sha256"]
        for name, row in revision_manifest["artifacts"].items()
    }
    eager_root = Path("results/training_step_oracle/qwen3_grpo_natural_transition_v0_2/eager_1").resolve()
    candidate_root = Path("results/training_step_oracle/qwen3_grpo_natural_transition_v0_2/compiled_1").resolve()
    candidate_result = json.loads((candidate_root / "result.json").read_text())
    baseline_evaluation = json.loads(
        Path("results/training_step_oracle/qwen3_grpo_natural_transition_v0_2/evaluation.json").read_text()
    )
    repair_root = Path(args.repair_root).resolve()
    treatments: dict[str, Any] = {}
    for name in manifest["selection"]["treatments"]:
        if name in revision_manifest["output_directories"]:
            directory = Path(revision_manifest["output_directories"][name]).resolve()
        else:
            directory = repair_root / name
        result_path = directory / "result.json"
        result = json.loads(result_path.read_text())
        repair = result.get("backward_singleton_repair", {})
        result_gates = {
            "repair_status_valid": repair.get("status") == "VALID_BACKWARD_SINGLETON_REPAIR",
            "all_embedded_repair_gates_true": bool(repair.get("gates"))
            and all(repair["gates"].values()),
            "base_transition_valid": result.get("valid") is True and result.get("verdict") == "VALID",
            "vector_artifacts_present": all(
                (directory / filename).is_file()
                for filename in ("clipped_gradients.safetensors", "parameter_updates.safetensors")
            ),
        }
        if not all(result_gates.values()):
            raise ValueError(f"invalid repair result for {name}: {result_gates}")
        profiles = {}
        for endpoint, summary_key, filename in (
            ("clipped_gradient", "clipped_gradient", "clipped_gradients.safetensors"),
            ("parameter_update", "parameter_update", "parameter_updates.safetensors"),
        ):
            result_hash = result["continuous"][summary_key]["tensor_hashes_sha256"]
            candidate_hash = candidate_result["continuous"][summary_key]["tensor_hashes_sha256"]
            if result_hash == candidate_hash:
                baseline = baseline_evaluation["profiles"][endpoint]
                profiles[endpoint] = {
                    "coordinates": sum(
                        int(row.get("coordinates", 0)) for row in baseline["per_parameter"]
                    ),
                    "changed_tensors": 0,
                    "candidate_to_eager_l2": baseline["B_effect_l2"],
                    "candidate_to_repair_l2": 0.0,
                    "eager_to_repair_l2": baseline["B_effect_l2"],
                    "candidate_to_repair_max_abs": 0.0,
                    "eager_to_repair_max_abs": baseline["B_max_abs_coordinate"],
                    "fractional_eager_distance_reduction": 0.0,
                    "cosine_repair_with_candidate_to_eager": None,
                    "repair_exactly_null": True,
                    "null_gate": "per-tensor hash digest equals compiled baseline",
                }
            else:
                profiles[endpoint] = vector_metrics(
                    eager_root / filename, candidate_root / filename, directory / filename
                )
        treatments[name] = {
            "kernel_family": repair["kernel_family"],
            "semantic_boundary": repair["semantic_boundary"],
            "result_gates": result_gates,
            "semantic": result["semantic"],
            "profiles": profiles,
        }
    gates = {
        "manifest_artifacts_exact": all(artifact_gates.values()),
        "revision_manifest_artifacts_exact": all(revision_artifact_gates.values()),
        "all_treatments_present": set(treatments) == set(manifest["selection"]["treatments"]),
        "all_results_valid": all(all(row["result_gates"].values()) for row in treatments.values()),
    }
    payload = {
        "schema_version": "forkcert.qwen3-backward-singleton-repair-evaluation.v0.1",
        "status": "VALID_BACKWARD_SINGLETON_REPAIR_EVALUATION" if all(gates.values()) else "INVALID_EVALUATION",
        "manifest": str(manifest_path),
        "revision_manifest": str(revision_manifest_path),
        "artifact_gates": artifact_gates,
        "revision_artifact_gates": revision_artifact_gates,
        "gates": gates,
        "treatments": treatments,
        "claim_limits": [
            "selected-state intervention impact only",
            "eager is a baseline rather than correctness authority",
            "no injection, necessity, sufficiency, population or long-run claim",
        ],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError(out)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": payload["status"], "gates": gates}, indent=2))
    if payload["status"] != "VALID_BACKWARD_SINGLETON_REPAIR_EVALUATION":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
