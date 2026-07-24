#!/usr/bin/env python
"""Evaluate all valid Qwen3 singleton backward repairs through revision 4."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from theory_oracle.evaluate_qwen3_backward_singleton_repairs_v0_1 import (
    sha256_file,
    vector_metrics,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifests", nargs=4, required=True)
    parser.add_argument("--repair-root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    manifests = [json.loads(Path(path).read_text()) for path in args.manifests]
    if not all(row["status"] == "FROZEN_PRE_EXECUTION" for row in manifests):
        raise ValueError("all manifests must be frozen")
    artifact_gates: dict[str, dict[str, bool]] = {}
    for manifest in manifests:
        artifact_gates[manifest["schema_version"]] = {
            name: sha256_file(Path(row["path"]).resolve()) == row["sha256"]
            for name, row in manifest["artifacts"].items()
        }

    root = Path(args.repair_root).resolve()
    directories = {
        "cast_view": root / "cast_view",
        "embedding_zero": root / "embedding_zero",
        "silu_mul": root / "silu_mul_v0_2",
        "silu_mul_backward": root / "silu_mul_backward_v0_2",
        "fp16_fp32_add": root / "fp16_fp32_add",
        "attention_safe_softmax": Path(manifests[2]["output"]).resolve(),
        **{
            name: Path(path).resolve()
            for name, path in manifests[3]["output_directories"].items()
        },
    }
    expected = {
        "cast_view",
        "embedding_zero",
        "silu_mul",
        "silu_mul_backward",
        "fp16_fp32_add",
        "attention_safe_softmax",
        "final_norm_backward",
        "embedding_norm_backward_prep",
    }
    if set(directories) != expected:
        raise ValueError("unexpected treatment directory map")

    eager_root = Path(
        "results/training_step_oracle/qwen3_grpo_natural_transition_v0_2/eager_1"
    ).resolve()
    candidate_root = Path(
        "results/training_step_oracle/qwen3_grpo_natural_transition_v0_2/compiled_1"
    ).resolve()
    candidate_result = json.loads((candidate_root / "result.json").read_text())
    baseline = json.loads(
        Path(
            "results/training_step_oracle/qwen3_grpo_natural_transition_v0_2/evaluation.json"
        ).read_text()
    )
    treatments: dict[str, Any] = {}
    for name, directory in directories.items():
        result = json.loads((directory / "result.json").read_text())
        repair = result.get("backward_singleton_repair", {})
        result_gates = {
            "repair_status_valid": repair.get("status") == "VALID_BACKWARD_SINGLETON_REPAIR",
            "all_embedded_repair_gates_true": bool(repair.get("gates"))
            and all(repair["gates"].values()),
            "base_transition_valid": result.get("valid") is True and result.get("verdict") == "VALID",
            "vectors_present": all(
                (directory / filename).is_file()
                for filename in ("clipped_gradients.safetensors", "parameter_updates.safetensors")
            ),
        }
        if not all(result_gates.values()):
            raise ValueError(f"invalid result {name}: {result_gates}")
        profiles = {}
        for endpoint, summary_key, filename in (
            ("clipped_gradient", "clipped_gradient", "clipped_gradients.safetensors"),
            ("parameter_update", "parameter_update", "parameter_updates.safetensors"),
        ):
            result_hash = result["continuous"][summary_key]["tensor_hashes_sha256"]
            candidate_hash = candidate_result["continuous"][summary_key]["tensor_hashes_sha256"]
            if result_hash == candidate_hash:
                reference = baseline["profiles"][endpoint]
                profiles[endpoint] = {
                    "coordinates": sum(
                        int(row.get("coordinates", 0)) for row in reference["per_parameter"]
                    ),
                    "changed_tensors": 0,
                    "candidate_to_eager_l2": reference["B_effect_l2"],
                    "candidate_to_repair_l2": 0.0,
                    "eager_to_repair_l2": reference["B_effect_l2"],
                    "candidate_to_repair_max_abs": 0.0,
                    "eager_to_repair_max_abs": reference["B_max_abs_coordinate"],
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
            "repair_revision": repair.get("repair_revision", "v0.1"),
            "result_gates": result_gates,
            "profiles": profiles,
        }
    gates = {
        "all_manifest_artifacts_exact": all(
            all(rows.values()) for rows in artifact_gates.values()
        ),
        "all_eight_treatments_present": set(treatments) == expected,
        "all_results_valid": all(
            all(row["result_gates"].values()) for row in treatments.values()
        ),
    }
    payload = {
        "schema_version": "forkcert.qwen3-backward-singleton-repair-evaluation.v0.2",
        "status": "VALID_BACKWARD_SINGLETON_REPAIR_EVALUATION_V0_2"
        if all(gates.values())
        else "INVALID_EVALUATION",
        "manifests": [str(Path(path).resolve()) for path in args.manifests],
        "artifact_gates": artifact_gates,
        "gates": gates,
        "treatments": treatments,
        "uninstantiated_singleton_family": (
            "triton_red_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_"
            "slice_backward_sum_view_30"
        ),
        "claim_limits": [
            "selected-state repair impact only",
            "eager is a baseline rather than correctness authority",
            "no injection, necessity, sufficiency, population, long-run or correctness claim",
        ],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError(out)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": payload["status"], "gates": gates}, indent=2))
    if payload["status"] != "VALID_BACKWARD_SINGLETON_REPAIR_EVALUATION_V0_2":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
