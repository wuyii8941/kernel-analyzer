#!/usr/bin/env python
"""Evaluate nested transition/evaluator repeats for T1a and T1b at one state."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
from pathlib import Path
from typing import Any

from theory_oracle.evaluate_qwen3_common_t1a_smoke_v0_1 import (
    evaluate_bank as evaluate_t1a_bank,
)
from theory_oracle.evaluate_qwen3_common_t1b_smoke_v0_1 import (
    bank_identity,
    build_bank as build_t1b_bank,
    evaluate_bank as evaluate_t1b_bank,
    reconstruct_model,
    tokenizer_identity,
)
from theory_oracle.qwen3_grpo_natural_transition_v0_2 import sha256_file


SCHEMA_VERSION = "forkcert.qwen3-calibration-state-endpoints.v0.1"
TASK_ENDPOINT_RANDOMNESS_SCOPE = {
    "transition_repeats": "identify paired transition-effect variability conditional on the frozen state and endpoint bank",
    "evaluator_repeats": "nested deterministic repeatability check; averaged within transition repeat",
    "t1a_bank_generation_repeats": "fresh reproducibility check using the same frozen seed; not independent rollout-bank samples",
    "t1a_bank_sampling_variance": "UNIDENTIFIED_ONE_FROZEN_BANK_PER_STATE",
    "t1a_cross_state_interpretation": "state-adaptive local policy-relative functional; H includes changing state and bank content",
    "t1b_cross_state_interpretation": "fixed correct-answer-bank functional under the frozen evaluator contract",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sample_variance(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = math.fsum(values) / len(values)
    return math.fsum((value - mean) ** 2 for value in values) / (len(values) - 1)


def endpoint_profile(
    arm_results: dict[str, list[dict[str, Any]]], value_key: str
) -> dict[str, Any]:
    transition_repeats = sorted(
        {int(row["transition_repeat"]) for rows in arm_results.values() for row in rows}
    )
    paired_rows = []
    for transition_repeat in transition_repeats:
        values_by_repeat: dict[str, dict[int, float]] = {}
        for arm in ("reference", "candidate"):
            selected = [
                row
                for row in arm_results[arm]
                if int(row["transition_repeat"]) == transition_repeat
            ]
            if any("evaluator_repeat" not in row for row in selected):
                raise ValueError("evaluator repeat identity is missing")
            repeat_ids = [int(row["evaluator_repeat"]) for row in selected]
            if len(repeat_ids) != len(set(repeat_ids)):
                raise ValueError("duplicate evaluator repeat identity")
            values_by_repeat[arm] = {
                int(row["evaluator_repeat"]): float(row[value_key])
                for row in selected
            }
        reference_ids = set(values_by_repeat["reference"])
        candidate_ids = set(values_by_repeat["candidate"])
        if not reference_ids or reference_ids != candidate_ids:
            raise ValueError("unbalanced evaluator repeats within transition repeat")
        evaluator_repeat_ids = sorted(reference_ids)
        if evaluator_repeat_ids != list(range(1, len(evaluator_repeat_ids) + 1)):
            raise ValueError("evaluator repeat identities are not consecutive from one")
        values = {
            arm: [values_by_repeat[arm][repeat] for repeat in evaluator_repeat_ids]
            for arm in ("reference", "candidate")
        }
        reference_mean = math.fsum(values["reference"]) / len(values["reference"])
        candidate_mean = math.fsum(values["candidate"]) / len(values["candidate"])
        evaluator_paired_effects = [
            candidate - reference
            for reference, candidate in zip(
                values["reference"], values["candidate"], strict=True
            )
        ]
        paired_rows.append(
            {
                "transition_repeat": transition_repeat,
                "evaluator_repeat_ids": evaluator_repeat_ids,
                "reference_evaluator_values": values["reference"],
                "candidate_evaluator_values": values["candidate"],
                "reference_evaluator_mean": reference_mean,
                "candidate_evaluator_mean": candidate_mean,
                "paired_effect": candidate_mean - reference_mean,
                "evaluator_paired_effect_variance": sample_variance(
                    evaluator_paired_effects
                ),
            }
        )
    effects = [row["paired_effect"] for row in paired_rows]
    return {
        "paired_transition_repeat_effects": paired_rows,
        "state_effect_signed_mean": math.fsum(effects) / len(effects),
        "N_transition_paired_effect_variance": sample_variance(effects),
        "evaluator_variability_is_separate": True,
        "B_status": "NOT_POPULATION_B_ONE_STATE",
        "H_status": "UNIDENTIFIABLE_ONE_STATE",
        "U_status": "NOT_ESTIMATED_ONE_TRAJECTORY_STATE",
    }


def validate_arm_artifacts(manifest: dict[str, Any]) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    for arm, arm_group in manifest["arms"].items():
        for spec in arm_group["transition_repeats"]:
            repeat = int(spec["transition_repeat"])
            checks[f"{arm}_{repeat}_result"] = (
                sha256_file(Path(spec["transition_result"]))
                == spec["transition_result_sha256"]
            )
            checks[f"{arm}_{repeat}_update"] = (
                sha256_file(Path(spec["parameter_updates"]))
                == spec["parameter_updates_sha256"]
            )
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--t1a-bank", required=True)
    parser.add_argument("--t1a-bank-repeat", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    bank_path = Path(args.t1a_bank).resolve()
    bank_repeat_path = Path(args.t1a_bank_repeat).resolve()
    manifest = load_json(manifest_path)
    bank = load_json(bank_path)
    bank_repeat = load_json(bank_repeat_path)
    if manifest.get("schema_version") != "forkcert.qwen3-calibration-state-endpoint-manifest.v0.1":
        raise ValueError("unsupported calibration endpoint manifest")
    if bank.get("schema_version") != "forkcert.qwen3-t1a-frozen-bank.v0.1":
        raise ValueError("unsupported T1a bank schema")

    arm_checks = validate_arm_artifacts(manifest)
    bank_checks = {
        "primary_valid": bool(bank.get("valid")),
        "repeat_valid": bool(bank_repeat.get("valid")),
        "primary_manifest_exact": bank["manifest"]["sha256"] == sha256_file(manifest_path),
        "repeat_manifest_exact": bank_repeat["manifest"]["sha256"] == sha256_file(manifest_path),
        "fresh_content_exact": bank["bank_sha256"] == bank_repeat["bank_sha256"],
    }
    if not all({**arm_checks, **bank_checks}.values()):
        raise ValueError(f"artifact identity failed: {arm_checks} {bank_checks}")

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("PYTHONHASHSEED", "0")
    import torch
    from transformers import AutoTokenizer

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one visible CUDA device is required")
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False

    snapshot_dir = Path(manifest["state_scope"]["snapshot_dir"])
    tokenizer = AutoTokenizer.from_pretrained(snapshot_dir, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    t1b_rows = build_t1b_bank(tokenizer, manifest["T1b"])
    t1b_digest = bank_identity(t1b_rows)
    evaluator_repeats = int(
        manifest["evaluation"]["evaluator_repeats_per_transition_repeat"]
    )
    t1a_results: dict[str, list[dict[str, Any]]] = {"reference": [], "candidate": []}
    t1b_results: dict[str, list[dict[str, Any]]] = {"reference": [], "candidate": []}
    reconstructions: dict[str, list[dict[str, Any]]] = {"reference": [], "candidate": []}
    for arm in ("reference", "candidate"):
        for transition_spec in manifest["arms"][arm]["transition_repeats"]:
            transition_repeat = int(transition_spec["transition_repeat"])
            for evaluator_repeat in range(1, evaluator_repeats + 1):
                model, reconstruction = reconstruct_model(
                    torch,
                    snapshot_dir,
                    transition_spec,
                    manifest["state_scope"]["pre_parameter_digest"],
                    manifest["state_scope"]["pre_buffer_digest"],
                )
                model = model.to("cuda")
                model.eval()
                t1a = evaluate_t1a_bank(
                    torch,
                    model,
                    bank["bank"]["rows"],
                    tokenizer.pad_token_id,
                    float(manifest["T1a"]["epsilon"]),
                )
                t1b = evaluate_t1b_bank(
                    torch,
                    model,
                    t1b_rows,
                    tokenizer.pad_token_id,
                    int(manifest["T1b"]["batch_size"]),
                )
                identity = {
                    "transition_repeat": transition_repeat,
                    "evaluator_repeat": evaluator_repeat,
                }
                t1a_results[arm].append({**identity, **t1a})
                t1b_results[arm].append({**identity, **t1b})
                reconstructions[arm].append({**identity, **reconstruction})
                del model
                gc.collect()
                torch.cuda.empty_cache()

    evaluator_exact: dict[str, bool] = {}
    for endpoint_name, endpoint_results in (("T1a", t1a_results), ("T1b", t1b_results)):
        for arm, rows in endpoint_results.items():
            for transition_repeat in (1, 2):
                digests = {
                    row["rows_digest"]
                    for row in rows
                    if int(row["transition_repeat"]) == transition_repeat
                }
                evaluator_exact[f"{endpoint_name}_{arm}_{transition_repeat}"] = len(digests) == 1

    informative_t1a = any(
        float(row["advantage"]) != 0.0 for row in bank["bank"]["rows"]
    )
    valid = all(arm_checks.values()) and all(bank_checks.values()) and all(
        evaluator_exact.values()
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "valid": valid,
        "verdict": "VALID_CALIBRATION_STATE_ENDPOINTS" if valid else "INVALID",
        "manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_build": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "evaluator_code_sha256": sha256_file(Path(__file__).resolve()),
        },
        "validity": {
            "arm_artifacts": arm_checks,
            "bank_artifacts": bank_checks,
            "evaluator_repeat_exact": evaluator_exact,
        },
        "randomness_decomposition": TASK_ENDPOINT_RANDOMNESS_SCOPE,
        "artifacts": {
            "T1a_bank": {"path": str(bank_path), "sha256": sha256_file(bank_path), "content_sha256": bank["bank_sha256"]},
            "T1a_bank_repeat": {"path": str(bank_repeat_path), "sha256": sha256_file(bank_repeat_path), "content_sha256": bank_repeat["bank_sha256"]},
            "T1b_bank_sha256": t1b_digest,
            "tokenizer": tokenizer_identity(snapshot_dir),
        },
        "reconstructions": reconstructions,
        "T1a": {
            "status": "MEASURED" if informative_t1a else "UNINSTANTIATED_ALL_GROUPS_TIED",
            "arm_results": t1a_results,
            "profile": endpoint_profile(t1a_results, "loss") if informative_t1a else None,
        },
        "T1b": {
            "status": "MEASURED",
            "arm_results": t1b_results,
            "profile": endpoint_profile(t1b_results, "mean_nll"),
        },
        "nesting": manifest["evaluation"]["nesting_rule"],
        "nonclaims": manifest["nonclaims"],
    }
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "verdict": payload["verdict"],
                "T1a_status": payload["T1a"]["status"],
                "T1a_state_effect": payload["T1a"]["profile"]["state_effect_signed_mean"] if payload["T1a"]["profile"] else None,
                "T1b_state_effect": payload["T1b"]["profile"]["state_effect_signed_mean"],
                "evaluator_exact": all(evaluator_exact.values()),
            },
            indent=2,
        )
    )
    if not valid:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
