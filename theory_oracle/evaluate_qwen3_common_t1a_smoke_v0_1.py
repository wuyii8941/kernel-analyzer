#!/usr/bin/env python
"""Evaluate one reproducible frozen T1a bank on exact matched post-states."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
from pathlib import Path
from typing import Any

try:
    from theory_oracle.evaluate_qwen3_common_t1b_smoke_v0_1 import reconstruct_model
    from theory_oracle.generate_qwen3_t1a_bank_v0_1 import score_completions
    from theory_oracle.qwen3_grpo_natural_transition_v0_2 import json_sha256, sha256_file
except ModuleNotFoundError:  # Direct script execution from theory_oracle/.
    from evaluate_qwen3_common_t1b_smoke_v0_1 import reconstruct_model
    from generate_qwen3_t1a_bank_v0_1 import score_completions
    from qwen3_grpo_natural_transition_v0_2 import json_sha256, sha256_file


SCHEMA_VERSION = "forkcert.qwen3-common-t1a-smoke-result.v0.1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--bank", required=True)
    parser.add_argument("--bank-repeat", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def grpo_components(
    torch: Any,
    new_logps: Any,
    old_logps: Any,
    advantages: Any,
    mask: Any,
    epsilon: float,
) -> tuple[Any, Any, Any]:
    ratio = torch.exp(new_logps - old_logps)
    clipped_ratio = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon)
    per_token_loss = -torch.minimum(
        ratio * advantages.unsqueeze(1),
        clipped_ratio * advantages.unsqueeze(1),
    )
    per_completion_loss = (per_token_loss * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)
    clipped = ((ratio < 1.0 - epsilon) & (advantages.unsqueeze(1) < 0)) | (
        (ratio > 1.0 + epsilon) & (advantages.unsqueeze(1) > 0)
    )
    return per_completion_loss.mean(), per_completion_loss, clipped


def validate_artifacts(manifest: dict[str, Any]) -> dict[str, bool]:
    state = manifest["state_scope"]
    checks = {
        "snapshot_metadata_sha256_exact": sha256_file(
            Path(state["snapshot_dir"]) / "forkcert_transition_snapshot.json"
        )
        == state["snapshot_metadata_sha256"]
    }
    for arm_name, arm in manifest["arms"].items():
        checks[f"{arm_name}_transition_result_sha256_exact"] = (
            sha256_file(Path(arm["transition_result"])) == arm["transition_result_sha256"]
        )
        checks[f"{arm_name}_parameter_updates_sha256_exact"] = (
            sha256_file(Path(arm["parameter_updates"])) == arm["parameter_updates_sha256"]
        )
    return checks


def prepare_group(torch: Any, rows: list[dict[str, Any]], pad_token_id: int) -> tuple[Any, ...]:
    max_prompt = max(len(row["prompt_ids"]) for row in rows)
    max_completion = max(len(row["completion_ids"]) for row in rows)
    prompt_ids = torch.full(
        (len(rows), max_prompt), pad_token_id, dtype=torch.long, device="cuda"
    )
    prompt_mask = torch.zeros_like(prompt_ids)
    completion_ids = torch.full(
        (len(rows), max_completion), pad_token_id, dtype=torch.long, device="cuda"
    )
    completion_mask = torch.zeros_like(completion_ids)
    old_logps = torch.zeros(
        (len(rows), max_completion), dtype=torch.float32, device="cuda"
    )
    advantages = torch.tensor(
        [row["advantage"] for row in rows], dtype=torch.float32, device="cuda"
    )
    for index, row in enumerate(rows):
        prompt = torch.tensor(row["prompt_ids"], dtype=torch.long, device="cuda")
        completion = torch.tensor(row["completion_ids"], dtype=torch.long, device="cuda")
        mask = torch.tensor(row["completion_mask"], dtype=torch.long, device="cuda")
        old = torch.tensor(row["old_per_token_logps"], dtype=torch.float32, device="cuda")
        prompt_ids[index, -len(prompt) :] = prompt
        prompt_mask[index, -len(prompt) :] = 1
        completion_ids[index, : len(completion)] = completion
        completion_mask[index, : len(mask)] = mask
        old_logps[index, : len(old)] = old
    return prompt_ids, prompt_mask, completion_ids, completion_mask, old_logps, advantages


def evaluate_bank(
    torch: Any,
    model: Any,
    bank_rows: list[dict[str, Any]],
    pad_token_id: int,
    epsilon: float,
) -> dict[str, Any]:
    group_ids = sorted({int(row["group_index"]) for row in bank_rows})
    output_rows: list[dict[str, Any]] = []
    for group_id in group_ids:
        rows = [row for row in bank_rows if int(row["group_index"]) == group_id]
        prepared = prepare_group(torch, rows, pad_token_id)
        prompt_ids, prompt_mask, completion_ids, completion_mask, old_logps, advantages = prepared
        new_logps = score_completions(
            torch,
            model,
            prompt_ids,
            prompt_mask,
            completion_ids,
            completion_mask,
        )
        _, per_completion, clipped = grpo_components(
            torch,
            new_logps,
            old_logps,
            advantages,
            completion_mask.float(),
            epsilon,
        )
        for index, row in enumerate(rows):
            valid = completion_mask[index].bool()
            output_rows.append(
                {
                    "group_index": group_id,
                    "dataset_index": row["dataset_index"],
                    "generation_index": row["generation_index"],
                    "advantage": row["advantage"],
                    "completion_loss": float(per_completion[index].item()),
                    "clip_count": int(clipped[index][valid].sum().item()),
                    "new_per_token_logps": new_logps[index][valid].cpu().tolist(),
                }
            )
        del prepared, new_logps, per_completion, clipped
    output_rows.sort(key=lambda row: (row["group_index"], row["generation_index"]))
    loss = math.fsum(float(row["completion_loss"]) for row in output_rows) / len(output_rows)
    return {
        "loss": loss,
        "clip_count": sum(int(row["clip_count"]) for row in output_rows),
        "rows": output_rows,
        "rows_digest": json_sha256(output_rows),
    }


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest).resolve()
    bank_path = Path(args.bank).resolve()
    bank_repeat_path = Path(args.bank_repeat).resolve()
    out_path = Path(args.out).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    bank_repeat = json.loads(bank_repeat_path.read_text(encoding="utf-8"))
    if manifest["schema_version"] != "forkcert.qwen3-t1a-selected-state-smoke-manifest.v0.1":
        raise ValueError("unsupported manifest schema")
    if bank["schema_version"] != "forkcert.qwen3-t1a-frozen-bank.v0.1":
        raise ValueError("unsupported bank schema")

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("PYTHONHASHSEED", "0")
    import torch
    from transformers import AutoTokenizer

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one visible CUDA device is required")
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False

    artifact_checks = validate_artifacts(manifest)
    bank_checks = {
        "primary_bank_valid": bool(bank["valid"]),
        "repeat_bank_valid": bool(bank_repeat["valid"]),
        "bank_manifest_matches": bank["manifest"]["sha256"] == sha256_file(manifest_path),
        "repeat_manifest_matches": bank_repeat["manifest"]["sha256"] == sha256_file(manifest_path),
        "fresh_bank_content_exact": bank["bank_sha256"] == bank_repeat["bank_sha256"],
        "bank_content_digest_self_valid": json_sha256(bank["bank"]) == bank["bank_sha256"],
        "bank_has_informative_advantage": any(
            float(row["advantage"]) != 0.0 for row in bank["bank"]["rows"]
        ),
    }
    if not all({**artifact_checks, **bank_checks}.values()):
        raise ValueError(f"artifact or bank validity failed: {artifact_checks} {bank_checks}")

    snapshot_dir = Path(manifest["state_scope"]["snapshot_dir"])
    tokenizer = AutoTokenizer.from_pretrained(snapshot_dir, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    repeats = int(manifest["endpoint"]["fresh_reconstructions_per_arm"])
    epsilon = float(manifest["endpoint"]["epsilon"])
    arm_results: dict[str, list[dict[str, Any]]] = {"reference": [], "candidate": []}
    reconstruction: dict[str, list[dict[str, Any]]] = {"reference": [], "candidate": []}
    for arm_name in ("reference", "candidate"):
        for repeat in range(repeats):
            model, reconstruction_row = reconstruct_model(
                torch,
                snapshot_dir,
                manifest["arms"][arm_name],
                manifest["state_scope"]["pre_parameter_digest"],
                manifest["state_scope"]["pre_buffer_digest"],
            )
            model = model.to("cuda")
            model.eval()
            result = evaluate_bank(
                torch,
                model,
                bank["bank"]["rows"],
                tokenizer.pad_token_id,
                epsilon,
            )
            result["repeat"] = repeat + 1
            arm_results[arm_name].append(result)
            reconstruction_row["repeat"] = repeat + 1
            reconstruction[arm_name].append(reconstruction_row)
            del model
            gc.collect()
            torch.cuda.empty_cache()

    repeat_exact = {
        arm_name: len({row["rows_digest"] for row in results}) == 1
        for arm_name, results in arm_results.items()
    }
    paired_shifts = [
        arm_results["candidate"][repeat]["loss"]
        - arm_results["reference"][repeat]["loss"]
        for repeat in range(repeats)
    ]
    valid = all(artifact_checks.values()) and all(bank_checks.values()) and all(repeat_exact.values())
    payload = {
        "schema_version": SCHEMA_VERSION,
        "valid": valid,
        "verdict": "VALID_SELECTED_STATE_T1A_SMOKE" if valid else "INVALID",
        "manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_build": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "attention_backend": "SDPA_MATH",
            "autocast_dtype": "float16",
            "model_storage_dtype": "float32",
            "evaluator_code_sha256": sha256_file(Path(__file__).resolve()),
        },
        "artifacts": {
            "checks": artifact_checks,
            "bank_checks": bank_checks,
            "bank": {"path": str(bank_path), "file_sha256": sha256_file(bank_path), "content_sha256": bank["bank_sha256"]},
            "bank_repeat": {"path": str(bank_repeat_path), "file_sha256": sha256_file(bank_repeat_path), "content_sha256": bank_repeat["bank_sha256"]},
            "informative_group_count": 8 - int(bank["summary"]["zero_advantage_groups"]),
            "total_group_count": 8,
        },
        "reconstruction": reconstruction,
        "endpoint": {
            "id": "T1a",
            "state_scope": manifest["state_scope"]["population_role"],
            "reference": arm_results["reference"],
            "candidate": arm_results["candidate"],
            "paired_candidate_minus_reference": paired_shifts,
            "mean_paired_shift": math.fsum(paired_shifts) / len(paired_shifts),
            "repeat_exact": repeat_exact,
            "direction": "candidate minus reference GRPO surrogate loss; sign is endpoint-relative, not harm",
        },
        "nonclaims": manifest["nonclaims"],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "verdict": payload["verdict"],
                "bank_content_sha256": bank["bank_sha256"],
                "reference_loss": [row["loss"] for row in arm_results["reference"]],
                "candidate_loss": [row["loss"] for row in arm_results["candidate"]],
                "paired_candidate_minus_reference": paired_shifts,
                "repeat_exact": repeat_exact,
                "informative_group_count": payload["artifacts"]["informative_group_count"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if not valid:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
