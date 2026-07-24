#!/usr/bin/env python
"""Evaluate exact matched post-states with one common eager T1b ruler."""

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
    from theory_oracle.qwen3_grpo_natural_transition_v0_2 import (
        json_sha256,
        named_tensor_hashes,
        sha256_file,
    )
except ModuleNotFoundError:  # Direct script execution from theory_oracle/.
    from qwen3_grpo_natural_transition_v0_2 import (
        json_sha256,
        named_tensor_hashes,
        sha256_file,
    )


SCHEMA_VERSION = "forkcert.qwen3-common-t1b-smoke-result.v0.1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def arithmetic_fields(index: int) -> dict[str, int]:
    start = 7 + index
    added = 3 + (index % 11)
    removed = 1 + (index % 5)
    return {
        "start": start,
        "added": added,
        "removed": removed,
        "result": start + added - removed,
    }


def build_bank(tokenizer: Any, config: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in range(config["start_index_inclusive"], config["stop_index_exclusive"]):
        fields = arithmetic_fields(index)
        prompt = config["prompt_template"].format(**fields)
        prefix = prompt + config["prompt_completion_separator"]
        completion = config["completion_template"].format(**fields)
        prefix_ids = tokenizer(prefix, add_special_tokens=True)["input_ids"]
        full_ids = tokenizer(prefix + completion, add_special_tokens=True)["input_ids"]
        prefix_stable = full_ids[: len(prefix_ids)] == prefix_ids
        if config["require_prompt_token_prefix_stability"] and not prefix_stable:
            raise ValueError(f"tokenized prompt is not a stable prefix at dataset index {index}")
        completion_ids = full_ids[len(prefix_ids) :]
        if not completion_ids:
            raise ValueError(f"empty completion target at dataset index {index}")
        if config["include_eos_in_target"]:
            if tokenizer.eos_token_id is None:
                raise ValueError("manifest requests EOS but tokenizer has no EOS token")
            full_ids = full_ids + [tokenizer.eos_token_id]
            completion_ids = completion_ids + [tokenizer.eos_token_id]
        rows.append(
            {
                "dataset_index": index,
                "fields": fields,
                "prompt": prompt,
                "completion": completion,
                "input_ids": full_ids,
                "prompt_token_count": len(prefix_ids),
                "completion_token_ids": completion_ids,
                "prefix_stable": prefix_stable,
            }
        )
    return rows


def bank_identity(rows: list[dict[str, Any]]) -> str:
    identity = [
        {
            "dataset_index": row["dataset_index"],
            "prompt": row["prompt"],
            "completion": row["completion"],
            "input_ids": row["input_ids"],
            "prompt_token_count": row["prompt_token_count"],
            "completion_token_ids": row["completion_token_ids"],
        }
        for row in rows
    ]
    return json_sha256(identity)


def tokenizer_identity(snapshot_dir: Path) -> dict[str, Any]:
    names = [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.json",
        "merges.txt",
    ]
    files = {
        name: sha256_file(snapshot_dir / name)
        for name in names
        if (snapshot_dir / name).is_file()
    }
    return {"files": files, "digest": json_sha256(files)}


def validate_input_artifacts(manifest: dict[str, Any]) -> dict[str, bool]:
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


def reconstruct_model(
    torch: Any,
    snapshot_dir: Path,
    arm: dict[str, Any],
    expected_pre_parameter_digest: str,
    expected_pre_buffer_digest: str,
) -> tuple[Any, dict[str, Any]]:
    from safetensors import safe_open
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        snapshot_dir,
        dtype=torch.float32,
        trust_remote_code=False,
        attn_implementation="sdpa",
        local_files_only=True,
    )
    named_parameters = list(model.named_parameters())
    named_buffers = list(model.named_buffers())
    _, pre_parameter_digest = named_tensor_hashes(named_parameters)
    _, pre_buffer_digest = named_tensor_hashes(named_buffers)

    update_path = Path(arm["parameter_updates"])
    with safe_open(update_path, framework="pt", device="cpu") as handle, torch.no_grad():
        parameter_names = [name for name, _ in named_parameters]
        if set(parameter_names) != set(handle.keys()):
            raise ValueError("saved update keys do not equal model parameter keys")
        for name, parameter in named_parameters:
            update = handle.get_tensor(name)
            if update.shape != parameter.shape or update.dtype != parameter.dtype:
                raise ValueError(f"saved update metadata mismatch for {name}")
            parameter.add_(update)

    _, post_parameter_digest = named_tensor_hashes(named_parameters)
    _, post_buffer_digest = named_tensor_hashes(named_buffers)
    checks = {
        "pre_parameter_digest_exact": pre_parameter_digest == expected_pre_parameter_digest,
        "pre_buffer_digest_exact": pre_buffer_digest == expected_pre_buffer_digest,
        "post_parameter_digest_exact": post_parameter_digest == arm["post_parameter_digest"],
        "post_buffer_digest_exact": post_buffer_digest == arm["post_buffer_digest"],
    }
    if not all(checks.values()):
        raise ValueError(f"post-state reconstruction failed: {checks}")
    return model, {
        "checks": checks,
        "pre_parameter_digest": pre_parameter_digest,
        "pre_buffer_digest": pre_buffer_digest,
        "post_parameter_digest": post_parameter_digest,
        "post_buffer_digest": post_buffer_digest,
    }


def evaluate_bank(
    torch: Any,
    model: Any,
    rows: list[dict[str, Any]],
    pad_token_id: int,
    batch_size: int,
) -> dict[str, Any]:
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from trl.trainer.grpo_trainer import selective_log_softmax

    model.eval()
    output_rows: list[dict[str, Any]] = []
    for offset in range(0, len(rows), batch_size):
        batch = rows[offset : offset + batch_size]
        max_length = max(len(row["input_ids"]) for row in batch)
        input_ids = torch.full(
            (len(batch), max_length), pad_token_id, dtype=torch.long, device="cuda"
        )
        attention_mask = torch.zeros_like(input_ids)
        target_mask = torch.zeros(
            (len(batch), max_length - 1), dtype=torch.bool, device="cuda"
        )
        for batch_index, row in enumerate(batch):
            length = len(row["input_ids"])
            input_ids[batch_index, :length] = torch.tensor(
                row["input_ids"], dtype=torch.long, device="cuda"
            )
            attention_mask[batch_index, :length] = 1
            target_start = row["prompt_token_count"] - 1
            target_stop = length - 1
            target_mask[batch_index, target_start:target_stop] = True

        with (
            torch.inference_mode(),
            torch.autocast("cuda", dtype=torch.float16),
            sdpa_kernel(SDPBackend.MATH),
        ):
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            ).logits[:, :-1, :]
            target_ids = input_ids[:, 1:]
            per_token_logps = selective_log_softmax(logits, target_ids)

        for batch_index, row in enumerate(batch):
            selected = per_token_logps[batch_index][target_mask[batch_index]].float()
            if selected.numel() != len(row["completion_token_ids"]):
                raise ValueError("completion mask length does not match frozen completion tokens")
            output_rows.append(
                {
                    "dataset_index": row["dataset_index"],
                    "completion_token_count": int(selected.numel()),
                    "completion_mean_nll": float((-selected).mean().item()),
                    "completion_logps": selected.cpu().tolist(),
                }
            )
        del logits, target_ids, per_token_logps, input_ids, attention_mask, target_mask

    mean_nll = math.fsum(row["completion_mean_nll"] for row in output_rows) / len(output_rows)
    return {
        "mean_nll": mean_nll,
        "rows": output_rows,
        "rows_digest": json_sha256(output_rows),
    }


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest).resolve()
    out_path = Path(args.out).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["schema_version"] != "forkcert.qwen3-common-evaluator-smoke-manifest.v0.1":
        raise ValueError("unsupported manifest schema")

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("PYTHONHASHSEED", "0")
    import torch
    from transformers import AutoTokenizer

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one visible CUDA device is required")
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False

    artifact_checks = validate_input_artifacts(manifest)
    if not all(artifact_checks.values()):
        raise ValueError(f"input artifact identity failed: {artifact_checks}")
    snapshot_dir = Path(manifest["state_scope"]["snapshot_dir"])
    tokenizer = AutoTokenizer.from_pretrained(snapshot_dir, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    bank = build_bank(tokenizer, manifest["endpoint"]["dataset"])
    bank_sha256 = bank_identity(bank)
    tokenizer_info = tokenizer_identity(snapshot_dir)

    evaluator = manifest["endpoint"]["evaluator"]
    repeats = int(evaluator["fresh_reconstructions_per_arm"])
    arm_results: dict[str, list[dict[str, Any]]] = {}
    reconstruction_results: dict[str, list[dict[str, Any]]] = {}
    for arm_name in ("reference", "candidate"):
        arm_results[arm_name] = []
        reconstruction_results[arm_name] = []
        for repeat in range(repeats):
            model, reconstruction = reconstruct_model(
                torch,
                snapshot_dir,
                manifest["arms"][arm_name],
                manifest["state_scope"]["pre_parameter_digest"],
                manifest["state_scope"]["pre_buffer_digest"],
            )
            model = model.to("cuda")
            result = evaluate_bank(
                torch,
                model,
                bank,
                tokenizer.pad_token_id,
                int(evaluator["batch_size"]),
            )
            result["repeat"] = repeat + 1
            arm_results[arm_name].append(result)
            reconstruction["repeat"] = repeat + 1
            reconstruction_results[arm_name].append(reconstruction)
            del model
            gc.collect()
            torch.cuda.empty_cache()

    repeat_checks = {
        arm_name: len({row["rows_digest"] for row in results}) == 1
        for arm_name, results in arm_results.items()
    }
    paired_shifts = [
        arm_results["candidate"][repeat]["mean_nll"]
        - arm_results["reference"][repeat]["mean_nll"]
        for repeat in range(repeats)
    ]
    valid = all(artifact_checks.values()) and all(repeat_checks.values())
    payload = {
        "schema_version": SCHEMA_VERSION,
        "valid": valid,
        "verdict": "VALID_SELECTED_STATE_T1B_SMOKE" if valid else "INVALID",
        "manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
            "status": manifest["status"],
        },
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
            "tokenizer": tokenizer_info,
            "bank_sha256": bank_sha256,
            "bank_size": len(bank),
            "bank_public_rows": [
                {
                    "dataset_index": row["dataset_index"],
                    "fields": row["fields"],
                    "prompt": row["prompt"],
                    "completion": row["completion"],
                    "input_ids": row["input_ids"],
                    "prompt_token_count": row["prompt_token_count"],
                    "completion_token_ids": row["completion_token_ids"],
                }
                for row in bank
            ],
        },
        "reconstruction": reconstruction_results,
        "endpoint": {
            "id": "T1b",
            "state_scope": manifest["state_scope"]["population_role"],
            "reference": arm_results["reference"],
            "candidate": arm_results["candidate"],
            "paired_candidate_minus_reference": paired_shifts,
            "mean_paired_shift": math.fsum(paired_shifts) / len(paired_shifts),
            "repeat_exact": repeat_checks,
            "interpretation": manifest["endpoint"]["paired_direction"],
        },
        "nonclaims": manifest["nonclaims"],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "verdict": payload["verdict"],
                "bank_sha256": bank_sha256,
                "reference_mean_nll": [x["mean_nll"] for x in arm_results["reference"]],
                "candidate_mean_nll": [x["mean_nll"] for x in arm_results["candidate"]],
                "paired_candidate_minus_reference": paired_shifts,
                "repeat_exact": repeat_checks,
            },
            indent=2,
            sort_keys=True,
        )
    )
    if not valid:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
