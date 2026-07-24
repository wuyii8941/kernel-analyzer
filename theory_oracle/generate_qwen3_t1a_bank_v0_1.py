#!/usr/bin/env python
"""Generate and score one frozen baseline-anchored Qwen3 T1a rollout bank."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import re
from pathlib import Path
from typing import Any

import numpy as np

try:
    from theory_oracle.qwen3_grpo_natural_transition_v0_2 import (
        json_sha256,
        named_tensor_hashes,
        sha256_file,
        tensor_sha256,
    )
except ModuleNotFoundError:  # Direct script execution from theory_oracle/.
    from qwen3_grpo_natural_transition_v0_2 import (
        json_sha256,
        named_tensor_hashes,
        sha256_file,
        tensor_sha256,
    )


SCHEMA_VERSION = "forkcert.qwen3-t1a-frozen-bank.v0.1"
NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--repeat", type=int, required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def derive_seed(payload: str) -> tuple[int, str]:
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF, digest.hex()


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


def numeric_reward(text: str, expected: float) -> tuple[float, float | None, bool]:
    predictions = NUMBER_RE.findall(text)
    if not predictions:
        return -1.0, None, False
    try:
        predicted = float(predictions[-1].replace(",", ""))
    except ValueError:
        return -1.0, None, False
    error = abs(predicted - expected)
    scale = max(1.0, abs(expected))
    exact = math.isclose(predicted, expected, rel_tol=1e-6, abs_tol=1e-6)
    return 1.0 / (1.0 + error / scale) + (1.0 if exact else 0.0), predicted, exact


def group_advantages(torch: Any, rewards: list[float]) -> list[float]:
    values = torch.tensor(rewards, dtype=torch.float32)
    centered = values - values.mean()
    std = values.std(correction=1) if values.numel() > 1 else torch.zeros((), dtype=torch.float32)
    return (centered / (std + 1e-4)).tolist()


def rng_fingerprint(torch: Any) -> dict[str, Any]:
    return {
        "torch_cpu": tensor_sha256(torch.random.get_rng_state()),
        "torch_cuda": [tensor_sha256(value) for value in torch.cuda.get_rng_state_all()],
        "python": json_sha256(random.getstate()),
        "numpy": json_sha256(np.random.get_state()),
    }


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


def score_completions(
    torch: Any,
    model: Any,
    prompt_ids: Any,
    prompt_mask: Any,
    completion_ids: Any,
    completion_mask: Any,
) -> Any:
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from trl.trainer.grpo_trainer import selective_log_softmax

    input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
    attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
    with (
        torch.inference_mode(),
        torch.autocast("cuda", dtype=torch.float16),
        sdpa_kernel(SDPBackend.MATH),
    ):
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            logits_to_keep=completion_ids.size(1) + 1,
            use_cache=False,
        )
        logits = outputs.logits[:, :-1, :]
        logits = logits[:, -completion_ids.size(1) :, :]
        logps = selective_log_softmax(logits, completion_ids)
    return logps.float()


def main() -> None:
    args = parse_args()
    if args.repeat <= 0:
        raise ValueError("repeat must be positive")
    manifest_path = Path(args.manifest).resolve()
    out_path = Path(args.out).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["schema_version"] not in {
        "forkcert.qwen3-t1a-selected-state-smoke-manifest.v0.1",
        "forkcert.qwen3-calibration-state-endpoint-manifest.v0.1",
    }:
        raise ValueError("unsupported manifest schema")

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("PYTHONHASHSEED", "0")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one visible CUDA device is required")
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False

    state = manifest["state_scope"]
    bank_config = manifest["bank"]
    snapshot_dir = Path(state["snapshot_dir"])
    seed, seed_digest = derive_seed(bank_config["seed_payload"])
    input_checks = {
        "snapshot_metadata_sha256_exact": sha256_file(
            snapshot_dir / "forkcert_transition_snapshot.json"
        )
        == state["snapshot_metadata_sha256"],
        "seed_payload_sha256_exact": seed_digest == bank_config["seed_payload_sha256"],
        "derived_seed_exact": seed == bank_config["seed"],
    }
    if not all(input_checks.values()):
        raise ValueError(f"input identity failed: {input_checks}")

    tokenizer = AutoTokenizer.from_pretrained(snapshot_dir, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        snapshot_dir,
        dtype=torch.float32,
        trust_remote_code=False,
        attn_implementation="sdpa",
        local_files_only=True,
    )
    _, pre_parameter_digest = named_tensor_hashes(list(model.named_parameters()))
    _, pre_buffer_digest = named_tensor_hashes(list(model.named_buffers()))
    state_checks = {
        "pre_parameter_digest_exact": pre_parameter_digest == state["pre_parameter_digest"],
        "pre_buffer_digest_exact": pre_buffer_digest == state["pre_buffer_digest"],
    }
    if not all(state_checks.values()):
        raise ValueError(f"pre-state identity failed: {state_checks}")
    model = model.to("cuda")
    model.eval()
    model.config.use_cache = False

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    rng_before = rng_fingerprint(torch)

    generation = bank_config["generation"]
    rows: list[dict[str, Any]] = []
    for group_index, dataset_index in enumerate(bank_config["prompt_indices"]):
        fields = arithmetic_fields(dataset_index)
        prompt = bank_config["prompt_template"].format(**fields)
        group_size = int(bank_config["num_generations_per_prompt"])
        encoded = tokenizer(
            [prompt] * group_size,
            return_tensors="pt",
            padding=True,
        ).to("cuda")
        prompt_width = int(encoded["input_ids"].shape[1])
        with (
            torch.inference_mode(),
            torch.autocast("cuda", dtype=torch.float16),
            sdpa_kernel(SDPBackend.MATH),
        ):
            generated = model.generate(
                **encoded,
                do_sample=bool(generation["do_sample"]),
                temperature=float(generation["temperature"]),
                top_p=float(generation["top_p"]),
                top_k=int(generation["top_k"]),
                repetition_penalty=float(generation["repetition_penalty"]),
                min_new_tokens=int(generation["min_new_tokens"]),
                max_new_tokens=int(generation["max_new_tokens"]),
                use_cache=bool(generation["use_cache"]),
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        completion_ids = generated[:, prompt_width:]
        if completion_ids.shape != (group_size, int(generation["max_new_tokens"])):
            raise ValueError(
                f"unexpected completion shape for group {group_index}: {tuple(completion_ids.shape)}"
            )
        completion_mask = torch.ones_like(completion_ids)
        old_logps = score_completions(
            torch,
            model,
            encoded["input_ids"],
            encoded["attention_mask"],
            completion_ids,
            completion_mask,
        )
        decoded = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        reward_rows = [numeric_reward(text, float(fields["result"])) for text in decoded]
        rewards = [value[0] for value in reward_rows]
        advantages = group_advantages(torch, rewards)
        for generation_index in range(group_size):
            valid_prompt_ids = encoded["input_ids"][generation_index][
                encoded["attention_mask"][generation_index].bool()
            ]
            reward, predicted, exact = reward_rows[generation_index]
            rows.append(
                {
                    "group_index": group_index,
                    "dataset_index": dataset_index,
                    "generation_index": generation_index,
                    "fields": fields,
                    "prompt": prompt,
                    "prompt_ids": valid_prompt_ids.cpu().tolist(),
                    "completion_ids": completion_ids[generation_index].cpu().tolist(),
                    "completion_mask": completion_mask[generation_index].cpu().tolist(),
                    "completion_text": decoded[generation_index],
                    "old_per_token_logps": old_logps[generation_index].cpu().tolist(),
                    "reward": reward,
                    "predicted": predicted,
                    "exact": exact,
                    "advantage": advantages[generation_index],
                }
            )
        del encoded, generated, completion_ids, completion_mask, old_logps

    torch.cuda.synchronize()
    rng_after = rng_fingerprint(torch)
    bank_content = {
        "state_identity": {
            "query_id": state["query_id"],
            "trajectory_id": state["trajectory_id"],
            "state_id": state["state_id"],
            "pre_parameter_digest": pre_parameter_digest,
            "pre_buffer_digest": pre_buffer_digest,
        },
        "generation": bank_config,
        "rows": rows,
    }
    bank_sha256 = json_sha256(bank_content)
    valid = all(input_checks.values()) and all(state_checks.values()) and len(rows) == 32
    payload = {
        "schema_version": SCHEMA_VERSION,
        "valid": valid,
        "verdict": "VALID_FROZEN_T1A_BANK" if valid else "INVALID",
        "repeat": args.repeat,
        "manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_build": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "evaluator_code_sha256": sha256_file(Path(__file__).resolve()),
        },
        "checks": {**input_checks, **state_checks},
        "tokenizer": tokenizer_identity(snapshot_dir),
        "rng": {"before_generation": rng_before, "after_bank": rng_after},
        "bank_sha256": bank_sha256,
        "bank": bank_content,
        "summary": {
            "rows": len(rows),
            "exact_reward_count": sum(bool(row["exact"]) for row in rows),
            "mean_reward": math.fsum(float(row["reward"]) for row in rows) / len(rows),
            "zero_advantage_groups": sum(
                all(rows[group * 4 + offset]["advantage"] == 0.0 for offset in range(4))
                for group in range(8)
            ),
        },
        "nonclaims": manifest["nonclaims"],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "verdict": payload["verdict"],
                "repeat": args.repeat,
                "bank_sha256": bank_sha256,
                "summary": payload["summary"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if not valid:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
