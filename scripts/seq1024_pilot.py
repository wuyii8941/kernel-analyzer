#!/usr/bin/env python3
"""Measure a real seq1024 Qwen3-1.7B training update before the long bank.

The pilot intentionally saves no model or gradient tensors.  Each successful
measurement includes forward, backward, and AdamW.step, so lazy optimizer state
and the true training-memory peak are included.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import random
import time
from pathlib import Path
from typing import Any


DATA_ROOT = Path("/data1/tzh").resolve()


def under_root(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if DATA_ROOT not in (resolved, *resolved.parents):
        raise ValueError(f"{label} must stay under {DATA_ROOT}: {resolved}")
    return resolved


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("/data1/tzh/models/Qwen/Qwen3-1.7B"))
    parser.add_argument("--output", type=Path, default=Path("results/final/seq1024_pilot.json"))
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--updates", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seq_len < 8 or args.updates < 1 or any(value < 1 for value in args.batch_sizes):
        raise ValueError("seq-len, updates, and batch sizes must be positive")
    model_path = under_root(args.model, "model")
    output_path = under_root(args.output, "output")

    os.environ.setdefault("HF_HOME", "/data1/tzh/cache/huggingface")
    os.environ.setdefault("HF_DATASETS_CACHE", "/data1/tzh/cache/huggingface/datasets")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/data1/tzh/cache/huggingface/transformers")
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/data1/tzh/cache/huggingface/hub")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import numpy as np
    import torch
    import transformers
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
    dataset = load_dataset(
        "Salesforce/wikitext",
        "wikitext-103-raw-v1",
        split="train",
        revision="main",
        download_mode="reuse_dataset_if_exists",
    )
    required = max(args.batch_sizes) * args.updates * args.seq_len
    token_ids: list[int] = []
    for row in dataset:
        text = str(row["text"]).strip()
        if text:
            token_ids.extend(tokenizer(text, add_special_tokens=False, return_attention_mask=False)["input_ids"])
        if len(token_ids) >= required:
            break
    if len(token_ids) < required:
        raise RuntimeError(f"token stream too short: {len(token_ids)} < {required}")

    result: dict[str, Any] = {
        "schema": "kernel-analyzer-seq1024-training-pilot-v1",
        "purpose": "choose a feasible long-horizon single-process F+B update configuration",
        "candidate_tensor_values_saved": False,
        "model": str(model_path),
        "seq_len": args.seq_len,
        "dtype": "bfloat16",
        "attention_implementation": "sdpa",
        "causal_label_alignment": "labels_equal_input_ids; exactly one internal model shift",
        "optimizer": {"name": "AdamW", "lr": args.lr, "betas": [0.9, 0.95], "weight_decay": 0.0, "foreach": False},
        "updates_per_batch_size": args.updates,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "gpu": torch.cuda.get_device_name(device),
            "gpu_total_bytes": torch.cuda.get_device_properties(device).total_memory,
        },
        "rows": [],
        "status": "RUNNING",
    }
    atomic_json(output_path, result)

    for batch_size in args.batch_sizes:
        print(f"pilot batch={batch_size}: loading model", flush=True)
        row: dict[str, Any] = {"batch_size": batch_size, "status": "RUNNING", "updates": []}
        result["rows"].append(row)
        atomic_json(output_path, result)
        model = None
        optimizer = None
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                dtype=torch.bfloat16,
                attn_implementation="sdpa",
                local_files_only=True,
            ).to(device)
            model.config.use_cache = False
            model.train()
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0, foreach=False
            )
            row["trainable_parameters"] = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
            row["allocated_after_load_bytes"] = torch.cuda.memory_allocated(device)
            row["reserved_after_load_bytes"] = torch.cuda.memory_reserved(device)

            for update in range(args.updates):
                offset = update * batch_size * args.seq_len
                blocks = [
                    token_ids[offset + index * args.seq_len : offset + (index + 1) * args.seq_len]
                    for index in range(batch_size)
                ]
                batch = torch.tensor(blocks, dtype=torch.long, device=device)
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize(device)
                started = time.perf_counter()
                # AutoModelForCausalLM performs the one-token label shift
                # internally.  Supplying pre-shifted labels would accidentally
                # train a two-token target and is therefore forbidden here.
                output = model(input_ids=batch, labels=batch, use_cache=False, return_dict=True)
                loss = output.loss
                loss.backward()
                finite_gradients = all(
                    parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
                    for parameter in model.parameters()
                )
                optimizer.step()
                torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - started
                loss_value = float(loss.detach().cpu())
                measurement = {
                    "update": update + 1,
                    "loss": loss_value,
                    "loss_finite": bool(torch.isfinite(loss.detach())),
                    "gradients_finite": finite_gradients,
                    "seconds": elapsed,
                    "tokens_per_second": batch_size * args.seq_len / elapsed,
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                }
                row["updates"].append(measurement)
                print(json.dumps({"batch": batch_size, **measurement}), flush=True)
                atomic_json(output_path, result)
                del output, loss, batch

            row["status"] = "OK"
            measured = row["updates"][1:] if len(row["updates"]) > 1 else row["updates"]
            row["steady_state_seconds_mean"] = sum(item["seconds"] for item in measured) / len(measured)
            row["steady_state_tokens_per_second_mean"] = sum(item["tokens_per_second"] for item in measured) / len(measured)
            row["peak_allocated_bytes"] = max(item["peak_allocated_bytes"] for item in row["updates"])
            row["peak_reserved_bytes"] = max(item["peak_reserved_bytes"] for item in row["updates"])
        except torch.cuda.OutOfMemoryError as error:
            row["status"] = "OOM"
            row["error"] = str(error).splitlines()[0]
            print(f"pilot batch={batch_size}: OOM", flush=True)
        finally:
            if optimizer is not None:
                del optimizer
            if model is not None:
                del model
            gc.collect()
            torch.cuda.empty_cache()
            atomic_json(output_path, result)
        if row["status"] == "OOM":
            break

    successful = [row for row in result["rows"] if row["status"] == "OK"]
    result["status"] = "COMPLETE" if successful else "NO_FEASIBLE_BATCH"
    result["selected_batch_size"] = max((row["batch_size"] for row in successful), default=None)
    result["selection_rule"] = "largest tested batch completing all real F+B+AdamW updates before first OOM"
    result["all_successful_values_finite"] = all(
        update["loss_finite"] and update["gradients_finite"]
        for row in successful
        for update in row["updates"]
    )
    atomic_json(output_path, result)
    print(json.dumps({"status": result["status"], "selected_batch_size": result["selected_batch_size"]}), flush=True)


if __name__ == "__main__":
    main()
