#!/usr/bin/env python3
"""Measure one real LM F+B peak before admitting a model to full capture."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from resource_preflight import resource_report


def require_data_paths(*paths: Path) -> None:
    root = Path("/data1/tzh")
    for path in paths:
        if not path.resolve().is_relative_to(root):
            raise RuntimeError(f"path outside /data1/tzh: {path}")


def run_arm(
    model_path: Path, ids_cpu: torch.Tensor, dtype: torch.dtype,
    device: str, shard_gpus: int,
) -> dict[str, object]:
    devices = [f"cuda:{index}" for index in range(shard_gpus)]
    torch.cuda.empty_cache()
    for item in devices:
        torch.cuda.reset_peak_memory_stats(item)
    memory_before = [torch.cuda.mem_get_info(item) for item in devices]
    start = time.monotonic()
    load_kwargs: dict[str, object] = {}
    if shard_gpus > 1:
        load_kwargs["device_map"] = "balanced"
        load_kwargs["max_memory"] = {
            index: f"{int(0.9 * memory_before[index][1] / 2**30)}GiB"
            for index in range(shard_gpus)
        }
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=dtype,
        local_files_only=True,
        trust_remote_code=False,
        attn_implementation="eager",
        **load_kwargs,
    )
    if shard_gpus == 1:
        model = model.to(device)
    model = model.train()
    model.config.use_cache = False
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    model.zero_grad(set_to_none=True)
    ids = ids_cpu.unsqueeze(0).to(device)
    loss = model(input_ids=ids, labels=ids, use_cache=False).loss
    if loss is None or not torch.isfinite(loss):
        raise RuntimeError("nonfinite or absent loss")
    loss.backward()
    present = 0
    finite = True
    for parameter in model.parameters():
        if parameter.grad is not None:
            present += 1
            finite = finite and bool(torch.isfinite(parameter.grad).all())
    for item in devices:
        torch.cuda.synchronize(item)
    per_gpu = [{
        "device": item,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(item) / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(item) / 2**30,
        "device_total_gib": memory_before[index][1] / 2**30,
        "device_free_before_gib": memory_before[index][0] / 2**30,
    } for index, item in enumerate(devices)]
    row = {
        "dtype": str(dtype),
        "loss": float(loss.detach().cpu()),
        "parameter_count": parameter_count,
        "parameter_gradients_present": present,
        "all_gradients_finite": finite,
        "peak_allocated_gib": max(row["peak_allocated_gib"] for row in per_gpu),
        "peak_reserved_gib": max(row["peak_reserved_gib"] for row in per_gpu),
        "device_total_gib": min(row["device_total_gib"] for row in per_gpu),
        "device_free_before_gib": min(row["device_free_before_gib"] for row in per_gpu),
        "per_gpu": per_gpu,
        "wall_seconds": time.monotonic() - start,
    }
    del loss, model
    gc.collect()
    torch.cuda.empty_cache()
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input-bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--parameter-estimate", type=int, required=True)
    parser.add_argument("--state", type=int, default=0)
    parser.add_argument("--shard-gpus", type=int, default=1)
    args = parser.parse_args()
    require_data_paths(args.model, args.input_bank, args.output)
    bank = json.loads(args.input_bank.read_text())
    state = bank["states"][args.state]
    ids_cpu = torch.tensor(state["token_ids"], dtype=torch.long)
    if hashlib.sha256(ids_cpu.numpy().tobytes()).hexdigest() != state["token_sha256"]:
        raise RuntimeError("input digest mismatch")
    preflight = resource_report(
        args.parameter_estimate, ids_cpu.numel(), 2, 500, args.shard_gpus
    )
    if not preflight["launch_allowed"]:
        raise RuntimeError(f"resource preflight failed: {preflight['failures']}")
    torch.manual_seed(41000 + args.state)
    torch.cuda.manual_seed_all(41000 + args.state)
    torch.backends.cuda.matmul.allow_tf32 = False
    rows = [
        run_arm(args.model, ids_cpu, torch.float32, args.device, args.shard_gpus),
        run_arm(args.model, ids_cpu, torch.bfloat16, args.device, args.shard_gpus),
    ]
    payload = {
        "schema": "kernel-analyzer-model-fb-memory-smoke-v1",
        "model": str(args.model.resolve()),
        "input_bank": str(args.input_bank.resolve()),
        "seq_len": ids_cpu.numel(),
        "state_id": args.state,
        "preflight": preflight,
        "rows": rows,
        "admission": {
            "all_finite": all(row["all_gradients_finite"] for row in rows),
            "peak_reserved_below_85pct": all(
                row["peak_reserved_gib"] < 0.85 * row["device_total_gib"] for row in rows
            ),
        },
        "tensor_values_saved": False,
    }
    payload["admitted"] = all(payload["admission"].values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "admitted": payload["admitted"], "rows": rows}))
    if not payload["admitted"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
