#!/usr/bin/env python3
"""Fail-closed storage, host-memory, and GPU-memory preflight.

All large or regenerable state must live below /data1/tzh.  Runners may import
``resource_report`` before loading a model and persist the returned compact
record beside their experiment design.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import torch


DATA_ROOT = Path("/data1/tzh").resolve()
CACHE_ENV = (
    "HF_HOME",
    "HUGGINGFACE_HUB_CACHE",
    "TRANSFORMERS_CACHE",
    "TORCHINDUCTOR_CACHE_DIR",
    "TRITON_CACHE_DIR",
    "XDG_CACHE_HOME",
    "PIP_CACHE_DIR",
    "TMPDIR",
)


def gib(value: int) -> float:
    return value / 2**30


def resident_memory() -> tuple[int, int]:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, raw = line.split(":", 1)
        values[key] = int(raw.strip().split()[0]) * 1024
    return values["MemTotal"], values["MemAvailable"]


def required_cache_paths() -> dict[str, str]:
    defaults = {
        "HF_HOME": DATA_ROOT / "cache/huggingface",
        "HUGGINGFACE_HUB_CACHE": DATA_ROOT / "cache/huggingface/hub",
        "TRANSFORMERS_CACHE": DATA_ROOT / "cache/huggingface/transformers",
        "TORCHINDUCTOR_CACHE_DIR": DATA_ROOT / "cache/torchinductor",
        "TRITON_CACHE_DIR": DATA_ROOT / "cache/triton",
        "XDG_CACHE_HOME": DATA_ROOT / "cache/xdg",
        "PIP_CACHE_DIR": DATA_ROOT / "cache/pip",
        "TMPDIR": DATA_ROOT / "cache/tmp",
    }
    return {name: str(Path(os.environ.get(name, value)).resolve()) for name, value in defaults.items()}


def parameter_budget(parameter_count: int, seq_len: int, dtype_bytes: int) -> dict[str, float]:
    # This is a launch gate, not a profiler.  The activation reserve is
    # deliberately conservative and must later be replaced by a measured peak.
    weights = parameter_count * dtype_bytes
    gradients = parameter_count * dtype_bytes
    master_or_reference = parameter_count * 4
    activation_reserve = max(4 * 2**30, int(parameter_count * dtype_bytes * min(seq_len / 256, 1.0) * 0.35))
    compiler_reserve = 6 * 2**30
    return {
        "weights_gib": gib(weights),
        "gradients_gib": gib(gradients),
        "fp32_reference_or_master_gib": gib(master_or_reference),
        "activation_reserve_gib": gib(activation_reserve),
        "compiler_reserve_gib": gib(compiler_reserve),
        "single_gpu_low_precision_peak_gib": gib(weights + gradients + activation_reserve + compiler_reserve),
        "aggregate_fp32_eager_peak_gib": gib(2 * master_or_reference + activation_reserve),
    }


def resource_report(
    parameter_count: int,
    seq_len: int,
    dtype_bytes: int,
    min_data_free_gib: int,
    shard_gpus: int = 1,
) -> dict[str, object]:
    cache_paths = required_cache_paths()
    bad_paths = {name: value for name, value in cache_paths.items() if not Path(value).is_relative_to(DATA_ROOT)}
    data_usage = shutil.disk_usage(DATA_ROOT)
    mem_total, mem_available = resident_memory()
    gpus = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(index)
            gpus.append({"device": index, "free_gib": gib(free), "total_gib": gib(total)})
    budget = parameter_budget(parameter_count, seq_len, dtype_bytes)
    weight_grad = budget["weights_gib"] + budget["gradients_gib"]
    per_gpu_low = weight_grad / shard_gpus + budget["activation_reserve_gib"] + budget["compiler_reserve_gib"]
    per_gpu_fp32 = (
        2 * budget["fp32_reference_or_master_gib"] / shard_gpus
        + budget["activation_reserve_gib"]
    )
    failures = []
    if bad_paths:
        failures.append("CACHE_PATH_OUTSIDE_DATA_ROOT")
    if gib(data_usage.free) < min_data_free_gib:
        failures.append("DATA_DISK_FREE_BELOW_FLOOR")
    if not gpus:
        failures.append("CUDA_UNAVAILABLE")
    elif shard_gpus > len(gpus):
        failures.append("INSUFFICIENT_VISIBLE_GPUS")
    else:
        safe_capacity = min(gpu["total_gib"] for gpu in gpus[:shard_gpus]) * 0.85
        if per_gpu_low > safe_capacity:
            failures.append("ESTIMATED_LOW_PRECISION_PEAK_EXCEEDS_GPU_GATE")
        if per_gpu_fp32 > safe_capacity:
            failures.append("ESTIMATED_FP32_PEAK_EXCEEDS_GPU_GATE")
    if gib(mem_available) < budget["aggregate_fp32_eager_peak_gib"] * 1.25:
        failures.append("HOST_MEMORY_HEADROOM_BELOW_GATE")
    report = {
        "schema": "kernel-analyzer-resource-preflight-v1",
        "data_root": str(DATA_ROOT),
        "cache_paths": cache_paths,
        "bad_cache_paths": bad_paths,
        "data_free_gib": gib(data_usage.free),
        "host_memory_total_gib": gib(mem_total),
        "host_memory_available_gib": gib(mem_available),
        "gpus": gpus,
        "model_budget": {
            "parameter_count": parameter_count,
            "seq_len": seq_len,
            "dtype_bytes": dtype_bytes,
            **budget,
            "shard_gpus": shard_gpus,
            "per_gpu_low_precision_peak_gib": per_gpu_low,
            "per_gpu_fp32_eager_peak_gib": per_gpu_fp32,
        },
        "failures": failures,
        "launch_allowed": not failures,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parameters", type=int, required=True)
    parser.add_argument("--seq-len", type=int, required=True)
    parser.add_argument("--dtype-bytes", type=int, choices=(2, 4), required=True)
    parser.add_argument("--min-data-free-gib", type=int, default=500)
    parser.add_argument("--shard-gpus", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = resource_report(
        args.parameters, args.seq_len, args.dtype_bytes,
        args.min_data_free_gib, args.shard_gpus,
    )
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")
    if not report["launch_allowed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
