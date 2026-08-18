#!/usr/bin/env python3
"""Resume one frozen full-coordinate case-audit batch for a model/shape cell."""

from __future__ import annotations

import argparse
import gzip
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/data1/tzh/miniconda3/envs/pt_nightly/bin/python")
CONFIG = {
    "qwen": ("qwen", "/data1/tzh/models/Qwen/Qwen3-1.7B", False),
    "phi4": ("phi", "/data1/tzh/models/microsoft/Phi-4-mini-instruct", True),
    "mamba": ("mamba", "/data1/tzh/models/state-spaces/mamba-130m-hf", False),
    "deepseek8b": (
        "deepseek8", "/data1/tzh/models/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B", False
    ),
}


def load_gzip(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def frozen_tasks(oracle_path: Path, output: Path, max_numel: int) -> list[str]:
    partial = output.with_name("." + output.name + ".partial")
    if partial.exists():
        checkpoint = load_gzip(partial)
        task_ids = checkpoint.get("bindings", {}).get("target_task_ids")
        if task_ids:
            return [str(value) for value in task_ids]
    oracle = load_gzip(oracle_path)
    positive = {
        str(row["task_id"]) for row in oracle["rows"]
        if row["verdict"] == "DIRECTIONAL_OPTIMIZATION_BIAS"
    }
    first = next(iter(oracle["states"].values()))["repeats"][0]["endpoint_metrics"]
    return sorted(
        task_id for task_id in positive
        if int(first[task_id]["error"]["directional_error_sketch"]["tensor_numel"])
        <= max_numel
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-prefix", choices=tuple(CONFIG), required=True)
    parser.add_argument("--sequence-length", type=int, choices=(64, 128, 256), required=True)
    parser.add_argument("--device-index", type=int, required=True)
    parser.add_argument("--max-numel", type=int, default=4096)
    args = parser.parse_args()

    architecture, model, graph_breaks = CONFIG[args.model_prefix]
    cell = f"{args.model_prefix}_seq{args.sequence_length}_r1"
    release = ROOT / "results/coverage/runtime_releases" / cell
    output = ROOT / "results/coverage/cases/full_coordinate" / (
        f"{args.model_prefix}_seq{args.sequence_length}_small.json.gz"
    )
    tasks = frozen_tasks(release / "same_dtype_oracle.json.gz", output, args.max_numel)
    if not tasks:
        raise RuntimeError(f"no frozen candidates selected for {cell}")

    cache = Path(f"/data1/tzh/cache/inductor/{args.model_prefix}_seq{args.sequence_length}_fullcoord")
    reference_cache = Path(str(cache) + "_reference")
    environment = os.environ.copy()
    environment.update({
        "CUDA_VISIBLE_DEVICES": str(args.device_index),
        "HF_HOME": "/data1/tzh/cache/huggingface",
        "HUGGINGFACE_HUB_CACHE": "/data1/tzh/cache/huggingface/hub",
        "XDG_CACHE_HOME": "/data1/tzh/cache/xdg",
        "TORCHINDUCTOR_CACHE_DIR": str(cache),
    })
    command = [
        str(PYTHON), str(ROOT / "scripts/run_same_dtype_semantic_oracle.py"),
        "--architecture", architecture, "--model", model,
        "--input-bank", str(ROOT / f"results/coverage/{args.model_prefix}_seq{args.sequence_length}_input_bank.json"),
        "--campaign", str(release / "campaign.json.gz"),
        "--inventory", str(release / "inventory.json.gz"),
        "--task-plan", str(release / "same_dtype_tasks.json.gz"),
        "--output", str(output), "--reference-cache-dir", str(reference_cache),
        "--states", "32", "--repeat", "2", "--sample-size", str(args.max_numel),
    ]
    if graph_breaks:
        command.append("--allow-graph-breaks")
    for task_id in tasks:
        command.extend(("--task-id", task_id))
    os.execve(command[0], command, environment)


if __name__ == "__main__":
    main()
