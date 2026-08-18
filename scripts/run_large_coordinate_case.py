#!/usr/bin/env python3
"""Run one large directional candidate with a streamed complete-coordinate Gram."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/data1/tzh/miniconda3/envs/pt_nightly/bin/python")
CONFIG = {
    "qwen": ("qwen", "/data1/tzh/models/Qwen/Qwen3-1.7B", False),
    "phi4": ("phi", "/data1/tzh/models/microsoft/Phi-4-mini-instruct", True),
    "mamba": ("mamba", "/data1/tzh/models/state-spaces/mamba-130m-hf", False),
    "deepseek8b": ("deepseek8", "/data1/tzh/models/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B", False),
}


def load(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def candidates(oracle: dict, threshold: int) -> list[tuple[str, int]]:
    positive = {str(row["task_id"]) for row in oracle["rows"]
                if row["verdict"] == "DIRECTIONAL_OPTIMIZATION_BIAS"}
    first = next(iter(oracle["states"].values()))["repeats"][0]["endpoint_metrics"]
    rows = []
    for task_id in positive:
        size = int(first[task_id]["error"]["directional_error_sketch"]["tensor_numel"])
        if size > threshold:
            rows.append((task_id, size))
    return sorted(rows, key=lambda row: (row[1], row[0]))


def canonical(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


def frozen_large_candidates(release: Path, threshold: int) -> list[tuple[str, int]]:
    manifest_path = release / f"large_candidate_manifest_gt{threshold}.json.gz"
    if manifest_path.exists():
        payload = load(manifest_path)
        if payload.get("threshold") != threshold:
            raise RuntimeError("large-candidate manifest threshold mismatch")
        return [(str(row["task_id"]), int(row["tensor_numel"]))
                for row in payload["rows"]]
    oracle_path = release / "same_dtype_oracle.json.gz"
    oracle = load(oracle_path)
    selected = candidates(oracle, threshold)
    payload = {
        "schema": "kernel-analyzer-large-candidate-size-manifest-v1",
        "threshold": threshold,
        "oracle_result_sha256": oracle["result_sha256"],
        "rows": [{"task_id": task_id, "tensor_numel": size}
                 for task_id, size in selected],
    }
    payload["result_sha256"] = canonical(payload)
    temporary = manifest_path.with_name("." + manifest_path.name + ".partial")
    with gzip.open(temporary, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
    temporary.replace(manifest_path)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-prefix", choices=tuple(CONFIG), required=True)
    parser.add_argument("--sequence-length", type=int, choices=(64, 128, 256), required=True)
    parser.add_argument("--candidate-index", type=int, required=True)
    parser.add_argument("--candidate-count", type=int, default=1)
    parser.add_argument("--max-batch-coordinates", type=int, default=700_000_000)
    parser.add_argument("--device-index", type=int, required=True)
    parser.add_argument("--small-threshold", type=int, default=4096)
    args = parser.parse_args()

    architecture, model, graph_breaks = CONFIG[args.model_prefix]
    cell = f"{args.model_prefix}_seq{args.sequence_length}_r1"
    release = ROOT / "results/coverage/runtime_releases" / cell
    selected = frozen_large_candidates(release, args.small_threshold)
    if not 0 <= args.candidate_index < len(selected) or args.candidate_count < 1:
        raise ValueError(f"candidate index outside [0,{len(selected)}): {args.candidate_index}")
    batch = selected[args.candidate_index:args.candidate_index + args.candidate_count]
    if len(batch) != args.candidate_count:
        raise ValueError("candidate batch extends past the frozen denominator")
    total_coordinates = sum(row[1] for row in batch)
    if total_coordinates > args.max_batch_coordinates:
        raise ValueError(
            f"batch has {total_coordinates} coordinates, above {args.max_batch_coordinates}"
        )
    token = hashlib.sha256("\0".join(row[0] for row in batch).encode()).hexdigest()[:16]
    result_dir = ROOT / "results/coverage/cases/full_coordinate/large" / cell
    stop = args.candidate_index + args.candidate_count
    output = result_dir / f"{args.candidate_index:04d}_{stop:04d}_{token}.json.gz"
    spool = Path(f"/data1/tzh/cache/kernel_analyzer_spool/{cell}/{args.candidate_index:04d}_{stop:04d}_{token}")
    cache = Path(f"/data1/tzh/cache/inductor/{cell}_large")
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
        "--output", str(output), "--reference-cache-dir", str(cache) + "_reference",
        "--states", "32", "--repeat", "2", "--sample-size", "4096",
        "--complete-coordinate-spool-dir", str(spool),
    ]
    for task_id, _coordinates in batch:
        command.extend(("--task-id", task_id))
    if graph_breaks:
        command.append("--allow-graph-breaks")
    print(json.dumps({"cell": cell, "candidate_index": args.candidate_index,
                      "candidate_count": len(batch), "large_candidate_count": len(selected),
                      "task_ids": [row[0] for row in batch],
                      "total_coordinates": total_coordinates,
                      "estimated_spool_bytes": total_coordinates * 32 * 4,
                      "output": str(output)}), flush=True)
    os.execve(command[0], command, environment)


if __name__ == "__main__":
    main()
