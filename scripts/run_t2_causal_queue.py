#!/usr/bin/env python3
"""Exhaust T2 sham/repair reachability for every full-coordinate T1 survivor."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.update_full_coordinate_audit import load_large_oracle_metadata  # noqa: E402

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


def compact_large_t1(path: Path, cell: str) -> tuple[Path, dict]:
    """Stage only T1 rows for a huge shard; retain the original result hash."""
    cache = Path("/data1/tzh/cache/kernel_analyzer/t1_compact") / cell
    cache.mkdir(parents=True, exist_ok=True)
    staged = cache / (path.name + ".rows.json.gz")
    if staged.exists():
        return staged, load(staged)
    metadata = load_large_oracle_metadata(path)
    payload = {
        "schema": metadata["schema"], "status": metadata["status"],
        "architecture": metadata["architecture"],
        "sequence_length": metadata["sequence_length"],
        "result_sha256": metadata["result_sha256"], "rows": metadata["rows"],
    }
    temporary = staged.with_name("." + staged.name + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
    temporary.replace(staged)
    return staged, payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-prefix", choices=tuple(CONFIG), required=True)
    parser.add_argument("--sequence-length", choices=(64, 128, 256), type=int, required=True)
    parser.add_argument("--device-index", type=int, required=True)
    parser.add_argument(
        "--input-bank", type=Path,
        help="override the frozen input-bank path when the canonical copy moved; "
             "its SHA-256 must still match the release capture",
    )
    parser.add_argument(
        "--artifact-name",
        help="process only the full-coordinate artifact with this basename; "
             "useful for running independent shards on separate GPUs",
    )
    parser.add_argument(
        "--reference-cache-dir", type=Path,
        help="override the T2 reference code cache; only use a cell-specific validated cache",
    )
    parser.add_argument(
        "--reuse-reference-cache", action="store_true",
        help="allow TorchInductor to reuse the selected reference code cache",
    )
    # Model/AOT compilation dominates startup, while T2 executes each repair
    # arm independently and retains only one selected carrier per task after
    # discovery.  A 32-task batch therefore removes repeated compilation
    # without weakening the denominator or materially increasing GPU memory.
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")

    architecture, model, graph_breaks = CONFIG[args.model_prefix]
    cell = f"{args.model_prefix}_seq{args.sequence_length}_r1"
    input_bank = args.input_bank or (
        ROOT / f"results/coverage/{args.model_prefix}_seq{args.sequence_length}_input_bank.json"
    )
    root = ROOT / "results/coverage/cases/full_coordinate"
    artifacts = [root / f"{args.model_prefix}_seq{args.sequence_length}_small.json.gz"]
    artifacts += sorted((root / "large" / cell).glob("*.json.gz"))
    artifacts = [path for path in artifacts if path.exists()]
    if args.artifact_name:
        artifacts = [path for path in artifacts if path.name == args.artifact_name]
        if not artifacts:
            raise FileNotFoundError(f"no full-coordinate artifact named {args.artifact_name!r}")
    out_dir = ROOT / "results/coverage/cases/causal" / cell
    out_dir.mkdir(parents=True, exist_ok=True)
    completed: set[str] = set()
    for path in out_dir.glob("*.json.gz"):
        payload = load(path)
        if payload.get("status") in {
            "COMPLETE_T2_T3_BATCH", "COMPLETE_T2_CAUSAL_REACH_BATCH"
        }:
            completed.update(str(row["task_id"]) for row in payload.get("rows", []))

    environment = os.environ.copy()
    environment.update({
        "CUDA_VISIBLE_DEVICES": str(args.device_index),
        "HF_HOME": "/data1/tzh/cache/huggingface",
        "HUGGINGFACE_HUB_CACHE": "/data1/tzh/cache/huggingface/hub",
        "XDG_CACHE_HOME": "/data1/tzh/cache/xdg",
        # The candidate arm must load the byte-identical frozen release.  The
        # worker switches to a separate reference cache only after validating
        # these wrappers.
        "TORCHINDUCTOR_CACHE_DIR": f"/data1/tzh/cache/torchinductor/frozen/{cell}",
    })
    for artifact in artifacts:
        t1_path = artifact
        if artifact.stat().st_size > 500_000_000:
            t1_path, t1 = compact_large_t1(artifact, cell)
        else:
            t1 = load(artifact)
        if t1.get("schema") != "kernel-analyzer-same-dtype-semantic-oracle-v1":
            continue
        survivors = [str(row["task_id"]) for row in t1["rows"]
                     if row["verdict"] == "DIRECTIONAL_OPTIMIZATION_BIAS"
                     and str(row["task_id"]) not in completed]
        for offset in range(0, len(survivors), args.batch_size):
            batch = survivors[offset:offset + args.batch_size]
            token = hashlib.sha256((t1["result_sha256"] + "\0" + "\0".join(batch)).encode()).hexdigest()[:16]
            output = out_dir / f"{artifact.stem.replace('.json','')}_{offset:04d}_{token}.json.gz"
            command = [
                str(PYTHON), str(ROOT / "scripts/run_same_dtype_causal_repair_batch.py"),
                "--architecture", architecture, "--model", model,
                "--input-bank", str(input_bank),
                "--release-dir", str(ROOT / "results/coverage/runtime_releases" / cell),
                "--full-coordinate-t1", str(t1_path), "--output", str(output),
                "--reference-cache-dir", str(
                    args.reference_cache_dir
                    or Path(f"/data1/tzh/cache/inductor/{cell}_t2_reference")
                ),
            ]
            if graph_breaks:
                command.append("--allow-graph-breaks")
            if args.reuse_reference_cache:
                command.append("--reuse-reference-cache")
            for task_id in batch:
                command.extend(("--task-id", task_id))
            print(json.dumps({"event": "T2_BATCH_START", "cell": cell,
                              "artifact": str(artifact), "tasks": batch}), flush=True)
            try:
                subprocess.run(command, cwd=ROOT, env=environment, check=True)
            except subprocess.CalledProcessError:
                # A worker can persist a complete fail-closed artifact before
                # raising on a terminal task (for example, no concrete
                # parameter carrier).  Accept only an artifact that is
                # readable, has the expected schema, and covers every task in
                # this batch.  Release/identity failures normally produce no
                # valid artifact and must still fail hard.
                if not output.exists():
                    raise
                try:
                    recovered = load(output)
                except Exception:
                    raise
                recovered_ids = {str(row["task_id"]) for row in recovered.get("rows", [])}
                expected_ids = set(batch)
                if recovered.get("schema") not in {
                    "kernel-analyzer-same-dtype-causal-repair-batch-v2",
                } or not expected_ids.issubset(recovered_ids):
                    raise
                print(json.dumps({
                    "event": "T2_BATCH_RECOVERED_FROM_COMPLETE_ARTIFACT",
                    "cell": cell, "output": str(output),
                    "task_count": len(expected_ids),
                }), flush=True)
            completed.update(batch)
    print(json.dumps({"event": "T2_QUEUE_COMPLETE", "cell": cell,
                      "completed": len(completed),
                      "artifact_name": args.artifact_name}), flush=True)
    # An artifact-scoped run is only a shard checkpoint.  It must not create
    # the cell-level marker consumed by T3, otherwise T3 could start while a
    # second T2 denominator shard is still pending.
    marker = (
        out_dir / f"shard_complete_{Path(args.artifact_name).stem}.json"
        if args.artifact_name else out_dir / "queue_complete.json"
    )
    marker.write_text(json.dumps({
        "schema": "kernel-analyzer-t2-causal-shard-complete-v1"
        if args.artifact_name else "kernel-analyzer-t2-causal-queue-complete-v1",
        "cell": cell, "artifact_name": args.artifact_name,
        "completed_task_ids": sorted(completed),
    }, sort_keys=True, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
