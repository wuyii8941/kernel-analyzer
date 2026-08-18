#!/usr/bin/env python3
"""Run paired T4 for every strict T3 survivor in one frozen cell."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess

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


def index_artifacts(paths: list[Path], *, t2: bool = False) -> dict[str, list[Path]]:
    """Decompress each prerequisite once instead of once per T3 survivor."""
    matches: dict[str, list[Path]] = {}
    for path in paths:
        payload = load(path)
        rows = payload.get("rows", [payload])
        for row in rows:
            if t2 and not row.get("causal_t2_positive", row.get("causal_t2_t3_positive", False)):
                continue
            if not t2 and row.get("verdict") != "DIRECTIONAL_OPTIMIZATION_BIAS":
                continue
            task_id = str(row["task_id"])
            matches.setdefault(task_id, []).append(path)
    return matches


def find_indexed(index: dict[str, list[Path]], task_id: str) -> Path:
    matches = index.get(task_id, [])
    if len(matches) != 1:
        raise RuntimeError(f"task {task_id} has {len(matches)} prerequisite artifacts")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-prefix", choices=tuple(CONFIG), required=True)
    parser.add_argument("--sequence-length", choices=(64, 128, 256), type=int, required=True)
    parser.add_argument("--device-index", type=int, required=True)
    args = parser.parse_args()
    architecture, model, graph_breaks = CONFIG[args.model_prefix]
    cell = f"{args.model_prefix}_seq{args.sequence_length}_r1"
    carrier_dir = ROOT / "results/coverage/cases/carrier" / cell
    if not (carrier_dir / "queue_complete.json").exists():
        raise RuntimeError("T3 queue is incomplete")
    t1_root = ROOT / "results/coverage/cases/full_coordinate"
    t1_paths = [t1_root / f"{args.model_prefix}_seq{args.sequence_length}_small.json.gz"]
    t1_paths += sorted((t1_root / "large" / cell).glob("*.json.gz"))
    t1_paths = [path for path in t1_paths if path.exists()]
    t2_paths = sorted((ROOT / "results/coverage/cases/causal" / cell).glob("*.json.gz"))
    t1_index = index_artifacts(t1_paths)
    t2_index = index_artifacts(t2_paths, t2=True)
    out_dir = ROOT / "results/coverage/cases/trajectory" / cell
    out_dir.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update({
        "CUDA_VISIBLE_DEVICES": str(args.device_index),
        "HF_HOME": "/data1/tzh/cache/huggingface",
        "HUGGINGFACE_HUB_CACHE": "/data1/tzh/cache/huggingface/hub",
        "XDG_CACHE_HOME": "/data1/tzh/cache/xdg",
        "TORCHINDUCTOR_CACHE_DIR": f"/data1/tzh/cache/torchinductor/frozen/{cell}",
    })
    passed = []
    for t3_path in sorted(carrier_dir.glob("*.json.gz")):
        t3 = load(t3_path)
        if t3.get("status") != "PASS_T3_COHERENT_REAL_CARRIER":
            continue
        task_id = str(t3["task_id"]); carrier = str(t3["carrier_parameter"])
        t1_path = find_indexed(t1_index, task_id)
        t2_path = find_indexed(t2_index, task_id)
        token = hashlib.sha256((task_id + "\0" + carrier).encode()).hexdigest()[:16]
        output = out_dir / f"{token}.json.gz"
        if output.exists():
            passed.append(task_id); continue
        command = [
            str(PYTHON), str(ROOT / "scripts/run_qwen128_rsqrt13_trajectory.py"),
            "--architecture", architecture, "--model", model,
            "--input-bank", str(ROOT / f"results/coverage/{args.model_prefix}_seq{args.sequence_length}_input_bank.json"),
            "--release-dir", str(ROOT / "results/coverage/runtime_releases" / cell),
            "--task-id", task_id, "--carrier", carrier,
            "--t1-artifact", str(t1_path), "--t2-artifact", str(t2_path),
            "--t3-artifact", str(t3_path), "--output", str(output),
        ]
        if graph_breaks:
            command.append("--allow-graph-breaks")
        print(json.dumps({"event": "T4_START", "cell": cell, "task_id": task_id,
                          "carrier": carrier}), flush=True)
        subprocess.run(command, cwd=ROOT, env=environment, check=True)
        passed.append(task_id)
    summary = {"schema": "kernel-analyzer-t4-trajectory-queue-summary-v1",
               "cell": cell, "t3_survivors_audited": sorted(passed)}
    (out_dir / "queue_complete.json").write_text(
        json.dumps(summary, sort_keys=True, separators=(",", ":")) + "\n"
    )
    print(json.dumps({"event": "T4_QUEUE_COMPLETE", **summary}), flush=True)


if __name__ == "__main__":
    main()
