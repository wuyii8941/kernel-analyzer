#!/usr/bin/env python3
"""Re-run the frozen 12-row consequence audit with real wall-clock timing.

The old consequence JSON files contain scientific results but no per-case
runtime.  This runner uses the same frozen case plans/releases and records the
actual process wall time separately.  It does not alter the scientific labels
or replace the original artifacts.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL_BY_ARCH = {
    "deepseek8b": "/data1/tzh/models/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
    "mamba": "/data1/tzh/models/state-spaces/mamba-130m-hf",
    "phi": "/data1/tzh/models/microsoft/Phi-4-mini-instruct",
    "qwen": "/data1/tzh/models/Qwen/Qwen3-1.7B",
}
PLAN_SLUG_BY_ARCH = {"deepseek8b": "deepseek8b", "mamba": "mamba", "phi": "phi4", "qwen": "qwen"}


def build_jobs(comparison: Path) -> list[dict[str, str]]:
    rows = json.loads(comparison.read_text())["rows"]
    jobs = []
    for row in rows:
        architecture = str(row["architecture"])
        release_name = Path(row["source"]).parts[-4]
        seq = int(release_name.split("_seq")[1].split("_")[0])
        plan_slug = PLAN_SLUG_BY_ARCH[architecture]
        case_id = str(row["case_id"])
        jobs.append({
            "case_id": case_id,
            "architecture": architecture,
            "sequence_length": str(seq),
            "model": MODEL_BY_ARCH[architecture],
            "input_bank": f"results/coverage/{plan_slug}_seq{seq}_input_bank.json",
            "release_dir": f"results/coverage/runtime_releases/{release_name}",
            "case_plan": f"results/property/joint_bias_formation_v1/negative_consequence_plans/{plan_slug}_seq{seq}.json",
        })
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", type=Path, default=ROOT / "results/property/joint_bias_formation_v1/oracle_baselines/comparison.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu-list", default="0,1,2,3")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--max-parallel", type=int, default=4)
    parser.add_argument(
        "--only-case",
        action="append",
        default=None,
        help="Run only the named frozen case id; may be supplied more than once.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logs = args.output_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    jobs = build_jobs(args.comparison)
    if args.only_case:
        requested = set(args.only_case)
        jobs = [job for job in jobs if job["case_id"] in requested]
        missing = requested.difference(job["case_id"] for job in jobs)
        if missing:
            raise ValueError("unknown case id(s): " + ", ".join(sorted(missing)))
    if not jobs:
        raise ValueError("no cases selected")
    gpus = [item.strip() for item in args.gpu_list.split(",") if item.strip()]
    if not gpus:
        raise ValueError("gpu-list cannot be empty")
    results = []
    for base in range(0, len(jobs), max(1, args.max_parallel)):
        wave = jobs[base:base + max(1, args.max_parallel)]
        running = []
        for offset, job in enumerate(wave):
            output = args.output_dir / f"{job['case_id']}.json"
            checkpoint = checkpoint_dir / f"{job['case_id']}.ckpt"
            log = logs / f"{job['case_id']}.log"
            command = [
                args.python, "scripts/run_bound_endpoint_consequence_v21.py",
                "--architecture", job["architecture"],
                "--model", job["model"],
                "--input-bank", job["input_bank"],
                "--release-dir", job["release_dir"],
                "--case-plan", job["case_plan"],
                "--case-id", job["case_id"],
                "--output", str(output),
                "--checkpoint", str(checkpoint),
                "--steps", "32",
                # CUDA_VISIBLE_DEVICES remaps the selected physical GPU to
                # logical device 0 inside this child process.
                "--device", "cuda:0",
            ]
            if job["architecture"] == "phi":
                command.append("--allow-graph-breaks")
            environment = os.environ.copy()
            environment.update({
                "CUDA_VISIBLE_DEVICES": gpus[offset % len(gpus)],
                "HF_HOME": "/data1/tzh/cache/huggingface",
                "HUGGINGFACE_HUB_CACHE": "/data1/tzh/cache/huggingface/hub",
                "TRANSFORMERS_CACHE": "/data1/tzh/cache/huggingface/transformers",
                "XDG_CACHE_HOME": "/data1/tzh/cache/xdg",
            })
            handle = log.open("w")
            started = time.monotonic()
            process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=handle, stderr=subprocess.STDOUT)
            running.append((job, command, output, checkpoint, log, handle, process, started))
        failed = []
        for job, command, output, checkpoint, log, handle, process, started in running:
            return_code = process.wait()
            handle.close()
            elapsed = time.monotonic() - started
            result_status = None
            if output.exists():
                try:
                    result_status = json.loads(output.read_text()).get("status")
                except Exception:
                    result_status = "MALFORMED_OUTPUT"
            results.append({
                **job,
                "command": command,
                "output": str(output),
                "log": str(log),
                "elapsed_seconds": elapsed,
                "return_code": return_code,
                "result_status": result_status,
            })
            if return_code != 0 or result_status != "COMPLETE":
                failed.append((job["case_id"], str(log)))
        if failed:
            raise RuntimeError("timed consequence failures: " + ", ".join(f"{case} ({log})" for case, log in failed))
    payload = {
        "schema": "kernel-analyzer-timed-negative-consequence-audit-v1",
        "status": "COMPLETE_12_CASES_WITH_WALL_TIMES",
        "selection_source": str(args.comparison),
        "selection_rule": "The exact frozen 12-row residual-nonzero, parameter-reachable sample from the Oracle comparison.",
        "steps": 32,
        "cases": results,
        "sum_case_wall_seconds": sum(float(row["elapsed_seconds"]) for row in results),
        "parallel_wave_count": (len(results) + max(1, args.max_parallel) - 1) // max(1, args.max_parallel),
        "claim_boundary": "Wall times measure these replay processes on the recorded host configuration; they are not a portable GPU-hour benchmark.",
    }
    target = args.output_dir / "timed_audit.json"
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": payload["status"], "cases": len(results), "sum_case_wall_seconds": payload["sum_case_wall_seconds"], "output": str(target)}))


if __name__ == "__main__":
    main()
