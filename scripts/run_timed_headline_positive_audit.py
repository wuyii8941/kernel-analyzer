#!/usr/bin/env python3
"""Time exact reproductions of the three frozen source-persistence headlines.

Each child writes to a cache-only result path.  The runner verifies that the
new 32-state scientific payload reproduces the corresponding retained result
before accepting its wall time.  These are case-specific validation costs,
not a claim that the three implementations have identical execution paths.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]
PT_NIGHTLY = "/data1/tzh/miniconda3/envs/pt_nightly/bin/python"
LIGER_PYTHON = "/data1/tzh/envs/liger/bin/python"


JOBS = (
    {
        "case_id": "liger_fused_ce_t128",
        "python": LIGER_PYTHON,
        "script": "scripts/run_liger_three_stage_reference.py",
        "source": "results/property/joint_bias_formation_v1/liger_three_stage_reference.json",
        "cache": "/data1/tzh/cache/kernel_analyzer/timed_headlines/liger",
        "extra": [],
    },
    {
        "case_id": "phi4_seq64_lmhead_dx",
        "python": PT_NIGHTLY,
        "script": "scripts/run_phi_three_stage_reference.py",
        "source": "results/property/joint_bias_formation_v1/phi_three_stage_reference.json",
        "cache": "/data1/tzh/cache/torchinductor/frozen/phi4_seq64_r1",
        "extra": ["--release-dir", "results/coverage/runtime_releases/phi4_seq64_r1"],
    },
    {
        "case_id": "qwen_seq256_lmhead_dx",
        "python": PT_NIGHTLY,
        "script": "scripts/run_qwen_three_stage_reference.py",
        "source": "results/property/joint_bias_formation_v1/qwen_three_stage_reference.json",
        "cache": "/data1/tzh/cache/torchinductor/bias_hotspot_qwen256_loss",
        "extra": ["--release-dir", "results/coverage/runtime_releases/qwen_seq256_r2"],
    },
)


def scientific_view(payload: dict) -> dict:
    return {
        "status": payload.get("status"),
        "case_id": payload.get("case_id"),
        "state_count": payload.get("state_count"),
        "state_order": payload.get("state_order"),
        "learning_rate": payload.get("learning_rate"),
        "reference_trajectory": payload.get("reference_trajectory"),
        "stages": payload.get("stages"),
        "rows": payload.get("rows"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu-list", default="0,1,2")
    parser.add_argument("--only-case", action="append")
    args = parser.parse_args()
    selected = list(JOBS)
    if args.only_case:
        requested = set(args.only_case)
        selected = [job for job in selected if job["case_id"] in requested]
        missing = requested.difference(job["case_id"] for job in selected)
        if missing:
            raise ValueError("unknown case id(s): " + ", ".join(sorted(missing)))
    gpus = [part.strip() for part in args.gpu_list.split(",") if part.strip()]
    if len(gpus) < len(selected):
        raise ValueError("one distinct GPU is required per concurrent headline job")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "logs").mkdir(exist_ok=True)
    running = []
    for index, job in enumerate(selected):
        output = args.output_dir / f"{job['case_id']}.json"
        log = args.output_dir / "logs" / f"{job['case_id']}.log"
        command = [
            job["python"], job["script"], *job["extra"],
            "--device", "cuda:0", "--output", str(output),
        ]
        environment = os.environ.copy()
        environment.update({
            "CUDA_VISIBLE_DEVICES": gpus[index],
            "HF_HOME": "/data1/tzh/cache/huggingface",
            "HUGGINGFACE_HUB_CACHE": "/data1/tzh/cache/huggingface/hub",
            "TRANSFORMERS_CACHE": "/data1/tzh/cache/huggingface/transformers",
            "TORCHINDUCTOR_CACHE_DIR": job["cache"],
            "PYTHONPATH": "src:archive/round1_code/src:scripts",
        })
        handle = log.open("w")
        started = time.monotonic()
        process = subprocess.Popen(
            command, cwd=ROOT, env=environment,
            stdout=handle, stderr=subprocess.STDOUT,
        )
        running.append((job, command, output, log, handle, process, started, gpus[index]))

    rows = []
    failures = []
    pending = list(running)
    while pending:
        remaining = []
        for job, command, output, log, handle, process, started, gpu in pending:
            return_code = process.poll()
            if return_code is None:
                remaining.append((job, command, output, log, handle, process, started, gpu))
                continue
            elapsed = time.monotonic() - started
            handle.close()
            reproduced = False
            status = "MISSING_OUTPUT"
            if output.exists():
                measured = json.loads(output.read_text(encoding="utf-8"))
                source = json.loads((ROOT / job["source"]).read_text(encoding="utf-8"))
                status = str(measured.get("status"))
                reproduced = scientific_view(measured) == scientific_view(source)
            row = {
                "case_id": job["case_id"],
                "command": command,
                "physical_gpu": gpu,
                "elapsed_seconds": elapsed,
                "return_code": return_code,
                "result_status": status,
                "scientific_result_reproduced": reproduced,
                "source": job["source"],
                "output": str(output),
                "log": str(log),
            }
            rows.append(row)
            if return_code != 0 or status != "COMPLETE_ORDERED_32_STATE_REFERENCE" or not reproduced:
                failures.append(job["case_id"])
        pending = remaining
        if pending:
            time.sleep(0.1)
    rows.sort(key=lambda row: next(
        index for index, job in enumerate(selected) if job["case_id"] == row["case_id"]
    ))
    payload = {
        "schema": "kernel-analyzer-timed-headline-positive-audit-v1",
        "status": "COMPLETE_3_HEADLINES_REPRODUCED_WITH_WALL_TIMES" if not failures and len(rows) == 3 else "INCOMPLETE",
        "cases": rows,
        "sum_case_wall_seconds": sum(row["elapsed_seconds"] for row in rows if row["scientific_result_reproduced"]),
        "claim_boundary": (
            "Measured process wall times for exact reproduction of each case's frozen "
            "32-state source-persistence certification.  The case-specific runners differ, "
            "so this is an observed validation-cost audit, not a portable kernel benchmark."
        ),
    }
    target = args.output_dir / "timed_headlines.json"
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "failures": failures, "output": str(target)}))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
