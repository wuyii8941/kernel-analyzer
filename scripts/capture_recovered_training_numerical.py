#!/usr/bin/env python3
"""Measure the original recovered Gemma/Llama endpoints with the shared runner."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from kernel_analyzer.training_numerical_analysis import analyze_artifact
from scripts.recover_training_numerical_runtime import CASES
from scripts.run_training_numerical_v2 import BASE, ROOT, hashes, save_new


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--recovery-dir", type=Path,
                        help="Explicit completed binding revision; the original failed recovery is preserved")
    args = parser.parse_args()
    case, folder, architecture, model, bank = CASES[args.case]
    recovery = args.recovery_dir or BASE / "recovery" / (args.case + "_tagged")
    status = json.loads((recovery / "status.json").read_text())
    if status["status"] != "BINDINGS_BUILT_REQUIRES_RUNTIME_RECAPTURE":
        raise SystemExit("Source binding not recovered")
    protocol = json.loads((BASE / "protocol.json").read_text())
    if hashes() != protocol["source_sha256"]:
        raise SystemExit("Frozen numerical analysis sources changed")
    old_run = ROOT / "results/property/three_mechanism_profiles_v1/runs" / folder
    prediction = json.loads((old_run / "prediction.json").read_text())
    target = prediction["target_region_id"] + ":" + prediction["requested_target_endpoint"]
    with gzip.open(recovery / "same_dtype_tasks.json.gz", "rt") as handle:
        tasks = json.load(handle)
    matched = [row for row in tasks["rows"] if row["task_id"] == target]
    if len(matched) != 1 or matched[0]["status"] != "EXACT_CANDIDATE_BUFFER_TO_AOT_SEMANTIC_ENDPOINT":
        # A failed preflight is an experimental record too. Do not silently
        # replace an old internal buffer with a larger semantic endpoint.
        rejected = BASE / "attempts" / (args.case + "_recovered_readback_preflight")
        save_new(rejected / "status.json", {
            "case_id": case, "status": "SOURCE_BOUNDARY_MISMATCH",
            "reason": "Original endpoint lacks exact recovered identity",
            "target": target, "expected_symbol": prediction["target_symbol"],
            "recovery_directory": str(recovery), "matched_tasks": matched,
            "tasks_sha256": hashlib.sha256((recovery / "same_dtype_tasks.json.gz").read_bytes()).hexdigest(),
            "replacement_used": False, "measurement_started": False,
            "claim_boundary": "A closed region proof does not establish the identity of this original internal buffer.",
        })
        raise SystemExit("Original endpoint lacks exact recovered identity")
    if matched[0]["symbol"] != prediction["target_symbol"]:
        raise SystemExit("Recovered implementation symbol differs from original case")
    out = BASE / "attempts" / (args.case + "_recovered_readback")
    release = out / "runtime_release"
    release.mkdir(parents=True, exist_ok=False)
    sources = {}
    for name in ("capture.json", "campaign.json.gz", "inventory.json.gz", "trace"):
        source = old_run / "runtime_release" / name
        (release / name).symlink_to(source.resolve(), target_is_directory=source.is_dir())
        if source.is_file():
            sources[str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
    (release / "same_dtype_tasks.json.gz").symlink_to((recovery / "same_dtype_tasks.json.gz").resolve())
    for source in (recovery / "same_dtype_tasks.json.gz", old_run / "prediction.json"):
        sources[str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
    case_plan = out / "case_plan.json"
    save_new(case_plan, {"cases": [{"case_id": case, "task_id": target,
              "carrier": prediction["carrier"], "reference_method": "AOT_REPLAY"}],
              "data_use": "EXISTING_FROZEN_CASE_RECOVERY_NOT_NEW_UNSEEN_IMPLEMENTATION"})
    command = [sys.executable, "scripts/capture_bound_endpoint_bias_formation_v21.py",
               "--architecture", architecture, "--model", model,
               "--input-bank", f"results/property/tcmp_allop_v1/input_banks/{bank}.json",
               "--state-bank", f"results/property/three_mechanism_profiles_v1/input_banks/{case}.json",
               "--release-dir", str(release), "--case-plan", str(case_plan),
               "--output-dir", str(out / "legacy"), "--spool-dir",
               f"/data1/tzh/cache/kernel-analyzer/readback_v2/{case}_recovered",
               "--device", args.device, "--states", "32", "--allow-graph-breaks",
               "--training-bias-profile-v2-output-dir", str(out / "raw")]
    save_new(out / "protocol.json", {**protocol, "selected_case": case,
             "command": command, "recovered_sources": sources,
             "new_implementation_confirmation": False})
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=f"{ROOT}/src:{ROOT}",
               HF_HOME="/data1/tzh/cache/huggingface", XDG_CACHE_HOME="/data1/tzh/cache/xdg",
               TRITON_CACHE_DIR="/data1/tzh/cache/triton",
               TORCHINDUCTOR_CACHE_DIR="/data1/tzh/cache/torchinductor", OMP_NUM_THREADS="4")
    with (out / "capture.log").open("x") as log:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    save_new(out / "status.json", {"case_id": case, "returncode": result.returncode,
             "status": "CAPTURE_RETURNED" if result.returncode == 0 else "EXECUTION_FAILED",
             "replacement_used": False})
    if result.returncode:
        raise SystemExit(result.returncode)
    raw_path = out / "raw" / (case + ".json")
    report = analyze_artifact(json.loads(raw_path.read_text()), protocol)
    report["provenance"]["raw_sha256"] = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    save_new(out / "recomputed.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
