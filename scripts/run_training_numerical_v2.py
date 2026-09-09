#!/usr/bin/env python3
"""Freeze, capture and recompute using existing family runners and one analyzer."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from kernel_analyzer.training_numerical_analysis import analyze_artifact

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/property/training_numerical_analysis_v2"
SOURCE_FILES = (
    "src/kernel_analyzer/update_write.py", "src/kernel_analyzer/training_numerical_analysis.py",
    "src/kernel_analyzer/training_equivalence.py", "src/kernel_analyzer/training_bias_profile.py",
    "src/kernel_analyzer/short_persistence.py",
    "scripts/run_training_bias_profile_v2_empirical.py",
    "scripts/capture_bound_endpoint_bias_formation_v21.py", "scripts/run_training_numerical_v2.py",
)
CASES = {
    "phi": "phi4_seq64_lmhead_dx", "liger": "liger_fused_ce_t128",
    "deepseek_norm": "deepseek8b_seq256_backward_1714_in_out_ptr0",
    "deepseek_attn": "deepseek8b_seq128_backward_1256_out_ptr0",
}


def save_new(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        json.dump(data, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


def hashes():
    return {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCE_FILES}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("command", choices=("freeze", "capture", "report"))
    parser.add_argument("--case", choices=CASES)
    parser.add_argument("--device", default="cuda:0")
    args=parser.parse_args()
    protocol_path=BASE/"protocol.json"
    if args.command=="freeze":
        save_new(protocol_path, {
            "schema": "training-numerical-analysis-readback-v2",
            "frozen_at": datetime.now(timezone.utc).isoformat(),
            "claim_scope": "FIXED_SUITE_UPDATE", "primary_stage": "PARAMETER_WRITE",
            "mandatory_endpoints": ["PARAMETER_WRITE_TOTAL_RMS"],
            "fixed_suite_margins": {"full_update_rms": .01},
            "policy_change": "Q-only fixed-suite update bound; direction projections remain descriptive, not the old three-interval policy. The 1% RMS margin is unchanged.",
            "parameter_representation": "FP32_MASTER_INITIALIZED_FROM_STORED_PARAMETER",
            "cases": CASES, "data_use": "EXISTING_CASE_RECAPTURE_NOT_NEW_IMPLEMENTATION_CONFIRMATION",
            "source_sha256": hashes(),
            "remaining_plan": ["GEMMA_LLAMA_EXECUTION_RECOVERY", "NEW_IMPLEMENTATION_CONFIRMATION", "NORMAL_LANGUAGE_MODEL_PAIRED_TRAINING"],
        })
        return
    protocol=json.loads(protocol_path.read_text())
    if args.command=="report":
        rows=[]
        for name, case in CASES.items():
            path=BASE/"raw"/(case+".json")
            if path.exists():
                raw=json.loads(path.read_text())
                row=analyze_artifact(raw,protocol)
                row["provenance"]["raw_sha256"]=hashlib.sha256(path.read_bytes()).hexdigest()
                rows.append(row)
            else:
                rows.append({"case_id":case,"measurement_status":"NOT_RUN","equivalence_decision":"NOT_ASSESSED"})
        stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        save_new(BASE/("report_"+stamp+".json"),{
            "status":"INCOMPLETE_FULL_RESEARCH_PLAN", "rows":rows,
            "remaining_plan": protocol["remaining_plan"],
        })
        print(json.dumps(rows,indent=2))
        return
    if args.case is None:
        parser.error("capture requires --case")
    if hashes()!=protocol["source_sha256"]:
        raise SystemExit("Frozen analysis/capture source changed; do not silently reuse this protocol.")
    case=CASES[args.case]; output=BASE/"raw"/(case+".json")
    if output.exists():
        raise SystemExit("Raw output already exists; preserving it.")
    if args.case in {"phi","liger"}:
        command=[sys.executable,"scripts/run_training_bias_profile_v2_empirical.py",
                 "--case",args.case,"--device",args.device,"--output",str(output)]
    else:
        length=256 if args.case=="deepseek_norm" else 128
        plan=f"results/property/tcmp_allop_v1/heldout/{case}/case_plan.json"
        command=[sys.executable,"scripts/capture_bound_endpoint_bias_formation_v21.py",
            "--architecture","deepseek8", "--model","/data1/tzh/models/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
            "--input-bank",f"results/coverage/deepseek8b_seq{length}_input_bank.json",
            "--release-dir",f"results/coverage/runtime_releases/deepseek8b_seq{length}_r1",
            "--case-plan",plan,"--states","32","--device",args.device,
            "--output-dir",str(BASE/"legacy"/case),
            "--spool-dir",f"/data1/tzh/cache/kernel-analyzer/readback_v2/{case}",
            "--training-bias-profile-v2-output-dir",str(BASE/"raw")]
    env=dict(os.environ)
    env.update({"PYTHONDONTWRITEBYTECODE":"1", "PYTHONPATH":str(ROOT/"src"),
                "HF_HOME":"/data1/tzh/cache/huggingface", "XDG_CACHE_HOME":"/data1/tzh/cache/xdg",
                "TRITON_CACHE_DIR":"/data1/tzh/cache/triton", "TORCHINDUCTOR_CACHE_DIR":"/data1/tzh/cache/torchinductor",
                "MPLCONFIGDIR":"/data1/tzh/cache/matplotlib", "OMP_NUM_THREADS":"4", "MKL_NUM_THREADS":"4"})
    log=BASE/"logs"/(case+".log"); log.parent.mkdir(parents=True,exist_ok=True)
    print(json.dumps({"event":"CAPTURE_STARTED","case":case,"command":command}),flush=True)
    with log.open("x") as handle:
        completed=subprocess.run(command,cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT)
    save_new(BASE/"execution"/(case+".json"),{
        "case_id":case,"command":command,"returncode":completed.returncode,
        "status":"CAPTURE_RETURNED" if completed.returncode==0 else "EXECUTION_FAILED",
        "log":str(log),"source_sha256":hashes(),
    })
    if completed.returncode:
        raise SystemExit(completed.returncode)
    raw=json.loads(output.read_text())
    report=analyze_artifact(raw,protocol)
    report["provenance"].update({"raw_sha256":hashlib.sha256(output.read_bytes()).hexdigest(),
                                 "protocol_sha256":hashlib.sha256(protocol_path.read_bytes()).hexdigest()})
    save_new(BASE/"recomputed"/(case+".json"),report)
    print(json.dumps(report,indent=2),flush=True)


if __name__=="__main__":
    main()
