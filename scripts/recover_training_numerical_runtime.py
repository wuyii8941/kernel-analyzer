#!/usr/bin/env python3
"""Recover missing semantic bindings for the original Gemma/Llama case IDs.

All builders retain their exact-identity gates. A failed binding is recorded,
not replaced by shape/name matching or a different numerical implementation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from scripts.run_training_numerical_v2 import BASE, ROOT, save_new


CASES={
    "llama":("llama32_text128_scan_0000","llama32_text128_scan_0000_profile_carrier",
             "generic","/data1/tzh/models/meta-llama/Llama-3.2-3B","llama32_3b_text128"),
    "gemma":("gemma4_text128_scan_0037","gemma4_text128_scan_0037",
             "gemma4","/data1/tzh/models/google/gemma-4-E2B","gemma4_e2b_text128"),
}


def main():
    p=argparse.ArgumentParser();p.add_argument("--case",choices=CASES,required=True)
    p.add_argument("--proof-tags", action="store_true",
                   help="Use the existing proof-ID propagation route, preserving earlier unrenamed attempts.")
    p.add_argument("--device",default="cuda:0");args=p.parse_args()
    case,folder,architecture,model,bank=CASES[args.case]
    old=ROOT/"results/property/three_mechanism_profiles_v1/runs"/folder/"runtime_release"
    suffix=args.case+("_tagged" if args.proof_tags else "")
    out=BASE/"recovery"/suffix
    proof=out/"proof.json.gz";aot=out/"aot.json.gz";math=out/"math.json.gz";bridge=out/"bridge.json.gz"
    commands=[
        ["scripts/capture_qwen_inductor_proof_ids.py","--architecture",architecture,"--model",model,
         "--input-bank",f"results/property/tcmp_allop_v1/input_banks/{bank}.json","--device",args.device,
         "--trace-dir",f"/data1/tzh/cache/kernel-analyzer/readback_v2/{suffix}_recovery_trace",
         "--output",str(proof),"--no-proof-node-renaming","--capture-unrenamed-aot","--allow-graph-breaks"],
        ["scripts/extract_standard_aot_capture.py","--proof-capture",str(proof),"--output",str(aot)],
        ["scripts/round2_vl_math.py","--capture",str(aot),"--output",str(math)],
        ["scripts/build_candidate_fb_bridge.py","--inventory",str(old/"inventory.json.gz"),
         "--aot",str(aot),"--math",str(math),"--proof-capture",str(proof),
         "--trace-dir",str(old/"trace"),"--output",str(bridge)],
        ["scripts/build_same_dtype_semantic_tasks.py","--inventory",str(old/"inventory.json.gz"),
         "--candidate-fb-bridge",str(bridge),"--proof-capture",str(proof),
         "--output",str(out/"same_dtype_tasks.json.gz")],
    ]
    if args.proof_tags:
        commands[0]=[x for x in commands[0] if x not in {"--no-proof-node-renaming","--capture-unrenamed-aot"}]
    env=dict(os.environ);env.update({"TORCHINDUCTOR_FORCE_DISABLE_CACHES":"1",
        "PYTHONDONTWRITEBYTECODE":"1","PYTHONPATH":f"{ROOT}/src:{ROOT}:{ROOT}/archive/round1_code/src",
        "XDG_CACHE_HOME":"/data1/tzh/cache/xdg","HF_HOME":"/data1/tzh/cache/huggingface",
        "TRITON_CACHE_DIR":"/data1/tzh/cache/triton","TORCHINDUCTOR_CACHE_DIR":"/data1/tzh/cache/torchinductor",
        "OMP_NUM_THREADS":"4"})
    save_new(out/"plan.json",{"case_id":case,"original_release":str(old),"commands":commands,
        "original_inventory_sha256":hashlib.sha256((old/"inventory.json.gz").read_bytes()).hexdigest(),
        "source_sha256":{c[0]:hashlib.sha256((ROOT/c[0]).read_bytes()).hexdigest() for c in commands},
        "identity_checks_relaxed":False,"bias_outcomes_used":False})
    for index,command in enumerate(commands):
        print(json.dumps({"event":"RECOVERY_STEP","case":case,"step":index}),flush=True)
        with (out/f"step{index}.log").open("x") as log:
            result=subprocess.run([sys.executable,*command],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            save_new(out/"status.json",{"case_id":case,"status":"UNRESOLVED_BINDING_RECOVERY",
                "failed_step":index,"returncode":result.returncode,"replacement_used":False})
            raise SystemExit(result.returncode)
    save_new(out/"status.json",{"case_id":case,"status":"BINDINGS_BUILT_REQUIRES_RUNTIME_RECAPTURE",
                                 "replacement_used":False,"bias_confirmed":False})


if __name__=="__main__":main()
