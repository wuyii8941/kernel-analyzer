#!/usr/bin/env python3
"""Rebuild bindings after checked math support, preserving failed attempts."""
import argparse
import gzip
import json
import os
import subprocess
import sys
from scripts.recover_training_numerical_runtime import CASES
from scripts.run_liger_single_boundary_collapse import file_sha256
from scripts.run_training_numerical_v2 import BASE, ROOT, save_new


def main():
    p = argparse.ArgumentParser(); p.add_argument("--case", choices=CASES, required=True)
    p.add_argument("--math", required=True); args = p.parse_args()
    case, folder, _, _, _ = CASES[args.case]
    from pathlib import Path
    math_path = Path(args.math).resolve()
    with gzip.open(math_path, "rt") as handle:
        derivation = json.load(handle)
    if derivation["status"] != "COMPLETE_AOT_FORWARD_BACKWARD_DERIVATION" or not all(derivation["gates"].values()):
        raise SystemExit("Mathematical binding gates remain incomplete")
    original = BASE / "recovery" / (args.case + "_tagged")
    out = BASE / "recovery" / (args.case + "_math_completed")
    out.mkdir(parents=True, exist_ok=False)
    for name in ("proof.json.gz", "aot.json.gz"):
        (out / name).symlink_to((original / name).resolve())
    (out / "math.json.gz").symlink_to(math_path)
    release = ROOT / "results/property/three_mechanism_profiles_v1/runs" / folder / "runtime_release"
    commands = [
        ["scripts/build_candidate_fb_bridge.py", "--inventory", str(release / "inventory.json.gz"),
         "--aot", str(out / "aot.json.gz"), "--math", str(out / "math.json.gz"),
         "--proof-capture", str(out / "proof.json.gz"), "--trace-dir", str(release / "trace"),
         "--output", str(out / "bridge.json.gz")],
        ["scripts/build_same_dtype_semantic_tasks.py", "--inventory", str(release / "inventory.json.gz"),
         "--candidate-fb-bridge", str(out / "bridge.json.gz"), "--proof-capture", str(out / "proof.json.gz"),
         "--output", str(out / "same_dtype_tasks.json.gz")],
    ]
    save_new(out / "plan.json", {
        "case_id": case, "commands": commands, "identity_gates_relaxed": False,
        "source_sha256": {name: file_sha256(ROOT / name) for name in (
            "scripts/round2_vl_math.py", *(c[0] for c in commands))},
        "math_sha256": file_sha256(math_path),
        "math_scope": "Nominal real arithmetic. GELU decimal coefficient model is explicit; binary coefficient rounding and finite-precision arithmetic are not declared exact.",
    })
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=f"{ROOT}/src:{ROOT}")
    for index, command in enumerate(commands):
        with (out / f"step{index}.log").open("x") as log:
            result = subprocess.run([sys.executable, *command], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            save_new(out / "status.json", {"case_id": case, "status": "UNRESOLVED_BINDING_RECOVERY", "failed_step": index, "returncode": result.returncode})
            raise SystemExit(result.returncode)
    save_new(out / "status.json", {"case_id": case,
        "status": "BINDINGS_BUILT_REQUIRES_RUNTIME_RECAPTURE", "replacement_used": False})


if __name__ == "__main__": main()
