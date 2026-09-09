#!/usr/bin/env python3
"""Resume offline binding after completed capture; preserve original logs."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from scripts.run_training_numerical_v2 import ROOT, save_new


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    plan = json.loads((args.directory / "plan.json").read_text())
    if (args.directory / "status.json").exists():
        raise SystemExit("Existing terminal status: do not overwrite a finished attempt")
    proof = args.directory / "proof.json.gz"
    with gzip.open(proof, "rt") as handle:
        data = json.load(handle)
    if data.get("status") != "COMPLETE_PROOF_ID_PROPAGATION_CAPTURE":
        raise SystemExit("Capture is not complete; cannot resume offline binding")
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1",
               PYTHONPATH=f"{ROOT}/src:{ROOT}:{ROOT}/archive/round1_code/src")
    save_new(args.directory / "resume.json", {
        "proof_sha256": hashlib.sha256(proof.read_bytes()).hexdigest(),
        "mode": "OFFLINE_BUILDERS_ONLY", "original_plan_preserved": True,
    })
    for index, command in enumerate(plan["commands"][1:], 1):
        digest = hashlib.sha256((ROOT / command[0]).read_bytes()).hexdigest()
        if digest != plan["source_sha256"][command[0]]:
            raise SystemExit(f"Builder changed: {command[0]}")
        with (args.directory / f"step{index}.log").open("x") as log:
            result = subprocess.run([sys.executable, *command], cwd=ROOT, env=env,
                                    stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            save_new(args.directory / "status.json", {
                "case_id": plan["case_id"], "status": "UNRESOLVED_BINDING_RECOVERY",
                "failed_step": index, "returncode": result.returncode,
                "replacement_used": False,
            })
            raise SystemExit(result.returncode)
    save_new(args.directory / "status.json", {
        "case_id": plan["case_id"], "status": "BINDINGS_BUILT_REQUIRES_RUNTIME_RECAPTURE",
        "replacement_used": False, "bias_confirmed": False,
    })


if __name__ == "__main__":
    main()
