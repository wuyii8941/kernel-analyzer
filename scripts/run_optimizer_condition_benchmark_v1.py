#!/usr/bin/env python3
"""Run the frozen checkpoint and AdamW-state comparison."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATHS = {
    "phi4": "/data1/tzh/models/microsoft/Phi-4-mini-instruct",
    "deepseek8b": "/data1/tzh/models/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
}
ARCHITECTURES = {"phi4": "phi", "deepseek8b": "deepseek8b"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--spool-root", type=Path, required=True)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    selected = set(args.models)
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(args.device)
    conditions = [
        row for row in protocol["conditions"]
        if row["name"] in {"warm_step_8", "warm_step_32", "warm_step_32_moments_reset"}
    ]
    for case in protocol["cases"]:
        model = case["model"]
        if model not in selected:
            continue
        sequence = int(case["sequence_length"])
        case_id = case["case_id"]
        release = ROOT / f"results/coverage/runtime_releases/{model}_seq{sequence}_r1"
        anchor_bank = ROOT / f"results/coverage/{model}_seq{sequence}_input_bank.json"
        state_bank = ROOT / f"results/property/tcmp_allop_v1/input_banks/{model}_seq{sequence}_trajectory4096.json"
        case_plan = ROOT / f"results/property/tcmp_allop_v1/heldout/{case_id}/case_plan.json"
        for condition in conditions:
            name = condition["name"]
            target = args.raw_root / name / f"{case_id}.json"
            if args.skip_existing and target.exists():
                print(json.dumps({"event": "CONDITION_SKIPPED_COMPLETE", "case_id": case_id, "condition": name}), flush=True)
                continue
            command = [
                str(args.python), "scripts/capture_bound_endpoint_bias_formation_v21.py",
                "--architecture", ARCHITECTURES[model],
                "--model", MODEL_PATHS[model],
                "--input-bank", str(anchor_bank),
                "--state-bank", str(state_bank),
                "--release-dir", str(release),
                "--case-plan", str(case_plan),
                "--output-dir", str(args.raw_root / "legacy" / name / case_id),
                "--spool-dir", str(args.spool_root / name / case_id),
                "--states", "32",
                "--warmup-steps", str(condition["warmup_steps"]),
                "--training-bias-profile-v2-output-dir", str(args.raw_root / name),
            ]
            if condition.get("reset_moments"):
                command.append("--reset-moments-at-measurement")
            # Current Phi rotary-position handling contains a data-dependent
            # length check in Transformers.  The frozen Phi releases were
            # captured with graph breaks permitted for this model-level code;
            # the target compiled backward output remains bound and checked.
            if model == "phi4":
                command.append("--allow-graph-breaks")
            print(json.dumps({"event": "CONDITION_STARTED", "case_id": case_id, "condition": name}), flush=True)
            subprocess.run(command, cwd=ROOT, env=env, check=True)
            if not target.exists():
                raise RuntimeError(f"missing expected result {target}")
            print(json.dumps({"event": "CONDITION_COMPLETE", "case_id": case_id, "condition": name}), flush=True)


if __name__ == "__main__":
    main()
