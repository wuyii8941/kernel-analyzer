#!/usr/bin/env python3
"""Run every frozen repeated paired-training consequence without early stopping."""

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


def case_model(case_id: str) -> tuple[str, int]:
    fields = case_id.split("_")
    model = fields[0]
    sequence = int(fields[1].removeprefix("seq"))
    return model, sequence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    selected = set(args.models)
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(args.device)
    # The four matched replays briefly keep several full target-parameter
    # gradients alive. Expandable CUDA segments avoid losing the final large
    # allocation to allocator fragmentation on the 48 GiB Phi run.
    env.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    for case in protocol["cases"]:
        case_id = case["case_id"]
        model, sequence = case_model(case_id)
        if model not in selected:
            continue
        release = ROOT / f"results/coverage/runtime_releases/{model}_seq{sequence}_r1"
        case_plan = ROOT / f"results/property/tcmp_allop_v1/heldout/{case_id}/case_plan.json"
        for repeat in case["repeats"]:
            index = int(repeat["repeat"])
            target = args.output_root / case_id / f"repeat{index}.json"
            if args.skip_existing and target.exists():
                print(json.dumps({"event": "REPEAT_SKIPPED_COMPLETE", "case_id": case_id, "repeat": index}), flush=True)
                continue
            checkpoint = args.checkpoint_root / case_id / f"repeat{index}.pt"
            command = [
                str(args.python), "scripts/run_bound_endpoint_consequence_v21.py",
                "--architecture", ARCHITECTURES[model],
                "--model", MODEL_PATHS[model],
                "--input-bank", repeat["input_bank"],
                "--release-dir", str(release),
                "--case-plan", str(case_plan),
                "--case-id", case_id,
                "--output", str(target),
                "--checkpoint", str(checkpoint),
                "--steps", str(protocol["steps_per_repeat"]),
                "--compact-long",
                "--checkpoint-every", str(protocol["steps_per_repeat"]),
            ]
            if model == "phi4":
                command.append("--allow-graph-breaks")
            print(json.dumps({"event": "REPEAT_STARTED", "case_id": case_id, "repeat": index}), flush=True)
            subprocess.run(command, cwd=ROOT, env=env, check=True)
            if not target.exists():
                raise RuntimeError(f"missing expected result {target}")
            checkpoint.unlink(missing_ok=True)
            print(json.dumps({"event": "REPEAT_COMPLETE", "case_id": case_id, "repeat": index}), flush=True)


if __name__ == "__main__":
    main()
