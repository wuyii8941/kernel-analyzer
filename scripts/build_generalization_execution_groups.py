#!/usr/bin/env python3
"""Combine frozen benchmark cases sharing one model and sequence length."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


MODEL_PATHS = {
    "qwen": "/data1/tzh/models/Qwen/Qwen3-1.7B",
    "phi4": "/data1/tzh/models/microsoft/Phi-4-mini-instruct",
    "deepseek8b": "/data1/tzh/models/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
    "mamba": "/data1/tzh/models/state-spaces/mamba-130m-hf",
}


def release_name(model: str, sequence_length: int) -> str:
    if model == "qwen" and sequence_length == 64:
        return "qwen_seq64_r2_recovered"
    if model == "mamba" and sequence_length in (64, 256):
        return f"mamba_seq{sequence_length}_r2"
    return f"{model}_seq{sequence_length}_r1"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--case-plan-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    groups = defaultdict(list)
    for row in protocol["cases"]:
        case_id = f"{row['model']}_seq{row['sequence_length']}_{row['task_id'].replace(':', '_')}"
        path = args.case_plan_root / case_id / "case_plan.json"
        plan = json.loads(path.read_text())
        groups[(row["model"], int(row["sequence_length"]))].extend(plan["cases"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    index = []
    for (model, sequence_length), cases in sorted(groups.items()):
        target = args.output_dir / f"{model}_seq{sequence_length}.json"
        payload = {
            "schema": "kernel-analyzer-generalization-execution-group-v1",
            "status": "FROZEN_FROM_GENERALIZATION_BENCHMARK_V1",
            "model": model,
            "sequence_length": sequence_length,
            "cases": cases,
        }
        target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        index.append({
            "model": model,
            "sequence_length": sequence_length,
            "case_count": len(cases),
            "case_plan": str(target),
            "model_path": MODEL_PATHS[model],
            "input_bank": f"results/coverage/{model}_seq{sequence_length}_input_bank.json",
            "runtime_release": f"results/coverage/runtime_releases/{release_name(model, sequence_length)}",
        })
    index_path = args.output_dir / "index.json"
    index_path.write_text(json.dumps({"groups": index}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"groups": len(index), "cases": sum(x["case_count"] for x in index)}))


if __name__ == "__main__":
    main()
