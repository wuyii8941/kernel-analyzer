#!/usr/bin/env python3
"""Build executable stage commands for the frozen sample-completion roster."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/property/sample_completion_v1/execution_manifest.json"


def load(path: Path) -> dict[str, Any]:
    if path.suffix == ".gz":
        import gzip
        with gzip.open(path, "rt", encoding="utf-8") as h:
            return json.load(h)
    return json.loads(path.read_text(encoding="utf-8"))


def binding(case_id: str, model: str) -> dict[str, Any] | None:
    if model == "qwen":
        matrix = load(ROOT / "results/property/bias_formation/hotspot_search/qwen_seq64_matrix.json")
        return next((row for row in matrix["cases"] if row["case_id"] == case_id), None)
    if model in {"phi4", "deepseek8b"}:
        directory = ROOT / "results/property/bias_formation/hotspot_search" / f"{model}_seq64_rescreen"
        for path in directory.glob("*.json"):
            payload = load(path)
            if payload.get("case_id") == case_id:
                item = dict(payload.get("binding", {}))
                item["case_id"] = case_id
                item["binding_artifact"] = str(path.relative_to(ROOT))
                return item
        return None
    if model == "mamba":
        plan = load(ROOT / "results/property/bias_formation/hotspot_search/mamba_seq64_capture_plan.json")
        return next((row for row in plan["cases"] if row["case_id"] == case_id), None)
    raise ValueError(model)


GROUPS = {
    "qwen": {
        "architecture": "qwen",
        "model": "/data1/tzh/models/Qwen/Qwen3-1.7B",
        "release_dir": "results/coverage/runtime_releases/qwen_seq64_r1",
        "input_bank": "results/coverage/qwen_seq64_input_bank.json",
        "case_ids": ["qwen_seq64_ce_dlogits", "qwen_seq64_l27_q_norm_vjp", "qwen_seq64_l27_k_norm_vjp", "qwen_seq64_l23_softmax_vjp"],
    },
    "phi4": {
        "architecture": "phi",
        "model": "/data1/tzh/models/microsoft/Phi-4-mini-instruct",
        "release_dir": "results/coverage/runtime_releases/phi4_seq64_r1",
        "input_bank": "results/coverage/phi4_seq64_input_bank.json",
        "case_ids": ["phi4_seq64_ce_dlogits", "phi4_seq64_final_norm_vjp", "phi4_seq64_l31_attention_dv", "phi4_seq64_l31_softmax_vjp"],
    },
    "deepseek8b": {
        "architecture": "deepseek8b",
        "model": "/data1/tzh/models/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
        "release_dir": "results/coverage/runtime_releases/deepseek8b_seq64_r1",
        "input_bank": "results/coverage/deepseek8b_seq64_input_bank.json",
        "case_ids": ["deepseek8b_seq64_ce_dlogits", "deepseek8b_seq64_final_norm_vjp", "deepseek8b_seq64_l35_attention_dv", "deepseek8b_seq64_l35_softmax_vjp"],
    },
    "mamba": {
        "architecture": "mamba",
        "model": "/data1/tzh/models/state-spaces/mamba-130m-hf",
        "release_dir": "results/coverage/runtime_releases/mamba_seq64_r1",
        "input_bank": "results/coverage/mamba_seq64_input_bank.json",
        "case_ids": ["mamba_seq64_ce_dlogits", "mamba_seq64_final_norm_vjp", "mamba_seq64_scan_recurrence", "mamba_seq64_scan_reduction"],
    },
}


def command(group: dict[str, Any], plan_path: str, stage: str) -> list[str]:
    common = [
        "python", "scripts/capture_bound_endpoint_bias_formation_v21.py",
        "--architecture", group["architecture"], "--model", group["model"],
        "--input-bank", group["input_bank"], "--release-dir", group["release_dir"],
        "--case-plan", plan_path,
        "--output-dir", f"/data1/tzh/cache/sample_completion_v1/{group['architecture']}/{stage}",
        "--spool-dir", f"/data1/tzh/cache/sample_completion_v1/{group['architecture']}/{stage}/spool",
        "--device", "cuda:0",
    ]
    if stage == "engineering":
        return common + ["--engineering-reach-only", "--states", "2"]
    if stage == "screen":
        return common + ["--screening-gram", "--states", "8"]
    if stage == "formal":
        return common + ["--states", "32"]
    raise ValueError(stage)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    roster = load(ROOT / "results/property/sample_completion_v1/roster.json")
    search_ids = {row["case_id"] for row in roster["search_units"]}
    groups: dict[str, Any] = {}
    for name, spec in GROUPS.items():
        rows = []
        for case_id in spec["case_ids"]:
            if case_id not in search_ids:
                raise RuntimeError(f"case {case_id} is not in the frozen roster")
            item = binding(case_id, name)
            if not item or not item.get("task_id"):
                raise RuntimeError(f"missing exact task binding for {case_id}")
            rows.append({"case_id": case_id, "task_id": item["task_id"], "carrier": item.get("carrier") or item.get("carrier_parameter"), "binding": item})
        plan_path = f"results/property/sample_completion_v1/case_plans/{name}.json"
        groups[name] = {
            "architecture": spec["architecture"],
            "model": spec["model"],
            "release_dir": spec["release_dir"],
            "input_bank": spec["input_bank"],
            "case_plan": plan_path,
            "case_count": len(rows),
            "cases": rows,
            "commands": {stage: command(spec, plan_path, stage) for stage in ("engineering", "screen", "formal")},
            "status": "READY_FOR_GPU" if Path(spec["model"]).exists() else "BLOCKED_MODEL_PATH",
        }
        target = ROOT / plan_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({"schema": "kernel-analyzer-sample-completion-case-plan-v1", "cases": rows}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result = {
        "schema": "kernel-analyzer-sample-completion-execution-manifest-v1",
        "status": "READY_FOR_GPU_BUT_NOT_EXECUTED",
        "groups": groups,
        "total_search_units_with_exact_command": sum(g["case_count"] for g in groups.values()),
        "gpu_observation": "nvidia-smi currently cannot communicate with the driver; no scientific run was started by this manifest.",
        "claim_boundary": "Commands and exact bindings are frozen; generated commands do not imply that any 8/16/32-step measurement has completed.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "units": result["total_search_units_with_exact_command"]}, sort_keys=True))


if __name__ == "__main__":
    main()
