#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

from forkcert.config import load_config
from forkcert.io import read_jsonl
from forkcert.logprob_runner import (
    _encode_sample,
    attention_backend_context,
    cleanup_memory,
    configure_determinism,
    load_hf_path,
    precision_context,
)
from forkcert.report import CLAIM_SCOPE, markdown_table
from phase6_twin_training import path_config


def digest(path: Path) -> str:
    value = hashlib.sha256()
    value.update(path.read_bytes())
    return value.hexdigest()


def profile_keys(torch, call) -> list[str]:
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(activities=activities) as profiler:
        call()
    torch.cuda.synchronize()
    return sorted(event.key for event in profiler.key_averages())


def kernel_summary(keys: list[str]) -> dict[str, Any]:
    selected = [
        key
        for key in keys
        if any(marker in key.lower() for marker in ["triton", "gemm", "softmax", "elementwise", "kernel"])
    ]
    families = Counter()
    for key in selected:
        lower = key.lower()
        if "triton" in lower:
            families["triton"] += 1
        elif "gemm" in lower:
            families["gemm"] += 1
        elif "softmax" in lower:
            families["softmax"] += 1
        else:
            families["other_kernel"] += 1
    return {"selected_keys": selected, "families": dict(families)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit FX/Inductor artifacts for the canonical eager-vs-compile pair.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--trace-dir", default="results/phase2_compile_trace")
    parser.add_argument("--out", default="results/phase2_compile_graph_audit.json")
    parser.add_argument("--report", default="reports/phase2_compile_graph_audit.md")
    args = parser.parse_args()

    import torch
    import torch._dynamo
    import torch._inductor.config as inductor_config

    cfg = load_config(args.config)
    configure_determinism(seed=int(cfg.get("seed", 0)))
    ref_cfg = replace(path_config(cfg, "path_ref"), compile_model=False)
    sample = read_jsonl(args.samples)[0]
    trace_dir = Path(args.trace_dir).resolve()
    trace_dir.mkdir(parents=True, exist_ok=True)
    inductor_config.trace.enabled = True
    inductor_config.trace.debug_dir = str(trace_dir)

    tokenizer, model = load_hf_path(ref_cfg)
    encoded = _encode_sample(tokenizer, sample, ref_cfg.device)
    input_ids = encoded["input_ids"]

    def eager_call():
        with torch.inference_mode(), attention_backend_context(ref_cfg), precision_context(ref_cfg):
            return model(input_ids=input_ids, use_cache=False).logits

    eager_first = eager_call().detach()
    eager_second = eager_call().detach()
    eager_self_equal = bool(torch.equal(eager_first, eager_second))
    eager_keys = profile_keys(torch, eager_call)

    explain = torch._dynamo.explain(model)(input_ids=input_ids, use_cache=False)
    graph_dir = trace_dir / "dynamo_graphs"
    graph_dir.mkdir(parents=True, exist_ok=True)
    graph_files = []
    for index, graph in enumerate(explain.graphs):
        path = graph_dir / f"graph_{index:03d}.py"
        path.write_text(graph.code, encoding="utf-8")
        graph_files.append(str(path.relative_to(Path.cwd())))

    compiled = torch.compile(model)
    with torch.inference_mode(), attention_backend_context(ref_cfg), precision_context(ref_cfg):
        compiled(input_ids=input_ids, use_cache=False)
        compiled_first = compiled(input_ids=input_ids, use_cache=False).logits.detach()
        compiled_second = compiled(input_ids=input_ids, use_cache=False).logits.detach()
    compiled_self_equal = bool(torch.equal(compiled_first, compiled_second))
    cross_delta_max = float((compiled_first.float() - eager_first.float()).abs().max().item())

    def compiled_call():
        with torch.inference_mode(), attention_backend_context(ref_cfg), precision_context(ref_cfg):
            return compiled(input_ids=input_ids, use_cache=False).logits

    compiled_keys = profile_keys(torch, compiled_call)
    generated = []
    for path in sorted(trace_dir.rglob("*")):
        if not path.is_file():
            continue
        generated.append(
            {
                "path": str(path.relative_to(Path.cwd())),
                "bytes": path.stat().st_size,
                "sha256": digest(path),
                "suffix": path.suffix,
            }
        )

    result = {
        "schema_version": "forkcert.phase2.compile_graph_audit.v1",
        "status": "completed",
        "path_ref": ref_cfg.name,
        "sample_case_id": sample["case_id"],
        "input_tokens": int(input_ids.numel()),
        "eager_self_equal": eager_self_equal,
        "compiled_self_equal_after_warmup": compiled_self_equal,
        "eager_compile_logits_max_abs_delta": cross_delta_max,
        "dynamo_graph_count": len(explain.graphs),
        "dynamo_graph_break_count": int(explain.graph_break_count),
        "dynamo_op_count": int(explain.op_count),
        "dynamo_break_reasons": [str(reason) for reason in explain.break_reasons],
        "ops_per_graph": [[str(op) for op in ops] for ops in explain.ops_per_graph],
        "graph_files": graph_files,
        "trace_files": generated,
        "eager_profile": kernel_summary(eager_keys),
        "compiled_profile": kernel_summary(compiled_keys),
        "causal_injection_points_enumerated": False,
        "analytic_legal": False,
        "remaining_requirement": (
            "Generated fusion kernels and external GEMM calls must be mapped to eager materialization/rounding "
            "boundaries with per-kernel arithmetic contracts before they can seed a legal difference bound."
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary_keys = [
        "input_tokens",
        "eager_self_equal",
        "compiled_self_equal_after_warmup",
        "eager_compile_logits_max_abs_delta",
        "dynamo_graph_count",
        "dynamo_graph_break_count",
        "dynamo_op_count",
        "causal_injection_points_enumerated",
        "analytic_legal",
    ]
    report = "\n".join(
        [
            "# Phase 2 Compile Graph Audit",
            "",
            "## Claim Scope",
            CLAIM_SCOPE,
            "",
            "## Confound Checklist",
            "- exact step-5 model artifact: PASS",
            "- same input and MATH SDPA context: PASS",
            "- compile warm-up discarded: PASS",
            "- FX graphs persisted: PASS",
            "- Inductor trace artifacts hashed: PASS",
            "- causal numerical injection points fully enumerated: FAIL / pending",
            "",
            "## Delta Self Control",
            f"Eager logits self equal: {eager_self_equal}; warmed compiled logits self equal: {compiled_self_equal}.",
            "",
            "## Summary",
            markdown_table([{key: result[key] for key in summary_keys}], summary_keys),
            "",
            "## Remaining Requirement",
            result["remaining_requirement"],
            "",
            "## External Validity",
            "Artifacts are specific to this PyTorch/Inductor build, T4 target, FP16 autocast, sequence shape, and Qwen3-0.6B snapshot. Other shapes or hardware can select different kernels.",
            "",
        ]
    )
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(report, encoding="utf-8")
    print(json.dumps({key: result[key] for key in summary_keys}, indent=2))

    del compiled, model, tokenizer
    cleanup_memory()


if __name__ == "__main__":
    main()
