#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

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
from phase15_measure_hf import path_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure conservative GEMM sum-absolute input bounds.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--max-samples", type=int, default=4)
    parser.add_argument("--out-json", default="results/phase2_operator_norms.json")
    parser.add_argument("--report", default="reports/phase2_operator_norms.md")
    args = parser.parse_args()

    import torch

    cfg = load_config(args.config)
    configure_determinism(seed=int(cfg.get("seed", 0)))
    config = path_config(cfg, "path_ref")
    samples = read_jsonl(args.samples)[: args.max_samples]
    tokenizer, model = load_hf_path(config)
    stats: dict[str, dict] = {}
    handles = []

    def make_hook(name, module):
        weight_abs_max = float(module.weight.detach().abs().max().item())
        in_features = int(module.in_features)

        def hook(_module, inputs):
            tensor = inputs[0].detach().float()
            input_l1 = tensor.abs().sum(dim=-1)
            row = stats.setdefault(
                name,
                {
                    "module": name,
                    "in_features_reduction_length": in_features,
                    "out_features": int(module.out_features),
                    "weight_abs_max": weight_abs_max,
                    "input_l1_max": 0.0,
                    "sum_abs_product_upper": 0.0,
                    "invocations": 0,
                },
            )
            value = float(input_l1.max().item())
            row["input_l1_max"] = max(float(row["input_l1_max"]), value)
            row["sum_abs_product_upper"] = max(float(row["sum_abs_product_upper"]), value * weight_abs_max)
            row["invocations"] += 1

        return hook

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and name != "lm_head":
            handles.append(module.register_forward_pre_hook(make_hook(name, module)))
    try:
        for sample in samples:
            encoded = _encode_sample(tokenizer, sample, config.device)
            with torch.inference_mode(), attention_backend_context(config), precision_context(config):
                model(input_ids=encoded["input_ids"])
    finally:
        for handle in handles:
            handle.remove()
        del model, tokenizer
        cleanup_memory()

    rows = sorted(stats.values(), key=lambda row: row["sum_abs_product_upper"], reverse=True)
    payload = {
        "measurement_kind": "conservative_linear_reduction_input_bound",
        "samples": len(samples),
        "modules": len(rows),
        "formula": "sum_i |x_i*w_ji| <= ||x||_1 * max_i,j |W_ji|",
        "input_norm_measured": True,
        "algorithm_order_known": False,
        "local_independence_established": False,
        "rows": rows,
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = "\n".join(
        [
            "# Phase 2 Operator Input Norms",
            "",
            "## Claim Scope",
            CLAIM_SCOPE,
            "",
            "## Confound Checklist",
            "- real model inputs measured: PASS",
            "- exact reduction lengths from Linear modules: PASS",
            "- conservative product-sum inequality: PASS",
            "- exact kernel reduction order: FAIL / not yet known",
            "- cross-source local independence: FAIL / not yet established",
            "",
            "## Delta Self Control",
            "This is a one-path norm measurement; Phase 1/A4 self controls remain authoritative.",
            "",
            "## External Validity",
            "Measured on the exact step-5 T4 FP16 snapshot with FP32 master weights and FP16 autocast.",
            "",
            "## Largest Conservative Injection Terms",
            markdown_table(rows[:20], list(rows[0].keys()) if rows else []),
            "",
            "These measurements close the input-norm evidence gap only. They do not make the Phase 2 bound semi-certified until algorithm order and local independence are justified.",
            "",
        ]
    )
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(report, encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ["samples", "modules", "input_norm_measured", "algorithm_order_known", "local_independence_established"]}, indent=2))


if __name__ == "__main__":
    main()
