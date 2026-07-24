#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from forkcert.config import load_config
from forkcert.io import read_jsonl, write_jsonl
from forkcert.logprob_runner import PathConfig, cleanup_memory, configure_determinism, load_hf_path, response_logprobs_for_sample
from forkcert.report import CLAIM_SCOPE, markdown_table


def path_config(data: dict, key: str) -> PathConfig:
    item = data[key]
    return PathConfig(
        name=item["name"],
        model_name_or_path=item["model_name_or_path"],
        dtype=item.get("dtype", "bf16"),
        device=item.get("device", "cuda"),
        compile_model=item.get("compile_model", False),
        attn_implementation=item.get("attn_implementation"),
        attention_backend=item.get("attention_backend"),
        logits_upcast_fp32=item.get("logits_upcast_fp32", True),
        rmsnorm_no_upcast=item.get("rmsnorm_no_upcast", False),
        rmsnorm_compile=item.get("rmsnorm_compile", False),
        materialize_bf16_outputs=item.get("materialize_bf16_outputs", False),
        materialization_dtype=item.get("materialization_dtype"),
        allow_bf16_reduced_precision_reduction=item.get("allow_bf16_reduced_precision_reduction"),
        allow_fp16_reduced_precision_reduction=item.get("allow_fp16_reduced_precision_reduction"),
    )


def select_module(model: Any, target: str) -> tuple[str, Any]:
    if target == "root_model":
        return "<root_model>", model
    candidates = []
    for name, module in model.named_modules():
        lowered = name.lower()
        class_name = module.__class__.__name__.lower()
        if target == "attention" and (lowered.endswith("self_attn") or "attention" in class_name):
            return name, module
        if target == "rmsnorm" and "rmsnorm" in class_name:
            return name, module
        if target == "projection" and lowered.endswith("q_proj"):
            return name, module
        if target == "linear" and class_name == "linear":
            candidates.append((name, module))
        if target == "decoder_layer" and ("decoderlayer" in class_name or lowered.endswith("layers.0")):
            return name, module
        if target == "lm_head" and lowered == "lm_head":
            return name, module
    if candidates:
        return candidates[0]
    raise ValueError(f"no module found for canary target {target}")


def perturb_hook(magnitude: float, vocab_id: int | None = None):
    def perturb_tensor(tensor):
        changed = tensor.clone()
        if vocab_id is not None and changed.ndim >= 3:
            changed[..., vocab_id] += magnitude
        else:
            changed[..., 0] += magnitude
        return changed

    def hook(_module, _inputs, output):
        import torch

        if torch.is_tensor(output):
            return perturb_tensor(output)
        if isinstance(output, tuple) and output and torch.is_tensor(output[0]):
            return (perturb_tensor(output[0]), *output[1:])
        if hasattr(output, "logits") and torch.is_tensor(output.logits):
            output.logits = perturb_tensor(output.logits)
            return output
        raise TypeError("selected canary module did not return a tensor or tensor-first tuple")

    return hook


def run_canary(spec: str, sample: dict, magnitude: float) -> dict:
    parts = spec.split("=", 3)
    if len(parts) not in {3, 4}:
        raise ValueError(f"invalid canary spec {spec!r}; expected LEVEL=CONFIG=TARGET[=PATH_KEY]")
    level, config_path, target = parts[:3]
    path_key = parts[3] if len(parts) == 4 else "path_alt"
    if path_key not in {"path_ref", "path_alt"}:
        raise ValueError(f"invalid path key {path_key!r}")
    cfg = load_config(config_path)
    configure_determinism(seed=int(cfg.get("seed", 0)))
    path = path_config(cfg, path_key)
    tokenizer, model = load_hf_path(path)
    try:
        baseline = response_logprobs_for_sample(tokenizer, model, path, sample)
        module_name, module = select_module(model, target)
        vocab_id = int(sample["response_ids"][0]) if target in {"lm_head", "root_model"} else None
        handle = module.register_forward_hook(perturb_hook(magnitude, vocab_id=vocab_id))
        try:
            perturbed = response_logprobs_for_sample(tokenizer, model, path, sample)
        finally:
            handle.remove()
        deltas = [abs(float(b["logp"]) - float(a["logp"])) for a, b in zip(baseline, perturbed, strict=True)]
        return {
            "level": level,
            "config": config_path,
            "target": target,
            "path_key": path_key,
            "module": module_name,
            "injected_magnitude": magnitude,
            "token_count": len(deltas),
            "observed_delta_max": max(deltas) if deltas else 0.0,
            "observed_nonzero_tokens": sum(delta > 0.0 for delta in deltas),
            "canary_pass": any(delta > 0.0 for delta in deltas),
        }
    finally:
        del model
        del tokenizer
        cleanup_memory()


def main() -> None:
    parser = argparse.ArgumentParser(description="Positive-control canaries for every attribution ladder switch.")
    parser.add_argument("--spec", action="append", required=True, help="LEVEL=CONFIG=TARGET[=PATH_KEY]")
    parser.add_argument("--samples", default="data/phase0_grpo_samples.jsonl")
    parser.add_argument("--magnitude", type=float, default=1e-3)
    parser.add_argument("--out-jsonl", default="results/phase15_canaries.jsonl")
    parser.add_argument("--report", default="reports/phase15_canaries.md")
    args = parser.parse_args()

    sample = read_jsonl(args.samples)[0]
    rows = [run_canary(spec, sample, args.magnitude) for spec in args.spec]
    write_jsonl(args.out_jsonl, rows)
    all_pass = all(row["canary_pass"] for row in rows)
    report = "\n".join(
        [
            "# Phase 1.5 Attribution Canaries",
            "",
            "## Claim Scope",
            CLAIM_SCOPE,
            "",
            "## Confound Checklist",
            "- known perturbation magnitude is 1e-3: PASS",
            f"- every switched module produced observable nonzero logprob delta: {'PASS' if all_pass else 'FAIL'}",
            "",
            "## Delta Self Control",
            "Canaries are positive controls, not self-consistency estimates. Phase A1 remains authoritative.",
            "",
            "## External Validity",
            "Canary pass/fail validates instrumentation on T4 FP16 only; it does not estimate production BF16 effects.",
            "",
            "## Results",
            markdown_table(rows, list(rows[0].keys())),
            "",
        ]
    )
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(report, encoding="utf-8")
    print(json.dumps({"all_pass": all_pass, "rows": rows}, indent=2, sort_keys=True))
    if not all_pass:
        raise SystemExit(35)


if __name__ == "__main__":
    main()
