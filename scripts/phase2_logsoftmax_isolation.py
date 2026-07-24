#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
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
from forkcert.stats import percentile
from phase15_measure_hf import path_config


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_equal(a, b) -> bool:
    import torch

    return bool(torch.equal(a, b))


def main() -> None:
    parser = argparse.ArgumentParser(description="Isolate FP16-vs-FP32 log_softmax on shared model logits.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--max-samples", type=int, default=4)
    parser.add_argument("--out", default="results/phase2_logsoftmax_isolation.json")
    parser.add_argument("--report", default="reports/phase2_logsoftmax_isolation.md")
    args = parser.parse_args()

    import torch

    cfg = load_config(args.config)
    configure_determinism(seed=int(cfg.get("seed", 0)))
    config = path_config(cfg, "path_ref")
    samples = read_jsonl(args.samples)[: args.max_samples]
    tokenizer, model = load_hf_path(config)
    totals: dict[str, Any] = {
        "samples": 0,
        "positions": 0,
        "vocabulary_outputs": 0,
        "fp16_equals_rounded_fp32": 0,
        "fp16_differs_rounded_fp32": 0,
        "max_fp16_vs_fp32_abs": 0.0,
        "max_fp16_vs_rounded_fp32_abs": 0.0,
        "max_abs_fp32_logprob": 0.0,
        "max_abs_fp16_logprob": 0.0,
        "target_delta_max": 0.0,
        "target_delta_sum": 0.0,
        "target_count": 0,
        "logits_dtype": None,
        "half_input_output_dtype": None,
        "float_input_output_dtype": None,
        "logits_self_equal": True,
        "fp16_logsoftmax_self_equal": True,
        "fp32_logsoftmax_self_equal": True,
    }
    first_pass: list[tuple[Any, Any, Any]] = []
    target_deltas: list[float] = []
    kernel_names: set[str] = set()
    try:
        for run in range(2):
            for sample_index, sample in enumerate(samples):
                encoded = _encode_sample(tokenizer, sample, config.device)
                input_ids = encoded["input_ids"]
                prompt_len = int(encoded["prompt_len"])
                with torch.inference_mode(), attention_backend_context(config), precision_context(config):
                    logits = model(input_ids=input_ids).logits[:, prompt_len - 1 : -1, :].contiguous()
                    if run == 0 and sample_index == 0:
                        activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
                        with torch.profiler.profile(activities=activities) as profiler:
                            _ = torch.nn.functional.log_softmax(logits, dim=-1)
                            _ = torch.nn.functional.log_softmax(logits.float(), dim=-1)
                        torch.cuda.synchronize()
                        kernel_names.update(
                            event.key
                            for event in profiler.key_averages()
                            if "softmax" in event.key.lower()
                        )
                    fp16_lp = torch.nn.functional.log_softmax(logits, dim=-1)
                    fp32_lp = torch.nn.functional.log_softmax(logits.float(), dim=-1)
                    rounded = fp32_lp.to(dtype=logits.dtype)

                if run == 0:
                    totals["samples"] += 1
                    totals["positions"] += int(logits.shape[1])
                    count = int(logits.numel())
                    equal = int((fp16_lp == rounded).sum().item())
                    totals["vocabulary_outputs"] += count
                    totals["fp16_equals_rounded_fp32"] += equal
                    totals["fp16_differs_rounded_fp32"] += count - equal
                    totals["max_fp16_vs_fp32_abs"] = max(
                        totals["max_fp16_vs_fp32_abs"],
                        float((fp16_lp.float() - fp32_lp).abs().max().item()),
                    )
                    totals["max_fp16_vs_rounded_fp32_abs"] = max(
                        totals["max_fp16_vs_rounded_fp32_abs"],
                        float((fp16_lp.float() - rounded.float()).abs().max().item()),
                    )
                    totals["max_abs_fp32_logprob"] = max(
                        totals["max_abs_fp32_logprob"], float(fp32_lp.abs().max().item())
                    )
                    totals["max_abs_fp16_logprob"] = max(
                        totals["max_abs_fp16_logprob"], float(fp16_lp.abs().max().item())
                    )
                    response = torch.tensor(sample["response_ids"], device=logits.device).view(1, -1, 1)
                    fp16_targets = fp16_lp.gather(-1, response).squeeze(-1).float()
                    fp32_targets = fp32_lp.gather(-1, response).squeeze(-1)
                    target_delta = (fp16_targets - fp32_targets).abs()
                    totals["target_delta_max"] = max(
                        totals["target_delta_max"], float(target_delta.max().item())
                    )
                    totals["target_delta_sum"] += float(target_delta.sum().item())
                    totals["target_count"] += int(target_delta.numel())
                    target_deltas.extend(float(value) for value in target_delta.cpu().tolist()[0])
                    totals["logits_dtype"] = str(logits.dtype)
                    totals["half_input_output_dtype"] = str(fp16_lp.dtype)
                    totals["float_input_output_dtype"] = str(fp32_lp.dtype)
                    first_pass.append((logits.detach().cpu(), fp16_lp.detach().cpu(), fp32_lp.detach().cpu()))
                else:
                    old_logits, old_fp16, old_fp32 = first_pass[sample_index]
                    totals["logits_self_equal"] &= tensor_equal(old_logits, logits.cpu())
                    totals["fp16_logsoftmax_self_equal"] &= tensor_equal(old_fp16, fp16_lp.cpu())
                    totals["fp32_logsoftmax_self_equal"] &= tensor_equal(old_fp32, fp32_lp.cpu())
                del logits, fp16_lp, fp32_lp, rounded
        totals["target_delta_mean"] = totals["target_delta_sum"] / totals["target_count"]
        totals["target_delta_p99"] = percentile(target_deltas, 99)
        totals["rounded_equality_rate"] = (
            totals["fp16_equals_rounded_fp32"] / totals["vocabulary_outputs"]
        )
    finally:
        del model, tokenizer
        cleanup_memory()

    header = Path(torch.__file__).parent / "include/ATen/native/cuda/PersistentSoftmax.cuh"
    evidence = {
        "schema_version": "forkcert.phase2.logsoftmax_isolation.v1",
        "status": "completed",
        **totals,
        "kernel_event_names": sorted(kernel_names),
        "pytorch_softmax_header": str(header),
        "pytorch_softmax_header_sha256": sha256(header) if header.exists() else None,
        "upstream_isolation": bool(totals["logits_self_equal"]),
        "output_rounding_only_observed": (
            totals["half_input_output_dtype"] == "torch.float16"
            and totals["fp16_differs_rounded_fp32"] == 0
        ),
        "autocast_promoted_half_input_output": (
            totals["logits_dtype"] == "torch.float16"
            and totals["half_input_output_dtype"] == "torch.float32"
        ),
        "analytic_legal": False,
        "analytic_legal_reason": (
            "Under canonical autocast, half-input log_softmax returns FP32, so this is an input-dispatch "
            "comparison rather than final FP16 output rounding. Kernel reduction order and CUDA exp/log "
            "ULP contracts are still required for an analytic legal bound."
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    summary_keys = [
        "samples",
        "positions",
        "vocabulary_outputs",
        "fp16_differs_rounded_fp32",
        "rounded_equality_rate",
        "target_delta_mean",
        "target_delta_p99",
        "target_delta_max",
        "max_fp16_vs_fp32_abs",
        "logits_dtype",
        "half_input_output_dtype",
        "float_input_output_dtype",
        "logits_self_equal",
        "fp16_logsoftmax_self_equal",
        "fp32_logsoftmax_self_equal",
        "analytic_legal",
    ]
    report = "\n".join(
        [
            "# Phase 2 Log-Softmax Isolation",
            "",
            "## Claim Scope",
            CLAIM_SCOPE,
            "",
            "## Confound Checklist",
            "- same model forward supplies both log-softmax paths: PASS",
            "- token and vocabulary dimensions identical: PASS",
            "- two independent measured calls per path: PASS",
            "- canonical autocast output dtype recorded: PASS",
            "- CUDA exp/log ULP contract established: FAIL / pending",
            "- large-vocabulary kernel reduction order established: FAIL / pending",
            "",
            "## Delta Self Control",
            f"Logits self equal: {evidence['logits_self_equal']}; FP16 log-softmax self equal: {evidence['fp16_logsoftmax_self_equal']}; FP32 log-softmax self equal: {evidence['fp32_logsoftmax_self_equal']}.",
            "",
            "## Summary",
            markdown_table([{key: evidence[key] for key in summary_keys}], summary_keys),
            "",
            "## Interpretation",
            evidence["analytic_legal_reason"],
            "",
            "## External Validity",
            "Measured with FP16 autocast on Tesla T4. It isolates the local output behavior of this installed PyTorch build only; native BF16 requires separate measurement and bounds.",
            "",
        ]
    )
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(report, encoding="utf-8")
    print(json.dumps({key: evidence[key] for key in summary_keys + ["kernel_event_names"]}, indent=2))


if __name__ == "__main__":
    main()
