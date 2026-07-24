#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import re
import warnings
from pathlib import Path

from forkcert.config import load_config
from forkcert.hooks import ActivationTensorRecorder
from forkcert.io import read_jsonl
from forkcert.logprob_runner import PathConfig, cleanup_memory, configure_determinism, load_hf_path, response_logprobs_for_sample


def path_config(data: dict, key: str) -> PathConfig:
    item = data[key]
    return PathConfig(
        name=item["name"],
        model_name_or_path=item["model_name_or_path"],
        dtype=item.get("dtype", "bf16"),
        autocast_dtype=item.get("autocast_dtype"),
        device=item.get("device", "cuda"),
        compile_model=item.get("compile_model", False),
        attn_implementation=item.get("attn_implementation"),
        attention_backend=item.get("attention_backend"),
        logits_upcast_fp32=item.get("logits_upcast_fp32", True),
        rmsnorm_reference=item.get("rmsnorm_reference", False),
        rmsnorm_no_upcast=item.get("rmsnorm_no_upcast", False),
        rmsnorm_compile=item.get("rmsnorm_compile", False),
        materialize_bf16_outputs=item.get("materialize_bf16_outputs", False),
        materialization_dtype=item.get("materialization_dtype"),
        allow_bf16_reduced_precision_reduction=item.get("allow_bf16_reduced_precision_reduction"),
        allow_fp16_reduced_precision_reduction=item.get("allow_fp16_reduced_precision_reduction"),
        model_training_mode=item.get("model_training_mode", False),
        gradient_checkpointing=item.get("gradient_checkpointing", False),
    )


LAYER_NAME_RE = re.compile(r"(?:^|\.)(?:layers|h|block)\.(\d+)$")


def transformer_layer_index(name: str) -> int | None:
    match = LAYER_NAME_RE.search(name)
    return int(match.group(1)) if match else None


def module_filter(name: str, _module) -> bool:
    # Full decoder-block outputs are residual-stream checkpoints. Submodule
    # invocation order is not a valid proxy for transformer depth.
    return transformer_layer_index(name) is not None


def compare_activation_l2(ref_records, alt_records) -> dict:
    import torch

    ref = {(record.name, record.invocation): record.tensor for record in ref_records}
    alt = {(record.name, record.invocation): record.tensor for record in alt_records}
    common = [(record.name, record.invocation) for record in ref_records if (record.name, record.invocation) in alt]
    invocation_rows = []
    for name, invocation in common:
        if ref[(name, invocation)].shape != alt[(name, invocation)].shape:
            continue
        diff = alt[(name, invocation)].float() - ref[(name, invocation)].float()
        ref_l2 = float(torch.linalg.vector_norm(ref[(name, invocation)].float()).item())
        diff_l2 = float(torch.linalg.vector_norm(diff).item())
        layer_index = transformer_layer_index(name)
        if layer_index is None:
            continue
        invocation_rows.append(
            {
                "name": name,
                "layer_index": layer_index,
                "invocation": invocation,
                "diff_l2": diff_l2,
                "diff_max_abs": float(diff.abs().max().item()) if diff.numel() else 0.0,
                "relative_l2": diff_l2 / ref_l2 if ref_l2 > 0 else 0.0,
            }
        )
    grouped: dict[int, list[dict]] = {}
    for row in invocation_rows:
        grouped.setdefault(int(row["layer_index"]), []).append(row)
    layerwise = []
    previous_diff = None
    for layer_index, rows in sorted(grouped.items()):
        diff_l2 = sum(float(row["diff_l2"]) for row in rows) / len(rows)
        ref_relative = sum(float(row["relative_l2"]) for row in rows) / len(rows)
        layerwise.append(
            {
                "layer_index": layer_index,
                "sample_invocations": len(rows),
                "diff_l2": diff_l2,
                "diff_max_abs": max(float(row["diff_max_abs"]) for row in rows),
                "relative_l2": ref_relative,
                "upstream_diff_l2": previous_diff,
                "observed_diff_increment_l2": diff_l2 - previous_diff if previous_diff is not None else diff_l2,
                "propagation_gain_from_previous": diff_l2 / previous_diff if previous_diff and previous_diff > 0 else None,
            }
        )
        previous_diff = diff_l2
    nonzero = [row for row in layerwise if row["diff_l2"] > 0]
    first = nonzero[0]["diff_l2"] if nonzero else 0.0
    last = nonzero[-1]["diff_l2"] if nonzero else 0.0
    return {
        "paired_activation_count": len(layerwise),
        "paired_activation_invocations": len(invocation_rows),
        "residual_layer_indexed": True,
        "local_injection_separated": False,
        "local_injection_note": (
            "Layer-to-layer error increments are observed residual-stream changes, not a causal separation "
            "of local implementation injection from transformed upstream error."
        ),
        "first_observed_diff_l2": first,
        "last_observed_diff_l2": last,
        "max_activation_diff_l2": max((row["diff_l2"] for row in layerwise), default=0.0),
        "mean_activation_diff_l2": sum(row["diff_l2"] for row in layerwise) / len(layerwise) if layerwise else 0.0,
        "propagation_gain_first_to_last": last / first if first > 0 else None,
        "layerwise_activation_diffs": layerwise,
    }


def measure_one_path(config: PathConfig, samples: list[dict], max_modules: int) -> tuple[list[list[dict]], list]:
    tokenizer, model = load_hf_path(config)
    recorder = ActivationTensorRecorder(module_filter, max_modules=max_modules).attach(model)
    try:
        all_rows = []
        for sample in samples:
            all_rows.append(response_logprobs_for_sample(tokenizer, model, config, sample))
        return all_rows, list(recorder.records)
    finally:
        recorder.close()
        del model
        del tokenizer
        cleanup_memory()


def measure_pair(ref_cfg: PathConfig, alt_cfg: PathConfig, samples: list[dict], max_modules: int) -> dict:
    ref_outputs, ref_records = measure_one_path(ref_cfg, samples, max_modules)
    alt_outputs, alt_records = measure_one_path(alt_cfg, samples, max_modules)
    deltas = []
    token_count = 0
    for sample, ref_rows, alt_rows in zip(samples, ref_outputs, alt_outputs):
        if len(ref_rows) != len(alt_rows):
            raise ValueError(f"token count mismatch for {sample['case_id']}")
        for ref_row, alt_row in zip(ref_rows, alt_rows):
            if ref_row["token_id"] != alt_row["token_id"]:
                raise ValueError(f"token id mismatch for {sample['case_id']} token {ref_row['token_index']}")
            deltas.append(abs(float(alt_row["logp"]) - float(ref_row["logp"])))
            token_count += 1
    activation = compare_activation_l2(ref_records, alt_records)
    final_delta = sum(deltas) / len(deltas) if deltas else 0.0
    return {
        "token_count": token_count,
        "measurement_kind": "paired_residual_stream_tensor_diff",
        **activation,
        "final_logprob_delta": final_delta,
        "max_logprob_delta": max(deltas) if deltas else 0.0,
        "recorded_modules_ref": len(ref_records),
        "recorded_modules_alt": len(alt_records),
        "sequential_model_loading": True,
    }


def infer_level(config_path: str) -> tuple[str, str, str]:
    name = Path(config_path).name
    if "sdpa" in name:
        return "L1", "attention backend", "algorithm_structure"
    if "logsoftmax" in name:
        return "L4", "log_softmax precision", "rounding_precision"
    if "rmsnorm" in name:
        return "L2", "RMSNorm fused/unfused", "materialization_points"
    if "materialization" in name:
        return "L3", "intermediate materialization", "materialization_points"
    if "matmul_reduction" in name:
        return "L5", "low-precision matmul reduction precision", "reduction_precision"
    if "debug_fp32_bf16" in name:
        return "debug", "dtype fp32 vs bf16", "precision_debug"
    return "L6", "torch.compile", "mixed"


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure one HF attribution ladder level with compact activation stats.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--samples", default="data/prompt_pairs.jsonl")
    parser.add_argument("--out-jsonl", default="results/phase15_measurements.jsonl")
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--max-modules", type=int, default=96)
    args = parser.parse_args()

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("PYTHONHASHSEED", "0")
    configure_determinism(seed=0)
    cfg = load_config(args.config)
    samples = read_jsonl(args.samples)[: args.max_samples]
    ref_cfg = path_config(cfg, "path_ref")
    alt_cfg = path_config(cfg, "path_alt")
    level, variable, mechanism = infer_level(args.config)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        measured = measure_pair(ref_cfg, alt_cfg, samples, args.max_modules)
    row = {
        "level": level,
        "variable": variable,
        "mechanism": mechanism,
        "path_ref": ref_cfg.name,
        "path_alt": alt_cfg.name,
        "config": args.config,
        "warnings": sorted({str(item.message) for item in caught}),
        "intervention": {
            "effective_by_construction": not (
                level == "L3"
                and ref_cfg.dtype == alt_cfg.dtype
                and (alt_cfg.materialization_dtype or alt_cfg.dtype) == alt_cfg.dtype
            ),
            "materialization_dtype": alt_cfg.materialization_dtype,
            "interpretation": (
                "controlled cross-format roundtrip sensitivity; not attribution to an observed backend materialization"
                if level == "L3"
                else "single configured path variable"
            ),
        },
        **measured,
    }
    out = Path(args.out_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps(row, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
