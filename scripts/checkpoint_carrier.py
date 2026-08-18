#!/usr/bin/env python3
"""Freeze a pilot gradient-difference direction across natural checkpoints."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path

import torch

from scripts.checkpoint_matrix import (
    build_model,
    capture_run,
    diff_pair,
    load_natural_validation,
    strip_tensors,
    under_root,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank-manifest", type=Path, default=Path("results/final/natural_bank.json"))
    parser.add_argument("--model", type=Path, default=Path("/data1/tzh/models/Qwen/Qwen3-1.7B"))
    parser.add_argument("--output", type=Path, default=Path("results/final/checkpoint_carrier.json"))
    parser.add_argument("--dataset", default="Salesforce/wikitext")
    parser.add_argument("--dataset-config", default="wikitext-103-raw-v1")
    parser.add_argument("--eval-split", default="validation")
    parser.add_argument("--eval-offset", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--variants", nargs="+", default=["sdpa_math", "sdpa_flash"])
    parser.add_argument("--min-step", type=int, default=0, help="evaluate step 0 plus checkpoints at or after this step")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.model = under_root(args.model, "model")
    args.bank_manifest = under_root(args.bank_manifest, "bank manifest")
    args.output = under_root(args.output, "output")
    os.environ.setdefault("HF_HOME", "/data1/tzh/cache/huggingface")
    os.environ.setdefault("HF_DATASETS_CACHE", "/data1/tzh/cache/huggingface/datasets")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/data1/tzh/cache/huggingface/transformers")
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/data1/tzh/cache/huggingface/hub")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    args.device = torch.device(args.device)
    bank = json.loads(args.bank_manifest.read_text())
    tokenizer_module = __import__("transformers", fromlist=["AutoTokenizer"])
    tokenizer = tokenizer_module.AutoTokenizer.from_pretrained(args.model, local_files_only=True, use_fast=True)
    inputs, eval_protocol = load_natural_validation(tokenizer, args)
    target_params = ["model.embed_tokens.weight"]
    bases: dict[str, torch.Tensor] = {}
    rows: list[dict] = []
    checkpoint_rows = [row for row in bank["checkpoints"] if row["step"] == 0 or row["step"] >= args.min_step]
    for checkpoint_row in checkpoint_rows:
        checkpoint = under_root(Path(checkpoint_row["path"]), "checkpoint")
        print(f"checkpoint step {checkpoint_row['step']}", flush=True)
        reference_model = build_model(args.model, checkpoint, "eager", torch.bfloat16, args.device)
        reference = capture_run(reference_model, inputs, "eager", target_params, capture_attention=False)
        del reference_model
        gc.collect()
        torch.cuda.empty_cache()
        variant_rows = {}
        for variant in args.variants:
            print(f"  variant {variant}", flush=True)
            try:
                candidate_model = build_model(args.model, checkpoint, variant, torch.bfloat16, args.device)
                candidate = capture_run(candidate_model, inputs, variant, target_params, capture_attention=False)
                pair = diff_pair(reference, candidate)
                raw = pair.pop("_raw_param_deltas")["model.embed_tokens.weight"]
                if variant not in bases:
                    bases[variant] = raw / (raw.norm() + 1e-30)
                    pilot = True
                else:
                    pilot = False
                projection = float((raw * bases[variant]).sum())
                norm = float(raw.norm())
                variant_rows[variant] = {
                    "status": "OK",
                    "loss_delta": pair["loss_delta"],
                    "embed_gradient_l2": norm,
                    "pilot_step": 0,
                    "is_pilot": pilot,
                    "carrier_projection": projection,
                    "carrier_cosine": projection / (norm + 1e-30),
                    "carrier_positive": projection > 0.0,
                    "parameter_diff": pair["parameter_carriers"]["model.embed_tokens.weight"],
                }
                del raw, candidate, candidate_model, pair
            except Exception as exc:
                variant_rows[variant] = {"status": "UNAVAILABLE", "error": repr(exc)}
                try:
                    del candidate_model
                except UnboundLocalError:
                    pass
            gc.collect()
            torch.cuda.empty_cache()
        del reference
        gc.collect()
        torch.cuda.empty_cache()
        rows.append({
            "checkpoint_step": checkpoint_row["step"],
            "checkpoint_parameter_sha256": checkpoint_row["parameter_sha256"],
            "variants": variant_rows,
        })
    summaries = {}
    for variant in args.variants:
        values = [row["variants"].get(variant, {}) for row in rows]
        valid = [value for value in values if value.get("status") == "OK"]
        heldout = [value for row, value in zip(rows, values) if row["checkpoint_step"] != 0 and value.get("status") == "OK"]
        summaries[variant] = {
            "valid_states": len(valid),
            "heldout_states": len(heldout),
            "heldout_positive": sum(value["carrier_positive"] for value in heldout),
            "heldout_positive_fraction": (sum(value["carrier_positive"] for value in heldout) / len(heldout)) if heldout else None,
            "heldout_mean_projection": (sum(value["carrier_projection"] for value in heldout) / len(heldout)) if heldout else None,
            "heldout_min_projection": min((value["carrier_projection"] for value in heldout), default=None),
            "heldout_mean_cosine": (sum(value["carrier_cosine"] for value in heldout) / len(heldout)) if heldout else None,
        }
    result = {
        "schema": "kernel-analyzer-checkpoint-carrier-v1",
        "subject": "Qwen3-1.7B tied embedding gradient",
        "bank_manifest": str(args.bank_manifest),
        "bank_protocol_sha256": bank["protocol_sha256"],
        "evaluation": eval_protocol,
        "reference": "eager BF16 with materialized causal mask",
        "pilot": {"checkpoint_step": 0, "parameter": "model.embed_tokens.weight", "candidate_independent": True},
        "variants": {
            "sdpa_math": {"changed": True, "mechanisms": ["backend", "reduction_schedule", "materialization"]},
            "sdpa_flash": {"changed": True, "mechanisms": ["backend", "online_reduction", "layout", "materialization"], "accumulator": "not directly observable from this interface"},
        },
        "rows": rows,
        "summary": summaries,
        "boundary": "Pilot direction is frozen at step 0; step 1/2/4/8/16 are held-out evolving states. This is a carrier screen, not a complete bias certificate without a live-weight intervention.",
    }
    result["result_sha256"] = hashlib.sha256(json.dumps(result, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "rows": len(rows), "summary": summaries}, sort_keys=True))


if __name__ == "__main__":
    main()
