#!/usr/bin/env python3
"""Compare a real full-step Inductor candidate with eager on bank checkpoints."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path

import torch

from scripts.checkpoint_matrix import build_model, load_natural_validation, under_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank-manifest", type=Path, default=Path("results/final/natural_bank.json"))
    parser.add_argument("--model", type=Path, default=Path("/data1/tzh/models/Qwen/Qwen3-1.7B"))
    parser.add_argument("--output", type=Path, default=Path("results/final/checkpoint_inductor.json"))
    parser.add_argument("--dataset", default="Salesforce/wikitext")
    parser.add_argument("--dataset-config", default="wikitext-103-raw-v1")
    parser.add_argument("--eval-split", default="validation")
    parser.add_argument("--eval-offset", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--min-step", type=int, default=0)
    parser.add_argument("--limit-checkpoints", type=int, default=1)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--tf32", action="store_true", help="enable TF32 matmul for FP32 reference/candidate")
    return parser.parse_args()


def capture_plain(model: torch.nn.Module, inputs: tuple[torch.Tensor, torch.Tensor], target_params: list[str]) -> tuple[float, dict[str, torch.Tensor]]:
    input_ids, labels = inputs
    model.zero_grad(set_to_none=True)
    output = model(input_ids=input_ids, labels=labels, use_cache=False, return_dict=False)[0]
    loss = output
    value = float(loss.detach().cpu())
    loss.backward()
    named = dict(model.named_parameters())
    grads = {name: named[name].grad.detach().cpu().clone() for name in target_params if named[name].grad is not None}
    return value, grads


def digest(value: torch.Tensor) -> str:
    return hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def pair_stats(reference: dict[str, torch.Tensor], candidate: dict[str, torch.Tensor]) -> dict:
    rows = {}
    all_names = sorted(set(reference) | set(candidate))
    for name in all_names:
        if name not in reference or name not in candidate:
            rows[name] = {"status": "MISSING"}
            continue
        delta = candidate[name].float() - reference[name].float()
        denom = candidate[name].float().norm() * reference[name].float().norm()
        rows[name] = {
            "rms": float(delta.square().mean().sqrt()),
            "l2": float(delta.norm()),
            "max_abs": float(delta.abs().max()),
            "mean": float(delta.mean()),
            "cosine": float((candidate[name].float() * reference[name].float()).sum() / (denom + 1e-30)),
            "nonzero": int((delta != 0).sum()),
            "reference_sha256": digest(reference[name]),
            "candidate_sha256": digest(candidate[name]),
        }
    return rows


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
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    if args.tf32 and args.dtype != "fp32":
        raise ValueError("--tf32 requires --dtype fp32")
    torch.backends.cuda.matmul.allow_tf32 = bool(args.tf32)
    torch.set_float32_matmul_precision("high" if args.tf32 else "highest")
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    bank = json.loads(args.bank_manifest.read_text())
    tokenizer_mod = __import__("transformers", fromlist=["AutoTokenizer"])
    tokenizer = tokenizer_mod.AutoTokenizer.from_pretrained(args.model, local_files_only=True, use_fast=True)
    inputs, eval_protocol = load_natural_validation(tokenizer, args)
    target_params = ["model.embed_tokens.weight"]
    target_params.extend(f"model.layers.{i}.self_attn.{part}.weight" for i in range(28) for part in ("q_proj", "k_proj", "v_proj", "o_proj"))
    rows = []
    carrier_base = None
    checkpoint_rows = [row for row in bank["checkpoints"] if row["step"] >= args.min_step]
    if args.limit_checkpoints is not None:
        checkpoint_rows = checkpoint_rows[: args.limit_checkpoints]
    for checkpoint_row in checkpoint_rows:
        checkpoint = under_root(Path(checkpoint_row["path"]), "checkpoint")
        print(f"checkpoint step {checkpoint_row['step']}", flush=True)
        model = None
        reference_grads = {}
        try:
            model = build_model(args.model, checkpoint, "eager", dtype, args.device)
            reference_loss, reference_grads = capture_plain(model, inputs, target_params)
            model.zero_grad(set_to_none=True)
            with torch.no_grad():
                eager_repeat = model(input_ids=inputs[0], labels=inputs[1], use_cache=False, return_dict=False)[0]
            eager_repeat_loss = float(eager_repeat.detach().cpu())

            class LossStep(torch.nn.Module):
                def __init__(self, subject: torch.nn.Module) -> None:
                    super().__init__()
                    self.subject = subject

                def forward(self, input_ids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
                    return self.subject(input_ids=input_ids, labels=labels, use_cache=False, return_dict=False)[0]

            audit = {"backend_compiles": 0, "runtime_invocations": 0, "graph_hashes": []}
            from torch._dynamo.backends.registry import lookup_backend

            inductor = lookup_backend("inductor")

            def backend(graph_module, example_inputs):
                audit["backend_compiles"] += 1
                audit["graph_hashes"].append(hashlib.sha256(graph_module.code.encode()).hexdigest())
                compiled = inductor(graph_module, example_inputs)

                def counted(*values):
                    audit["runtime_invocations"] += 1
                    return compiled(*values)

                return counted

            candidate = torch.compile(LossStep(model), backend=backend, fullgraph=True, dynamic=False)
            candidate_loss_tensor = candidate(inputs[0], inputs[1])
            candidate_loss = float(candidate_loss_tensor.detach().cpu())
            candidate_repeat = candidate(inputs[0], inputs[1])
            candidate_repeat_loss = float(candidate_repeat.detach().cpu())
            candidate_loss_tensor.backward()
            named = dict(model.named_parameters())
            candidate_grads = {name: named[name].grad.detach().cpu().clone() for name in target_params if named[name].grad is not None}
            embed_delta = candidate_grads["model.embed_tokens.weight"].float() - reference_grads["model.embed_tokens.weight"].float()
            if carrier_base is None:
                carrier_base = embed_delta / (embed_delta.norm() + 1e-30)
                carrier_is_pilot = True
            else:
                carrier_is_pilot = False
            carrier_projection = float((embed_delta * carrier_base).sum())
            carrier_norm = float(embed_delta.norm())
            rows.append({
                "checkpoint_step": checkpoint_row["step"],
                "checkpoint_parameter_sha256": checkpoint_row["parameter_sha256"],
                "status": "OK",
                "reference_loss": reference_loss,
                "eager_repeat_loss": eager_repeat_loss,
                "candidate_loss": candidate_loss,
                "candidate_repeat_loss": candidate_repeat_loss,
                "loss_delta": candidate_loss - reference_loss,
                "candidate_repeat_delta": candidate_repeat_loss - candidate_loss,
                "carrier": {
                    "parameter": "model.embed_tokens.weight",
                    "pilot_step": 0,
                    "is_pilot": carrier_is_pilot,
                    "projection": carrier_projection,
                    "cosine": carrier_projection / (carrier_norm + 1e-30),
                    "positive": carrier_projection > 0.0,
                },
                "compile_audit": audit,
                "parameter_carriers": pair_stats(reference_grads, candidate_grads),
            })
            del candidate, candidate_grads, candidate_loss_tensor
        except Exception as exc:
            rows.append({
                "checkpoint_step": checkpoint_row["step"],
                "checkpoint_parameter_sha256": checkpoint_row["parameter_sha256"],
                "status": "UNAVAILABLE",
                "error": repr(exc),
            })
        finally:
            reference_grads.clear()
            del model
            gc.collect()
            torch.cuda.empty_cache()
    result = {
        "schema": "kernel-analyzer-evolving-inductor-v1",
        "subject": "Qwen3-1.7B full training step",
        "bank_manifest": str(args.bank_manifest),
        "bank_protocol_sha256": bank["protocol_sha256"],
        "evaluation": eval_protocol,
        "dtype": args.dtype,
        "tf32": bool(args.tf32),
        "reference": f"eager {args.dtype.upper()} full step",
        "candidate": {"backend": "torch.compile/Inductor", "changed": True, "mechanisms": ["fusion", "reduction_schedule", "materialization", "layout"], "accumulator": "not inferred"},
        "rows": rows,
        "boundary": "This measures full-step compiled-vs-eager differences on already closed F+B semantic units; generated-region attribution remains bounded by the static implementation atlas.",
    }
    result["result_sha256"] = hashlib.sha256(json.dumps(result, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "rows": len(rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
