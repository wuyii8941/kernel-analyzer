#!/usr/bin/env python3
"""Measure real attention implementation differences on evolving checkpoints.

The mathematical proof ledger is intentionally not rebuilt here.  This script
loads each already-proven Qwen checkpoint, runs the same natural validation
batch through an eager reference and real SDPA backends, and records only the
changed F+B attention regions plus selected parameter-gradient carriers.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Any


DATA_ROOT = Path("/data1/tzh").resolve()


def under_root(path: Path, label: str) -> Path:
    value = path.expanduser().resolve()
    if DATA_ROOT not in (value, *value.parents):
        raise ValueError(f"{label} must stay under {DATA_ROOT}: {value}")
    return value


def tensor_digest(value: Any) -> str:
    import torch

    tensor = value.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(repr(tuple(tensor.shape)).encode())
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def tensor_summary(value: Any) -> dict[str, Any]:
    import torch

    tensor = value.detach().float().cpu()
    return {
        "shape": list(tensor.shape),
        "dtype": str(value.dtype),
        "l2": float(tensor.norm()),
        "max_abs": float(tensor.abs().max()),
        "mean": float(tensor.mean()),
        "sha256": tensor_digest(value),
        "tensor": tensor,
    }


def compare_tensors(left: Any, right: Any) -> dict[str, Any]:
    import torch

    delta = left.float() - right.float()
    denom = left.float().norm() * right.float().norm()
    return {
        "max_abs": float(delta.abs().max()),
        "rms": float(delta.square().mean().sqrt()),
        "mean": float(delta.mean()),
        "l2": float(delta.norm()),
        "cosine": float((left.float() * right.float()).sum() / (denom + 1e-30)),
        "nonzero": int((delta != 0).sum()),
    }


def load_natural_validation(tokenizer: Any, args: argparse.Namespace) -> tuple[Any, dict[str, Any]]:
    import torch
    from datasets import load_dataset

    dataset = load_dataset(
        args.dataset,
        args.dataset_config,
        split=args.eval_split,
        revision="main",
        download_mode="reuse_dataset_if_exists",
    )
    tokens: list[int] = []
    documents = 0
    required = args.seq_len + 1
    for row in dataset:
        text = str(row["text"]).strip()
        if not text:
            continue
        documents += 1
        tokens.extend(tokenizer(text, add_special_tokens=False, return_attention_mask=False)["input_ids"])
        if len(tokens) >= args.eval_offset + required:
            break
    start = args.eval_offset
    if len(tokens) < start + required:
        raise RuntimeError("validation token stream is too short")
    block = tokens[start : start + required]
    ids = torch.tensor([block[:-1]], dtype=torch.long, device=args.device)
    labels = torch.tensor([block[1:]], dtype=torch.long, device=args.device)
    return (ids, labels), {
        "split": args.eval_split,
        "offset": start,
        "documents_consumed": documents,
        "token_sha256": hashlib.sha256(json.dumps(block, separators=(",", ":")).encode()).hexdigest(),
        "seq_len": args.seq_len,
    }


def backend_context(variant: str):
    if variant == "eager":
        return nullcontext()
    from torch.nn.attention import SDPBackend, sdpa_kernel

    if variant == "sdpa_math":
        return sdpa_kernel(SDPBackend.MATH)
    if variant == "sdpa_flash":
        return sdpa_kernel(SDPBackend.FLASH_ATTENTION)
    raise ValueError(variant)


def build_model(model_path: Path, checkpoint: Path, variant: str, dtype: Any, device: Any):
    import torch
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    config._attn_implementation = "eager" if variant == "eager" else "sdpa"
    model = AutoModelForCausalLM.from_config(config)
    state = load_file(str(checkpoint), device="cpu")
    incompatible = model.load_state_dict(state, strict=True, assign=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"checkpoint mismatch: {incompatible}")
    del state
    model = model.to(device=device, dtype=dtype)
    model.config.use_cache = False
    model.eval()
    return model


def capture_run(model: Any, inputs: tuple[Any, Any], variant: str, target_params: list[str], capture_attention: bool = True) -> dict[str, Any]:
    import torch

    module_names = [name for name, module in model.named_modules() if capture_attention and name.startswith("model.layers.") and name.endswith(".self_attn")]
    named_modules = dict(model.named_modules())
    forward_values: dict[str, Any] = {}
    backward_values: dict[str, Any] = {}

    def forward_hook(name: str):
        def hook(_module, _args, output):
            value = output[0] if isinstance(output, (tuple, list)) else output
            if torch.is_tensor(value):
                forward_values[name] = value.detach().float().cpu()
        return hook

    def backward_hook(name: str):
        def hook(_module, _grad_input, grad_output):
            value = grad_output[0] if isinstance(grad_output, (tuple, list)) else grad_output
            if torch.is_tensor(value):
                backward_values[name] = value.detach().float().cpu()
        return hook

    handles = []
    for name in module_names:
        handles.append(named_modules[name].register_forward_hook(forward_hook(name)))
        handles.append(named_modules[name].register_full_backward_hook(backward_hook(name)))
    input_ids, labels = inputs
    model.zero_grad(set_to_none=True)
    # The SDPA path receives an explicit causal-mask mapping from newer
    # Transformers releases.  Supplying ``None`` for a full-attention layer
    # lets PyTorch use its native ``is_causal=True`` branch, which is the only
    # route that can select the real CUDA flash kernel.  Eager keeps the normal
    # materialized-mask reference path.
    attention_kwargs = {}
    if variant != "eager":
        attention_kwargs["attention_mask"] = {"full_attention": None}
    with backend_context(variant):
        output = model(input_ids=input_ids, labels=labels, use_cache=False, return_dict=True, **attention_kwargs)
        loss = output.loss
        loss_value = float(loss.detach().cpu())
        loss.backward()
    for handle in handles:
        handle.remove()

    params: dict[str, dict[str, Any]] = {}
    for name in target_params:
        parameter = dict(model.named_parameters()).get(name)
        if parameter is None or parameter.grad is None:
            params[name] = {"present": False}
            continue
        value = parameter.grad.detach().cpu()
        params[name] = {"present": True, **tensor_summary(value)}
    return {
        "loss": loss_value,
        "forward": forward_values,
        "backward": backward_values,
        "params": params,
        "parameter_count": sum(1 for _ in model.parameters()),
    }


def diff_pair(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "loss_delta": candidate["loss"] - reference["loss"],
        "attention_regions": {},
        "parameter_carriers": {},
    }
    for field in ("forward", "backward"):
        names = sorted(set(reference[field]) | set(candidate[field]))
        for name in names:
            if name not in reference[field] or name not in candidate[field]:
                result["attention_regions"].setdefault(name, {})[field] = {"status": "MISSING"}
                continue
            result["attention_regions"].setdefault(name, {})[field] = compare_tensors(
                candidate[field][name], reference[field][name]
            )
    for name in sorted(set(reference["params"]) | set(candidate["params"])):
        left, right = reference["params"].get(name, {}), candidate["params"].get(name, {})
        if not left.get("present") or not right.get("present"):
            result["parameter_carriers"][name] = {"status": "MISSING"}
            continue
        result["parameter_carriers"][name] = compare_tensors(left["tensor"], right["tensor"])
        if name == "model.embed_tokens.weight":
            result.setdefault("_raw_param_deltas", {})[name] = right["tensor"] - left["tensor"]
    return result


def strip_tensors(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: strip_tensors(item)
            for key, item in value.items()
            if key not in {"tensor", "_raw_param_deltas"}
        }
    if isinstance(value, list):
        return [strip_tensors(item) for item in value]
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank-manifest", type=Path, default=Path("results/final/natural_bank.json"))
    parser.add_argument("--model", type=Path, default=Path("/data1/tzh/models/Qwen/Qwen3-1.7B"))
    parser.add_argument("--output", type=Path, default=Path("results/final/checkpoint_matrix.json"))
    parser.add_argument("--dataset", default="Salesforce/wikitext")
    parser.add_argument("--dataset-config", default="wikitext-103-raw-v1")
    parser.add_argument("--eval-split", default="validation")
    parser.add_argument("--eval-offset", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit-checkpoints", type=int, default=None)
    parser.add_argument("--min-step", type=int, default=0, help="measure only checkpoints at or after this step")
    parser.add_argument("--variants", nargs="+", default=["eager", "sdpa_math", "sdpa_flash"])
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

    import torch
    from transformers import AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    args.device = torch.device(args.device)
    dtype = torch.bfloat16
    bank = json.loads(args.bank_manifest.read_text())
    checkpoint_rows = [row for row in bank["checkpoints"] if row["step"] >= args.min_step]
    if args.limit_checkpoints is not None:
        checkpoint_rows = checkpoint_rows[: args.limit_checkpoints]
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, use_fast=True)
    inputs, eval_protocol = load_natural_validation(tokenizer, args)

    target_params = ["model.embed_tokens.weight"]
    target_params.extend(
        name for name in (f"model.layers.{i}.self_attn.{part}.weight" for i in range(28) for part in ("q_proj", "k_proj", "v_proj", "o_proj"))
    )
    rows: list[dict[str, Any]] = []
    for checkpoint_row in checkpoint_rows:
        checkpoint = under_root(Path(checkpoint_row["path"]), "checkpoint")
        print(f"checkpoint step {checkpoint_row['step']}", flush=True)
        reference = None
        reference_variant = None
        pair_by_variant: dict[str, Any] = {}
        for variant in args.variants:
            try:
                print(f"  variant {variant}", flush=True)
                model = build_model(args.model, checkpoint, variant, dtype, args.device)
                run = capture_run(model, inputs, variant, target_params)
                if variant == "eager":
                    reference = run
                    reference_variant = variant
                elif reference is None:
                    raise RuntimeError("eager must be included as the first reference variant")
                else:
                    pair = diff_pair(reference, run)
                    pair["candidate_variant"] = variant
                    pair["reference_variant"] = reference_variant
                    pair_by_variant[variant] = strip_tensors(pair)
                del run, model
                gc.collect()
                torch.cuda.empty_cache()
            except Exception as exc:
                pair_by_variant[variant] = {"candidate_variant": variant, "status": "UNAVAILABLE", "error": repr(exc)}
                try:
                    del model
                except UnboundLocalError:
                    pass
                gc.collect()
                torch.cuda.empty_cache()
        if reference is None:
            raise RuntimeError(f"reference eager failed for step {checkpoint_row['step']}")
        rows.append({
            "checkpoint_step": checkpoint_row["step"],
            "checkpoint_parameter_sha256": checkpoint_row["parameter_sha256"],
            "reference_loss": reference["loss"],
            "reference_forward_regions": len(reference["forward"]),
            "reference_backward_regions": len(reference["backward"]),
            "variants": pair_by_variant,
        })
        del reference
        gc.collect()
        torch.cuda.empty_cache()

    result = {
        "schema": "kernel-analyzer-evolving-checkpoint-matrix-v1",
        "subject": "Qwen3-1.7B dense causal LM",
        "bank_manifest": str(args.bank_manifest),
        "bank_protocol_sha256": bank["protocol_sha256"],
        "model": str(args.model),
        "evaluation": eval_protocol,
        "dtype": "bfloat16",
        "variants": {
            "eager": {"role": "reference", "changed": False},
            "sdpa_math": {"role": "candidate", "changed": True, "mechanisms": ["backend", "reduction_schedule", "materialization"]},
            "sdpa_flash": {"role": "candidate", "changed": True, "mechanisms": ["backend", "online_reduction", "layout", "materialization"], "accumulator": "not directly observable from this interface"},
        },
        "rows": rows,
        "boundary": "No mathematical derivations are recomputed. Rows compare the same closed attention F+B semantic regions at evolving weights; unavailable backends remain explicit.",
    }
    result["result_sha256"] = hashlib.sha256(json.dumps(result, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "checkpoints": len(rows), "variants": args.variants}, sort_keys=True))


if __name__ == "__main__":
    main()
