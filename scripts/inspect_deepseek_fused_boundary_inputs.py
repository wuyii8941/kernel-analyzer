#!/usr/bin/env python3
"""Inventory the real tensors entering the DeepSeek fused embedding boundary.

This is a source-discovery probe only.  It does not assign a root cause and it
does not replace any candidate/reference path.  The inventory is useful for
choosing the next single-input intervention without confusing an upstream
producer with the fused boundary itself.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch._inductor.codecache import PyCodeCache
from transformers import AutoModelForCausalLM

from scripts.qwen_candidate_step import LossStep, configure_candidate_runtime


SYMBOL = "triton_red_fused__to_copy_add_div_embedding_dense_backward_expand_mul_nll_loss_forward_pow_sum_view_22"


def _tensor_meta(value: torch.Tensor) -> dict[str, Any]:
    x = value.detach().float()
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "device": str(value.device),
        "numel": int(value.numel()),
        "min": float(x.min().item()) if x.numel() else 0.0,
        "max": float(x.max().item()) if x.numel() else 0.0,
        "mean": float(x.mean().item()) if x.numel() else 0.0,
        "rms": float(torch.linalg.vector_norm(x).item()) / max(float(x.numel()) ** 0.5, 1.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input-bank", type=Path, required=True)
    parser.add_argument("--state-index", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    bank = json.loads(args.input_bank.read_text(encoding="utf-8"))
    states = bank["states"]
    state = states[args.state_index]
    values = state.get("token_ids", state.get("input_ids"))
    if values is None:
        raise KeyError("state has no token_ids/input_ids")

    configure_candidate_runtime(27890)
    device = torch.device(args.device)
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model), dtype=torch.bfloat16, attn_implementation="eager", local_files_only=True,
    ).to(device).train()
    model.config.use_cache = False
    target_name = "model.embed_tokens.weight"
    target = dict(model.named_parameters())[target_name]
    start = len(PyCodeCache.modules)
    compiled = torch.compile(LossStep(model), backend="inductor", fullgraph=True, dynamic=False)
    warm = torch.tensor([states[0].get("token_ids", states[0].get("input_ids"))], dtype=torch.long, device=device)
    model.zero_grad(set_to_none=True)
    compiled(warm).backward()
    torch.cuda.synchronize(device)

    target_kernel = None
    for module in list(PyCodeCache.modules)[start:]:
        for name, value in vars(module).items():
            if SYMBOL in str(name) and callable(getattr(value, "run", None)):
                target_kernel = value
                break
        if target_kernel is not None:
            break
    if target_kernel is None:
        raise RuntimeError("generated fused boundary kernel not found")

    captured: dict[str, Any] = {}
    original_run = target_kernel.run

    def wrapped(*call_args: Any, **kwargs: Any) -> Any:
        if "tensors" not in captured:
            captured["tensors"] = [
                _tensor_meta(value) if isinstance(value, torch.Tensor) else {"type": type(value).__name__, "repr": repr(value)}
                for value in call_args
            ]
            captured["kwargs"] = {key: repr(value) for key, value in kwargs.items()}
        return original_run(*call_args, **kwargs)

    target_kernel.run = wrapped
    try:
        ids = torch.tensor([values], dtype=torch.long, device=device)
        model.zero_grad(set_to_none=True)
        loss = compiled(ids)
        loss.backward()
        torch.cuda.synchronize(device)
    finally:
        target_kernel.run = original_run

    result = {
        "schema": "kernel-analyzer-deepseek-fused-boundary-input-inventory-v1",
        "status": "COMPLETE_SOURCE_DISCOVERY_ONLY",
        "model": str(args.model),
        "input_bank": str(args.input_bank),
        "state_id": state.get("state_id", args.state_index),
        "kernel_symbol": SYMBOL,
        "target_parameter": target_name,
        "loss": float(loss.detach().cpu().item()),
        "target_parameter_shape": list(target.shape),
        "target_parameter_dtype": str(target.dtype),
        "call_argument_inventory": captured.get("tensors", []),
        "call_kwargs": captured.get("kwargs", {}),
        "claim_boundary": "ABI inventory only; it does not identify which input producer causes the original region effect and is not a bias result.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
