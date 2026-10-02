#!/usr/bin/env python3
"""Run the Qwen3 eager-vs-SDPA boundary with SDPA forced to math mode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import run_qwen3_attention_backend_isolated_probe as isolated
from run_qwen3_attention_backend_isolated_probe import run


_ORIGINAL_SDPA = isolated.sdpa_attention_forward


def math_sdpa(*args, **kwargs):
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        return _ORIGINAL_SDPA(*args, **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("/data1/tzh/models/Qwen/Qwen3-1.7B"))
    parser.add_argument("--input-bank", type=Path, default=Path("results/coverage/qwen_seq128_input_bank.json"))
    parser.add_argument("--layer", type=int, default=13)
    parser.add_argument("--parameter", choices=("q_proj", "v_proj"), default="q_proj")
    parser.add_argument("--states", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    original = isolated.sdpa_attention_forward
    isolated.sdpa_attention_forward = math_sdpa
    try:
        result = run(args)
    finally:
        isolated.sdpa_attention_forward = original
    result["schema"] = "kernel-analyzer-qwen3-attention-backend-math-isolated-v1"
    result["reference"] = "same model and attention inputs with SDPA forced to the math backend"
    result["comparison_scope"]["single_changed_boundary"] = "eager attention versus forced-math SDPA interface"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
