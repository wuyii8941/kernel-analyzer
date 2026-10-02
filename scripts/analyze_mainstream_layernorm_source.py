#!/usr/bin/env python3
"""Check the arithmetic source of BF16 LayerNorm discrepancies.

The public mainstream scan uses a generic LayerNorm probe for models whose
architectures expose LayerNorm rather than RMSNorm.  This follow-up keeps the
same generated inputs, weights and cotangents, and compares eager/compiled
outputs and input gradients with explicit BF16/FP32 formulas.  It is a
behavioural source check; it does not claim an instruction-level cause or a
training consequence.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


MODEL_SPECS = {
    "bert_base_uncased": "/data1/tzh/models/google-bert/bert-base-uncased",
    "roberta_base": "/data1/tzh/models/FacebookAI/roberta-base",
    "gpt_neo_125m": "/data1/tzh/models/EleutherAI/gpt-neo-125m",
    "deberta_v3_small": "/data1/tzh/models/microsoft/deberta-v3-small",
    "electra_small": "/data1/tzh/models/google/electra-small-discriminator",
    "opt_350m": "/data1/tzh/models/facebook/opt-350m",
    "pythia_410m": "/data1/tzh/models/EleutherAI/pythia-410m",
    "falcon_rw_1b": "/data1/tzh/models/tiiuae/falcon-rw-1b",
    "bloom_560m": "/data1/tzh/models/bigscience/bloom-560m",
    "gpt2": "/data1/tzh/models/openai-community/gpt2",
}


def _nested(config: Any, name: str, default: Any = None) -> Any:
    raw = getattr(config, "__dict__", {})
    if name in raw and raw[name] is not None:
        return raw[name]
    try:
        value = getattr(config, name, None)
    except Exception:
        value = None
    if value is not None:
        return value
    return default


def _metric(left: Any, right: Any) -> dict[str, Any]:
    import torch

    delta = (left.detach().float() - right.detach().float()).double()
    right_norm = float(torch.linalg.vector_norm(right.detach().float()).item())
    return {
        "exact": bool(torch.count_nonzero(delta).item() == 0),
        "nonzero_coordinates": int(torch.count_nonzero(delta).item()),
        "relative_l2": float(torch.linalg.vector_norm(delta).item()) / right_norm
        if right_norm else 0.0,
        "max_abs": float(delta.abs().max().item()) if delta.numel() else 0.0,
        "signed_mean": float(delta.mean().item()) if delta.numel() else 0.0,
    }


def _inputs(torch: Any, index: int, hidden: int, device: Any) -> Any:
    generator = torch.Generator(device=device)
    generator.manual_seed(1000 + index)
    x = torch.randn((2, hidden), generator=generator, device=device, dtype=torch.bfloat16)
    scale = torch.ones(hidden, device=device, dtype=torch.bfloat16)
    scale[: max(1, hidden // 8)] = 0.03125
    return x * scale


def _formula(x: Any, weight: Any, bias: Any, eps: float, *, fp32: bool) -> Any:
    import torch

    work = x.float() if fp32 else x
    w = weight.float() if fp32 else weight
    b = bias.float() if fp32 else bias
    mean = work.mean(dim=-1, keepdim=True)
    centered = work - mean
    variance = (centered * centered).mean(dim=-1, keepdim=True)
    output = centered * torch.rsqrt(variance + eps)
    output = output * w + b
    return output.to(dtype=x.dtype) if fp32 else output


def _analytic_input_grad(x: Any, weight: Any, cotangent: Any, eps: float) -> Any:
    """Closed-form FP32 input VJP for biased-variance LayerNorm."""
    import torch

    work = x.float()
    w = weight.float()
    g = cotangent.float()
    count = work.shape[-1]
    mean = work.mean(dim=-1, keepdim=True)
    centered = work - mean
    inv = torch.rsqrt((centered * centered).mean(dim=-1, keepdim=True) + eps)
    normalized = centered * inv
    weighted = g * w
    return (inv / count) * (
        count * weighted
        - weighted.sum(dim=-1, keepdim=True)
        - normalized * (weighted * normalized).sum(dim=-1, keepdim=True)
    )


def _run_model(name: str, path: str, samples: int, device: Any) -> dict[str, Any]:
    import torch
    from transformers import AutoConfig

    torch._dynamo.reset()
    config = AutoConfig.from_pretrained(path, local_files_only=True)
    hidden = _nested(config, "hidden_size")
    if hidden is None:
        hidden = _nested(config, "n_embd", _nested(config, "n_embed"))
    if hidden is None:
        raise ValueError("model config has no hidden-size alias")
    eps = float(_nested(config, "layer_norm_eps", _nested(config, "layer_norm_epsilon", 1e-5)))
    hidden = int(hidden)

    norm = torch.nn.LayerNorm(hidden, eps=eps, elementwise_affine=True).to(
        device=device, dtype=torch.bfloat16
    ).eval()
    with torch.no_grad():
        norm.weight.copy_(1.0 + 0.01 * torch.sin(torch.arange(hidden, device=device)))
        norm.bias.copy_(0.01 * torch.cos(torch.arange(hidden, device=device)))
    candidate = torch.compile(norm, backend="inductor", fullgraph=True, dynamic=False)
    warm = _inputs(torch, 0, hidden, device)
    candidate(warm)
    warm.requires_grad_(True)
    candidate(warm).float().sum().backward()

    rows = []
    for index in range(samples):
        x = _inputs(torch, index, hidden, device)
        cotangents = [
            torch.ones_like(x),
            torch.where(
                torch.arange(hidden, device=device).remainder(2).view(1, -1) == 0,
                torch.ones_like(x), -torch.ones_like(x),
            ),
        ]
        with torch.no_grad():
            ref_output = norm(x)
            cand_output = candidate(x)
            bf16_output = _formula(x, norm.weight, norm.bias, eps, fp32=False)
            fp32_output = _formula(x, norm.weight, norm.bias, eps, fp32=True)
        row: dict[str, Any] = {
            "sample": index,
            "output_candidate_vs_reference": _metric(cand_output, ref_output),
            "output_reference_vs_bf16_formula": _metric(ref_output, bf16_output),
            "output_candidate_vs_fp32_formula": _metric(cand_output, fp32_output),
            "backward": [],
        }
        for cotangent in cotangents:
            x_ref = x.detach().clone().requires_grad_(True)
            x_cand = x.detach().clone().requires_grad_(True)
            ref = norm(x_ref)
            cand = candidate(x_cand)
            ref_grad = torch.autograd.grad((ref.float() * cotangent.float()).sum(), x_ref)[0]
            cand_grad = torch.autograd.grad((cand.float() * cotangent.float()).sum(), x_cand)[0]
            analytic = _analytic_input_grad(x, norm.weight, cotangent, eps)
            analytic_cast = analytic.to(dtype=x.dtype)
            row["backward"].append({
                "cotangent": "ones" if bool(torch.all(cotangent == 1)) else "alternating",
                "candidate_vs_reference": _metric(cand_grad, ref_grad),
                "reference_vs_analytic_fp32": _metric(ref_grad, analytic),
                "candidate_vs_analytic_fp32": _metric(cand_grad, analytic),
                "reference_vs_analytic_fp32_cast": _metric(ref_grad, analytic_cast),
                "candidate_vs_analytic_fp32_cast": _metric(cand_grad, analytic_cast),
            })
        rows.append(row)
    return {
        "model_path": path,
        "model_type": getattr(config, "model_type", None),
        "hidden_size": hidden,
        "eps": eps,
        "samples": samples,
        "rows": rows,
        "all_candidate_differs_from_reference": any(
            row["output_candidate_vs_reference"]["nonzero_coordinates"] > 0
            or any(item["candidate_vs_reference"]["nonzero_coordinates"] > 0 for item in row["backward"])
            for row in rows
        ),
        "all_reference_matches_bf16_formula": all(
            row["output_reference_vs_bf16_formula"]["exact"] for row in rows
        ),
        "all_candidate_matches_fp32_formula": all(
            row["output_candidate_vs_fp32_formula"]["exact"] for row in rows
        ),
        "all_candidate_backward_matches_analytic": all(
            item["candidate_vs_analytic_fp32"]["exact"]
            for row in rows for item in row["backward"]
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--models", nargs="+", choices=tuple(MODEL_SPECS), default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.samples < 4:
        raise SystemExit("--samples must be >= 4")
    os.environ.setdefault("TRITON_CACHE_DIR", "/data1/tzh/cache/triton_mainstream_bias_scan")
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/data1/tzh/cache/torchinductor_mainstream_bias_scan")
    import torch

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise SystemExit("a CUDA device is required")
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    if not output.is_relative_to(root):
        raise ValueError("output must be inside the repository")
    if output.exists():
        raise FileExistsError(output)
    selected = (
        [(name, MODEL_SPECS[name]) for name in args.models]
        if args.models is not None else list(MODEL_SPECS.items())
    )
    payload: dict[str, Any] = {
        "schema": "mainstream-layernorm-source-analysis-v1",
        "status": "COMPLETE",
        "selected_models": [name for name, _ in selected],
        "candidate_route": "torch.compile backend=inductor fullgraph=True",
        "reference_route": "eager torch.nn.LayerNorm",
        "input_policy": "identical generated BF16 inputs, synthetic BF16 affine parameters",
        "interpretation": (
            "Formula matches localize the discrepancy to LayerNorm arithmetic, reduction or "
            "materialization behaviour. They do not identify a particular generated instruction "
            "and do not establish a training-level consequence."
        ),
        "models": {},
    }
    for name, path in selected:
        try:
            payload["models"][name] = _run_model(name, path, args.samples, device)
            print(name, "OK", payload["models"][name]["all_candidate_differs_from_reference"])
        except Exception as error:
            payload["models"][name] = {
                "status": "UNRESOLVED_MEASUREMENT",
                "error": f"{type(error).__name__}: {error}",
            }
            print(name, "UNRESOLVED_MEASUREMENT", error)
        torch.cuda.empty_cache()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
