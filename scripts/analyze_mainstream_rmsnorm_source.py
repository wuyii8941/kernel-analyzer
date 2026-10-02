#!/usr/bin/env python3
"""Validate the arithmetic source of the mainstream BF16 RMSNorm signal.

This keeps the same inputs and norm weights for an eager reference and an
Inductor candidate, then compares both with the explicit Transformers formula.
It distinguishes casting the normalized value before the weight multiply from
keeping that multiply in FP32.  The result is a behavioral attribution, not an
instruction-level proof or a training claim.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
from typing import Any


MODEL_SPECS = {
    "qwen2_0p5b": {
        "path": "/data1/tzh/models/Qwen/Qwen2.5-0.5B-Instruct",
        "module": "transformers.models.qwen2.modeling_qwen2",
        "norm": "Qwen2RMSNorm",
    },
    "tinyllama_1p1b": {
        "path": "/data1/tzh/models/TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "module": "transformers.models.llama.modeling_llama",
        "norm": "LlamaRMSNorm",
    },
    "qwen3_1p7b": {
        "path": "/data1/tzh/models/Qwen/Qwen3-1.7B",
        "module": "transformers.models.qwen3.modeling_qwen3",
        "norm": "Qwen3RMSNorm",
    },
    "qwen3_vl_reranker_2b": {
        "path": "/data1/tzh/models/Qwen/Qwen3-VL-Reranker-2B",
        "module": "transformers.models.qwen3_vl.modeling_qwen3_vl",
        "norm": "Qwen3VLTextRMSNorm",
    },
    "deepseek_qwen3_8b": {
        "path": "/data1/tzh/models/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
        "module": "transformers.models.qwen3.modeling_qwen3",
        "norm": "Qwen3RMSNorm",
    },
    "llama3_2_3b": {
        "path": "/data1/tzh/models/meta-llama/Llama-3.2-3B",
        "module": "transformers.models.llama.modeling_llama",
        "norm": "LlamaRMSNorm",
    },
    "gemma3_4b": {
        "path": "/data1/tzh/models/google/gemma-3-4b-pt",
        "module": "transformers.models.gemma3.modeling_gemma3",
        "norm": "Gemma3RMSNorm",
        "weight_is_offset": True,
    },
    "gemma4_e2b": {
        "path": "/data1/tzh/models/google/gemma-4-E2B",
        "module": "transformers.models.gemma4.modeling_gemma4",
        "norm": "Gemma4RMSNorm",
    },
    "phi4_mini": {
        "path": "/data1/tzh/models/microsoft/Phi-4-mini-instruct",
        "module": "transformers.models.phi3.modeling_phi3",
        "norm": "Phi3RMSNorm",
    },
    "granite_moe": {
        "path": "/data1/tzh/models/ibm-granite/granite-3.1-1b-a400m-base",
        "module": "transformers.models.granitemoe.modeling_granitemoe",
        "norm": "GraniteMoeRMSNorm",
    },
    "olmoe_1b": {
        "path": "/data1/tzh/models/allenai/OLMoE-1B-7B-0125",
        "module": "transformers.models.olmoe.modeling_olmoe",
        "norm": "OlmoeRMSNorm",
    },
    "mamba_130m": {
        "path": "/data1/tzh/models/state-spaces/mamba-130m-hf",
        "module": "transformers.models.mamba.modeling_mamba",
        "norm": "MambaRMSNorm",
    },
    "ministral3_3b": {
        "path": "/data1/tzh/models/mistralai/Ministral-3-3B-Base-2512",
        "module": "transformers.models.mistral3.modeling_mistral3",
        "norm": "Mistral3RMSNorm",
    },
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
    for child_name in ("text_config", "llm_config", "language_config"):
        child = getattr(config, child_name, None)
        if child is None:
            continue
        child_raw = getattr(child, "__dict__", {})
        if name in child_raw and child_raw[name] is not None:
            return child_raw[name]
        try:
            value = getattr(child, name, None)
        except Exception:
            value = None
        if value is not None:
            return value
    return default


def _metric(left: Any, right: Any) -> dict[str, Any]:
    import torch

    delta = (left.float() - right.float()).double()
    right_norm = float(torch.linalg.vector_norm(right.float()).item())
    return {
        "exact": bool(torch.count_nonzero(delta).item() == 0),
        "nonzero_coordinates": int(torch.count_nonzero(delta).item()),
        "relative_l2": float(torch.linalg.vector_norm(delta).item()) / right_norm
        if right_norm else 0.0,
        "max_abs": float(delta.abs().max().item()) if delta.numel() else 0.0,
        "signed_mean": float(delta.mean().item()) if delta.numel() else 0.0,
    }


def _explicit(x: Any, weight: Any, eps: float, *, keep_weight_fp32: bool,
              weight_is_offset: bool = False):
    import torch

    y = x.float()
    variance = y.pow(2).mean(-1, keepdim=True)
    normalized = y * torch.rsqrt(variance + eps)
    multiplier = (1.0 + weight.float()) if weight_is_offset else weight.float()
    if keep_weight_fp32:
        return (normalized * multiplier).to(x.dtype)
    return normalized.to(x.dtype) * multiplier.to(x.dtype)


def _analytic_backward(x: Any, cotangent: Any, weight: Any, eps: float,
                       *, weight_is_offset: bool = False):
    """Closed-form FP32 RMSNorm derivative with one explicit dot reduction."""
    import torch

    hidden = x.shape[-1]
    xf = x.float()
    multiplier = (1.0 + weight.float()) if weight_is_offset else weight.float()
    inv = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    scaled_cotangent = cotangent.float() * multiplier
    dot = (scaled_cotangent * xf).sum(-1, keepdim=True)
    gradient = scaled_cotangent * inv - xf * dot * (inv ** 3) / hidden
    return gradient.to(x.dtype)


def _run_model(name: str, spec: dict[str, Any], samples: int, device: Any) -> dict[str, Any]:
    import torch
    from transformers import AutoConfig

    torch._dynamo.reset()
    config = AutoConfig.from_pretrained(spec["path"], local_files_only=True)
    hidden = int(_nested(config, "hidden_size"))
    eps = float(_nested(config, "rms_norm_eps", _nested(config, "layer_norm_eps", 1e-6)))
    module = importlib.import_module(spec["module"])
    norm_cls = getattr(module, spec["norm"])
    norm = norm_cls(hidden, eps).to(device=device, dtype=torch.bfloat16).eval()
    # Use a checkpoint norm vector whenever the scanner can load one, and the
    # same declared synthetic fallback otherwise.
    index_path = Path(spec["path"]) / "model.safetensors.index.json"
    weight = None
    weight_source = "synthetic_declared"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        candidates = [
            key for key in index.get("weight_map", {})
            if key.endswith("input_layernorm.weight")
            or key.endswith("post_attention_layernorm.weight")
            or key.endswith("norm_f.weight")
            or key.endswith("final_layernorm.weight")
        ]
        if candidates:
            try:
                from safetensors import safe_open
                key = sorted(candidates)[0]
                shard = Path(spec["path"]) / index["weight_map"][key]
                with safe_open(str(shard), framework="pt", device="cpu") as handle:
                    loaded = handle.get_tensor(key)
                if tuple(loaded.shape) == (hidden,):
                    weight = loaded
                    weight_source = f"checkpoint:{key}"
            except Exception:
                weight = None
    with torch.no_grad():
        if weight is None:
            norm.weight.copy_(1.0 + 0.01 * torch.sin(torch.arange(hidden, device=device)))
        else:
            norm.weight.copy_(weight.to(device=device, dtype=torch.bfloat16))
    candidate = torch.compile(norm, backend="inductor", fullgraph=True, dynamic=False)
    # Warm both compiler modes before recording rows.  Otherwise a guard
    # recompilation caused only by requires_grad could be mistaken for a
    # numerical source difference.
    warm_x = torch.randn((2, hidden), device=device, dtype=torch.bfloat16, requires_grad=True)
    warm_y = candidate(warm_x)
    torch.autograd.grad(warm_y.sum(), warm_x)
    rows = []
    backward_rows = []
    for index in range(samples):
        generator = torch.Generator(device=device)
        generator.manual_seed(20260917 + index)
        x = torch.randn((2, hidden), generator=generator, device=device, dtype=torch.bfloat16)
        scale = torch.ones(hidden, device=device, dtype=torch.bfloat16)
        scale[: max(1, hidden // 8)] = 0.03125
        x = x * scale
        with torch.no_grad():
            observed_reference = norm(x)
            observed_candidate = candidate(x)
            bf16_formula = _explicit(
                x, norm.weight, eps, keep_weight_fp32=False,
                weight_is_offset=bool(spec.get("weight_is_offset", False)),
            )
            fp32_formula = _explicit(
                x, norm.weight, eps, keep_weight_fp32=True,
                weight_is_offset=bool(spec.get("weight_is_offset", False)),
            )
        # Use the same cotangent for four independently traced calls.  This
        # validates the backward stage for every nonzero scanner signal rather
        # than inferring it from the forward output alone.
        cotangent = torch.randn(
            observed_reference.shape, generator=generator, device=device,
            dtype=torch.bfloat16,
        )
        grad_rows = []
        for fn in (norm, candidate):
            x_grad = x.detach().clone().requires_grad_(True)
            y_grad = fn(x_grad)
            grad_rows.append(torch.autograd.grad((y_grad * cotangent).sum(), x_grad)[0].detach())
        for fn_formula in (
            lambda z: _explicit(
                z, norm.weight, eps, keep_weight_fp32=False,
                weight_is_offset=bool(spec.get("weight_is_offset", False)),
            ),
            lambda z: _explicit(
                z, norm.weight, eps, keep_weight_fp32=True,
                weight_is_offset=bool(spec.get("weight_is_offset", False)),
            ),
        ):
            x_grad = x.detach().clone().requires_grad_(True)
            y_grad = fn_formula(x_grad)
            grad_rows.append(torch.autograd.grad((y_grad * cotangent).sum(), x_grad)[0].detach())
        analytic_gradient = _analytic_backward(
            x, cotangent, norm.weight, eps,
            weight_is_offset=bool(spec.get("weight_is_offset", False)),
        )
        rows.append({
            "sample": index,
            "reference_vs_bf16_weight_formula": _metric(observed_reference, bf16_formula),
            "candidate_vs_reference": _metric(observed_candidate, observed_reference),
            "candidate_vs_fp32_weight_formula": _metric(observed_candidate, fp32_formula),
        })
        backward_rows.append({
            "sample": index,
            "reference_vs_bf16_weight_formula": _metric(grad_rows[0], grad_rows[2]),
            "candidate_vs_reference": _metric(grad_rows[1], grad_rows[0]),
            "candidate_vs_fp32_weight_formula": _metric(grad_rows[1], grad_rows[3]),
            "reference_vs_analytic_fp32_derivative": _metric(grad_rows[0], analytic_gradient),
            "candidate_vs_analytic_fp32_derivative": _metric(grad_rows[1], analytic_gradient),
        })
    return {
        "model_path": spec["path"],
        "module": spec["module"],
        "weight_source": weight_source,
        "hidden_size": hidden,
        "samples": samples,
        "rows": rows,
        "backward_rows": backward_rows,
        "all_reference_matches_bf16_weight_formula": all(
            row["reference_vs_bf16_weight_formula"]["exact"] for row in rows
        ),
        "all_candidate_matches_fp32_weight_formula": all(
            row["candidate_vs_fp32_weight_formula"]["exact"] for row in rows
        ),
        "candidate_fp32_formula_exact_rows": sum(
            row["candidate_vs_fp32_weight_formula"]["exact"] for row in rows
        ),
        "reference_bf16_formula_exact_rows": sum(
            row["reference_vs_bf16_weight_formula"]["exact"] for row in rows
        ),
        "all_candidate_differs_from_reference": all(
            row["candidate_vs_reference"]["nonzero_coordinates"] > 0 for row in rows
        ),
        "max_candidate_reference_relative_l2": max(
            row["candidate_vs_reference"]["relative_l2"] for row in rows
        ),
        "all_backward_reference_matches_bf16_weight_formula": all(
            row["reference_vs_bf16_weight_formula"]["exact"] for row in backward_rows
        ),
        "all_backward_candidate_matches_fp32_weight_formula": all(
            row["candidate_vs_fp32_weight_formula"]["exact"] for row in backward_rows
        ),
        "all_backward_candidate_differs_from_reference": all(
            row["candidate_vs_reference"]["nonzero_coordinates"] > 0 for row in backward_rows
        ),
        "backward_candidate_fp32_formula_exact_rows": sum(
            row["candidate_vs_fp32_weight_formula"]["exact"] for row in backward_rows
        ),
        "all_backward_candidate_matches_analytic_fp32_derivative": all(
            row["candidate_vs_analytic_fp32_derivative"]["exact"] for row in backward_rows
        ),
        "backward_candidate_analytic_exact_rows": sum(
            row["candidate_vs_analytic_fp32_derivative"]["exact"] for row in backward_rows
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
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
    results: dict[str, Any] = {
        "schema": "mainstream-rmsnorm-source-analysis-v1",
        "status": "COMPLETE",
        "candidate_route": "torch.compile backend=inductor fullgraph=True",
        "reference_route": "eager Transformers RMSNorm helper",
        "input_policy": "identical generated BF16 inputs and checkpoint/synthetic norm weights",
        "interpretation": (
            "The eager formula casts the normalized FP32 value to BF16 before multiplying "
            "the BF16 weight.  A candidate matching the explicit FP32-weight formula while "
            "the reference matches the BF16-weight formula identifies a behavioral arithmetic "
            "difference at the intermediate cast/multiply boundary."
        ),
        "models": {},
    }
    for name, spec in MODEL_SPECS.items():
        try:
            results["models"][name] = _run_model(name, spec, args.samples, device)
            print(name, "OK", results["models"][name]["all_candidate_matches_fp32_weight_formula"])
        except Exception as error:
            results["models"][name] = {
                "status": "UNRESOLVED_MEASUREMENT",
                "error": f"{type(error).__name__}: {error}",
            }
            print(name, "UNRESOLVED_MEASUREMENT", error)
        torch.cuda.empty_cache()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
