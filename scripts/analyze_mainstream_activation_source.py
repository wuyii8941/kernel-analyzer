#!/usr/bin/env python3
"""Attribute tiny FP32 activation differences in the mainstream scan.

Inductor and eager activation calls are evaluated on identical inputs.  SiLU
is compared with its algebraically equivalent exponential spelling.  GELU
uses the configured native tanh approximation and a separately written
reference expression, so that the report distinguishes a source spelling
change from the compiler/native implementation path.  The report is a
behavioural check only; it is not an instruction-level or training claim.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any


MODEL_SPECS = {
    "bert_base_uncased": ("/data1/tzh/models/google-bert/bert-base-uncased", "gelu"),
    "roberta_base": ("/data1/tzh/models/FacebookAI/roberta-base", "gelu"),
    "gpt_neo_125m": ("/data1/tzh/models/EleutherAI/gpt-neo-125m", "gelu_new"),
    "deberta_v3_small": ("/data1/tzh/models/microsoft/deberta-v3-small", "gelu"),
    "electra_small": ("/data1/tzh/models/google/electra-small-discriminator", "gelu"),
    "opt_350m": ("/data1/tzh/models/facebook/opt-350m", "relu"),
    "pythia_410m": ("/data1/tzh/models/EleutherAI/pythia-410m", "gelu"),
    "falcon_rw_1b": ("/data1/tzh/models/tiiuae/falcon-rw-1b", "gelu"),
    "bloom_560m": ("/data1/tzh/models/bigscience/bloom-560m", "gelu"),
    "gpt2": ("/data1/tzh/models/openai-community/gpt2", "gelu_new"),
    "qwen2_0p5b": ("/data1/tzh/models/Qwen/Qwen2.5-0.5B-Instruct", "silu"),
    "tinyllama_1p1b": ("/data1/tzh/models/TinyLlama/TinyLlama-1.1B-Chat-v1.0", "silu"),
    "qwen3_1p7b": ("/data1/tzh/models/Qwen/Qwen3-1.7B", "silu"),
    "qwen3_vl_reranker_2b": ("/data1/tzh/models/Qwen/Qwen3-VL-Reranker-2B", "silu"),
    "deepseek_qwen3_8b": ("/data1/tzh/models/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B", "silu"),
    "llama3_2_3b": ("/data1/tzh/models/meta-llama/Llama-3.2-3B", "silu"),
    "gemma3_4b": ("/data1/tzh/models/google/gemma-3-4b-pt", "gelu_pytorch_tanh"),
    "gemma4_e2b": ("/data1/tzh/models/google/gemma-4-E2B", "gelu_pytorch_tanh"),
    "phi4_mini": ("/data1/tzh/models/microsoft/Phi-4-mini-instruct", "silu"),
    "granite_moe": ("/data1/tzh/models/ibm-granite/granite-3.1-1b-a400m-base", "silu"),
    "olmoe_1b": ("/data1/tzh/models/allenai/OLMoE-1B-7B-0125", "silu"),
    "mamba_130m": ("/data1/tzh/models/state-spaces/mamba-130m-hf", "silu"),
    "ministral3_3b": ("/data1/tzh/models/mistralai/Ministral-3-3B-Base-2512", "silu"),
}


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


def _inputs(torch: Any, index: int, hidden: int, *, requires_grad: bool, dtype: Any):
    generator = torch.Generator(device="cuda")
    generator.manual_seed(2000 + index)
    x = torch.randn((2, hidden), generator=generator, device="cuda", dtype=dtype)
    scale = torch.ones(hidden, device="cuda", dtype=dtype)
    scale[: max(1, hidden // 8)] = 0.03125
    x = x * scale
    x.requires_grad_(requires_grad)
    return x


def _silu_exp(x: Any) -> Any:
    import torch

    return x / (1.0 + torch.exp(-x))


def _silu_sigmoid(x: Any) -> Any:
    import torch

    return x * torch.sigmoid(x)


def _gelu_python(x: Any) -> Any:
    import torch

    coefficient = math.sqrt(2.0 / math.pi)
    return x * 0.5 * (
        1.0 + torch.tanh(coefficient * (x + 0.044715 * torch.pow(x, 3.0)))
    )


def _gelu_exact(x: Any) -> Any:
    import torch

    # PyTorch's GELU uses an FP32 intermediate for BF16 inputs on this route;
    # cast only at the declared output boundary so the explicit formula has
    # the same semantic precision as the registered eager implementation.
    work = x.float()
    return (0.5 * work * (1.0 + torch.erf(work / math.sqrt(2.0)))).to(dtype=x.dtype)


def _analytic_derivative(x: Any, hidden_act: str) -> Any:
    """FP32 derivative for the explicit activation semantics used below."""
    import torch

    work = x.float()
    if hidden_act == "silu":
        sigmoid = torch.sigmoid(work)
        return sigmoid + work * sigmoid * (1.0 - sigmoid)
    if hidden_act == "relu":
        return (work > 0).to(work.dtype)
    if hidden_act == "gelu":
        z = work / math.sqrt(2.0)
        return 0.5 * (1.0 + torch.erf(z)) + work * torch.exp(-0.5 * work.square()) / math.sqrt(2.0 * math.pi)
    coefficient = math.sqrt(2.0 / math.pi)
    inner = coefficient * (work + 0.044715 * torch.pow(work, 3.0))
    tanh_inner = torch.tanh(inner)
    inner_derivative = coefficient * (1.0 + 3.0 * 0.044715 * torch.pow(work, 2.0))
    return 0.5 * (1.0 + tanh_inner) + 0.5 * work * (1.0 - tanh_inner.square()) * inner_derivative


def _run(name: str, hidden_act: str, hidden: int, samples: int, device: Any, dtype: Any) -> dict[str, Any]:
    import torch
    from transformers.activations import ACT2FN

    torch._dynamo.reset()
    class Activation(torch.nn.Module):
        def forward(self, x):
            # Use the same Transformers activation registry as the scanner.
            # In particular, ``gelu_new`` is a distinct implementation from
            # ``gelu_pytorch_tanh``; replacing it with F.gelu would invalidate
            # the source comparison and could hide the actual compiled/eager
            # discrepancy.
            return ACT2FN[hidden_act](x)

    # Keep output-only and autograd-enabled compilations separate. Mixing
    # requires_grad and no-grad calls in one compiled callable can exercise
    # guard recompilation rather than one stable arithmetic path.
    reference_nograd = Activation().to(device=device).eval()
    candidate_nograd = torch.compile(
        reference_nograd, backend="inductor", fullgraph=True, dynamic=False
    )
    with torch.no_grad():
        candidate_nograd(_inputs(torch, 0, hidden, requires_grad=False, dtype=dtype))
    # Keep autograd compilation separate from the no-grad output route.  This
    # prevents guard recompilation from being mistaken for a backward effect.
    reference_grad = Activation().to(device=device).eval()
    candidate_grad = torch.compile(reference_grad, backend="inductor", fullgraph=True, dynamic=False)
    warm_grad = _inputs(torch, 0, hidden, requires_grad=True, dtype=dtype)
    candidate_grad(warm_grad).float().sum().backward()
    rows = []
    for index in range(samples):
        x0 = _inputs(torch, index, hidden, requires_grad=False, dtype=dtype)
        with torch.no_grad():
            reference_output = reference_nograd(x0)
            candidate_output = candidate_nograd(x0)
            if hidden_act == "silu":
                explicit = _silu_exp(x0)
                alternate = _silu_sigmoid(x0)
            elif hidden_act == "relu":
                explicit = torch.relu(x0)
                alternate = None
            elif hidden_act == "gelu":
                explicit = _gelu_exact(x0)
                alternate = None
            else:
                explicit = _gelu_python(x0)
                alternate = None
        rows.append({
            "sample": index,
            "output_candidate_vs_reference": _metric(candidate_output, reference_output),
            "output_reference_vs_explicit": _metric(reference_output, explicit),
            "output_candidate_vs_explicit": _metric(candidate_output, explicit),
            "output_reference_vs_alternate": (
                _metric(reference_output, alternate) if alternate is not None else None
            ),
            "output_candidate_vs_alternate": (
                _metric(candidate_output, alternate) if alternate is not None else None
            ),
            "backward": [],
        })
        cotangents = [
            torch.ones_like(x0),
            torch.where(
                torch.arange(hidden, device=device).remainder(2).view(1, -1) == 0,
                torch.ones_like(x0), -torch.ones_like(x0),
            ),
        ]
        for cotangent in cotangents:
            x_ref = x0.detach().clone().requires_grad_(True)
            x_cand = x0.detach().clone().requires_grad_(True)
            ref_grad = torch.autograd.grad(
                (reference_grad(x_ref).float() * cotangent.float()).sum(), x_ref
            )[0]
            cand_grad = torch.autograd.grad(
                (candidate_grad(x_cand).float() * cotangent.float()).sum(), x_cand
            )[0]
            analytic = _analytic_derivative(x0, hidden_act) * cotangent.float()
            rows[-1]["backward"].append({
                "cotangent": "ones" if bool(torch.all(cotangent == 1)) else "alternating",
                "candidate_vs_reference": _metric(cand_grad, ref_grad),
                "reference_vs_analytic_fp32": _metric(ref_grad, analytic),
                "candidate_vs_analytic_fp32": _metric(cand_grad, analytic),
            })
    return {
        "hidden_act": hidden_act,
        "hidden_size": hidden,
        "dtype": str(dtype).split(".")[-1],
        "samples": samples,
        "rows": rows,
        "max_output_candidate_reference_relative_l2": max(
            row["output_candidate_vs_reference"]["relative_l2"] for row in rows
        ),
        "all_candidate_output_differs": any(
            row["output_candidate_vs_reference"]["nonzero_coordinates"] > 0 for row in rows
        ),
        "all_candidate_backward_differs": any(
            item["candidate_vs_reference"]["nonzero_coordinates"] > 0
            for row in rows for item in row["backward"]
        ),
        "all_reference_output_matches_explicit": all(
            row["output_reference_vs_explicit"]["exact"] for row in rows
        ),
        "candidate_matches_explicit_everywhere": all(
            row["output_candidate_vs_explicit"]["exact"] for row in rows
        ),
        "candidate_closer_to_sigmoid_spelling": (
            hidden_act == "silu"
            and sum(
                row["output_candidate_vs_alternate"]["relative_l2"]
                < row["output_candidate_vs_explicit"]["relative_l2"]
                for row in rows
            )
            > samples / 2
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument(
        "--models", nargs="+", choices=tuple(MODEL_SPECS), default=None,
        help="Optional subset of registered activation model specs; default scans all.",
    )
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
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    if not output.is_relative_to(root):
        raise ValueError("output must be inside the repository")
    if output.exists():
        raise FileExistsError(output)
    payload: dict[str, Any] = {
        "schema": "mainstream-activation-source-analysis-v1",
        "status": "COMPLETE",
        "candidate_route": "torch.compile backend=inductor fullgraph=True",
        "reference_route": "Transformers ACT2FN eager module",
        "input_policy": f"identical generated {args.dtype} inputs with the scanner's scale mixture",
        "interpretation": (
            "A nonzero candidate/reference result is a compiler/native evaluation difference. "
            "For SiLU, the eager reference is compared with both x/(1+exp(-x)) and x*sigmoid(x); "
            "the compiled result is reported against both without forcing an exact attribution. "
            "For exact `gelu`, the explicit comparison uses the erf formula; for `gelu_new` and "
            "`gelu_pytorch_tanh`, it uses the declared tanh approximation.  Disagreement with an "
            "explicit spelling means the native route cannot be reduced to that spelling on this "
            "runtime.  Neither observation is a training-level bias claim."
        ),
        "models": {},
    }
    selected_models = (
        [(name, MODEL_SPECS[name]) for name in args.models]
        if args.models is not None else list(MODEL_SPECS.items())
    )
    payload["selected_models"] = [name for name, _ in selected_models]
    for name, (path, hidden_act) in selected_models:
        try:
            # The scan uses the configured hidden size when available; a small
            # explicit override keeps this source probe bounded and reproducible.
            from transformers import AutoConfig

            config = AutoConfig.from_pretrained(path, local_files_only=True)
            hidden = int(getattr(config, "hidden_size", args.hidden_size))
            payload["models"][name] = _run(name, hidden_act, hidden, args.samples, device, dtype)
            print(name, "OK", payload["models"][name]["all_candidate_output_differs"])
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
