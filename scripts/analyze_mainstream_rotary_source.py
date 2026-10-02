#!/usr/bin/env python3
"""Attribute the mainstream BF16 rotary signal to an arithmetic contract.

The scan found a nonzero output difference for compiled versus eager rotary
helpers.  This follow-up keeps q/k/cos/sin identical and compares the observed
outputs with two explicit expressions: BF16 stepwise arithmetic and FP32
intermediate arithmetic followed by BF16 output storage.  It is a behavioral
source validation, not an instruction-level proof and not a training claim.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
from typing import Any


MODEL_SPECS = {
    "pythia_410m": {
        "path": "/data1/tzh/models/EleutherAI/pythia-410m",
        "module": "transformers.models.gpt_neox.modeling_gpt_neox",
        "rotary": "apply_rotary_pos_emb",
    },
    "falcon_rw_1b": {
        "path": "/data1/tzh/models/tiiuae/falcon-rw-1b",
        "module": "transformers.models.falcon.modeling_falcon",
        "rotary": "apply_rotary_pos_emb",
    },
    "qwen2_0p5b": {
        "path": "/data1/tzh/models/Qwen/Qwen2.5-0.5B-Instruct",
        "module": "transformers.models.qwen2.modeling_qwen2",
        "norm": "Qwen2RMSNorm",
        "rotary": "apply_rotary_pos_emb",
    },
    "tinyllama_1p1b": {
        "path": "/data1/tzh/models/TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "module": "transformers.models.llama.modeling_llama",
        "norm": "LlamaRMSNorm",
        "rotary": "apply_rotary_pos_emb",
    },
    "qwen3_1p7b": {
        "path": "/data1/tzh/models/Qwen/Qwen3-1.7B",
        "module": "transformers.models.qwen3.modeling_qwen3",
        "norm": "Qwen3RMSNorm",
        "rotary": "apply_rotary_pos_emb",
    },
    "qwen3_vl_reranker_2b": {
        "path": "/data1/tzh/models/Qwen/Qwen3-VL-Reranker-2B",
        "module": "transformers.models.qwen3_vl.modeling_qwen3_vl",
        "norm": "Qwen3VLTextRMSNorm",
        "rotary": "apply_rotary_pos_emb",
    },
    "deepseek_qwen3_8b": {
        "path": "/data1/tzh/models/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
        "module": "transformers.models.qwen3.modeling_qwen3",
        "norm": "Qwen3RMSNorm",
        "rotary": "apply_rotary_pos_emb",
    },
    "llama3_2_3b": {
        "path": "/data1/tzh/models/meta-llama/Llama-3.2-3B",
        "module": "transformers.models.llama.modeling_llama",
        "norm": "LlamaRMSNorm",
        "rotary": "apply_rotary_pos_emb",
    },
    "gemma3_4b": {
        "path": "/data1/tzh/models/google/gemma-3-4b-pt",
        "module": "transformers.models.gemma3.modeling_gemma3",
        "norm": "Gemma3RMSNorm",
        "rotary": "apply_rotary_pos_emb",
    },
    "gemma4_e2b": {
        "path": "/data1/tzh/models/google/gemma-4-E2B",
        "module": "transformers.models.gemma4.modeling_gemma4",
        "norm": "Gemma4RMSNorm",
        "rotary": "apply_rotary_pos_emb",
        "single": True,
    },
    "phi4_mini": {
        "path": "/data1/tzh/models/microsoft/Phi-4-mini-instruct",
        "module": "transformers.models.phi3.modeling_phi3",
        "norm": "Phi3RMSNorm",
        "rotary": "apply_rotary_pos_emb",
    },
    "granite_moe": {
        "path": "/data1/tzh/models/ibm-granite/granite-3.1-1b-a400m-base",
        "module": "transformers.models.granitemoe.modeling_granitemoe",
        "norm": "GraniteMoeRMSNorm",
        "rotary": "apply_rotary_pos_emb",
    },
    "olmoe_1b": {
        "path": "/data1/tzh/models/allenai/OLMoE-1B-7B-0125",
        "module": "transformers.models.olmoe.modeling_olmoe",
        "norm": "OlmoeRMSNorm",
        "rotary": "apply_rotary_pos_emb",
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
        if child is not None:
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


def _rotate_half(x: Any) -> Any:
    half = x.shape[-1] // 2
    import torch

    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _explicit(q: Any, k: Any, cos: Any, sin: Any, *, fp32: bool, single: bool):
    dtype = q.dtype
    if fp32:
        q_work, k_work = q.float(), k.float()
        cos_work, sin_work = cos.float(), sin.float()
    else:
        q_work, k_work = q, k
        cos_work, sin_work = cos, sin
    cos_work = cos_work.unsqueeze(1)
    sin_work = sin_work.unsqueeze(1)
    q_out = q_work * cos_work + _rotate_half(q_work) * sin_work
    if single:
        return (q_out.to(dtype) if fp32 else q_out,)
    k_out = k_work * cos_work + _rotate_half(k_work) * sin_work
    if fp32:
        return q_out.to(dtype), k_out.to(dtype)
    return q_out, k_out


def _run_model(name: str, spec: dict[str, Any], samples: int, device: Any) -> dict[str, Any]:
    import torch
    from transformers import AutoConfig

    torch._dynamo.reset()
    config = AutoConfig.from_pretrained(spec["path"], local_files_only=True)
    heads = int(_nested(config, "num_attention_heads"))
    head_dim = _nested(config, "head_dim")
    if head_dim is None:
        head_dim = int(_nested(config, "hidden_size")) // heads
    head_dim = int(head_dim)
    module = importlib.import_module(spec["module"])
    rotary_fn = getattr(module, spec["rotary"])
    single = bool(spec.get("single", False))

    class Wrapper(torch.nn.Module):
        def forward(self, q, k, cos, sin):
            if single:
                return rotary_fn(q, cos, sin)
            return rotary_fn(q, k, cos, sin)

    reference = Wrapper().to(device=device).eval()
    candidate = torch.compile(reference, backend="inductor", fullgraph=True, dynamic=False)
    rows = []
    for index in range(samples):
        generator = torch.Generator(device=device)
        generator.manual_seed(20260917 + index)
        q = torch.randn((2, min(heads, 8), 32, head_dim), generator=generator,
                        device=device, dtype=torch.bfloat16)
        k = torch.randn_like(q)
        phase = torch.randn((2, 32, head_dim), generator=generator,
                            device=device, dtype=torch.float32)
        cos = torch.cos(phase).to(torch.bfloat16)
        sin = torch.sin(phase).to(torch.bfloat16)
        with torch.no_grad():
            observed_reference = reference(q, k, cos, sin)
            observed_candidate = candidate(q, k, cos, sin)
            bf16_formula = _explicit(q, k, cos, sin, fp32=False, single=single)
            fp32_formula = _explicit(q, k, cos, sin, fp32=True, single=single)
        ref_values = observed_reference if isinstance(observed_reference, tuple) else (observed_reference,)
        cand_values = observed_candidate if isinstance(observed_candidate, tuple) else (observed_candidate,)
        rows.append({
            "sample": index,
            "reference_vs_bf16_formula": [_metric(a, b) for a, b in zip(ref_values, bf16_formula)],
            "candidate_vs_reference": [_metric(a, b) for a, b in zip(cand_values, ref_values)],
            "candidate_vs_fp32_formula": [_metric(a, b) for a, b in zip(cand_values, fp32_formula)],
        })
    return {
        "model_path": spec["path"],
        "module": spec["module"],
        "model_type": getattr(config, "model_type", None),
        "head_count_used": min(heads, 8),
        "head_dim": head_dim,
        "samples": samples,
        "single_tensor_helper": single,
        "rows": rows,
        "all_reference_matches_bf16_formula": all(
            item["exact"] for row in rows for item in row["reference_vs_bf16_formula"]
        ),
        "all_candidate_matches_fp32_formula": all(
            item["exact"] for row in rows for item in row["candidate_vs_fp32_formula"]
        ),
        "all_candidate_differs_from_reference": all(
            item["nonzero_coordinates"] > 0
            for row in rows for item in row["candidate_vs_reference"]
        ),
        "max_candidate_reference_relative_l2": max(
            item["relative_l2"] for row in rows for item in row["candidate_vs_reference"]
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument(
        "--models", nargs="+", choices=tuple(MODEL_SPECS), default=None,
        help="Optional subset of registered rotary model specs; default scans all.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ.setdefault("TRITON_CACHE_DIR", "/data1/tzh/cache/triton_mainstream_bias_scan")
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/data1/tzh/cache/torchinductor_mainstream_bias_scan")

    import torch

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise SystemExit("a CUDA device is required")
    results: dict[str, Any] = {
        "schema": "mainstream-rotary-source-analysis-v1",
        "status": "COMPLETE",
        "candidate_route": "torch.compile backend=inductor fullgraph=True",
        "reference_route": "eager Transformers rotary helper",
        "input_policy": "identical generated BF16 q/k/cos/sin operands",
        "interpretation": (
            "A candidate that matches the explicit FP32-intermediate expression while the "
            "eager reference matches the BF16-stepwise expression has a behavioral attribution "
            "to intermediate arithmetic/materialization. This does not prove a particular GPU "
            "instruction and does not establish a training-level consequence."
        ),
        "models": {},
    }
    selected_models = (
        [(name, MODEL_SPECS[name]) for name in args.models]
        if args.models is not None else list(MODEL_SPECS.items())
    )
    results["selected_models"] = [name for name, _ in selected_models]
    for name, spec in selected_models:
        try:
            results["models"][name] = _run_model(name, spec, args.samples, device)
            print(name, "OK", results["models"][name]["all_candidate_matches_fp32_formula"])
        except Exception as error:
            results["models"][name] = {
                "status": "UNRESOLVED_MEASUREMENT",
                "error": f"{type(error).__name__}: {error}",
            }
            print(name, "UNRESOLVED_MEASUREMENT", error)
        torch.cuda.empty_cache()
    args.output = args.output.resolve()
    if not args.output.is_relative_to(Path(__file__).resolve().parents[1]):
        raise ValueError("output must be inside the repository")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
