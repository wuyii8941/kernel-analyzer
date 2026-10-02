#!/usr/bin/env python3
"""Attribute the source of generated BF16 projection differences.

The companion mainstream scan uses a small linear projection for every local
model configuration.  This probe keeps operands identical and compares the
observed eager/Inductor outputs with direct BF16 matmul and FP32-accumulate
then BF16-store expressions.  A match is a behavioral arithmetic attribution;
it is not an instruction-level proof and does not establish a training effect.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


MODEL_SPECS = {
    "qwen2_0p5b": "/data1/tzh/models/Qwen/Qwen2.5-0.5B-Instruct",
    "tinyllama_1p1b": "/data1/tzh/models/TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "qwen3_1p7b": "/data1/tzh/models/Qwen/Qwen3-1.7B",
    "qwen3_vl_reranker_2b": "/data1/tzh/models/Qwen/Qwen3-VL-Reranker-2B",
    "deepseek_qwen3_8b": "/data1/tzh/models/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
    "llama3_2_3b": "/data1/tzh/models/meta-llama/Llama-3.2-3B",
    "gemma3_4b": "/data1/tzh/models/google/gemma-3-4b-pt",
    "gemma4_e2b": "/data1/tzh/models/google/gemma-4-E2B",
    "phi4_mini": "/data1/tzh/models/microsoft/Phi-4-mini-instruct",
    "granite_moe": "/data1/tzh/models/ibm-granite/granite-3.1-1b-a400m-base",
    "olmoe_1b": "/data1/tzh/models/allenai/OLMoE-1B-7B-0125",
    "mamba_130m": "/data1/tzh/models/state-spaces/mamba-130m-hf",
    "ministral3_3b": "/data1/tzh/models/mistralai/Ministral-3-3B-Base-2512",
}


def _nested(config: Any, name: str, default: Any = None) -> Any:
    raw = getattr(config, "__dict__", {})
    if name in raw and raw[name] is not None:
        return raw[name]
    value = getattr(config, name, None)
    if value is not None:
        return value
    for child_name in ("text_config", "llm_config", "language_config"):
        child = getattr(config, child_name, None)
        if child is not None:
            value = getattr(child, name, None)
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


def _run_model(name: str, path: str, samples: int, device: Any) -> dict[str, Any]:
    import torch
    from transformers import AutoConfig

    # Each model has a different feature dimension.  Reset Dynamo's process
    # cache so a guard compiled for one configuration cannot consume the
    # global recompile budget of the next one.
    torch._dynamo.reset()
    config = AutoConfig.from_pretrained(path, local_files_only=True)
    hidden = int(_nested(config, "hidden_size"))
    out_features = min(hidden, 1024)

    class Projection(torch.nn.Module):
        def __init__(self):
            super().__init__()
            generator = torch.Generator(device=device)
            generator.manual_seed(9000 + hidden + out_features)
            self.weight = torch.nn.Parameter(
                torch.randn((out_features, hidden), generator=generator,
                            device=device, dtype=torch.bfloat16)
                / max(1.0, hidden ** 0.5), requires_grad=False
            )

        def forward(self, x):
            return torch.nn.functional.linear(x, self.weight)

    reference = Projection().eval()
    candidate = torch.compile(reference, backend="inductor", fullgraph=True, dynamic=False)
    rows = []
    for index in range(samples):
        generator = torch.Generator(device=device)
        generator.manual_seed(20260918 + index)
        x = torch.randn((2, hidden), generator=generator, device=device, dtype=torch.bfloat16)
        x[:, : max(1, hidden // 8)] *= 0.03125
        with torch.no_grad():
            observed_reference = reference(x)
            observed_candidate = candidate(x)
            direct_bf16 = torch.nn.functional.linear(x, reference.weight)
            fp32_accumulate = torch.nn.functional.linear(
                x.float(), reference.weight.float()
            ).to(torch.bfloat16)
        rows.append({
            "sample": index,
            "reference_vs_direct_bf16": _metric(observed_reference, direct_bf16),
            "candidate_vs_reference": _metric(observed_candidate, observed_reference),
            "candidate_vs_fp32_accumulate": _metric(observed_candidate, fp32_accumulate),
            "reference_vs_fp32_accumulate": _metric(observed_reference, fp32_accumulate),
        })
    return {
        "model_path": path,
        "model_type": getattr(config, "model_type", None),
        "hidden_size": hidden,
        "output_width": out_features,
        "samples": samples,
        "rows": rows,
        "all_reference_matches_direct_bf16": all(
            row["reference_vs_direct_bf16"]["exact"] for row in rows
        ),
        "all_candidate_matches_fp32_accumulate": all(
            row["candidate_vs_fp32_accumulate"]["exact"] for row in rows
        ),
        "all_candidate_differs_from_reference": all(
            row["candidate_vs_reference"]["nonzero_coordinates"] > 0 for row in rows
        ),
        "max_candidate_reference_relative_l2": max(
            row["candidate_vs_reference"]["relative_l2"] for row in rows
        ),
    }


def main() -> int:
    import argparse
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ.setdefault("TRITON_CACHE_DIR", "/data1/tzh/cache/triton_mainstream_bias_scan")
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/data1/tzh/cache/torchinductor_mainstream_bias_scan")
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
        "schema": "mainstream-linear-source-analysis-v1",
        "status": "COMPLETE",
        "candidate_route": "torch.compile backend=inductor fullgraph=True",
        "reference_route": "eager torch.nn.functional.linear",
        "input_policy": "identical generated BF16 operands and declared synthetic projection weights",
        "interpretation": (
            "A candidate matching the explicit FP32-accumulate/BF16-store expression while "
            "the eager reference matches direct BF16 matmul is a behavioral attribution to "
            "the accumulation/materialization contract. No row-level match is an instruction "
            "trace or a training-level causal proof."
        ),
        "models": {},
    }
    for name, path in MODEL_SPECS.items():
        try:
            results["models"][name] = _run_model(name, path, args.samples, device)
            print(name, "OK", results["models"][name]["all_candidate_matches_fp32_accumulate"])
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
