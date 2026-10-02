#!/usr/bin/env python3
"""Run the public bias checker on operators used by local mainstream models.

This scan deliberately uses small operator modules rather than loading full model
weights.  The model config selects the real hidden size, epsilon, activation and
attention shape; when available, one checkpoint norm vector is loaded without
constructing the model.  The candidate is an Inductor-compiled implementation
and the reference is the corresponding eager PyTorch implementation.  Results
are conditional on the generated input distribution and are never promoted to a
model-wide or training-loss claim.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
from typing import Any, Callable


MODEL_SPECS = {
    "bert_base_uncased": {
        "path": "/data1/tzh/models/google-bert/bert-base-uncased",
        "norm_kind": "layernorm",
        "activation": "gelu",
    },
    "roberta_base": {
        "path": "/data1/tzh/models/FacebookAI/roberta-base",
        "norm_kind": "layernorm",
        "activation": "gelu",
    },
    "gpt_neo_125m": {
        "path": "/data1/tzh/models/EleutherAI/gpt-neo-125m",
        "norm_kind": "layernorm",
        "activation": "gelu_new",
    },
    "deberta_v3_small": {
        "path": "/data1/tzh/models/microsoft/deberta-v3-small",
        "norm_kind": "layernorm",
        "activation": "gelu",
    },
    "electra_small": {
        "path": "/data1/tzh/models/google/electra-small-discriminator",
        "norm_kind": "layernorm",
        "activation": "gelu",
    },
    "opt_350m": {
        "path": "/data1/tzh/models/facebook/opt-350m",
        "norm_kind": "layernorm",
        "activation": "relu",
    },
    "pythia_410m": {
        "path": "/data1/tzh/models/EleutherAI/pythia-410m",
        "norm_kind": "layernorm",
        "activation": "gelu",
        "rotary": ("transformers.models.gpt_neox.modeling_gpt_neox", "apply_rotary_pos_emb"),
    },
    "falcon_rw_1b": {
        "path": "/data1/tzh/models/tiiuae/falcon-rw-1b",
        "norm_kind": "layernorm",
        "activation": "gelu",
        "rotary": ("transformers.models.falcon.modeling_falcon", "apply_rotary_pos_emb"),
    },
    "bloom_560m": {
        "path": "/data1/tzh/models/bigscience/bloom-560m",
        "norm_kind": "layernorm",
        "activation": "gelu",
    },
    "gpt2": {
        "path": "/data1/tzh/models/openai-community/gpt2",
        "norm_kind": "layernorm",
        "activation": "gelu_new",
    },
    "qwen2_0p5b": {
        "path": "/data1/tzh/models/Qwen/Qwen2.5-0.5B-Instruct",
        "norm": ("transformers.models.qwen2.modeling_qwen2", "Qwen2RMSNorm"),
        "rotary": ("transformers.models.qwen2.modeling_qwen2", "apply_rotary_pos_emb"),
    },
    "tinyllama_1p1b": {
        "path": "/data1/tzh/models/TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "norm": ("transformers.models.llama.modeling_llama", "LlamaRMSNorm"),
        "rotary": ("transformers.models.llama.modeling_llama", "apply_rotary_pos_emb"),
    },
    "qwen3_1p7b": {
        "path": "/data1/tzh/models/Qwen/Qwen3-1.7B",
        "norm": ("transformers.models.qwen3.modeling_qwen3", "Qwen3RMSNorm"),
        "rotary": ("transformers.models.qwen3.modeling_qwen3", "apply_rotary_pos_emb"),
    },
    "qwen3_vl_reranker_2b": {
        "path": "/data1/tzh/models/Qwen/Qwen3-VL-Reranker-2B",
        "norm": ("transformers.models.qwen3_vl.modeling_qwen3_vl", "Qwen3VLTextRMSNorm"),
        "rotary": ("transformers.models.qwen3_vl.modeling_qwen3_vl", "apply_rotary_pos_emb"),
    },
    "deepseek_qwen3_8b": {
        "path": "/data1/tzh/models/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
        "norm": ("transformers.models.qwen3.modeling_qwen3", "Qwen3RMSNorm"),
        "rotary": ("transformers.models.qwen3.modeling_qwen3", "apply_rotary_pos_emb"),
    },
    "llama3_2_3b": {
        "path": "/data1/tzh/models/meta-llama/Llama-3.2-3B",
        "norm": ("transformers.models.llama.modeling_llama", "LlamaRMSNorm"),
        "rotary": ("transformers.models.llama.modeling_llama", "apply_rotary_pos_emb"),
    },
    "gemma3_4b": {
        "path": "/data1/tzh/models/google/gemma-3-4b-pt",
        "norm": ("transformers.models.gemma3.modeling_gemma3", "Gemma3RMSNorm"),
        "rotary": ("transformers.models.gemma3.modeling_gemma3", "apply_rotary_pos_emb"),
    },
    "gemma4_e2b": {
        "path": "/data1/tzh/models/google/gemma-4-E2B",
        "norm": ("transformers.models.gemma4.modeling_gemma4", "Gemma4RMSNorm"),
        "rotary": ("transformers.models.gemma4.modeling_gemma4", "apply_rotary_pos_emb"),
        "rotary_single": True,
    },
    "phi4_mini": {
        "path": "/data1/tzh/models/microsoft/Phi-4-mini-instruct",
        "norm": ("transformers.models.phi3.modeling_phi3", "Phi3RMSNorm"),
        "rotary": ("transformers.models.phi3.modeling_phi3", "apply_rotary_pos_emb"),
    },
    "granite_moe": {
        "path": "/data1/tzh/models/ibm-granite/granite-3.1-1b-a400m-base",
        "norm": ("transformers.models.granitemoe.modeling_granitemoe", "GraniteMoeRMSNorm"),
        "rotary": ("transformers.models.granitemoe.modeling_granitemoe", "apply_rotary_pos_emb"),
    },
    "olmoe_1b": {
        "path": "/data1/tzh/models/allenai/OLMoE-1B-7B-0125",
        "norm": ("transformers.models.olmoe.modeling_olmoe", "OlmoeRMSNorm"),
        "rotary": ("transformers.models.olmoe.modeling_olmoe", "apply_rotary_pos_emb"),
    },
    "mamba_130m": {
        "path": "/data1/tzh/models/state-spaces/mamba-130m-hf",
        "norm": ("transformers.models.mamba.modeling_mamba", "MambaRMSNorm"),
        "softplus_probe": True,
    },
    "ministral3_3b": {
        "path": "/data1/tzh/models/mistralai/Ministral-3-3B-Base-2512",
        "norm": ("transformers.models.mistral3.modeling_mistral3", "Mistral3RMSNorm"),
    },
}


def _nested(config: Any, name: str, default: Any = None) -> Any:
    raw_config = getattr(config, "__dict__", {})
    if name in raw_config and raw_config[name] is not None:
        return raw_config[name]
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
            child_value = getattr(child, name, None)
            if child_value is not None:
                return child_value
    return default


def _load_norm_class(spec: dict[str, Any]) -> type:
    if spec.get("norm_kind") == "layernorm":
        import torch
        return torch.nn.LayerNorm
    module_name, class_name = spec["norm"]
    return getattr(importlib.import_module(module_name), class_name)


def _load_one_norm_weight(model_path: str, hidden: int):
    """Load one norm vector without constructing the full model."""
    index_path = Path(model_path) / "model.safetensors.index.json"
    if not index_path.is_file():
        return None, "synthetic_weight_no_index"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    candidates = [
        name for name in index.get("weight_map", {})
        if name.endswith("input_layernorm.weight")
        or name.endswith("post_attention_layernorm.weight")
        or name.endswith("norm_f.weight")
        or name.endswith("final_layernorm.weight")
    ]
    if not candidates:
        return None, "synthetic_weight_no_norm_key"
    key = sorted(candidates)[0]
    try:
        from safetensors import safe_open
        shard = Path(model_path) / index["weight_map"][key]
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            weight = handle.get_tensor(key)
        if tuple(weight.shape) != (hidden,):
            return None, f"synthetic_weight_shape_{tuple(weight.shape)}"
        return weight, f"checkpoint:{key}"
    except Exception as error:
        return None, f"synthetic_weight_load_error:{type(error).__name__}"


def _make_inputs(torch: Any, shape: tuple[int, ...], dtype: Any, seed_base: int):
    def make(index: int):
        generator = torch.Generator(device="cuda")
        generator.manual_seed(seed_base + index)
        x = torch.randn(shape, generator=generator, device="cuda", dtype=dtype)
        # Include a deterministic scale mixture to exercise both ordinary and
        # small-magnitude activation paths without using model activations as
        # an unrecorded source of randomness.
        if len(shape) >= 1:
            scale = torch.ones(shape[-1], device="cuda", dtype=dtype)
            scale[: max(1, shape[-1] // 8)] = 0.03125
            x = x * scale
        return (x,)

    return make


def _make_sdpa_inputs(torch: Any, heads: int, head_dim: int, seq: int, seed_base: int, dtype: Any):
    def make(index: int):
        generator = torch.Generator(device="cuda")
        generator.manual_seed(seed_base + index)
        shape = (2, heads, seq, head_dim)
        return tuple(torch.randn(shape, generator=generator, device="cuda", dtype=dtype)
                     for _ in range(3))

    return make


def _make_softmax_inputs(torch: Any, heads: int, seq: int, seed_base: int, dtype: Any):
    """Generate attention-shaped logits for an isolated softmax comparison."""

    def make(index: int):
        generator = torch.Generator(device="cuda")
        generator.manual_seed(seed_base + index)
        logits = torch.randn(
            (2, heads, seq, seq), generator=generator, device="cuda", dtype=dtype
        ) * 4
        # Keep a deterministic range mixture so the reduction sees both
        # ordinary and sharply peaked rows without using model activations.
        logits[..., 0] = logits[..., 0] + 12
        return (logits,)

    return make


def _make_rotary_inputs(torch: Any, heads: int, head_dim: int, seq: int, seed_base: int, dtype: Any):
    """Generate q/k tensors and a broadcastable cos/sin pair for RoPE."""

    def make(index: int):
        generator = torch.Generator(device="cuda")
        generator.manual_seed(seed_base + index)
        q, k = (
            torch.randn((2, heads, seq, head_dim), generator=generator, device="cuda", dtype=dtype)
            for _ in range(2)
        )
        phase = torch.randn((2, seq, head_dim), generator=generator, device="cuda", dtype=torch.float32)
        return q, k, torch.cos(phase).to(dtype=dtype), torch.sin(phase).to(dtype=dtype)

    return make


def _make_softplus_inputs(torch: Any, width: int, seed_base: int, dtype: Any):
    """Exercise both smooth and threshold branches of Mamba's softplus."""

    def make(index: int):
        generator = torch.Generator(device="cuda")
        generator.manual_seed(seed_base + index)
        x = torch.randn((2, width), generator=generator, device="cuda", dtype=dtype)
        x = x * 3
        x[:, : max(1, width // 8)] = -4
        x[:, max(1, width // 8): max(2, width // 4)] = 4
        x[:, max(2, width // 4): max(3, width // 4 + 1)] = 20
        return (x,)

    return make


def _compile(module: Any):
    import torch

    return torch.compile(module, backend="inductor", fullgraph=True, dynamic=False)


def _run_check(check_bias: Callable[..., dict[str, Any]], candidate: Any, reference: Any,
               make_inputs: Callable[[int], Any], samples: int, backward: bool) -> dict[str, Any]:
    report = check_bias(
        candidate,
        reference,
        make_inputs,
        samples=samples,
        calibration_samples=samples // 2,
        check_backward=backward,
    )
    stages = report.get("stages", {})
    return {
        "status": report.get("status"),
        "measurement_status": report.get("measurement_status"),
        "stages": {
            name: {
                "total_rms": data.get("total_rms"),
                "aligned_ratio_of_sums": data.get("aligned_ratio_of_sums"),
                "aligned_projection_interval": data.get("aligned_projection_interval"),
                "aligned_estimand": data.get("aligned_estimand"),
                "paired_sufficient_statistics": data.get("paired_sufficient_statistics"),
                "sample_indices": data.get("sample_indices"),
                "endpoint_alpha": data.get("endpoint_alpha"),
                "direction": data.get("direction"),
                "systematic_bias_reasons": data.get("systematic_bias_reasons", []),
                "decision": data.get("decision"),
                "errors": data.get("errors", []),
            }
            for name, data in stages.items()
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument(
        "--models", nargs="+", choices=tuple(MODEL_SPECS), default=None,
        help="Optional subset of registered model specs; default scans all.",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/property/mainstream_model_bias_scan_v1/scan_20260917.json"),
    )
    args = parser.parse_args()
    if args.samples < 4:
        raise SystemExit("--samples must be >= 4")

    os.environ.setdefault("TRITON_CACHE_DIR", "/data1/tzh/cache/triton_mainstream_bias_scan")
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/data1/tzh/cache/torchinductor_mainstream_bias_scan")

    import torch
    from transformers import AutoConfig
    from transformers.activations import ACT2FN
    from kernel_analyzer import check_bias

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise SystemExit("a CUDA device is required")
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32

    class ActivationModule(torch.nn.Module):
        def __init__(self, fn):
            super().__init__()
            self.fn = fn

        def forward(self, x):
            return self.fn(x)

    class LinearModule(torch.nn.Module):
        """A model-shaped projection probe with a declared synthetic weight.

        The probe is intentionally small enough to run for every config while
        preserving the same BF16 input/weight arithmetic used by a projection
        in a decoder block.  It is a generated-input operator result, not a
        claim about any particular checkpoint layer.
        """

        def __init__(self, in_features: int, out_features: int, *, dtype: Any):
            super().__init__()
            generator = torch.Generator(device=device)
            generator.manual_seed(9000 + in_features + out_features)
            weight = torch.randn(
                (out_features, in_features), generator=generator,
                device=device, dtype=dtype,
            ) / max(1.0, in_features ** 0.5)
            self.weight = torch.nn.Parameter(weight, requires_grad=False)

        def forward(self, x):
            return torch.nn.functional.linear(x, self.weight)

    class SdpaModule(torch.nn.Module):
        def forward(self, q, k, v):
            return torch.nn.functional.scaled_dot_product_attention(
                q, k, v, is_causal=True
            )

    class SoftmaxModule(torch.nn.Module):
        def forward(self, logits):
            return torch.softmax(logits, dim=-1)

    class SoftplusModule(torch.nn.Module):
        def forward(self, x):
            # Mamba's declared activation uses the standard beta=1,
            # threshold=20 branch.  This is an operator probe, not a claim
            # about every occurrence of softplus in a model.
            return torch.nn.functional.softplus(x, beta=1.0, threshold=20.0)

    class RotaryModule(torch.nn.Module):
        def __init__(self, apply_fn, single_input=False):
            super().__init__()
            self.apply_fn = apply_fn
            self.single_input = single_input

        def forward(self, q, k, cos, sin):
            if self.single_input:
                # Gemma 4's helper rotates one tensor and therefore has the
                # signature (x, cos, sin), unlike the q/k helpers used by the
                # other decoder families.  Keep a matched four-input wrapper
                # and deliberately ignore k on both sides.
                return self.apply_fn(q, cos, sin)
            return self.apply_fn(q, k, cos, sin)

    results: dict[str, Any] = {
        "schema": "kernel-analyzer-mainstream-model-operator-bias-scan-v1",
        "device": str(device),
        "samples": args.samples,
        "dtype": args.dtype,
        "candidate_route": "torch.compile backend=inductor fullgraph=True",
        "reference_route": "eager PyTorch/model implementation",
        "scope": "DECLARED_MODEL_CONFIG_INPUTS_ONLY",
        "cases": {},
        "notes": [
            "No full model is constructed; one checkpoint norm vector is loaded when an index is available, otherwise a declared synthetic vector is used.",
            "A confirmed result is conditional on this generated input distribution and compiled route.",
            "No result is promoted to a model-wide training or loss claim.",
        ],
    }

    selected_models = (
        [(name, MODEL_SPECS[name]) for name in args.models]
        if args.models is not None else list(MODEL_SPECS.items())
    )
    results["selected_models"] = [name for name, _ in selected_models]
    for model_name, spec in selected_models:
        case: dict[str, Any] = {"model_path": spec["path"], "operators": {}}
        try:
            # Shape-specialized graphs are intentionally compiled per model;
            # clear Dynamo's global guard cache so one model's dimensions do
            # not turn a later model into a spurious unresolved measurement.
            torch._dynamo.reset()
            config = AutoConfig.from_pretrained(spec["path"], local_files_only=True)
            hidden_value = _nested(config, "hidden_size")
            if hidden_value is None:
                hidden_value = _nested(config, "n_embd")
            if hidden_value is None:
                hidden_value = _nested(config, "n_embed")
            if hidden_value is None:
                hidden_value = _nested(config, "d_model")
            if hidden_value is None:
                raise ValueError("model config has no hidden-size alias")
            hidden = int(hidden_value)
            eps = float(_nested(
                config, "rms_norm_eps",
                _nested(config, "layer_norm_eps",
                        _nested(config, "layer_norm_epsilon", 1e-6)),
            ))
            hidden_act = str(
                _nested(config, "hidden_act", None)
                or _nested(config, "activation_function", None)
                or _nested(config, "hidden_activation", None)
                or spec.get("activation", "silu")
            )
            heads = _nested(config, "num_attention_heads", _nested(config, "n_head"))
            head_dim = _nested(config, "head_dim")
            if head_dim is None and heads:
                head_dim = hidden // int(heads)
            case["config"] = {
                "model_type": getattr(config, "model_type", None),
                "hidden_size": hidden,
                "rms_norm_eps": eps,
                "hidden_act": hidden_act,
                "num_attention_heads": int(heads) if heads else None,
                "head_dim": int(head_dim) if head_dim else None,
            }

            norm_cls = _load_norm_class(spec)
            norm = norm_cls(hidden, eps).to(device=device, dtype=dtype).eval()
            checkpoint_weight, weight_source = _load_one_norm_weight(spec["path"], hidden)
            with torch.no_grad():
                if checkpoint_weight is None:
                    norm.weight.copy_(1.0 + 0.01 * torch.sin(torch.arange(hidden, device=device)))
                else:
                    norm.weight.copy_(checkpoint_weight.to(device=device, dtype=dtype))
            case["norm_weight_source"] = weight_source
            norm_candidate = _compile(norm)
            norm_inputs = _make_inputs(torch, (2, hidden), dtype, 1000)
            norm_inputs(0)
            norm_operator = "layernorm" if spec.get("norm_kind") == "layernorm" else "rmsnorm"
            case["operators"][norm_operator] = _run_check(
                check_bias, norm_candidate, norm, norm_inputs, args.samples, backward=True
            )

            act_fn = ACT2FN[hidden_act]
            activation = ActivationModule(act_fn).to(device=device).eval()
            activation_candidate = _compile(activation)
            activation_inputs = _make_inputs(torch, (2, hidden), dtype, 2000)
            activation_inputs(0)
            case["operators"]["activation"] = _run_check(
                check_bias, activation_candidate, activation, activation_inputs, args.samples, backward=True
            )

            # A bounded projection probe adds a representative GEMM family
            # without constructing the full model.  The output width is
            # declared in the result so it cannot be mistaken for a layer
            # checkpoint measurement.
            projection_width = min(hidden, 1024)
            projection = LinearModule(hidden, projection_width, dtype=dtype).eval()
            projection_candidate = _compile(projection)
            projection_inputs = _make_inputs(torch, (2, hidden), dtype, 5000)
            projection_inputs(0)
            case["operators"]["linear_projection"] = _run_check(
                check_bias, projection_candidate, projection, projection_inputs,
                args.samples, backward=True
            )
            case["operators"]["linear_projection"]["declared_output_width"] = projection_width

            if spec.get("softplus_probe"):
                softplus = SoftplusModule().to(device=device).eval()
                softplus_candidate = _compile(softplus)
                softplus_inputs = _make_softplus_inputs(torch, hidden, 6000, dtype)
                softplus_inputs(0)
                case["operators"]["softplus"] = _run_check(
                    check_bias, softplus_candidate, softplus, softplus_inputs,
                    args.samples, backward=True
                )

            if heads and head_dim:
                softmax = SoftmaxModule().to(device=device).eval()
                softmax_candidate = _compile(softmax)
                softmax_inputs = _make_softmax_inputs(
                    torch, min(int(heads), 8), 32, 3500, dtype
                )
                softmax_inputs(0)
                case["operators"]["softmax"] = _run_check(
                    check_bias, softmax_candidate, softmax, softmax_inputs, args.samples, backward=True
                )

                sdpa = SdpaModule().to(device=device).eval()
                sdpa_candidate = _compile(sdpa)
                sdpa_inputs = _make_sdpa_inputs(
                    torch, min(int(heads), 8), int(head_dim), 32, 3000, dtype
                )
                sdpa_inputs(0)
                case["operators"]["scaled_dot_product_attention"] = _run_check(
                    check_bias, sdpa_candidate, sdpa, sdpa_inputs, args.samples, backward=True
                )
            else:
                case["operators"]["scaled_dot_product_attention"] = {
                    "status": "NOT_APPLICABLE",
                    "measurement_status": "NOT_ASSESSED",
                    "reason": "model config has no text attention head shape",
                }

            rotary_spec = spec.get("rotary")
            if rotary_spec and heads and head_dim:
                rotary_fn = getattr(importlib.import_module(rotary_spec[0]), rotary_spec[1])
                rotary = RotaryModule(
                    rotary_fn, single_input=bool(spec.get("rotary_single", False))
                ).to(device=device).eval()
                rotary_candidate = _compile(rotary)
                rotary_inputs = _make_rotary_inputs(
                    torch, min(int(heads), 8), int(head_dim), 32, 4000, dtype
                )
                rotary_inputs(0)
                case["operators"]["rotary_position_embedding"] = _run_check(
                    # The public rotary helper returns a (q, k) tuple.  The
                    # lightweight checker deliberately limits its automatic
                    # backward probe to a single tensor output, so this case
                    # records the declared output bias without manufacturing a
                    # scalar cotangent for a multi-output Jacobian.
                    check_bias, rotary_candidate, rotary, rotary_inputs, args.samples, backward=False
                )
            else:
                case["operators"]["rotary_position_embedding"] = {
                    "status": "NOT_APPLICABLE",
                    "measurement_status": "NOT_ASSESSED",
                    "reason": "model config or implementation has no rotary position embedding adapter",
                }
        except Exception as error:
            case["status"] = "UNRESOLVED_MEASUREMENT"
            case["error"] = f"{type(error).__name__}: {error}"
        results["cases"][model_name] = case
        print(model_name, case.get("status", "OK"),
              {name: data.get("status") for name, data in case.get("operators", {}).items()})

        # Release compiled graphs and module storage before the next model.
        for value in list(locals().values()):
            if value is not None and value.__class__.__module__.startswith("torch"):
                del value
        torch.cuda.empty_cache()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
