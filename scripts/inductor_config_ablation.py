#!/usr/bin/env python3
"""Causally ablate Inductor reduction splitting and kernel fusion."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

from scripts.long_horizon_trigger import build_model, load_eval_states, load_milestone, under_root


PARAMETERS = (
    "model.layers.19.self_attn.q_norm.weight",
    "model.layers.19.self_attn.k_norm.weight",
)
VARIANTS = {
    "default": {},
    "no_split_reductions": {"split_reductions": False},
    "minimal_fusion": {"max_fusion_size": 1, "epilogue_fusion": False},
    "materialize_reused_intermediates": {
        "max_fusion_size": 1,
        "epilogue_fusion": False,
        "realize_reads_threshold": 0,
        "realize_opcount_threshold": 0,
        "realize_acc_reads_threshold": 0,
    },
}


def run(model: Any, inputs: tuple[Any, Any], program: Any | None) -> tuple[float, dict[str, Any]]:
    model.zero_grad(set_to_none=True)
    input_ids, labels = inputs
    loss = (
        model(input_ids=input_ids, labels=labels, use_cache=False, return_dict=False)[0]
        if program is None else program(input_ids, labels)
    )
    loss.backward()
    gradients = {
        name: dict(model.named_parameters())[name].grad.detach().float().cpu().clone()
        for name in PARAMETERS
    }
    return float(loss.detach().float().cpu()), gradients


def contrast(reference: Any, candidate: Any, default_direction: Any) -> dict[str, Any]:
    delta = candidate - reference
    l2 = float(delta.norm())
    projection = float((delta * default_direction).sum())
    return {
        "exact": bool((delta == 0).all()),
        "numel": delta.numel(),
        "nonzero": int((delta != 0).sum()),
        "l2": l2,
        "rms": l2 / math.sqrt(delta.numel()),
        "max_abs": float(delta.abs().max()),
        "signed_mean": float(delta.mean()),
        "default_direction_projection": projection,
        "default_direction_residual_ratio": projection,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, default=Path("results/final/long_horizon_bank.json"))
    parser.add_argument("--model", type=Path, default=Path("/data1/tzh/models/Qwen/Qwen3-1.7B"))
    parser.add_argument("--step", type=int, default=256)
    parser.add_argument("--state", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("results/final/inductor_config_ablation.json"))
    args = parser.parse_args()
    bank_path = under_root(args.bank, "bank")
    model_path = under_root(args.model, "model")
    output_path = under_root(args.output, "output")

    os.environ.setdefault("HF_HOME", "/data1/tzh/cache/huggingface")
    os.environ.setdefault("HF_DATASETS_CACHE", "/data1/tzh/cache/huggingface/datasets")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/data1/tzh/cache/huggingface/transformers")
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/data1/tzh/cache/huggingface/hub")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    import torch
    from torch._dynamo.backends.registry import lookup_backend
    from torch._inductor import config
    from transformers import AutoTokenizer

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")

    bank = json.loads(bank_path.read_text())
    milestone = next(row for row in bank["milestones"] if int(row["step"]) == args.step)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
    states, protocol = load_eval_states(tokenizer, 1024, args.state + 1, device)
    inputs = states[args.state]
    model = build_model(model_path, device)
    load_milestone(model, milestone, model_path)

    class LossStep(torch.nn.Module):
        def __init__(self, subject: Any) -> None:
            super().__init__()
            self.subject = subject

        def forward(self, input_ids: Any, labels: Any) -> Any:
            return self.subject(
                input_ids=input_ids, labels=labels, use_cache=False, return_dict=False
            )[0]

    eager = [run(model, inputs, None) for _ in range(2)]
    captures: dict[str, list[tuple[float, dict[str, Any]]]] = {}
    audits = {}
    for variant, settings in VARIANTS.items():
        audit = {"backend_compiles": 0, "graph_sha256": []}
        inductor = lookup_backend("inductor")

        def backend(graph_module: Any, example_inputs: list[Any]) -> Any:
            audit["backend_compiles"] += 1
            audit["graph_sha256"].append(
                hashlib.sha256(graph_module.code.encode()).hexdigest()
            )
            return inductor(graph_module, example_inputs)

        with config.patch(settings):
            program = torch.compile(
                LossStep(model), backend=backend, fullgraph=True, dynamic=False
            )
            warm_loss, warm_gradients = run(model, inputs, program)
            del warm_loss, warm_gradients
            captures[variant] = [run(model, inputs, program) for _ in range(2)]
            torch.cuda.synchronize(device)
        audits[variant] = audit
        del program
        gc.collect()
        torch.cuda.empty_cache()

    default_directions = {}
    for name in PARAMETERS:
        delta = captures["default"][0][1][name] - eager[0][1][name]
        default_directions[name] = delta / delta.norm().clamp_min(1e-30)

    rows = []
    for variant, repeats in captures.items():
        for repeat, (loss, gradients) in enumerate(repeats):
            item = {
                "variant": variant,
                "config": VARIANTS[variant],
                "repeat": repeat,
                "eager_loss": eager[repeat][0],
                "candidate_loss": loss,
                "loss_delta": loss - eager[repeat][0],
                "parameters": {},
            }
            for name in PARAMETERS:
                value = contrast(eager[repeat][1][name], gradients[name], default_directions[name])
                default_l2 = float(
                    (captures["default"][repeat][1][name] - eager[repeat][1][name]).norm()
                )
                value["default_direction_residual_ratio"] = (
                    value["default_direction_projection"] / (default_l2 + 1e-30)
                )
                value["default_direction_removal_fraction"] = (
                    1.0 - value["default_direction_residual_ratio"]
                )
                item["parameters"][name] = value
            rows.append(item)

    repeat_gates = {}
    for variant, repeats in {"eager": eager, **captures}.items():
        repeat_gates[variant] = {
            "loss_exact": repeats[0][0] == repeats[1][0],
            "parameter_exact": {
                name: bool((repeats[0][1][name] == repeats[1][1][name]).all())
                for name in PARAMETERS
            },
        }
    output = {
        "schema": "kernel-analyzer-inductor-config-ablation-v1",
        "subject": "Qwen3-1.7B seq1024 Inductor mechanism ablation",
        "checkpoint_step": args.step,
        "state_id": args.state,
        "evaluation": protocol,
        "parameters": list(PARAMETERS),
        "variants": VARIANTS,
        "compile_audits": audits,
        "rows": rows,
        "repeat_gates": repeat_gates,
        "full_gradient_tensors_saved": False,
        "boundary": (
            "Single frozen checkpoint/state mechanism screen. A configuration effect must be "
            "replayed on held-out checkpoints/states before becoming a causal case."
        ),
    }
    output["result_sha256"] = hashlib.sha256(
        json.dumps(output, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output_path), "variants": list(VARIANTS)}, sort_keys=True))
    del captures, eager, model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
