#!/usr/bin/env python3
"""Separate eager→AOT and AOT→Inductor contributions for trigger gradients."""

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


def metrics(left: Any, right: Any) -> dict[str, Any]:
    delta = right - left
    l2 = float(delta.norm())
    return {
        "exact": bool((delta == 0).all()),
        "numel": delta.numel(),
        "nonzero": int((delta != 0).sum()),
        "l2": l2,
        "rms": l2 / math.sqrt(delta.numel()),
        "max_abs": float(delta.abs().max()),
        "signed_mean": float(delta.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, default=Path("results/final/long_horizon_bank.json"))
    parser.add_argument("--model", type=Path, default=Path("/data1/tzh/models/Qwen/Qwen3-1.7B"))
    parser.add_argument("--step", type=int, default=256)
    parser.add_argument("--state", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("results/final/backend_stage_diagnosis.json"))
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

    programs = {
        "eager": None,
        "aot_eager": torch.compile(LossStep(model), backend="aot_eager", fullgraph=True, dynamic=False),
        "inductor": torch.compile(LossStep(model), backend="inductor", fullgraph=True, dynamic=False),
    }
    # Warm compiled stages before measurement.
    for stage in ("aot_eager", "inductor"):
        warm_loss, warm_gradients = run(model, inputs, programs[stage])
        del warm_loss, warm_gradients
    torch.cuda.synchronize(device)

    captures: dict[str, list[tuple[float, dict[str, Any]]]] = {}
    for stage, program in programs.items():
        captures[stage] = [run(model, inputs, program) for _ in range(2)]
        torch.cuda.synchronize(device)

    comparisons = {}
    for left_stage, right_stage in (
        ("eager", "aot_eager"), ("aot_eager", "inductor"), ("eager", "inductor")
    ):
        key = f"{right_stage}_minus_{left_stage}"
        comparisons[key] = []
        for repeat in range(2):
            left_loss, left = captures[left_stage][repeat]
            right_loss, right = captures[right_stage][repeat]
            comparisons[key].append({
                "repeat": repeat,
                "loss_delta": right_loss - left_loss,
                "parameters": {name: metrics(left[name], right[name]) for name in PARAMETERS},
            })

    repeat_gates = {
        stage: {
            "loss_exact": values[0][0] == values[1][0],
            "parameters": {
                name: metrics(values[0][1][name], values[1][1][name])
                for name in PARAMETERS
            },
        }
        for stage, values in captures.items()
    }
    output = {
        "schema": "kernel-analyzer-backend-stage-diagnosis-v1",
        "subject": "Qwen3-1.7B seq1024 trigger decomposition by compiler stage",
        "checkpoint_step": args.step,
        "state_id": args.state,
        "evaluation": protocol,
        "parameters": list(PARAMETERS),
        "comparisons": comparisons,
        "repeat_gates": repeat_gates,
        "full_gradient_tensors_saved": False,
        "boundary": (
            "AOT-eager isolates graph capture/decomposition from Inductor code generation at one "
            "frozen checkpoint/state; it is a mechanism discriminator, not a natural-case certificate."
        ),
    }
    output["result_sha256"] = hashlib.sha256(
        json.dumps(output, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output_path), "comparisons": list(comparisons)}, sort_keys=True))
    del captures, programs, model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
