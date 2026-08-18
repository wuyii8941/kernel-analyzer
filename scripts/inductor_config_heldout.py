#!/usr/bin/env python3
"""Replay selected Inductor mechanism controls on the frozen long grid."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
from pathlib import Path
from typing import Any

from scripts.long_horizon_trigger import (
    build_model, load_eval_states, load_milestone, percentile, under_root,
)


PARAMETERS = (
    "model.layers.19.self_attn.q_norm.weight",
    "model.layers.19.self_attn.k_norm.weight",
)
VARIANTS = {
    "default": {},
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
    named = dict(model.named_parameters())
    return float(loss.detach().float().cpu()), {
        name: named[name].grad.detach().float().cpu().clone() for name in PARAMETERS
    }


def summarize(rows: list[dict[str, Any]], states: int, bootstrap: int = 2000) -> dict[str, Any]:
    output = []
    heldout_steps = (64, 256, 1024, 2048, 4096)
    for variant in VARIANTS:
        if variant == "default":
            continue
        for parameter in PARAMETERS:
            selected = [
                row for row in rows
                if row["variant"] == variant and row["parameter"] == parameter
                and row["step"] in heldout_steps
            ]
            checkpoints = []
            for step in heldout_steps:
                values = [row for row in selected if row["step"] == step]
                paired = [sum(
                    row["paired_direction_removal"]
                    for row in values if row["state"] == state
                ) / 2 for state in range(states)]
                frozen = [sum(
                    row["frozen_direction_removal"]
                    for row in values if row["state"] == state
                ) / 2 for state in range(states)]
                checkpoints.append({
                    "step": step,
                    "paired_mean_absolute_removal": sum(paired) / states,
                    "frozen_mean_absolute_removal": sum(frozen) / states,
                    "paired_state_means": paired,
                    "frozen_state_means": frozen,
                })
            generator = random.Random(int.from_bytes(
                hashlib.sha256(f"{variant}:{parameter}".encode()).digest()[:8], "little"
            ))
            paired_bootstrap = []
            frozen_bootstrap = []
            for _ in range(bootstrap):
                sampled = [generator.randrange(states) for _ in range(states)]
                paired_bootstrap.append(sum(
                    row["paired_state_means"][state]
                    for row in checkpoints for state in sampled
                ) / (len(checkpoints) * states))
                frozen_bootstrap.append(sum(
                    row["frozen_state_means"][state]
                    for row in checkpoints for state in sampled
                ) / (len(checkpoints) * states))
            output.append({
                "variant": variant,
                "parameter": parameter,
                "checkpoints": checkpoints,
                "positive_paired_checkpoints": sum(
                    row["paired_mean_absolute_removal"] > 0 for row in checkpoints
                ),
                "positive_frozen_checkpoints": sum(
                    row["frozen_mean_absolute_removal"] > 0 for row in checkpoints
                ),
                "paired_mean_absolute_removal": sum(
                    row["paired_mean_absolute_removal"] for row in checkpoints
                ) / len(checkpoints),
                "frozen_mean_absolute_removal": sum(
                    row["frozen_mean_absolute_removal"] for row in checkpoints
                ) / len(checkpoints),
                "paired_cluster_bootstrap_lower_95": percentile(paired_bootstrap, 0.025),
                "frozen_cluster_bootstrap_lower_95": percentile(frozen_bootstrap, 0.025),
            })
    for row in output:
        row["heldout_gate"] = bool(
            row["positive_paired_checkpoints"] >= 4
            and row["positive_frozen_checkpoints"] >= 4
            and row["paired_cluster_bootstrap_lower_95"] > 0
            and row["frozen_cluster_bootstrap_lower_95"] > 0
        )
    return {"rows": output, "passed": [
        {"variant": row["variant"], "parameter": row["parameter"]}
        for row in output if row["heldout_gate"]
    ]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, default=Path("results/final/long_horizon_bank.json"))
    parser.add_argument("--model", type=Path, default=Path("/data1/tzh/models/Qwen/Qwen3-1.7B"))
    parser.add_argument("--output", type=Path, default=Path("results/final/inductor_config_heldout.json"))
    args = parser.parse_args()
    bank_path = under_root(args.bank, "bank")
    model_path = under_root(args.model, "model")
    output_path = under_root(args.output, "output")
    for key, value in {
        "HF_HOME": "/data1/tzh/cache/huggingface",
        "HF_DATASETS_CACHE": "/data1/tzh/cache/huggingface/datasets",
        "TRANSFORMERS_CACHE": "/data1/tzh/cache/huggingface/transformers",
        "HUGGINGFACE_HUB_CACHE": "/data1/tzh/cache/huggingface/hub",
        "HF_DATASETS_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
    }.items():
        os.environ.setdefault(key, value)

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
    milestones = bank["milestones"]
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
    states, protocol = load_eval_states(tokenizer, 1024, 8, device)
    model = build_model(model_path, device)

    class LossStep(torch.nn.Module):
        def __init__(self, subject: Any) -> None:
            super().__init__(); self.subject = subject
        def forward(self, input_ids: Any, labels: Any) -> Any:
            return self.subject(input_ids=input_ids, labels=labels, use_cache=False, return_dict=False)[0]

    references = {}
    reference_repeat_exact = []
    for milestone in milestones:
        load_milestone(model, milestone, model_path)
        for state, inputs in enumerate(states):
            first = run(model, inputs, None)
            second = run(model, inputs, None)
            references[(int(milestone["step"]), state)] = first
            reference_repeat_exact.append({
                "step": int(milestone["step"]), "state": state,
                "loss_exact": first[0] == second[0],
                "parameters_exact": {
                    name: bool((first[1][name] == second[1][name]).all()) for name in PARAMETERS
                },
            })
            del second

    raw = {}
    audits = {}
    default_deltas = {}
    frozen_directions = {}
    del model
    gc.collect(); torch.cuda.empty_cache()
    for variant, settings in VARIANTS.items():
        # Reconstruct from the immutable source checkpoint for every variant;
        # load_milestone(step=0) intentionally performs no reload.
        model = build_model(model_path, device)
        audit = {"backend_compiles": 0, "graph_sha256": []}
        inductor = lookup_backend("inductor")
        def backend(graph_module: Any, example_inputs: list[Any]) -> Any:
            audit["backend_compiles"] += 1
            audit["graph_sha256"].append(hashlib.sha256(graph_module.code.encode()).hexdigest())
            return inductor(graph_module, example_inputs)
        with config.patch(settings):
            program = torch.compile(LossStep(model), backend=backend, fullgraph=True, dynamic=False)
            warm = run(model, states[0], program)
            del warm
            for milestone in milestones:
                step = int(milestone["step"])
                load_milestone(model, milestone, model_path)
                for state, inputs in enumerate(states):
                    reference = references[(step, state)]
                    for repeat in range(2):
                        candidate = run(model, inputs, program)
                        raw[(variant, step, state, repeat)] = candidate
                        if variant == "default":
                            default_deltas[(step, state, repeat)] = {
                                name: candidate[1][name] - reference[1][name] for name in PARAMETERS
                            }
                            if step == 0 and state == 0 and repeat == 0:
                                frozen_directions = {
                                    name: delta / delta.norm().clamp_min(1e-30)
                                    for name, delta in default_deltas[(step, state, repeat)].items()
                                }
            torch.cuda.synchronize(device)
        audits[variant] = audit
        del program, model
        gc.collect(); torch.cuda.empty_cache()

    rows = []
    for (variant, step, state, repeat), candidate in raw.items():
        reference = references[(step, state)]
        default = default_deltas[(step, state, repeat)]
        for name in PARAMETERS:
            delta = candidate[1][name] - reference[1][name]
            default_l2 = float(default[name].norm())
            local_direction = default[name] / default[name].norm().clamp_min(1e-30)
            paired_projection = float((delta * local_direction).sum())
            frozen_projection = float((delta * frozen_directions[name]).sum())
            default_frozen_projection = float((default[name] * frozen_directions[name]).sum())
            rows.append({
                "variant": variant, "step": step, "state": state, "repeat": repeat,
                "parameter": name, "loss_delta": candidate[0] - reference[0],
                "delta_l2": float(delta.norm()),
                "paired_direction_removal_fraction": 1 - paired_projection / (default_l2 + 1e-30),
                "frozen_direction_removal_fraction": 1 - frozen_projection / (default_frozen_projection + 1e-30),
                "paired_direction_removal": default_l2 - paired_projection,
                "frozen_direction_removal": default_frozen_projection - frozen_projection,
                "default_delta_l2": default_l2,
                "default_frozen_projection": default_frozen_projection,
            })
    summary = summarize(rows, 8)
    output = {
        "schema": "kernel-analyzer-inductor-config-heldout-v1",
        "subject": "Qwen3-1.7B seq1024 fusion/materialization held-out causal replay",
        "bank": str(bank_path), "evaluation": protocol, "variants": VARIANTS,
        "milestones": [int(row["step"]) for row in milestones], "repeats": 2,
        "parameters": list(PARAMETERS), "compile_audits": audits,
        "reference_repeat_exact": reference_repeat_exact, "rows": rows, "summary": summary,
        "full_gradient_tensors_saved": False,
        "boundary": (
            "The frozen direction comes only from step0/state0 default Inductor. Held-out gates exclude "
            "step0. A passing control is an implementation-structure cause, not proof of a single kernel bug."
        ),
    }
    output["result_sha256"] = hashlib.sha256(
        json.dumps(output, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output_path), "passed": summary["passed"]}, sort_keys=True))
    del raw, references, default_deltas
    gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
