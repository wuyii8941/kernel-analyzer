#!/usr/bin/env python3
"""Test whether residual readback removes the aligned AdamW8bit write effect.

The histories were frozen for the earlier compensation probe.  Their RMS
outcomes have already been observed, so this is a predeclared post-hoc source
link, not a new blind case confirmation.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from kernel_analyzer.bias_evidence import binary_prevalence_summary, statewise_aligned_summary
from scripts.run_adamw8bit_error_compensation_probe import compensated_transition
from scripts.run_adamw8bit_population_update import BANKS, MODEL, ROOT, TARGET, transition_sequence

SOURCE = ROOT / "results/property/numerical_coverage_v1/adamw8bit_error_compensation_probe_v1/protocol.json"


def load(path: Path):
    return json.loads(path.read_text())


def save_new(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def freeze(output: Path):
    if output.exists():
        raise ValueError("freeze requires a new output directory")
    source = load(SOURCE)
    protocol = {
        "schema": "adamw8bit-bias-source-link-v1",
        "status": "FROZEN_BEFORE_ALIGNED_SOURCE_LINK_MEASUREMENT",
        "data_use": "POST_HOC_SOURCE_LINK_ON_PREVIOUSLY_FROZEN_HISTORIES",
        "history_source": str(SOURCE.relative_to(ROOT)),
        "unit_population_indices": source["unit_population_indices"],
        "unit_count": source["unit_count"],
        "history_length": source["history_length"],
        "target_parameter": TARGET,
        "conditions": ["DEFAULT_ADAMW8BIT", "RESIDUAL_READBACK", "FP32_ADAMW"],
        "primary_predictions": [
            "default candidate-minus-FP32 write has positive FP32-write inner product in more than half of histories",
            "residual readback reduces the absolute statewise aligned gain in more than half of histories",
        ],
        "primary_rules": {
            "positive_default": "one-sided exact 95% lower prevalence bound exceeds 0.5",
            "aligned_reduction": "one-sided exact 95% lower prevalence bound exceeds 0.5",
        },
        "not_claimed": [
            "new blind case discovery", "nonzero full-vector mean",
            "mean bias as the unique loss mediator", "cross-checkpoint generalization",
        ],
    }
    save_new(output / "protocol.json", protocol)


def tokens(protocol: dict, population_index: int):
    source = load(SOURCE)
    descriptor = source["population_index"][population_index]
    bank = load(BANKS[int(descriptor["bank_index"])])
    return bank["states"][int(descriptor["row_index"])]["token_ids"]


def run(output: Path, device: str):
    import torch
    from transformers import AutoModelForCausalLM

    protocol = load(output / "protocol.json")
    if protocol.get("status") != "FROZEN_BEFORE_ALIGNED_SOURCE_LINK_MEASUREMENT":
        raise ValueError("invalid protocol")
    torch.manual_seed(20260914)
    torch.cuda.manual_seed_all(20260914)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float32, local_files_only=True,
    ).to(device).eval()
    model.config.use_cache = False
    parameters = dict(model.named_parameters())
    target = parameters[TARGET]
    for name, parameter in parameters.items():
        parameter.requires_grad_(name == TARGET)
    base = target.detach().float().cpu().clone()
    unit_dir = output / "units"
    unit_dir.mkdir(exist_ok=True)
    for index, population_indices in enumerate(protocol["unit_population_indices"]):
        path = unit_dir / f"unit-{index:03d}.json"
        if path.exists():
            continue
        gradients = []
        for population_index in population_indices:
            model.zero_grad(set_to_none=True)
            input_ids = torch.tensor([tokens(protocol, population_index)], dtype=torch.long, device=device)
            model(input_ids=input_ids, labels=input_ids).loss.backward()
            gradients.append(target.grad.detach().float().cpu().clone())
        default, reference = transition_sequence(base, gradients, device)
        compensated, _ = compensated_transition(base, gradients, device)
        repair = reference.double().reshape(-1)
        default_effect = (default - reference).double().reshape(-1)
        compensated_effect = (compensated - reference).double().reshape(-1)
        b = float(torch.dot(repair, repair))
        default_inner = float(torch.dot(default_effect, repair))
        compensated_inner = float(torch.dot(compensated_effect, repair))
        row = {
            "schema": "adamw8bit-bias-source-link-unit-v1",
            "unit_id": f"source-link-history-{index:03d}",
            "population_indices": population_indices,
            "repair_energy": b,
            "default_effect_energy": float(torch.dot(default_effect, default_effect)),
            "compensated_effect_energy": float(torch.dot(compensated_effect, compensated_effect)),
            "default_effect_repair_inner_product": default_inner,
            "compensated_effect_repair_inner_product": compensated_inner,
            "default_aligned_gain": default_inner / b,
            "compensated_aligned_gain": compensated_inner / b,
            "absolute_aligned_gain_reduced": abs(compensated_inner) < abs(default_inner),
        }
        if not all(math.isfinite(value) for key, value in row.items() if isinstance(value, float)) or b <= 0:
            raise RuntimeError("invalid sufficient statistics")
        save_new(path, row)
        print(json.dumps({"event": "UNIT_COMPLETE", "unit": index + 1,
                          "aligned_reduced": row["absolute_aligned_gain_reduced"]}), flush=True)
    summarize(output)


def summarize(output: Path):
    protocol = load(output / "protocol.json")
    unit_dir = output / "units"
    rows = [load(unit_dir / f"unit-{index:03d}.json") for index in range(protocol["unit_count"])]
    default_summary = statewise_aligned_summary(
        [row["default_effect_repair_inner_product"] for row in rows],
        [row["repair_energy"] for row in rows],
    )
    compensated_summary = statewise_aligned_summary(
        [row["compensated_effect_repair_inner_product"] for row in rows],
        [row["repair_energy"] for row in rows],
    )
    reduction = binary_prevalence_summary(
        [row["absolute_aligned_gain_reduced"] for row in rows],
        null_probability=.5, alpha=.05,
        estimand="PROBABILITY_RESIDUAL_READBACK_REDUCES_ABSOLUTE_ALIGNED_GAIN",
    )
    result = {
        "schema": "adamw8bit-bias-source-link-result-v1",
        "status": "COMPLETE",
        "default_aligned": default_summary,
        "compensated_aligned": compensated_summary,
        "absolute_aligned_reduction_prevalence": reduction,
        "prediction_results": {
            "positive_default": default_summary["positive_direction_frequency"]["decision"] == "DIRECTION_PREVALENCE_CONFIRMED",
            "aligned_reduction": reduction["decision"] == "PREVALENCE_ABOVE_NULL",
        },
        "claim_boundary": protocol["not_claimed"],
    }
    save_new(output / "result.json", result)
    print(json.dumps({"status": result["status"], "predictions": result["prediction_results"]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "run", "summarize"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT):
        parser.error("output must be inside kernel-analyzer")
    if args.action == "freeze":
        freeze(output)
    elif args.action == "run":
        run(output, args.device)
    else:
        summarize(output)


if __name__ == "__main__":
    main()
