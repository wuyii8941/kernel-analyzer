#!/usr/bin/env python3
"""Prospectively test BF16 residual compensation on real gradient histories."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

from kernel_analyzer.adamw8bit_error_compensation import (
    AdamW8bitErrorCompensated,
    optimizer_state_storage_bytes,
)
from kernel_analyzer.population_direction import population_positive_direction_prevalence
from scripts.run_adamw8bit_population_update import (
    BANKS,
    MODEL,
    ROOT,
    TARGET,
    empirical_population,
    load,
    save_new,
    sha,
    tensor_sha,
    tokens_for_population_index,
    transition_sequence,
)


DESIGN_SEED = 20260912
UNIT_COUNT = 16
HISTORY_LENGTH = 8


def draws(population_size: int) -> list[list[int]]:
    generator = random.Random(DESIGN_SEED)
    return [[generator.randrange(population_size) for _ in range(HISTORY_LENGTH)]
            for _ in range(UNIT_COUNT)]


def freeze(output: Path) -> None:
    if output.exists():
        raise ValueError("freeze requires a new output directory")
    population = empirical_population()
    sources = [
        Path(__file__).resolve(),
        ROOT / "src/kernel_analyzer/adamw8bit_error_compensation.py",
        ROOT / "src/kernel_analyzer/population_direction.py",
        ROOT / "src/kernel_analyzer/training_equivalence.py",
        ROOT / "scripts/run_adamw8bit_population_update.py",
        MODEL / "config.json", MODEL / "model.safetensors", *BANKS,
    ]
    protocol = {
        "schema": "adamw8bit-error-compensation-probe-v1",
        "status": "FROZEN_BEFORE_MODIFICATION_MEASUREMENT",
        "data_use": "PROSPECTIVE_MECHANISM_MODIFICATION_CONFIRMATION",
        "question": (
            "Does adding the previous block-quantization residual back into each "
            "AdamW moment recurrence reduce actual parameter-write RMS?"
        ),
        "population_definition": (
            "length-8 histories drawn iid with replacement from the fixed 8192-token-block "
            "population at one Mamba checkpoint"
        ),
        "unit_count": UNIT_COUNT,
        "history_length": HISTORY_LENGTH,
        "design_seed": DESIGN_SEED,
        "population_index": population,
        "unit_population_indices": draws(len(population)),
        "target_parameter": TARGET,
        "conditions": ["FP32_ADAMW", "ADAMW8BIT_BLOCK256",
                       "ADAMW8BIT_BLOCK256_BF16_ERROR_COMPENSATION"],
        "optimizer": {"lr": 1e-3, "betas": [0.9, 0.999], "eps": 1e-8,
                      "weight_decay": 0.01},
        "primary_endpoint": (
            "prevalence of histories where compensated parameter-write RMS to FP32 "
            "is lower than default block256 RMS"
        ),
        "primary_rule": (
            "exact one-sided 95% lower bound for improvement prevalence exceeds one half"
        ),
        "null_improvement_probability": 0.5,
        "alpha": 0.05,
        "secondary_endpoints": [
            "mean parameter-write RMS by condition",
            "optimizer-state tensor bytes after the final step",
        ],
        "not_claimed": ["TRAINING_LOSS_IMPROVEMENT", "PRODUCTION_THROUGHPUT_IMPROVEMENT"],
        "source_sha256": {str(path.resolve()): sha(path) for path in sources},
    }
    save_new(output / "protocol.json", protocol)


def verify_protocol(output: Path) -> dict:
    protocol = load(output / "protocol.json")
    for name, expected in protocol.get("source_sha256", {}).items():
        path = Path(name)
        if not path.is_file() or sha(path) != expected:
            raise ValueError("frozen source or input changed: " + name)
    return protocol


def compensated_transition(base, gradients, device: str):
    import torch

    settings = dict(lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    parameter = torch.nn.Parameter(base.to(device).clone())
    optimizer = AdamW8bitErrorCompensated(
        [parameter], block_size=256, compensation_dtype=torch.bfloat16, **settings
    )
    final = None
    for gradient_cpu in gradients:
        with torch.no_grad():
            parameter.copy_(base.to(device))
        parameter.grad = gradient_cpu.to(device).clone()
        optimizer.step()
        final = (parameter.detach() - base.to(device)).cpu()
    return final, optimizer


def run(output: Path, device: str) -> None:
    import torch
    from transformers import AutoModelForCausalLM

    protocol = verify_protocol(output)
    torch.manual_seed(161803)
    torch.cuda.manual_seed_all(161803)
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

    for unit_index, indices in enumerate(protocol["unit_population_indices"]):
        path = unit_dir / f"unit-{unit_index:03d}.json"
        if path.exists():
            continue
        gradients = []
        for population_index in indices:
            model.zero_grad(set_to_none=True)
            tokens = torch.tensor(
                [tokens_for_population_index(protocol, population_index)],
                dtype=torch.long, device=device,
            )
            model(input_ids=tokens, labels=tokens).loss.backward()
            gradients.append(target.grad.detach().float().cpu().clone())
        default_write, reference_write = transition_sequence(base, gradients, device)
        compensated, compensated_optimizer = compensated_transition(base, gradients, device)
        default_effect = (default_write - reference_write).double().reshape(-1)
        compensated_effect = (compensated - reference_write).double().reshape(-1)
        repair = reference_write.double().reshape(-1)
        repair_energy = float(torch.dot(repair, repair))
        default_rms = math.sqrt(float(torch.dot(default_effect, default_effect)) / repair_energy)
        compensated_rms = math.sqrt(
            float(torch.dot(compensated_effect, compensated_effect)) / repair_energy
        )
        save_new(path, {
            "schema": "adamw8bit-error-compensation-unit-v1",
            "unit_id": f"iid-compensation-history-{unit_index:03d}",
            "population_indices": indices,
            "default_write_rms": default_rms,
            "compensated_write_rms": compensated_rms,
            "default_minus_compensated_rms": default_rms - compensated_rms,
            "compensated_state_storage_bytes": optimizer_state_storage_bytes(
                compensated_optimizer
            ),
            "candidate_write_sha256": tensor_sha(default_write),
            "reference_write_sha256": tensor_sha(reference_write),
            "compensated_write_sha256": tensor_sha(compensated),
        })
        print(json.dumps({"event": "COMPENSATION_UNIT_COMPLETE", "unit": unit_index + 1,
                          "improved": compensated_rms < default_rms}), flush=True)

    rows = [load(unit_dir / f"unit-{index:03d}.json") for index in range(UNIT_COUNT)]
    endpoint = population_positive_direction_prevalence(
        [row["default_minus_compensated_rms"] for row in rows],
        null_positive_probability=protocol["null_improvement_probability"],
        alpha=protocol["alpha"],
    )
    result = {
        "schema": "adamw8bit-error-compensation-probe-result-v1",
        "status": "COMPLETE",
        "primary_endpoint": endpoint,
        "prediction_result": (
            "CONFIRMED" if endpoint["decision"] == "DIRECTION_PREVALENCE_CONFIRMED"
            else "NOT_CONFIRMED"
        ),
        "mean_write_rms": {
            "default_block256": math.fsum(row["default_write_rms"] for row in rows) / len(rows),
            "compensated_block256": math.fsum(row["compensated_write_rms"] for row in rows) / len(rows),
        },
        "state_storage_bytes": sorted({row["compensated_state_storage_bytes"] for row in rows}),
        "training_status": "NOT_MEASURED",
        "claim_boundary": protocol["not_claimed"],
    }
    save_new(output / "result.json", result)
    print(json.dumps({"prediction_result": result["prediction_result"],
                      "primary_endpoint": endpoint}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "run"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(Path("/data1/tzh")):
        parser.error("output must be under /data1/tzh")
    if args.action == "freeze":
        freeze(args.output)
    else:
        run(args.output, args.device)


if __name__ == "__main__":
    main()
