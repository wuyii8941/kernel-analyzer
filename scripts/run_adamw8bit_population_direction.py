#!/usr/bin/env python3
"""Confirm an AdamW8bit repair-aligned direction on new iid histories.

This is a prospective follow-up after the separate RMS-prevalence result.  It
uses new independently drawn optimizer histories and one predeclared primary
endpoint: whether the candidate-minus-reference parameter write has positive
inner product with the FP32 AdamW write in more than half of histories.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

from kernel_analyzer.population_direction import (
    population_positive_direction_prevalence,
)
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


DESIGN_SEED = 20260911
UNIT_COUNT = 32
HISTORY_LENGTH = 8


def independent_draws(population_size: int) -> list[list[int]]:
    generator = random.Random(DESIGN_SEED)
    return [
        [generator.randrange(population_size) for _ in range(HISTORY_LENGTH)]
        for _ in range(UNIT_COUNT)
    ]


def freeze(output: Path) -> None:
    if output.exists():
        raise ValueError("freeze requires a new output directory")
    population = empirical_population()
    sources = [
        Path(__file__).resolve(),
        ROOT / "scripts/run_adamw8bit_population_update.py",
        ROOT / "src/kernel_analyzer/population_direction.py",
        ROOT / "src/kernel_analyzer/training_equivalence.py",
        MODEL / "config.json",
        MODEL / "model.safetensors",
        *BANKS,
    ]
    protocol = {
        "schema": "adamw8bit-population-direction-v1",
        "status": "FROZEN_BEFORE_DIRECTION_MEASUREMENT",
        "case_id": "mamba_x_proj_adamw8bit_iid_optimizer_history_direction",
        "contrast_id": "ADAMW8BIT_BLOCK256_MINUS_FP32_ADAMW",
        "population_definition": (
            "length-8 optimizer gradient histories drawn independently with replacement "
            "from the fixed 8192-token-block empirical population at one Mamba checkpoint"
        ),
        "independent_unit": "ONE_INDEPENDENTLY_DRAWN_LENGTH_8_GRADIENT_HISTORY",
        "draws_are_iid_with_replacement": True,
        "unit_count": UNIT_COUNT,
        "history_length": HISTORY_LENGTH,
        "design_seed": DESIGN_SEED,
        "population_index": population,
        "unit_population_indices": independent_draws(len(population)),
        "model": str(MODEL),
        "target_parameter": TARGET,
        "candidate": {"implementation": "torchao.optim.AdamW8bit", "block_size": 256},
        "reference": {"implementation": "torch.optim.AdamW", "foreach": False, "fused": False},
        "optimizer": {"lr": 1e-3, "betas": [0.9, 0.999], "eps": 1e-8,
                      "weight_decay": 0.01, "amsgrad": False},
        "primary_endpoint": (
            "sign of inner product between candidate-minus-reference parameter write "
            "and FP32 AdamW parameter write"
        ),
        "positive_direction_interpretation": (
            "candidate adds a component in the direction of the FP32 AdamW write"
        ),
        "null_positive_probability": 0.5,
        "alpha": 0.05,
        "primary_rule": (
            "the exact one-sided 95% lower bound for positive-direction prevalence "
            "must exceed one half"
        ),
        "data_use": "PROSPECTIVE_DIRECTION_POPULATION_CONFIRMATION_AFTER_RMS_RESULT",
        "not_claimed": [
            "VECTOR_MEAN_NONZERO",
            "MEAN_DIRECTIONAL_MAGNITUDE_BOUND",
            "RECURSIVE_TRAINING_CAUSATION",
            "CROSS_CHECKPOINT_GENERALIZATION",
        ],
        "source_sha256": {str(path.resolve()): sha(path) for path in sources},
    }
    save_new(output / "protocol.json", protocol)


def verify_protocol(output: Path) -> dict[str, Any]:
    protocol = load(output / "protocol.json")
    if protocol.get("schema") != "adamw8bit-population-direction-v1":
        raise ValueError("unexpected protocol schema")
    for name, expected in protocol["source_sha256"].items():
        path = Path(name)
        if not path.is_file() or sha(path) != expected:
            raise ValueError("frozen source or input changed: " + name)
    if protocol.get("draws_are_iid_with_replacement") is not True:
        raise ValueError("iid with-replacement sampling was not frozen")
    return protocol


def run(output: Path, device: str) -> None:
    import torch
    from transformers import AutoModelForCausalLM

    protocol = verify_protocol(output)
    torch.manual_seed(314159)
    torch.cuda.manual_seed_all(314159)
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
    for unit_index, population_indices in enumerate(protocol["unit_population_indices"]):
        path = unit_dir / f"unit-{unit_index:03d}.json"
        if path.exists():
            continue
        gradients = []
        for population_index in population_indices:
            model.zero_grad(set_to_none=True)
            tokens = torch.tensor(
                [tokens_for_population_index(protocol, population_index)],
                dtype=torch.long, device=device,
            )
            loss = model(input_ids=tokens, labels=tokens).loss
            loss.backward()
            gradients.append(target.grad.detach().float().cpu().clone())
        first = transition_sequence(base, gradients, device)
        second = transition_sequence(base, gradients, device)
        if tensor_sha(first[0]) != tensor_sha(second[0]) or tensor_sha(first[1]) != tensor_sha(second[1]):
            raise RuntimeError(f"optimizer transition was not repeatable for unit {unit_index}")
        effect = (first[0] - first[1]).double().reshape(-1)
        reference = first[1].double().reshape(-1)
        inner = float(torch.dot(effect, reference))
        x = float(torch.dot(effect, effect))
        b = float(torch.dot(reference, reference))
        if not all(math.isfinite(value) for value in (inner, x, b)) or b <= 0.0:
            raise RuntimeError(f"invalid sufficient statistics for unit {unit_index}")
        save_new(path, {
            "schema": "adamw8bit-population-direction-unit-v1",
            "unit_id": f"iid-direction-history-{unit_index:03d}",
            "population_indices": population_indices,
            "effect_repair_inner_product": inner,
            "effect_energy": x,
            "repair_energy": b,
            "aligned_ratio": inner / b,
            "statewise_rms": math.sqrt(x / b),
            "repeatability": "EXACT",
        })
        print(json.dumps({"event": "DIRECTION_UNIT_COMPLETE", "unit": unit_index + 1,
                          "positive": inner > 0.0}), flush=True)

    rows = [load(unit_dir / f"unit-{index:03d}.json") for index in range(UNIT_COUNT)]
    endpoint = population_positive_direction_prevalence(
        [row["effect_repair_inner_product"] for row in rows],
        null_positive_probability=protocol["null_positive_probability"],
        alpha=protocol["alpha"],
    )
    result = {
        "schema": "adamw8bit-population-direction-result-v1",
        "status": "COMPLETE",
        "case_id": protocol["case_id"],
        "primary_endpoint": endpoint,
        "descriptive": {
            "aligned_ratio_of_sums": (
                math.fsum(row["effect_repair_inner_product"] for row in rows)
                / math.fsum(row["repair_energy"] for row in rows)
            ),
            "mean_statewise_rms": math.fsum(row["statewise_rms"] for row in rows) / len(rows),
        },
        "claim_scope": protocol["population_definition"],
        "data_use": protocol["data_use"],
        "not_claimed": protocol["not_claimed"],
    }
    save_new(output / "result.json", result)
    print(json.dumps({"decision": endpoint["decision"],
                      "positive_count": endpoint["positive_count"]}))


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
