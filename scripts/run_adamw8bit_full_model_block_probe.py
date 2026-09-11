#!/usr/bin/env python3
"""Test whether smaller AdamW8bit blocks reduce full-model write error.

This result-aware mechanism follow-up uses new iid gradient histories.  It does
not choose another optimizer modification.  Its purpose is to determine
whether the previously observed block64 reduction was confined to one tensor
or also holds for the full declared model parameter set.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

from kernel_analyzer.training_equivalence import exact_binomial_one_sided_bounds


ROOT = Path(__file__).resolve().parents[1]
MODEL = Path("/data1/tzh/models/state-spaces/mamba-130m-hf")
BANKS = tuple(sorted((ROOT / "results/property/numerical_coverage_v1").glob(
    "mamba_adamw8bit_confirm_bank_*_v1.json"
)))
PRIOR_MECHANISM = ROOT / (
    "results/property/numerical_coverage_v1/torchao_adamw8bit_block_mechanism_v1/summary.json"
)
PRIOR_TRAINING = ROOT / (
    "results/property/numerical_coverage_v1/mamba_adamw8bit_training_confirmation_v1/verification.json"
)
DESIGN_SEED = 20260911
UNIT_COUNT = 16
HISTORY_LENGTH = 8


def load(path: Path) -> Any:
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_new(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def population_index() -> list[dict[str, Any]]:
    rows = []
    for bank_index, path in enumerate(BANKS):
        for row_index, row in enumerate(load(path)["states"]):
            rows.append({"bank_index": bank_index, "row_index": row_index,
                         "token_sha256": row["token_sha256"]})
    return rows


def draws(population_size: int) -> list[list[int]]:
    generator = random.Random(DESIGN_SEED)
    return [[generator.randrange(population_size) for _ in range(HISTORY_LENGTH)]
            for _ in range(UNIT_COUNT)]


def freeze(output: Path) -> None:
    if output.exists():
        raise ValueError("freeze requires a new output directory")
    prior = load(PRIOR_MECHANISM)
    if prior.get("prediction_result") != "CONFIRMED" or prior.get(
        "recommended_training_variant"
    ) != 64:
        raise ValueError("the prior block-size prediction is not confirmed")
    population = population_index()
    sources = [Path(__file__).resolve(), PRIOR_MECHANISM, PRIOR_TRAINING,
               MODEL / "config.json", MODEL / "model.safetensors", *BANKS]
    protocol = {
        "schema": "adamw8bit-full-model-block-probe-v1",
        "status": "FROZEN_BEFORE_NEW_GRADIENT_HISTORIES",
        "data_use": "RESULT_AWARE_MECHANISM_FOLLOWUP_WITH_NEW_IID_HISTORIES",
        "question": (
            "Does block64 reduce full-model actual parameter-write RMS relative to "
            "block256 on more than half of iid optimizer histories?"
        ),
        "frozen_prediction_source": str(PRIOR_MECHANISM),
        "known_training_result_source": str(PRIOR_TRAINING),
        "known_training_result": "BLOCK64_LOSS_IMPROVEMENT_NOT_CONFIRMED",
        "model": str(MODEL),
        "declared_parameter_scope": "ALL_MODEL_PARAMETERS_UPDATED_BY_THE_OPTIMIZER",
        "reference": "torch.optim.AdamW(foreach=False,fused=False)",
        "candidates": {"block64": "torchao.optim.AdamW8bit(block_size=64)",
                       "block256": "torchao.optim.AdamW8bit(block_size=256)"},
        "unit_count": UNIT_COUNT,
        "history_length": HISTORY_LENGTH,
        "design_seed": DESIGN_SEED,
        "draws_are_iid_with_replacement": True,
        "empirical_population_size": len(population),
        "population_index": population,
        "unit_population_indices": draws(len(population)),
        "optimizer": {"lr": 1e-4, "betas": [0.9, 0.999], "eps": 1e-8,
                      "weight_decay": 0.01, "amsgrad": False},
        "primary_endpoint": "INDICATOR_FULL_MODEL_RMS_BLOCK64_LT_BLOCK256",
        "primary_rule": (
            "one-sided exact binomial lower bound for the probability of a lower "
            "full-model RMS; CONFIRMED only when the 95% lower bound exceeds 0.5"
        ),
        "secondary_diagnostic": (
            "first-order inner product between the final-step gradient and each "
            "candidate-minus-reference parameter write"
        ),
        "not_claimed": ["TRAINING_LOSS_CAUSATION", "CROSS_CHECKPOINT_GENERALIZATION"],
        "source_sha256": {str(path.resolve()): sha(path) for path in sources},
    }
    save_new(output / "protocol.json", protocol)


def verify_protocol(output: Path) -> dict[str, Any]:
    protocol = load(output / "protocol.json")
    if protocol.get("schema") != "adamw8bit-full-model-block-probe-v1":
        raise ValueError("unexpected protocol")
    for name, expected in protocol["source_sha256"].items():
        path = Path(name)
        if not path.is_file() or sha(path) != expected:
            raise ValueError("frozen source or input changed: " + name)
    return protocol


def token_ids(protocol: dict[str, Any], population_index_value: int) -> list[int]:
    descriptor = protocol["population_index"][population_index_value]
    row = load(BANKS[descriptor["bank_index"]])["states"][descriptor["row_index"]]
    if row["token_sha256"] != descriptor["token_sha256"]:
        raise ValueError("token identity changed")
    return row["token_ids"]


def make_parameter_copies(model, device: str):
    import torch
    names = [name for name, _ in model.named_parameters()]
    base = [parameter.detach() for _, parameter in model.named_parameters()]
    copies = {
        key: [torch.nn.Parameter(value.detach().clone().to(device)) for value in base]
        for key in ("block64", "block256", "reference")
    }
    settings = dict(lr=1e-4, betas=(0.9, 0.999), eps=1e-8,
                    weight_decay=0.01, amsgrad=False)
    from torchao.optim import AdamW8bit
    optimizers = {
        "block64": AdamW8bit(copies["block64"], block_size=64, **settings),
        "block256": AdamW8bit(copies["block256"], block_size=256, **settings),
        "reference": torch.optim.AdamW(
            copies["reference"], foreach=False, fused=False, **settings
        ),
    }
    return names, base, copies, optimizers


def run_unit(model, protocol: dict[str, Any], indices: list[int], device: str) -> dict:
    import torch
    names, base, copies, optimizers = make_parameter_copies(model, device)
    final_gradients = None
    for population_index_value in indices:
        model.zero_grad(set_to_none=True)
        tokens = torch.tensor([token_ids(protocol, population_index_value)],
                              dtype=torch.long, device=device)
        loss = model(input_ids=tokens, labels=tokens).loss
        loss.backward()
        gradient_values = [parameter.grad for _, parameter in model.named_parameters()]
        if any(value is None for value in gradient_values):
            raise RuntimeError("a declared model parameter has no gradient")
        gradients = [value.detach() for value in gradient_values]
        for key in copies:
            with torch.no_grad():
                for target, source in zip(copies[key], base):
                    target.copy_(source)
            for target, gradient in zip(copies[key], gradients):
                target.grad = gradient.detach().clone()
            optimizers[key].step()
        final_gradients = gradients

    repair_energy = 0.0
    effect_energy = {"block64": 0.0, "block256": 0.0}
    effect_gradient_inner = {"block64": 0.0, "block256": 0.0}
    for index, source in enumerate(base):
        reference_write = copies["reference"][index].detach() - source
        repair_energy += float(torch.sum(reference_write.double().square()).item())
        for key in effect_energy:
            write = copies[key][index].detach() - source
            effect = write - reference_write
            effect_energy[key] += float(torch.sum(effect.double().square()).item())
            effect_gradient_inner[key] += float(torch.sum(
                effect.double() * final_gradients[index].double()
            ).item())
    if not math.isfinite(repair_energy) or repair_energy <= 0:
        raise RuntimeError("invalid full-model repair energy")
    rms = {key: math.sqrt(value / repair_energy) for key, value in effect_energy.items()}
    return {
        "parameter_count": len(names),
        "coordinate_count": sum(value.numel() for value in base),
        "repair_write_energy": repair_energy,
        "effect_energy": effect_energy,
        "full_model_write_rms": rms,
        "effect_final_gradient_inner_product": effect_gradient_inner,
        "block64_lower_rms": rms["block64"] < rms["block256"],
    }


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
    unit_dir = output / "units"
    unit_dir.mkdir(exist_ok=True)
    for index, indices in enumerate(protocol["unit_population_indices"]):
        path = unit_dir / f"unit-{index:03d}.json"
        if path.exists():
            continue
        row = run_unit(model, protocol, indices, device)
        row.update({"schema": "adamw8bit-full-model-block-unit-v1",
                    "unit_id": f"iid-history-{index:03d}",
                    "population_indices": indices})
        save_new(path, row)
        torch.cuda.empty_cache()
        print(json.dumps({"event": "FULL_MODEL_BLOCK_UNIT_COMPLETE", "unit": index + 1,
                          "rms": row["full_model_write_rms"]}), flush=True)

    rows = [load(unit_dir / f"unit-{index:03d}.json") for index in range(UNIT_COUNT)]
    successes = sum(bool(row["block64_lower_rms"]) for row in rows)
    lower, upper = exact_binomial_one_sided_bounds(successes, UNIT_COUNT, alpha=0.05)
    result = {
        "schema": "adamw8bit-full-model-block-probe-result-v1",
        "status": "COMPLETE",
        "independent_unit_count": UNIT_COUNT,
        "block64_lower_rms_count": successes,
        "one_sided_probability_bounds": [lower, upper],
        "prediction_result": "CONFIRMED" if lower > 0.5 else "NOT_CONFIRMED",
        "mean_full_model_write_rms": {
            key: math.fsum(row["full_model_write_rms"][key] for row in rows) / UNIT_COUNT
            for key in ("block64", "block256")
        },
        "relation_to_training_result": (
            "If confirmed, lower full-model one-step RMS still did not establish the "
            "previously tested 1024-step loss improvement; trajectory response remains open."
        ),
        "training_outcome_recomputed_here": False,
    }
    save_new(output / "result.json", result)
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "run"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(Path("/data1/tzh")):
        parser.error("output must be under /data1/tzh")
    (freeze if args.action == "freeze" else run)(args.output, **({} if args.action == "freeze" else {"device": args.device}))


if __name__ == "__main__":
    main()
