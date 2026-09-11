#!/usr/bin/env python3
"""Freeze and run an iid optimizer-state population study for AdamW8bit.

Each inference unit independently draws a short gradient history with
replacement from a frozen empirical token-block population.  Candidate and
reference receive identical gradients and identical FP32 parameter values;
their optimizer states evolve naturally within the unit and are reset between
units.  The primary endpoint is the population prevalence of final parameter
writes whose RMS difference is at least the declared update margin.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

from kernel_analyzer.training_numerical_analysis import (
    analyze_population_exceedance_artifact,
)


ROOT = Path(__file__).resolve().parents[1]
MODEL = Path("/data1/tzh/models/state-spaces/mamba-130m-hf")
TARGET = "backbone.layers.0.mixer.x_proj.weight"
BANKS = tuple(sorted((ROOT / "results/property/numerical_coverage_v1").glob(
    "mamba_adamw8bit_confirm_bank_*_v1.json"
)))
DESIGN_SEED = 20260910
UNIT_COUNT = 32
HISTORY_LENGTH = 8


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> Any:
    return json.loads(path.read_text())


def save_new(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def empirical_population() -> list[dict[str, Any]]:
    rows = []
    for bank_index, path in enumerate(BANKS):
        for row_index, row in enumerate(load(path)["states"]):
            rows.append({
                "bank_index": bank_index,
                "row_index": row_index,
                "state_id": str(row["state_id"]),
                "token_sha256": row["token_sha256"],
            })
    if not rows:
        raise ValueError("no frozen token blocks are available")
    return rows


def independent_draws(population_size: int) -> list[list[int]]:
    if population_size < 1:
        raise ValueError("population_size must be positive")
    generator = random.Random(DESIGN_SEED)
    return [
        [generator.randrange(population_size) for _ in range(HISTORY_LENGTH)]
        for _ in range(UNIT_COUNT)
    ]


def freeze(output: Path) -> None:
    if output.exists():
        raise ValueError("freeze requires a new output directory")
    if len(BANKS) != 8:
        raise ValueError("the declared empirical population requires eight frozen banks")
    population = empirical_population()
    sources = [
        Path(__file__).resolve(),
        ROOT / "src/kernel_analyzer/training_numerical_analysis.py",
        ROOT / "src/kernel_analyzer/training_equivalence.py",
        MODEL / "config.json",
        MODEL / "model.safetensors",
        *BANKS,
    ]
    protocol = {
        "schema": "adamw8bit-population-update-v1",
        "status": "FROZEN_BEFORE_UPDATE_MEASUREMENT",
        "case_id": "mamba_x_proj_adamw8bit_iid_optimizer_history",
        "contrast_id": "ADAMW8BIT_BLOCK256_MINUS_FP32_ADAMW",
        "claim_scope": "DECLARED_STATE_POPULATION_UPDATE",
        "population_estimand": "STATEWISE_RMS_EXCEEDANCE_PROBABILITY",
        "population_definition": (
            "length-8 optimizer gradient histories drawn independently with replacement "
            "from the fixed 8192-token-block empirical population at one Mamba checkpoint"
        ),
        "independent_unit": "ONE_INDEPENDENTLY_DRAWN_LENGTH_8_GRADIENT_HISTORY",
        "draws_are_iid_with_replacement": True,
        "unit_count": UNIT_COUNT,
        "history_length": HISTORY_LENGTH,
        "design_seed": DESIGN_SEED,
        "empirical_population_size": len(population),
        "population_index": population,
        "unit_population_indices": independent_draws(len(population)),
        "model": str(MODEL),
        "target_parameter": TARGET,
        "candidate": {"implementation": "torchao.optim.AdamW8bit", "block_size": 256},
        "reference": {"implementation": "torch.optim.AdamW", "foreach": False, "fused": False},
        "optimizer": {"lr": 1e-3, "betas": [0.9, 0.999], "eps": 1e-8,
                      "weight_decay": 0.01, "amsgrad": False},
        "primary_stage": "PARAMETER_WRITE",
        "statewise_rms_margin": 0.01,
        "maximum_exceedance_probability": 0.05,
        "repair_energy_floor": 1e-30,
        "alpha": 0.05,
        "primary_rule": (
            "exact one-sided Clopper-Pearson bounds for the prevalence of independent "
            "units with statewise parameter-write RMS at or above 1%"
        ),
        "not_claimed": [
            "MEAN_ENERGY_Q_POPULATION_EQUIVALENCE",
            "RECURSIVE_TRAINING_STATE_POPULATION",
            "CROSS_CHECKPOINT_GENERALIZATION",
        ],
        "data_use": "PROSPECTIVE_POPULATION_CONFIRMATION_AFTER_CASE_DISCOVERY",
        "source_sha256": {str(path.resolve()): sha(path) for path in sources},
    }
    save_new(output / "protocol.json", protocol)


def verify_protocol(output: Path) -> dict[str, Any]:
    protocol = load(output / "protocol.json")
    if protocol.get("schema") != "adamw8bit-population-update-v1":
        raise ValueError("unexpected protocol schema")
    for name, expected in protocol["source_sha256"].items():
        path = Path(name)
        if not path.is_file() or sha(path) != expected:
            raise ValueError("frozen source or input changed: " + name)
    if protocol.get("draws_are_iid_with_replacement") is not True:
        raise ValueError("iid with-replacement sampling was not frozen")
    return protocol


def tensor_sha(value) -> str:
    return hashlib.sha256(
        value.detach().contiguous().cpu().numpy().tobytes()
    ).hexdigest()


def tokens_for_population_index(protocol: dict[str, Any], index: int) -> list[int]:
    descriptor = protocol["population_index"][index]
    bank = load(BANKS[int(descriptor["bank_index"])])
    row = bank["states"][int(descriptor["row_index"])]
    if row["token_sha256"] != descriptor["token_sha256"]:
        raise ValueError("token population identity changed")
    return row["token_ids"]


def transition_sequence(base, gradients, device: str):
    import torch
    from torchao.optim import AdamW8bit

    settings = dict(lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                    weight_decay=0.01, amsgrad=False)
    candidate_parameter = torch.nn.Parameter(base.to(device).clone())
    reference_parameter = torch.nn.Parameter(base.to(device).clone())
    candidate = AdamW8bit([candidate_parameter], block_size=256, **settings)
    reference = torch.optim.AdamW(
        [reference_parameter], foreach=False, fused=False, **settings
    )
    final = None
    for gradient_cpu in gradients:
        with torch.no_grad():
            candidate_parameter.copy_(base.to(device))
            reference_parameter.copy_(base.to(device))
        gradient = gradient_cpu.to(device)
        candidate_parameter.grad = gradient.clone()
        reference_parameter.grad = gradient.clone()
        candidate.step(); reference.step()
        final = (
            (candidate_parameter.detach() - base.to(device)).cpu(),
            (reference_parameter.detach() - base.to(device)).cpu(),
        )
    return final


def run(output: Path, device: str) -> None:
    import torch
    from transformers import AutoModelForCausalLM

    protocol = verify_protocol(output)
    torch.manual_seed(271828)
    torch.cuda.manual_seed_all(271828)
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
        gradient_digests = []
        for population_index in population_indices:
            model.zero_grad(set_to_none=True)
            tokens = torch.tensor(
                [tokens_for_population_index(protocol, population_index)],
                dtype=torch.long, device=device,
            )
            loss = model(input_ids=tokens, labels=tokens).loss
            loss.backward()
            gradient = target.grad.detach().float().cpu().clone()
            gradients.append(gradient)
            gradient_digests.append(tensor_sha(gradient))
        first = transition_sequence(base, gradients, device)
        second = transition_sequence(base, gradients, device)
        if tensor_sha(first[0]) != tensor_sha(second[0]) or tensor_sha(first[1]) != tensor_sha(second[1]):
            raise RuntimeError(f"optimizer transition was not repeatable for unit {unit_index}")
        effect = (first[0] - first[1]).double().reshape(-1)
        repair = first[1].double().reshape(-1)
        x = float(torch.dot(effect, effect))
        b = float(torch.dot(repair, repair))
        a = float(torch.dot(effect, repair))
        if not all(math.isfinite(value) for value in (x, b, a)) or b <= 1e-30:
            raise RuntimeError(f"invalid sufficient statistics for unit {unit_index}")
        save_new(path, {
            "schema": "adamw8bit-population-unit-v1",
            "unit_id": f"iid-history-{unit_index:03d}",
            "population_indices": population_indices,
            "gradient_sha256": gradient_digests,
            "effect_energy": x,
            "repair_energy": b,
            "effect_repair_inner_product": a,
            "statewise_rms": math.sqrt(x / b),
            "candidate_write_sha256": tensor_sha(first[0]),
            "reference_write_sha256": tensor_sha(first[1]),
            "repeatability": "EXACT",
        })
        print(json.dumps({"event": "POPULATION_UNIT_COMPLETE", "unit": unit_index + 1,
                          "statewise_rms": math.sqrt(x / b)}), flush=True)

    rows = [load(unit_dir / f"unit-{index:03d}.json") for index in range(UNIT_COUNT)]
    raw = {
        "schema": "adamw8bit-population-raw-v1",
        "status": "COMPLETE",
        "case_id": protocol["case_id"],
        "contrast_id": protocol["contrast_id"],
        "runtime_boundary": {
            "kind": "OPTIMIZER_PARAMETER_AND_MOMENT_TRANSITION",
            "candidate_backend": "TORCH_COMPILE_GENERATED_TRITON",
        },
        "parameter_write_protocol": {
            "version": "optimizer-implementation-readback-v1",
            "measurement": "parameter_after_step_minus_parameter_before_step",
        },
        "determinism": {"all_exact": True},
        "state_ids": [row["unit_id"] for row in rows],
        "inference_unit_ids": [row["unit_id"] for row in rows],
        "original_coordinate_statistics": {"PARAMETER_WRITE": [{
            "effect_energy": row["effect_energy"],
            "repair_energy": row["repair_energy"],
            "effect_repair_inner_product": row["effect_repair_inner_product"],
        } for row in rows]},
    }
    save_new(output / "raw.json", raw)
    analysis = analyze_population_exceedance_artifact(raw, protocol)
    analysis["descriptive_only"] = {
        "mean_energy_rms_ratio": math.sqrt(
            math.fsum(row["effect_energy"] for row in rows)
            / math.fsum(row["repair_energy"] for row in rows)
        ),
        "aligned_ratio_of_sums": (
            math.fsum(row["effect_repair_inner_product"] for row in rows)
            / math.fsum(row["repair_energy"] for row in rows)
        ),
        "not_a_mean_energy_population_certificate": True,
    }
    save_new(output / "analysis.json", analysis)
    print(json.dumps({"decision": analysis["equivalence_decision"],
                      "endpoint": analysis["mandatory_endpoints"][0]}))


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
