#!/usr/bin/env python3
"""Locate AdamW8bit compensation effects with complementary parameter groups."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = Path("/data1/tzh/models/state-spaces/mamba-130m-hf")
BANKS = tuple(sorted((ROOT / "results/property/numerical_coverage_v1").glob(
    "mamba_adamw8bit_confirm_bank_*_v1.json"
)))
KEY_PARAMETERS = (
    "backbone.embeddings.weight",
    "backbone.layers.23.mixer.out_proj.weight",
)
UNIT_COUNT = 8
HISTORY_LENGTH = 8
DESIGN_SEED = 20260913


def load(path: Path):
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_new(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def population() -> list[dict]:
    result = []
    for bank_index, path in enumerate(BANKS):
        for row_index, row in enumerate(load(path)["states"]):
            result.append({
                "bank_index": bank_index, "row_index": row_index,
                "token_sha256": row["token_sha256"],
            })
    return result


def freeze(output: Path) -> None:
    if output.exists():
        raise ValueError("freeze needs a new output directory")
    entries = population()
    generator = random.Random(DESIGN_SEED)
    draws = [[generator.randrange(len(entries)) for _ in range(HISTORY_LENGTH)]
             for _ in range(UNIT_COUNT)]
    sources = [
        Path(__file__).resolve(), ROOT / "src/kernel_analyzer/compensation_control.py",
        ROOT / "src/kernel_analyzer/adamw8bit_error_compensation.py",
        MODEL / "config.json", MODEL / "model.safetensors", *BANKS,
    ]
    protocol = {
        "schema": "selective-parameter-compensation-probe-v1",
        "status": "FROZEN_BEFORE_GRADIENT_EXECUTION",
        "data_use": "RESULT_AWARE_MECHANISM_LOCALIZATION_WITH_NEW_IID_HISTORIES",
        "question": (
            "Do the predeclared embedding and final mixer output projection account "
            "for the compensation effect across multiple optimizer histories?"
        ),
        "model": str(MODEL), "unit_count": UNIT_COUNT,
        "history_length": HISTORY_LENGTH, "design_seed": DESIGN_SEED,
        "draws_are_iid_with_replacement": True,
        "empirical_population_size": len(entries), "population": entries,
        "unit_population_indices": draws,
        "key_parameters": list(KEY_PARAMETERS),
        "conditions": ["FP32_REFERENCE", "OFF", "ALL", "KEY_ONLY", "REST_ONLY"],
        "optimizer": {"lr": 1e-4, "betas": [0.9, 0.999], "eps": 1e-8,
                      "weight_decay": 0.01},
        "primary_outputs": [
            "fraction of OFF effect energy in key parameters",
            "write RMS for OFF, ALL, KEY_ONLY, REST_ONLY",
            "exact complementary-path checks",
        ],
        "claim_boundary": [
            "FIXED_CHECKPOINT_EMPIRICAL_TOKEN_POPULATION",
            "OPTIMIZER_RESPONSE_WITH_PARAMETERS_RESET_BEFORE_EACH_WRITE",
            "NO_TRAINING_LOSS_OR_CROSS_CHECKPOINT_CLAIM",
        ],
        "source_sha256": {str(path): sha(path) for path in sources},
    }
    save_new(output / "protocol.json", protocol)
    print(json.dumps({"status": protocol["status"], "units": UNIT_COUNT}))


def verify(output: Path) -> dict:
    protocol = load(output / "protocol.json")
    for name, expected in protocol["source_sha256"].items():
        path = Path(name)
        if not path.is_file() or sha(path) != expected:
            raise ValueError("frozen input changed: " + name)
    return protocol


def tokens(protocol: dict, index: int) -> list[int]:
    descriptor = protocol["population"][index]
    row = load(BANKS[descriptor["bank_index"]])["states"][descriptor["row_index"]]
    if row["token_sha256"] != descriptor["token_sha256"]:
        raise ValueError("token identity changed")
    return row["token_ids"]


def run_unit(model, protocol: dict, indices: list[int], device: str) -> dict:
    import torch
    from kernel_analyzer.compensation_control import TensorScalarCompensationControl

    named = list(model.named_parameters())
    names = [name for name, _ in named]
    if not set(protocol["key_parameters"]).issubset(names):
        raise ValueError("predeclared key parameter absent")
    key_mask = [name in protocol["key_parameters"] for name in names]
    base = [parameter.detach() for _, parameter in named]
    modes = ("OFF", "ALL", "KEY_ONLY", "REST_ONLY", "FP32_REFERENCE")
    copies = {
        mode: [torch.nn.Parameter(value.clone().to(device)) for value in base]
        for mode in modes
    }
    settings = dict(lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)

    def controlled(mode: str):
        parameters = copies[mode]
        if mode in {"OFF", "ALL"}:
            return [TensorScalarCompensationControl(
                parameters, compensation_enabled=mode == "ALL", **settings,
            )]
        key = [p for p, selected in zip(parameters, key_mask) if selected]
        rest = [p for p, selected in zip(parameters, key_mask) if not selected]
        return [
            TensorScalarCompensationControl(
                key, compensation_enabled=mode == "KEY_ONLY", **settings,
            ),
            TensorScalarCompensationControl(
                rest, compensation_enabled=mode == "REST_ONLY", **settings,
            ),
        ]

    optimizers = {mode: controlled(mode) for mode in modes if mode != "FP32_REFERENCE"}
    optimizers["FP32_REFERENCE"] = [torch.optim.AdamW(
        copies["FP32_REFERENCE"], foreach=False, fused=False, **settings,
    )]
    for population_index in indices:
        model.zero_grad(set_to_none=True)
        input_ids = torch.tensor(
            [tokens(protocol, population_index)], dtype=torch.long, device=device,
        )
        model(input_ids=input_ids, labels=input_ids).loss.backward()
        gradients = [parameter.grad.detach() for _, parameter in named]
        if any(value is None for value in gradients):
            raise RuntimeError("declared model parameter has no gradient")
        for mode in modes:
            with torch.no_grad():
                for target, source in zip(copies[mode], base):
                    target.copy_(source)
            for target, gradient in zip(copies[mode], gradients):
                target.grad = gradient.clone()
            for optimizer in optimizers[mode]:
                optimizer.step()

    energies = {mode: {"KEY": 0.0, "REST": 0.0} for mode in modes[:-1]}
    repair = {"KEY": 0.0, "REST": 0.0}
    complement = {
        "key_only_key_matches_all": True,
        "key_only_rest_matches_off": True,
        "rest_only_key_matches_off": True,
        "rest_only_rest_matches_all": True,
    }
    for index, selected in enumerate(key_mask):
        group = "KEY" if selected else "REST"
        source = base[index]
        reference_write = copies["FP32_REFERENCE"][index].detach() - source
        repair[group] += float(reference_write.double().square().sum().item())
        writes = {}
        for mode in modes[:-1]:
            writes[mode] = copies[mode][index].detach() - source
            effect = writes[mode] - reference_write
            energies[mode][group] += float(effect.double().square().sum().item())
        if selected:
            complement["key_only_key_matches_all"] &= bool(torch.equal(
                writes["KEY_ONLY"], writes["ALL"],
            ))
            complement["rest_only_key_matches_off"] &= bool(torch.equal(
                writes["REST_ONLY"], writes["OFF"],
            ))
        else:
            complement["key_only_rest_matches_off"] &= bool(torch.equal(
                writes["KEY_ONLY"], writes["OFF"],
            ))
            complement["rest_only_rest_matches_all"] &= bool(torch.equal(
                writes["REST_ONLY"], writes["ALL"],
            ))
    repair_total = repair["KEY"] + repair["REST"]
    totals = {mode: value["KEY"] + value["REST"] for mode, value in energies.items()}
    return {
        "repair_energy": repair, "effect_energy": energies,
        "write_rms": {mode: math.sqrt(value / repair_total) for mode, value in totals.items()},
        "off_effect_energy_fraction_in_key": energies["OFF"]["KEY"] / totals["OFF"],
        "compensation_energy_reduction": {
            mode: totals["OFF"] - totals[mode] for mode in ("ALL", "KEY_ONLY", "REST_ONLY")
        },
        "complement_checks": complement,
        "all_complement_checks": all(complement.values()),
    }


def run(output: Path, device: str, worker: int, workers: int) -> None:
    import torch
    from transformers import AutoModelForCausalLM
    protocol = verify(output)
    torch.manual_seed(314159)
    torch.cuda.manual_seed_all(314159)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float32, local_files_only=True,
    ).to(device).eval()
    model.config.use_cache = False
    for index in range(worker, protocol["unit_count"], workers):
        path = output / "units" / f"unit-{index:03d}.json"
        if path.exists():
            continue
        row = run_unit(model, protocol, protocol["unit_population_indices"][index], device)
        row.update({"schema": "selective-parameter-compensation-unit-v1",
                    "unit": index, "population_indices": protocol["unit_population_indices"][index]})
        save_new(path, row)
        torch.cuda.empty_cache()
        print(json.dumps({"event": "UNIT_COMPLETE", "unit": index,
                          "write_rms": row["write_rms"]}), flush=True)


def summarize(output: Path) -> None:
    protocol = verify(output)
    rows = [load(output / "units" / f"unit-{index:03d}.json")
            for index in range(protocol["unit_count"])]
    modes = ("OFF", "ALL", "KEY_ONLY", "REST_ONLY")
    result = {
        "schema": protocol["schema"], "status": "COMPLETE",
        "unit_count": len(rows),
        "all_complement_checks": all(row["all_complement_checks"] for row in rows),
        "mean_write_rms": {mode: math.fsum(row["write_rms"][mode] for row in rows) / len(rows)
                           for mode in modes},
        "key_fraction_of_off_effect_energy": {
            "values": [row["off_effect_energy_fraction_in_key"] for row in rows],
            "mean": math.fsum(row["off_effect_energy_fraction_in_key"] for row in rows) / len(rows),
            "minimum": min(row["off_effect_energy_fraction_in_key"] for row in rows),
            "maximum": max(row["off_effect_energy_fraction_in_key"] for row in rows),
        },
        "key_only_closer_than_rest_only_count": sum(
            row["write_rms"]["KEY_ONLY"] < row["write_rms"]["REST_ONLY"] for row in rows
        ),
        "all_compensation_best_count": sum(
            row["write_rms"]["ALL"] < min(row["write_rms"]["KEY_ONLY"], row["write_rms"]["REST_ONLY"])
            for row in rows
        ),
        "scope": protocol["claim_boundary"],
    }
    save_new(output / "result.json", result)
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "run", "summarize"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--worker", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT):
        parser.error("output must stay inside kernel-analyzer")
    if args.action == "freeze":
        freeze(output)
    elif args.action == "run":
        run(output, args.device, args.worker, args.workers)
    else:
        summarize(output)


if __name__ == "__main__":
    main()
