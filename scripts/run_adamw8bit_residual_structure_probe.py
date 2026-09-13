#!/usr/bin/env python3
"""Test coordinate and time structure of first-moment residual feedback."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
from pathlib import Path

from scripts.run_adamw8bit_population_update import (
    MODEL, ROOT, TARGET, load, save_new, sha, tensor_sha,
    tokens_for_population_index,
)

SOURCE_PROTOCOL = ROOT / "results/property/result_analysis_v2/selective_parameter_compensation/protocol.json"


def freeze(output: Path) -> None:
    if output.exists():
        raise ValueError("freeze requires a new directory")
    from kernel_analyzer.structured_residual_control import ExternalFirstMomentResidualControl
    sources = [
        Path(__file__).resolve(), Path(inspect.getfile(ExternalFirstMomentResidualControl)),
        ROOT / "src/kernel_analyzer/compensation_control.py",
        ROOT / "src/kernel_analyzer/adamw8bit_error_compensation.py",
        ROOT / "scripts/run_adamw8bit_population_update.py",
        SOURCE_PROTOCOL, MODEL / "config.json", MODEL / "model.safetensors",
    ]
    source = load(SOURCE_PROTOCOL)
    protocol = {
        "schema": "adamw8bit-first-moment-residual-structure-v1",
        "status": "FROZEN_BEFORE_STRUCTURE_EXECUTION",
        "data_use": "RESULT_AWARE_MECHANISM_ELIMINATION_ON_FIXED_REAL_GRADIENT_HISTORIES",
        "target_parameter": TARGET,
        "unit_population_indices": source["unit_population_indices"],
        "population": source["population"],
        "history_length": source["history_length"],
        "conditions": ["OFF", "CORRECT", "COORDINATE_ROLL", "TIME_REVERSE", "FP32_REFERENCE"],
        "coordinate_transform": "cyclic roll by one within every 256-coordinate block",
        "time_transform": "keep initial zero read, reverse the seven observed first-moment residual reads",
        "controlled_scope": "FIRST_MOMENT_RESIDUAL_ONLY; SECOND_MOMENT_COMPENSATION_REMAINS_CORRECT",
        "primary_outputs": [
            "write RMS relative to FP32 by condition",
            "correct compensation improvement over coordinate and time rearrangements",
            "exact first-moment residual read energy preservation",
        ],
        "not_claimed": ["TRAINING_LOSS_EFFECT", "ONLINE_DEPLOYABILITY_OF_TIME_REVERSE", "MEAN_BIAS_CAUSALITY"],
        "optimizer": {"lr": 1e-3, "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.01},
        "source_sha256": {str(path.resolve()): sha(path) for path in sources},
    }
    save_new(output / "protocol.json", protocol)


def checked(output: Path) -> dict:
    protocol = load(output / "protocol.json")
    for name, expected in protocol["source_sha256"].items():
        if sha(Path(name)) != expected:
            raise ValueError("frozen dependency changed: " + name)
    return protocol


def population_tokens(protocol: dict, index: int) -> list[int]:
    # Reuse the source protocol's immutable bank descriptor and resolver.
    return tokens_for_population_index({"population_index": protocol["population"]}, index)


def run_sequence(base, gradients, condition: str, residual_reads, device: str):
    import torch
    from kernel_analyzer.compensation_control import TensorScalarCompensationControl
    from kernel_analyzer.structured_residual_control import ExternalFirstMomentResidualControl

    settings = dict(lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    parameter = torch.nn.Parameter(base.to(device).clone())
    if condition == "FP32_REFERENCE":
        optimizer = torch.optim.AdamW([parameter], foreach=False, fused=False, **settings)
    elif condition in {"OFF", "CORRECT_BASELINE"}:
        optimizer = TensorScalarCompensationControl(
            [parameter], compensation_enabled=condition == "CORRECT_BASELINE", **settings,
        )
    else:
        optimizer = ExternalFirstMomentResidualControl(
            [parameter], first_residual_reads=residual_reads,
            arrangement=condition, **settings,
        )
    final = None
    observed_reads = []
    for gradient in gradients:
        if condition == "CORRECT_BASELINE":
            state = optimizer.state[parameter]
            observed_reads.append(
                torch.zeros_like(base) if not state else state["exp_avg_compensation"].detach().clone()
            )
        with torch.no_grad():
            parameter.copy_(base.to(device))
        parameter.grad = gradient.to(device).clone()
        optimizer.step()
        final = (parameter.detach() - base.to(device)).cpu()
    return final, observed_reads


def run(output: Path, device: str) -> None:
    import torch
    from transformers import AutoModelForCausalLM
    protocol = checked(output)
    torch.manual_seed(271828); torch.cuda.manual_seed_all(271828)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32, local_files_only=True).to(device).eval()
    model.config.use_cache = False
    parameters = dict(model.named_parameters()); target = parameters[TARGET]
    for name, parameter in parameters.items():
        parameter.requires_grad_(name == TARGET)
    base = target.detach().float().cpu().clone()
    for unit, indices in enumerate(protocol["unit_population_indices"]):
        path = output / "units" / f"unit-{unit:03d}.json"
        if path.exists():
            continue
        gradients = []
        for index in indices:
            model.zero_grad(set_to_none=True)
            ids = torch.tensor([population_tokens(protocol, index)], dtype=torch.long, device=device)
            model(input_ids=ids, labels=ids).loss.backward()
            gradients.append(target.grad.detach().float().cpu().clone())
        _, reads = run_sequence(base, gradients, "CORRECT_BASELINE", None, device)
        writes = {}
        for condition in protocol["conditions"]:
            writes[condition], _ = run_sequence(base, gradients, condition, reads, device)
        reference = writes["FP32_REFERENCE"].double().reshape(-1)
        denominator = float(torch.dot(reference, reference))
        rms = {}
        for condition, write in writes.items():
            effect = write.double().reshape(-1) - reference
            rms[condition] = math.sqrt(float(torch.dot(effect, effect)) / denominator)
        energies = [float(value.double().square().sum()) for value in reads]
        from kernel_analyzer.structured_residual_control import roll_within_blocks
        rolled_values = [roll_within_blocks(value, 256) for value in reads]
        rolled = [float(value.double().square().sum()) for value in rolled_values]
        reverse = [energies[0], *reversed(energies[1:])]
        save_new(path, {
            "schema": "adamw8bit-first-moment-residual-structure-unit-v1",
            "unit": unit, "population_indices": indices, "write_rms": rms,
            "residual_read_energy": energies,
            "coordinate_roll_energy": rolled,
            "time_reverse_energy": reverse,
            "coordinate_multiset_exact": all(torch.equal(
                roll_within_blocks(value, 256).roll(-1, dims=-1), value
            ) for value in reads),
            "time_multiset_energy_exact": sorted(energies[1:]) == sorted(reverse[1:]),
            "write_sha256": {name: tensor_sha(value) for name, value in writes.items()},
        })
        print(json.dumps({"event": "UNIT_COMPLETE", "unit": unit, "write_rms": rms}), flush=True)


def summarize(output: Path) -> None:
    protocol = checked(output)
    rows = [load(output / "units" / f"unit-{i:03d}.json") for i in range(len(protocol["unit_population_indices"]))]
    conditions = protocol["conditions"]
    mean = {name: math.fsum(row["write_rms"][name] for row in rows) / len(rows) for name in conditions}
    result = {
        "schema": protocol["schema"], "status": "COMPLETE", "unit_count": len(rows),
        "mean_write_rms": mean,
        "correct_better_than_coordinate_count": sum(row["write_rms"]["CORRECT"] < row["write_rms"]["COORDINATE_ROLL"] for row in rows),
        "correct_better_than_time_count": sum(row["write_rms"]["CORRECT"] < row["write_rms"]["TIME_REVERSE"] for row in rows),
        "all_energy_checks_exact": all(row["coordinate_multiset_exact"] and row["time_multiset_energy_exact"] for row in rows),
        "scope": protocol["not_claimed"],
    }
    save_new(output / "result.json", result)
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "run", "summarize"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:2")
    args = parser.parse_args(); output = args.output.resolve()
    if not output.is_relative_to(ROOT):
        parser.error("output must stay inside kernel-analyzer")
    if args.action == "freeze": freeze(output)
    elif args.action == "run": run(output, args.device)
    else: summarize(output)


if __name__ == "__main__":
    main()
