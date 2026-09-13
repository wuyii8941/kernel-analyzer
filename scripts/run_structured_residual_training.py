#!/usr/bin/env python3
"""Paired training attribution for parameter, coordinate, and temporal residual structure."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import time
from pathlib import Path

from scripts import run_adamw8bit_error_compensation_training as original

ROOT = Path(__file__).resolve().parents[1]
HISTORICAL = ROOT / "results/property/result_analysis_v1/same_path_training"
KEY_PARAMETERS = (
    "backbone.embeddings.weight",
    "backbone.layers.23.mixer.out_proj.weight",
)
CONDITIONS = ("KEY_ONLY", "REST_ONLY", "COORDINATE_ROLL", "ONE_EXTRA_STEP_LAG")


def load(path: Path):
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_new(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


class OptimizerBundle:
    def __init__(self, optimizers):
        self.optimizers = list(optimizers)

    def zero_grad(self, set_to_none=True):
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def step(self):
        for optimizer in self.optimizers:
            optimizer.step()


def freeze(output: Path) -> None:
    if output.exists():
        raise ValueError("freeze requires a new directory")
    from kernel_analyzer.compensation_control import TensorScalarCompensationControl
    from kernel_analyzer.online_residual_control import OnlineFirstMomentResidualControl
    old = load(HISTORICAL / "protocol.json")
    sources = [
        Path(__file__).resolve(), Path(inspect.getfile(TensorScalarCompensationControl)),
        Path(inspect.getfile(OnlineFirstMomentResidualControl)),
        ROOT / "src/kernel_analyzer/adamw8bit_error_compensation.py",
        ROOT / "scripts/run_adamw8bit_error_compensation_training.py",
        HISTORICAL / "protocol.json", HISTORICAL / "summary.json",
        ROOT / "results/property/result_analysis_v3/residual_structure/result.json",
        ROOT / "results/property/result_analysis_v3/residual_structure/verification.json",
        original.MODEL / "config.json", original.MODEL / "model.safetensors",
        original.EVAL_BANK,
    ]
    sources.extend(Path(path) for path in old["train_banks"])
    sources.extend(HISTORICAL / "runs" / f"stream_{stream:02d}_{mode}.json"
                   for stream in range(8) for mode in ("OFF", "ON"))
    protocol = {
        "schema": "structured-residual-training-attribution-v1",
        "status": "FROZEN_BEFORE_NEW_CONDITION_TRAINING",
        "data_use": "RESULT_AWARE_SAME_STREAM_MECHANISM_ATTRIBUTION_NOT_EXTERNAL_CONFIRMATION",
        "stream_count": 8, "steps": old["steps"],
        "train_banks": old["train_banks"], "evaluation_steps": old["evaluation_steps"],
        "evaluation_states": old["evaluation_states"], "optimizer": old["optimizer"],
        "historical_conditions": ["OFF", "ON"], "new_conditions": list(CONDITIONS),
        "key_parameters": list(KEY_PARAMETERS),
        "condition_definitions": {
            "KEY_ONLY": "correct compensation only for the two predeclared key parameters",
            "REST_ONLY": "correct compensation for every parameter except the two key parameters",
            "COORDINATE_ROLL": "first-moment residual rolled by one coordinate inside each 256-coordinate block; second moment correct",
            "ONE_EXTRA_STEP_LAG": "first-moment residual read with one additional causal step lag; second moment correct",
        },
        "family_alpha": 0.05, "contrast_count": 4, "simultaneous_interval_level": 0.9875,
        "loss_margin": 0.01,
        "frozen_contrasts": [
            "KEY_ONLY_MINUS_ON inside [-0.01,+0.01]",
            "OFF_MINUS_REST_ONLY inside [-0.01,+0.01]",
            "COORDINATE_ROLL_MINUS_ON lower bound above +0.01",
            "ONE_EXTRA_STEP_LAG_MINUS_ON lower bound above +0.01",
        ],
        "tracked_outputs": [
            "evaluation loss at frozen checkpoints", "final key-parameter norms",
            "final non-key parameter norm", "final optimizer first/second state norms",
            "runtime and peak allocated memory",
        ],
        "not_claimed": [
            "UNSEEN_DATA_CONFIRMATION", "CROSS_MODEL_GENERALIZATION",
            "TIME_REVERSE_ONLINE_DEPLOYABILITY", "MEAN_BIAS_AS_UNIQUE_CAUSE",
        ],
        "source_sha256": {str(path.resolve()): sha(path) for path in sources},
    }
    save_new(output / "protocol.json", protocol)
    print(json.dumps({"status": protocol["status"], "new_runs": 8 * len(CONDITIONS)}))


def checked(output: Path) -> dict:
    protocol = load(output / "protocol.json")
    for name, expected in protocol["source_sha256"].items():
        if not Path(name).is_file() or sha(Path(name)) != expected:
            raise ValueError("frozen dependency changed: " + name)
    return protocol


def make_optimizer(condition: str, model, settings):
    from kernel_analyzer.compensation_control import TensorScalarCompensationControl
    from kernel_analyzer.online_residual_control import OnlineFirstMomentResidualControl
    named = list(model.named_parameters())
    if not set(KEY_PARAMETERS).issubset(dict(named)):
        raise ValueError("predeclared key parameters are absent")
    if condition in {"KEY_ONLY", "REST_ONLY"}:
        key = [parameter for name, parameter in named if name in KEY_PARAMETERS]
        rest = [parameter for name, parameter in named if name not in KEY_PARAMETERS]
        key_enabled = condition == "KEY_ONLY"
        return OptimizerBundle([
            TensorScalarCompensationControl(key, compensation_enabled=key_enabled,
                                            block_size=256, **settings),
            TensorScalarCompensationControl(rest, compensation_enabled=not key_enabled,
                                            block_size=256, **settings),
        ])
    return OnlineFirstMomentResidualControl(
        model.parameters(), arrangement=condition, block_size=256, **settings,
    )


def state_norms(optimizer) -> dict:
    from torchao.optim.subclass_8bit import OptimState8bit
    optimizers = optimizer.optimizers if isinstance(optimizer, OptimizerBundle) else [optimizer]
    energies = {"first": 0.0, "second": 0.0, "first_compensation": 0.0,
                "second_compensation": 0.0}
    for item in optimizers:
        for state in item.state.values():
            for key, output in (("exp_avg", "first"), ("exp_avg_sq", "second")):
                if key not in state:
                    continue
                value = state[key]
                value = value.dequantize() if isinstance(value, OptimState8bit) else value.float()
                energies[output] += float(value.double().square().sum())
                compensation = state.get(key + "_compensation")
                if compensation is not None:
                    energies[output + "_compensation"] += float(compensation.double().square().sum())
    return {key: math.sqrt(value) for key, value in energies.items()}


def run_condition(protocol: dict, stream: int, condition: str, device: str) -> dict:
    import torch
    from transformers import AutoModelForCausalLM
    seed = 193000 + stream
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    model = AutoModelForCausalLM.from_pretrained(
        original.MODEL, dtype=torch.float32, local_files_only=True,
    ).to(device)
    model.config.use_cache = False
    settings = dict(lr=protocol["optimizer"]["lr"], betas=tuple(protocol["optimizer"]["betas"]),
                    eps=protocol["optimizer"]["eps"], weight_decay=protocol["optimizer"]["weight_decay"])
    optimizer = make_optimizer(condition, model, settings)
    train_rows = load(Path(protocol["train_banks"][stream]))["states"]
    eval_rows = load(original.EVAL_BANK)["states"][:protocol["evaluation_states"]]
    evaluations = {"0": original.evaluate(model, eval_rows, device)}
    parameter_norms = {}
    losses = []
    torch.cuda.reset_peak_memory_stats(device); started = time.perf_counter(); model.train()
    for index, row in enumerate(train_rows):
        step = index + 1
        torch.manual_seed(1000000 * stream + 193000 + index)
        torch.cuda.manual_seed_all(1000000 * stream + 193000 + index)
        tokens = torch.tensor([row["token_ids"]], dtype=torch.long, device=device)
        optimizer.zero_grad(set_to_none=True)
        loss = model(input_ids=tokens, labels=tokens).loss
        if not torch.isfinite(loss):
            raise RuntimeError(f"nonfinite loss at step {step}")
        loss.backward(); optimizer.step(); losses.append(float(loss.detach().cpu()))
        if step in protocol["evaluation_steps"]:
            torch.cuda.synchronize(); evaluations[str(step)] = original.evaluate(model, eval_rows, device)
            with torch.no_grad():
                key_energy = sum(float(dict(model.named_parameters())[name].double().square().sum()) for name in KEY_PARAMETERS)
                total_energy = sum(float(parameter.double().square().sum()) for parameter in model.parameters())
            parameter_norms[str(step)] = {"key": math.sqrt(key_energy),
                                           "non_key": math.sqrt(max(total_energy - key_energy, 0.0))}
            model.train()
        if step % 128 == 0:
            print(json.dumps({"event": "STRUCTURED_TRAINING", "stream": stream,
                              "condition": condition, "step": step}), flush=True)
    torch.cuda.synchronize(); elapsed = time.perf_counter() - started
    digest = hashlib.sha256()
    for parameter in model.parameters():
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return {
        "schema": "structured-residual-training-run-v1", "status": "COMPLETE",
        "stream": stream, "condition": condition, "training_loss": losses,
        "evaluation_loss_by_step": evaluations, "parameter_norm_by_step": parameter_norms,
        "final_optimizer_state_norm": state_norms(optimizer),
        "elapsed_seconds": elapsed, "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "final_parameter_sha256": digest.hexdigest(),
    }


def run(output: Path, device: str, worker: int, workers: int) -> None:
    protocol = checked(output)
    tasks = [(stream, condition) for stream in range(8) for condition in CONDITIONS]
    for index, (stream, condition) in enumerate(tasks):
        if index % workers != worker:
            continue
        path = output / "runs" / f"stream_{stream:02d}_{condition}.json"
        if path.exists():
            continue
        print(json.dumps({"event": "START", "stream": stream, "condition": condition}), flush=True)
        save_new(path, run_condition(protocol, stream, condition, device))


def interval(values, probability: float) -> list[float]:
    import scipy.stats
    center = sum(values) / len(values)
    variance = sum((value - center) ** 2 for value in values) / (len(values) - 1)
    half = float(scipy.stats.t.ppf((1 + probability) / 2, len(values) - 1)) * math.sqrt(variance / len(values))
    return [center - half, center + half]


def summarize(output: Path) -> None:
    protocol = checked(output); endpoint = str(protocol["steps"])
    values = {condition: [] for condition in ("OFF", "ON", *CONDITIONS)}
    runs = []
    for stream in range(8):
        for mode in ("OFF", "ON"):
            row = load(HISTORICAL / "runs" / f"stream_{stream:02d}_{mode}.json")
            values[mode].append(sum(row["evaluation_loss_by_step"][endpoint]) / len(row["evaluation_loss_by_step"][endpoint]))
        for condition in CONDITIONS:
            row = load(output / "runs" / f"stream_{stream:02d}_{condition}.json")
            values[condition].append(sum(row["evaluation_loss_by_step"][endpoint]) / len(row["evaluation_loss_by_step"][endpoint]))
            runs.append(row)
    contrasts = {
        "KEY_ONLY_MINUS_ON": [a-b for a,b in zip(values["KEY_ONLY"], values["ON"])],
        "OFF_MINUS_REST_ONLY": [a-b for a,b in zip(values["OFF"], values["REST_ONLY"])],
        "COORDINATE_ROLL_MINUS_ON": [a-b for a,b in zip(values["COORDINATE_ROLL"], values["ON"])],
        "ONE_EXTRA_STEP_LAG_MINUS_ON": [a-b for a,b in zip(values["ONE_EXTRA_STEP_LAG"], values["ON"])],
    }
    probability = protocol["simultaneous_interval_level"]
    results = {name: {"values": row, "mean": sum(row)/len(row),
                      "simultaneous_interval": interval(row, probability)}
               for name, row in contrasts.items()}
    margin = protocol["loss_margin"]
    results["KEY_ONLY_MINUS_ON"]["decision"] = "EQUIVALENT" if (
        results["KEY_ONLY_MINUS_ON"]["simultaneous_interval"][0] > -margin and
        results["KEY_ONLY_MINUS_ON"]["simultaneous_interval"][1] < margin) else "NOT_ESTABLISHED"
    results["OFF_MINUS_REST_ONLY"]["decision"] = "EQUIVALENT" if (
        results["OFF_MINUS_REST_ONLY"]["simultaneous_interval"][0] > -margin and
        results["OFF_MINUS_REST_ONLY"]["simultaneous_interval"][1] < margin) else "NOT_ESTABLISHED"
    for name in ("COORDINATE_ROLL_MINUS_ON", "ONE_EXTRA_STEP_LAG_MINUS_ON"):
        results[name]["decision"] = "MATERIAL_WORSENING" if results[name]["simultaneous_interval"][0] > margin else "NOT_ESTABLISHED"
    result = {
        "schema": protocol["schema"], "status": "COMPLETE", "stream_count": 8,
        "mean_final_evaluation_loss": {name: sum(row)/len(row) for name,row in values.items()},
        "contrasts": results,
        "trajectory_metrics": {
            condition: {
                "parameter_norm_by_step": [row["parameter_norm_by_step"] for row in runs if row["condition"] == condition],
                "final_optimizer_state_norm": [row["final_optimizer_state_norm"] for row in runs if row["condition"] == condition],
            } for condition in CONDITIONS
        },
        "scope": protocol["not_claimed"],
    }
    save_new(output / "result.json", result); print(json.dumps(result["contrasts"], indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "run", "summarize"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--worker", type=int, default=0); parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(); output = args.output.resolve()
    if not output.is_relative_to(ROOT):
        parser.error("all outputs must stay inside kernel-analyzer")
    if args.action == "freeze": freeze(output)
    elif args.action == "run": run(output, args.device, args.worker, args.workers)
    else: summarize(output)


if __name__ == "__main__":
    main()
