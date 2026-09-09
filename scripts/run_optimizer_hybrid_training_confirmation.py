#!/usr/bin/env python3
"""Prospective training confirmation for the FP32-first-moment AdamW8bit variant."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MODEL = Path("/data1/tzh/models/state-spaces/mamba-130m-hf")
EVAL_BANK = ROOT / "results/coverage/mamba_seq64_input_bank.json"
PILOT_BANK = ROOT / "results/property/numerical_coverage_v1/mamba_adamw8bit_training_pilot_bank_v1.json"
COMPONENT = ROOT / "results/property/numerical_coverage_v1/torchao_adamw8bit_component_mechanism_v1/summary.json"
PRIOR_TRAINING = ROOT / "results/property/numerical_coverage_v1/mamba_adamw8bit_training_confirmation_v1/verification.json"
CONDITIONS = ("ADAMW8BIT_BLOCK256", "FP32_FIRST_MOMENT_BLOCK256", "FP32_ADAMW")
STREAMS = 8
STEPS = 1024
EVALUATION_STEPS = (0, 256, 512, 768, 1024)
DESIGN_SEED = 20260911
SELECTED_STARTS = (171109, 139468, 94002, 150795, 103965, 178837, 114780, 158161)
MINIMUM_SEPARATION = 1200
MATERIAL_IMPROVEMENT_MARGIN = 0.01
PILOT_STEPS = 8


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def load(path: Path) -> Any:
    return json.loads(path.read_text())


def save_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def bank_paths() -> list[Path]:
    return [ROOT / ("results/property/numerical_coverage_v1/"
                   f"mamba_adamw8bit_hybrid_confirm_bank_{index:02d}_v1.json")
            for index in range(STREAMS)]


def validate_starts() -> None:
    generator = random.Random(DESIGN_SEED)
    chosen = []
    for value in generator.sample(range(92000, 180000), 88000):
        if all(abs(value - prior) >= MINIMUM_SEPARATION for prior in chosen):
            chosen.append(value)
            if len(chosen) == STREAMS:
                break
    if tuple(chosen) != SELECTED_STARTS:
        raise ValueError("frozen stream selection is not reproducible")


def make_optimizer(condition: str, parameters, settings):
    import torch
    from torchao.optim import AdamW8bit
    from kernel_analyzer.optimizer_state_repairs import AdamWFirstMomentFP32

    if condition == "FP32_ADAMW":
        return torch.optim.AdamW(parameters, foreach=False, fused=False, **settings)
    if condition == "FP32_FIRST_MOMENT_BLOCK256":
        return AdamWFirstMomentFP32(parameters, block_size=256, **settings)
    return AdamW8bit(parameters, block_size=256, **settings)


def evaluate(model, rows, device: str) -> list[float]:
    import torch

    model.eval()
    values = []
    with torch.no_grad():
        for row in rows:
            tokens = torch.tensor([row.get("input_ids", row.get("token_ids"))],
                                  dtype=torch.long, device=device)
            values.append(float(model(input_ids=tokens, labels=tokens).loss.detach().cpu()))
    return values


def run_condition(condition: str, train_rows, eval_rows, steps: int,
                  evaluation_steps: tuple[int, ...], stream_index: int, device: str):
    import torch
    from transformers import AutoModelForCausalLM

    seed = 193000 + stream_index
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float32, local_files_only=True,
    ).to(device)
    model.config.use_cache = False
    settings = dict(lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    optimizer = make_optimizer(condition, model.parameters(), settings)
    evaluations = {"0": evaluate(model, eval_rows, device)}
    losses = []
    model.train()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for index, row in enumerate(train_rows[:steps]):
        step = index + 1
        step_seed = 1000000 * stream_index + 151000 + index
        torch.manual_seed(step_seed)
        torch.cuda.manual_seed_all(step_seed)
        tokens = torch.tensor([row["token_ids"]], dtype=torch.long, device=device)
        optimizer.zero_grad(set_to_none=True)
        loss = model(input_ids=tokens, labels=tokens).loss
        if not torch.isfinite(loss):
            raise RuntimeError(f"nonfinite training loss at step {step}")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if step in evaluation_steps:
            torch.cuda.synchronize()
            evaluations[str(step)] = evaluate(model, eval_rows, device)
            model.train()
        if step % 128 == 0:
            print(json.dumps({"event": "HYBRID_TRAINING", "stream": stream_index,
                              "condition": condition, "step": step}), flush=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    digest = hashlib.sha256()
    for parameter in model.parameters():
        digest.update(parameter.detach().float().cpu().contiguous().numpy().tobytes())
    result = {
        "condition": condition,
        "status": "COMPLETE",
        "training_loss": losses,
        "evaluation_loss_by_step": evaluations,
        "elapsed_seconds_including_evaluation": elapsed,
        "steps_per_second_with_evaluation_overhead": steps / elapsed,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "final_parameter_sha256": digest.hexdigest(),
    }
    del optimizer, model
    torch.cuda.empty_cache()
    return result


def pilot(output: Path, device: str, cache_dir: Path) -> None:
    if output.exists():
        raise ValueError("pilot output already exists")
    resolved = cache_dir.resolve()
    if not resolved.is_relative_to(Path("/data1/tzh/cache")):
        raise ValueError("cache must remain under /data1/tzh/cache")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(resolved)
    rows = load(PILOT_BANK)["states"]
    eval_rows = load(EVAL_BANK)["states"][:4]
    record = run_condition("FP32_FIRST_MOMENT_BLOCK256", rows, eval_rows,
                           PILOT_STEPS, (), -1, device)
    save_new(output, {
        "schema": "optimizer-hybrid-training-pilot-v1",
        "status": "FEASIBLE" if len(record["training_loss"]) == PILOT_STEPS else "FAILED",
        "purpose": "EXECUTION_FEASIBILITY_ONLY_LOSS_NOT_USED_FOR_SELECTION",
        "steps": PILOT_STEPS,
        "record": record,
    })


def freeze(output: Path, pilot_path: Path) -> None:
    if output.exists():
        raise ValueError("freeze requires a new output directory")
    validate_starts()
    component = load(COMPONENT)
    prior = load(PRIOR_TRAINING)
    pilot_record = load(pilot_path)
    if not component.get("primary_parameter_write_reduction"):
        raise ValueError("component modification did not reduce parameter-write distortion")
    if prior.get("primary", {}).get("decision") != "MATERIAL_EFFECT":
        raise ValueError("prior default optimizer training effect is unavailable")
    if pilot_record.get("status") != "FEASIBLE":
        raise ValueError("hybrid optimizer feasibility failed")
    banks = bank_paths()
    for index, (path, start) in enumerate(zip(banks, SELECTED_STARTS)):
        document = load(path)
        if document.get("start_block") != start or len(document.get("states", [])) != STEPS:
            raise ValueError(f"hybrid stream bank {index} differs from design")
    dependencies = [Path(__file__).resolve(), pilot_path, COMPONENT, PRIOR_TRAINING,
                    EVAL_BANK, MODEL / "config.json", MODEL / "model.safetensors",
                    ROOT / "src/kernel_analyzer/optimizer_state_repairs.py", *banks]
    protocol = {
        "schema": "optimizer-hybrid-training-confirmation-v1",
        "status": "FROZEN_BEFORE_NEW_TRAINING_STREAMS",
        "conditions": list(CONDITIONS),
        "stream_count": STREAMS,
        "steps": STEPS,
        "evaluation_steps": list(EVALUATION_STEPS),
        "train_banks": [str(path) for path in banks],
        "evaluation_bank": str(EVAL_BANK),
        "evaluation_states": 32,
        "stream_selection": {"seed": DESIGN_SEED, "selected_start_blocks": list(SELECTED_STARTS),
                             "minimum_start_block_separation": MINIMUM_SEPARATION},
        "optimizer": {"lr": 1e-4, "betas": [0.9, 0.999], "eps": 1e-8,
                      "weight_decay": 0.01},
        "primary_endpoint": "MEAN_FIXED_EVALUATION_LOSS_AT_STEP_1024",
        "primary_contrast": "ADAMW8BIT_BLOCK256_MINUS_FP32_FIRST_MOMENT_BLOCK256",
        "material_improvement_margin": MATERIAL_IMPROVEMENT_MARGIN,
        "primary_decision": (
            "MATERIAL_IMPROVEMENT if the two-sided 95% paired t interval is entirely "
            "above +0.01; DETECTABLE_IMPROVEMENT if entirely above zero; otherwise NOT_CONFIRMED"
        ),
        "secondary": "both optimizer variants relative to FP32 AdamW and intermediate checkpoints",
        "multiplicity": "one primary modification contrast; all other comparisons are secondary",
        "data_use": "RESULT_AWARE_MODIFICATION_WITH_NEW_UNTOUCHED_TOKEN_STREAMS",
        "prior_strict_component_prediction_result": component.get("prediction_result"),
        "prior_component_primary_write_reduction": component.get("primary_parameter_write_reduction"),
        "source_sha256": {str(path.resolve()): sha(path) for path in dependencies},
    }
    save_new(output / "protocol.json", protocol)


def verify(output: Path) -> dict[str, Any]:
    protocol = load(output / "protocol.json")
    for name, expected in protocol["source_sha256"].items():
        path = Path(name)
        if not path.is_file() or sha(path) != expected:
            raise ValueError("frozen dependency changed: " + name)
    return protocol


def condition_order(index: int):
    shift = index % len(CONDITIONS)
    return CONDITIONS[shift:] + CONDITIONS[:shift]


def capture_stream(output: Path, stream_index: int, device: str, cache_dir: Path) -> None:
    if not 0 <= stream_index < STREAMS:
        raise ValueError("stream index is out of range")
    protocol = verify(output)
    destination = output / "streams" / f"stream_{stream_index:02d}.json"
    failure = output / "streams" / f"stream_{stream_index:02d}_failure.json"
    if destination.exists() or failure.exists():
        raise ValueError("stream result already exists")
    resolved = cache_dir.resolve()
    if not resolved.is_relative_to(Path("/data1/tzh/cache")):
        raise ValueError("cache must remain under /data1/tzh/cache")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(resolved)
    try:
        train_rows = load(Path(protocol["train_banks"][stream_index]))["states"]
        eval_rows = load(EVAL_BANK)["states"][:protocol["evaluation_states"]]
        records = [run_condition(condition, train_rows, eval_rows, STEPS, EVALUATION_STEPS,
                                 stream_index, device) for condition in condition_order(stream_index)]
        save_new(destination, {
            "schema": "optimizer-hybrid-training-stream-v1",
            "status": "COMPLETE",
            "stream_index": stream_index,
            "train_bank": protocol["train_banks"][stream_index],
            "condition_execution_order": list(condition_order(stream_index)),
            "protocol_sha256": sha(output / "protocol.json"),
            "records": records,
        })
    except Exception as error:
        save_new(failure, {
            "schema": "optimizer-hybrid-training-failure-v1",
            "status": "FAILED",
            "stream_index": stream_index,
            "error_type": type(error).__name__,
            "error": str(error),
            "protocol_sha256": sha(output / "protocol.json"),
        })
        raise


def t_interval(values: list[float]):
    import scipy.stats

    if len(values) < 2 or not all(math.isfinite(value) for value in values):
        raise ValueError("at least two finite independent values are required")

    center = math.fsum(values) / len(values)
    variance = math.fsum((value - center) ** 2 for value in values) / (len(values) - 1)
    critical = float(scipy.stats.t.ppf(0.975, len(values) - 1))
    half = 0 if variance == 0 else critical * math.sqrt(variance / len(values))
    return [center - half, center + half]


def summarize(protocol: dict[str, Any], streams: list[dict[str, Any]]):
    if len(streams) != STREAMS:
        raise ValueError("all frozen streams are required")
    improvements = []
    rows = []
    for expected, stream in enumerate(sorted(streams, key=lambda row: row["stream_index"])):
        if stream.get("stream_index") != expected or stream.get("status") != "COMPLETE":
            raise ValueError("stream records are incomplete or reordered")
        records = {row["condition"]: row for row in stream["records"]}
        if set(records) != set(CONDITIONS):
            raise ValueError("condition records are incomplete")
        for record in records.values():
            if record.get("status") != "COMPLETE" or len(record.get("training_loss", [])) != STEPS:
                raise ValueError("condition record is incomplete")
            if set(record.get("evaluation_loss_by_step", {})) != {
                    str(step) for step in EVALUATION_STEPS}:
                raise ValueError("evaluation checkpoints are incomplete")
        endpoint = {condition: math.fsum(records[condition]["evaluation_loss_by_step"][str(STEPS)]) /
                    len(records[condition]["evaluation_loss_by_step"][str(STEPS)])
                    for condition in CONDITIONS}
        improvement = endpoint["ADAMW8BIT_BLOCK256"] - endpoint["FP32_FIRST_MOMENT_BLOCK256"]
        improvements.append(improvement)
        rows.append({"stream_index": stream["stream_index"], "final_mean_evaluation_loss": endpoint,
                     "default_minus_hybrid": improvement,
                     "default_minus_fp32": endpoint["ADAMW8BIT_BLOCK256"] - endpoint["FP32_ADAMW"],
                     "hybrid_minus_fp32": endpoint["FP32_FIRST_MOMENT_BLOCK256"] - endpoint["FP32_ADAMW"]})
    interval = t_interval(improvements)
    margin = float(protocol["material_improvement_margin"])
    decision = ("MATERIAL_IMPROVEMENT" if interval[0] > margin else
                "DETECTABLE_IMPROVEMENT" if interval[0] > 0 else "NOT_CONFIRMED")
    return {"schema": "optimizer-hybrid-training-summary-v1", "streams": rows,
            "primary": {"paired_values": improvements,
                        "mean": math.fsum(improvements) / len(improvements),
                        "interval_95": interval, "material_margin": margin,
                        "decision": decision},
            "scope": "Eight new nonoverlapping WikiText streams at one fixed Mamba checkpoint"}


def report(output: Path) -> None:
    protocol = verify(output)
    failures = list((output / "streams").glob("stream_*_failure.json"))
    if failures:
        raise ValueError("failed stream records exist; confirmation is incomplete")
    streams = [load(output / "streams" / f"stream_{index:02d}.json")
               for index in range(STREAMS)]
    save_new(output / "summary.json", summarize(protocol, streams))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("pilot", "freeze", "capture-stream", "report"))
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--pilot-output", type=Path)
    parser.add_argument("--stream-index", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args()
    if args.command == "pilot":
        pilot(args.pilot_output.resolve(), args.device, args.cache_dir)
    elif args.command == "freeze":
        freeze(args.output_root.resolve(), args.pilot_output.resolve())
    elif args.command == "capture-stream":
        capture_stream(args.output_root.resolve(), args.stream_index, args.device, args.cache_dir)
    else:
        report(args.output_root.resolve())


if __name__ == "__main__":
    os.environ.setdefault("HF_HOME", "/data1/tzh/cache/huggingface")
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/data1/tzh/cache/huggingface/hub")
    os.environ.setdefault("XDG_CACHE_HOME", "/data1/tzh/cache/xdg")
    main()
