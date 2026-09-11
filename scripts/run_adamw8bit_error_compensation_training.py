#!/usr/bin/env python3
"""Prospective training confirmation of AdamW8bit recurrence compensation."""

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
PROBE = ROOT / ("results/property/numerical_coverage_v1/"
                "adamw8bit_error_compensation_probe_v1/result.json")
CONDITIONS = ("FP32_ADAMW", "ADAMW8BIT_BLOCK256", "ADAMW8BIT_COMPENSATED_BLOCK256")
STREAMS = 8
STEPS = 1024
EVALUATION_STEPS = (0, 256, 512, 768, 1024)
DESIGN_SEED = 20260913
START_RANGE = (200000, 500000)
MINIMUM_SEPARATION = 1200
MATERIAL_IMPROVEMENT_MARGIN = 0.01


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> Any:
    return json.loads(path.read_text())


def save_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def selected_start_blocks() -> list[int]:
    generator = random.Random(DESIGN_SEED)
    chosen: list[int] = []
    for value in generator.sample(range(*START_RANGE), START_RANGE[1] - START_RANGE[0]):
        if all(abs(value - prior) >= MINIMUM_SEPARATION for prior in chosen):
            chosen.append(value)
            if len(chosen) == STREAMS:
                return chosen
    raise RuntimeError("unable to select nonoverlapping streams")


def bank_paths() -> list[Path]:
    return [ROOT / ("results/property/numerical_coverage_v1/"
                    f"mamba_adamw8bit_compensation_bank_{index:02d}_v1.json")
            for index in range(STREAMS)]


def build_banks() -> None:
    """Tokenize the source corpus once and write all preregistered streams."""
    os.environ.setdefault("HF_DATASETS_CACHE", "/data1/tzh/cache/huggingface/datasets")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from datasets import load_dataset
    from transformers import AutoTokenizer

    paths = bank_paths()
    if any(path.exists() for path in paths):
        raise ValueError("one or more compensation training banks already exist")
    starts = selected_start_blocks()
    sequence_length = 64
    required = (max(starts) + STEPS) * (sequence_length + 1)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, use_fast=True)
    dataset = load_dataset(
        "Salesforce/wikitext", "wikitext-103-raw-v1", split="train",
        download_mode="reuse_dataset_if_exists",
    )
    tokens: list[int] = []
    documents = 0
    for row in dataset:
        source = str(row["text"]).strip()
        if not source:
            continue
        tokens.extend(tokenizer(source, add_special_tokens=False,
                                return_attention_mask=False)["input_ids"])
        documents += 1
        if len(tokens) >= required:
            break
    if len(tokens) < required:
        raise RuntimeError("cached WikiText stream is too short for the frozen design")
    import numpy as np
    for bank_index, (path, start_block) in enumerate(zip(paths, starts)):
        states = []
        for state_index in range(STEPS):
            start = (start_block + state_index) * (sequence_length + 1)
            values = tokens[start:start + sequence_length]
            encoded = np.asarray(values, dtype=np.int64).tobytes()
            states.append({
                "state_id": f"mamba-adamw8bit-compensation-{bank_index:02d}-{state_index:04d}",
                "role": "TRAJECTORY",
                "order_within_role": state_index,
                "token_ids": values,
                "token_sha256": hashlib.sha256(encoded).hexdigest(),
            })
        save_new(path, {
            "schema": "kernel-analyzer-tcmp-input-bank-v1",
            "cell_id": f"mamba-adamw8bit-compensation-{bank_index:02d}",
            "model_path": str(MODEL),
            "sequence_length": sequence_length,
            "dataset": "Salesforce/wikitext:wikitext-103-raw-v1:train",
            "nonempty_documents_consumed": documents,
            "states": states,
            "start_block": start_block,
            "splits": {"TRAJECTORY": STEPS},
        })
    print(json.dumps({"event": "COMPENSATION_BANKS_BUILT", "starts": starts,
                      "documents": documents}), flush=True)


def make_optimizer(condition: str, parameters, settings):
    import torch
    if condition == "FP32_ADAMW":
        return torch.optim.AdamW(parameters, foreach=False, fused=False, **settings)
    if condition == "ADAMW8BIT_BLOCK256":
        from torchao.optim import AdamW8bit
        return AdamW8bit(parameters, block_size=256, **settings)
    from kernel_analyzer.adamw8bit_error_compensation import AdamW8bitErrorCompensated
    return AdamW8bitErrorCompensated(parameters, block_size=256, **settings)


def condition_order(stream_index: int) -> tuple[str, ...]:
    shift = stream_index % len(CONDITIONS)
    return CONDITIONS[shift:] + CONDITIONS[:shift]


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


def run_condition(protocol: dict[str, Any], stream_index: int, condition: str,
                  device: str, *, steps: int | None = None) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM

    limit = int(protocol["steps"] if steps is None else steps)
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
    settings = dict(lr=protocol["optimizer"]["lr"],
                    betas=tuple(protocol["optimizer"]["betas"]),
                    eps=protocol["optimizer"]["eps"],
                    weight_decay=protocol["optimizer"]["weight_decay"])
    optimizer = make_optimizer(condition, model.parameters(), settings)
    train_rows = load(Path(protocol["train_banks"][stream_index]))["states"][:limit]
    eval_rows = load(EVAL_BANK)["states"][:int(protocol["evaluation_states"])]
    torch.cuda.reset_peak_memory_stats(device)
    evaluations = {"0": evaluate(model, eval_rows, device)}
    losses = []
    model.train()
    started = time.perf_counter()
    for index, row in enumerate(train_rows):
        step = index + 1
        step_seed = 1000000 * stream_index + 193000 + index
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
        if step in protocol["evaluation_steps"] or step == limit:
            torch.cuda.synchronize()
            evaluations[str(step)] = evaluate(model, eval_rows, device)
            model.train()
        if step % 128 == 0:
            print(json.dumps({"event": "COMPENSATION_TRAINING", "stream": stream_index,
                              "condition": condition, "step": step,
                              "loss": losses[-1]}), flush=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    digest = hashlib.sha256()
    for parameter in model.parameters():
        digest.update(parameter.detach().float().cpu().contiguous().numpy().tobytes())
    result = {
        "condition": condition,
        "status": "COMPLETE",
        "training_steps": limit,
        "training_loss": losses,
        "evaluation_loss_by_step": evaluations,
        "elapsed_seconds_including_evaluation": elapsed,
        "training_steps_per_second_with_evaluation_overhead": limit / elapsed,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "final_parameter_sha256": digest.hexdigest(),
    }
    del optimizer, model
    torch.cuda.empty_cache()
    return result


def pilot(output: Path, device: str, cache_dir: Path) -> None:
    if output.exists():
        raise ValueError("pilot output already exists")
    banks = bank_paths()
    if not banks[0].is_file():
        raise ValueError("build the frozen training banks before the feasibility pilot")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache_dir.resolve())
    provisional = {
        "steps": 8,
        "evaluation_steps": [],
        "evaluation_states": 2,
        "train_banks": [str(path) for path in banks],
        "optimizer": {"lr": 1e-4, "betas": [0.9, 0.999], "eps": 1e-8,
                      "weight_decay": 0.01},
    }
    rows = [run_condition(provisional, 0, condition, device, steps=8)
            for condition in CONDITIONS]
    save_new(output, {
        "schema": "adamw8bit-error-compensation-full-model-pilot-v1",
        "status": "FEASIBLE_NO_OUTCOME_SELECTION",
        "steps": 8,
        "records": rows,
    })


def freeze(output: Path, pilot_path: Path) -> None:
    if output.exists():
        raise ValueError("freeze requires a new output directory")
    if load(PROBE).get("prediction_result") != "CONFIRMED":
        raise ValueError("optimizer-history compensation probe is not confirmed")
    if load(pilot_path).get("status") != "FEASIBLE_NO_OUTCOME_SELECTION":
        raise ValueError("full-model feasibility pilot is incomplete")
    starts = selected_start_blocks()
    banks = bank_paths()
    for index, (path, start) in enumerate(zip(banks, starts)):
        document = load(path)
        if document.get("start_block") != start or len(document.get("states", [])) != STEPS:
            raise ValueError(f"bank {index} differs from frozen random design")
    module = ROOT / "src/kernel_analyzer/adamw8bit_error_compensation.py"
    paths = [Path(__file__).resolve(), module, EVAL_BANK, PROBE, pilot_path,
             MODEL / "config.json", MODEL / "model.safetensors", *banks]
    protocol = {
        "schema": "adamw8bit-error-compensation-training-v1",
        "status": "FROZEN_BEFORE_INDEPENDENT_TRAINING_STREAMS",
        "model": str(MODEL),
        "conditions": list(CONDITIONS),
        "stream_count": STREAMS,
        "steps": STEPS,
        "evaluation_steps": list(EVALUATION_STEPS),
        "evaluation_bank": str(EVAL_BANK),
        "evaluation_states": 32,
        "train_banks": [str(path) for path in banks],
        "stream_selection": {
            "method": "SEEDED_WITHOUT_REPLACEMENT_AND_MINIMUM_BLOCK_SEPARATION",
            "seed": DESIGN_SEED,
            "start_range": list(START_RANGE),
            "minimum_start_block_separation": MINIMUM_SEPARATION,
            "selected_start_blocks": starts,
            "disjoint_from_prior_confirmation_ranges": True,
        },
        "optimizer": {"lr": 1e-4, "betas": [0.9, 0.999], "eps": 1e-8,
                      "weight_decay": 0.01},
        "parameter_dtype": "float32",
        "pairing": "same checkpoint, tokens, and per-step RNG within each stream",
        "independent_unit": "NONOVERLAPPING_RANDOMLY_SELECTED_WIKITEXT_TOKEN_STREAM",
        "primary_endpoint": "MEAN_FIXED_EVALUATION_LOSS_AT_STEP_1024",
        "primary_contrast": (
            "ADAMW8BIT_BLOCK256_MINUS_ADAMW8BIT_COMPENSATED_BLOCK256"
        ),
        "material_improvement_margin": MATERIAL_IMPROVEMENT_MARGIN,
        "primary_decision": (
            "MATERIAL_IMPROVEMENT if the two-sided 95% paired t interval is entirely "
            "above +0.01; DETECTABLE_IMPROVEMENT if entirely above zero; "
            "otherwise NOT_CONFIRMED"
        ),
        "secondary_contrasts": [
            "ADAMW8BIT_BLOCK256_MINUS_FP32_ADAMW",
            "ADAMW8BIT_COMPENSATED_BLOCK256_MINUS_FP32_ADAMW",
        ],
        "multiplicity": (
            "one preregistered primary modification contrast; reference contrasts, "
            "intermediate checkpoints, memory, and speed are secondary"
        ),
        "condition_execution": (
            "each condition runs to completion; order rotates by stream to avoid "
            "interleaving optimizer compilation state"
        ),
        "data_use": "PROSPECTIVE_TRAINING_CONFIRMATION_AFTER_WRITE_LEVEL_PROBE",
        "pilot_used_only_for_execution_feasibility": True,
        "not_claimed": [
            "PRODUCTION_READY_OPTIMIZER",
            "CROSS_MODEL_GENERALIZATION",
            "UNIVERSAL_TRAINING_IMPROVEMENT",
        ],
        "source_sha256": {str(path.resolve()): sha(path) for path in paths},
    }
    save_new(output / "protocol.json", protocol)


def verify_protocol(output: Path) -> dict[str, Any]:
    protocol = load(output / "protocol.json")
    if protocol.get("schema") != "adamw8bit-error-compensation-training-v1":
        raise ValueError("unexpected protocol")
    for name, expected in protocol.get("source_sha256", {}).items():
        path = Path(name)
        if not path.is_file() or sha(path) != expected:
            raise ValueError("frozen training dependency changed: " + name)
    return protocol


def capture_stream(output: Path, stream_index: int, device: str, cache_dir: Path) -> None:
    if not 0 <= stream_index < STREAMS:
        raise ValueError("stream index is out of range")
    protocol = verify_protocol(output)
    destination = output / "streams" / f"stream_{stream_index:02d}.json"
    failure = output / "streams" / f"stream_{stream_index:02d}_failure.json"
    if destination.exists() or failure.exists():
        raise ValueError("stream output already exists")
    resolved_cache = cache_dir.resolve()
    if not resolved_cache.is_relative_to(Path("/data1/tzh/cache")):
        raise ValueError("cache must remain under /data1/tzh/cache")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(resolved_cache)
    try:
        rows = [run_condition(protocol, stream_index, condition, device)
                for condition in condition_order(stream_index)]
        save_new(destination, {
            "schema": "adamw8bit-error-compensation-training-stream-v1",
            "status": "COMPLETE",
            "stream_index": stream_index,
            "train_bank": protocol["train_banks"][stream_index],
            "condition_execution_order": list(condition_order(stream_index)),
            "protocol_sha256": sha(output / "protocol.json"),
            "records": rows,
        })
    except Exception as error:
        save_new(failure, {
            "schema": "adamw8bit-error-compensation-training-failure-v1",
            "status": "FAILED",
            "stream_index": stream_index,
            "error_type": type(error).__name__,
            "error": str(error),
            "protocol_sha256": sha(output / "protocol.json"),
        })
        raise


def t_interval(values: list[float]) -> list[float]:
    import scipy.stats
    if len(values) < 2 or not all(math.isfinite(value) for value in values):
        raise ValueError("at least two finite independent values are required")
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    if variance == 0:
        return [mean, mean]
    critical = float(scipy.stats.t.ppf(0.975, len(values) - 1))
    half = critical * math.sqrt(variance / len(values))
    return [mean - half, mean + half]


def summarize(protocol: dict[str, Any], streams: list[dict[str, Any]]) -> dict[str, Any]:
    paired = []
    compensated_reference = []
    default_reference = []
    rows = []
    for expected, stream in enumerate(sorted(streams, key=lambda row: row["stream_index"])):
        if stream.get("stream_index") != expected or stream.get("status") != "COMPLETE":
            raise ValueError("stream records are incomplete or reordered")
        records = {row["condition"]: row for row in stream["records"]}
        if set(records) != set(CONDITIONS):
            raise ValueError("condition records are incomplete")
        endpoint = {
            condition: sum(records[condition]["evaluation_loss_by_step"][str(STEPS)]) /
            len(records[condition]["evaluation_loss_by_step"][str(STEPS)])
            for condition in CONDITIONS
        }
        gain = endpoint["ADAMW8BIT_BLOCK256"] - endpoint["ADAMW8BIT_COMPENSATED_BLOCK256"]
        d_default = endpoint["ADAMW8BIT_BLOCK256"] - endpoint["FP32_ADAMW"]
        d_compensated = endpoint["ADAMW8BIT_COMPENSATED_BLOCK256"] - endpoint["FP32_ADAMW"]
        paired.append(gain)
        default_reference.append(d_default)
        compensated_reference.append(d_compensated)
        rows.append({
            "stream_index": expected,
            "final_mean_evaluation_loss": endpoint,
            "default_minus_compensated": gain,
            "default_minus_reference": d_default,
            "compensated_minus_reference": d_compensated,
            "runtime_and_memory": {
                condition: {
                    "steps_per_second_with_evaluation_overhead":
                        records[condition]["training_steps_per_second_with_evaluation_overhead"],
                    "peak_allocated_bytes": records[condition]["peak_allocated_bytes"],
                } for condition in CONDITIONS
            },
        })
    interval = t_interval(paired)
    margin = float(protocol["material_improvement_margin"])
    if interval[0] > margin:
        decision = "MATERIAL_IMPROVEMENT"
    elif interval[0] > 0:
        decision = "DETECTABLE_IMPROVEMENT"
    else:
        decision = "NOT_CONFIRMED"
    return {
        "schema": "adamw8bit-error-compensation-training-summary-v1",
        "status": "COMPLETE",
        "streams": rows,
        "primary": {
            "contrast": protocol["primary_contrast"],
            "paired_values": paired,
            "mean": sum(paired) / len(paired),
            "interval_95": interval,
            "material_improvement_margin": margin,
            "decision": decision,
        },
        "secondary": {
            "default_minus_reference_mean": sum(default_reference) / len(default_reference),
            "default_minus_reference_interval_95": t_interval(default_reference),
            "compensated_minus_reference_mean": sum(compensated_reference) / len(compensated_reference),
            "compensated_minus_reference_interval_95": t_interval(compensated_reference),
        },
        "claim_scope": (
            "Eight preregistered nonoverlapping WikiText token streams at one fixed "
            "Mamba checkpoint and optimizer setting"
        ),
    }


def report(output: Path) -> None:
    protocol = verify_protocol(output)
    streams = []
    for index in range(STREAMS):
        path = output / "streams" / f"stream_{index:02d}.json"
        if not path.is_file():
            raise ValueError(f"missing stream {index}")
        streams.append(load(path))
    save_new(output / "summary.json", summarize(protocol, streams))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build-banks", "pilot", "freeze",
                                            "capture-stream", "report"))
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--pilot-output", type=Path)
    parser.add_argument("--stream-index", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args()
    for path in (args.output_root, args.pilot_output, args.cache_dir):
        if path is not None and not path.resolve().is_relative_to(Path("/data1/tzh")):
            parser.error("all outputs and caches must remain under /data1/tzh")
    if args.command == "build-banks":
        build_banks()
    elif args.command == "pilot":
        if args.pilot_output is None or args.cache_dir is None:
            parser.error("pilot requires --pilot-output and --cache-dir")
        pilot(args.pilot_output, args.device, args.cache_dir)
    elif args.command == "freeze":
        if args.output_root is None or args.pilot_output is None:
            parser.error("freeze requires --output-root and --pilot-output")
        freeze(args.output_root, args.pilot_output)
    elif args.command == "capture-stream":
        if args.output_root is None or args.stream_index is None or args.cache_dir is None:
            parser.error("capture-stream requires --output-root, --stream-index, and --cache-dir")
        capture_stream(args.output_root, args.stream_index, args.device, args.cache_dir)
    else:
        if args.output_root is None:
            parser.error("report requires --output-root")
        report(args.output_root)


if __name__ == "__main__":
    os.environ.setdefault("HF_HOME", "/data1/tzh/cache/huggingface")
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/data1/tzh/cache/huggingface/hub")
    os.environ.setdefault("XDG_CACHE_HOME", "/data1/tzh/cache/xdg")
    main()
