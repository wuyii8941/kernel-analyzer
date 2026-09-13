#!/usr/bin/env python3
"""Prospective paired training confirmation on iid sampled token-stream starts."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MODEL = Path("/data1/tzh/models/state-spaces/mamba-130m-hf")
EVAL_BANK = ROOT / "results/coverage/mamba_seq64_input_bank.json"
RESULT_ROOT = ROOT / "results/property/result_analysis_v4/iid_training_confirmation"
STREAMS = 8
STEPS = 1024
EVALUATION_STEPS = (0, 256, 512, 768, 1024)
EVALUATION_STATES = 32
DESIGN_SEED = 20260914
START_POPULATION = (500000, 700000)
SEQUENCE_LENGTH = 64
MATERIAL_MARGIN = 0.01
CONDITIONS = ("OFF", "ON")


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


def selected_starts() -> list[int]:
    generator = random.Random(DESIGN_SEED)
    width = START_POPULATION[1] - START_POPULATION[0]
    return [START_POPULATION[0] + generator.randrange(width) for _ in range(STREAMS)]


def bank_paths() -> list[Path]:
    return [RESULT_ROOT / "input_banks" / f"stream_{index:02d}.json"
            for index in range(STREAMS)]


def build_banks() -> None:
    paths = bank_paths()
    if any(path.exists() for path in paths):
        raise ValueError("one or more iid confirmation banks already exist")
    os.environ.setdefault("HF_DATASETS_CACHE", "/data1/tzh/cache/huggingface/datasets")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from datasets import load_dataset
    from transformers import AutoTokenizer

    starts = selected_starts()
    required = (max(starts) + STEPS) * (SEQUENCE_LENGTH + 1)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, use_fast=True)
    dataset = load_dataset(
        "Salesforce/wikitext", "wikitext-103-raw-v1", split="train",
        download_mode="reuse_dataset_if_exists",
    )
    tokens: list[int] = []
    documents = 0
    for row in dataset:
        text = str(row["text"]).strip()
        if not text:
            continue
        tokens.extend(tokenizer(text, add_special_tokens=False,
                                return_attention_mask=False)["input_ids"])
        documents += 1
        if len(tokens) >= required:
            break
    if len(tokens) < required:
        raise RuntimeError("cached WikiText stream is too short for the frozen start population")
    import numpy as np
    for stream, (path, start_block) in enumerate(zip(paths, starts)):
        states = []
        for step in range(STEPS):
            start = (start_block + step) * (SEQUENCE_LENGTH + 1)
            values = tokens[start:start + SEQUENCE_LENGTH]
            states.append({
                "state_id": f"mamba-adamw8bit-iid-confirm-{stream:02d}-{step:04d}",
                "role": "TRAJECTORY",
                "order_within_role": step,
                "token_ids": values,
                "token_sha256": hashlib.sha256(
                    np.asarray(values, dtype=np.int64).tobytes()
                ).hexdigest(),
            })
        save_new(path, {
            "schema": "kernel-analyzer-training-input-bank-v1",
            "stream": stream,
            "start_block": start_block,
            "start_population": list(START_POPULATION),
            "draw_method": "IID_UNIFORM_WITH_REPLACEMENT",
            "design_seed": DESIGN_SEED,
            "dataset": "Salesforce/wikitext:wikitext-103-raw-v1:train",
            "sequence_length": SEQUENCE_LENGTH,
            "states": states,
        })
    print(json.dumps({"event": "BANKS_BUILT", "starts": starts,
                      "unique_starts": len(set(starts)), "documents": documents}))


def freeze() -> None:
    protocol_path = RESULT_ROOT / "protocol.json"
    if protocol_path.exists():
        raise ValueError("protocol already exists")
    from kernel_analyzer.compensation_control import CompensationControl
    from scripts import run_adamw8bit_error_compensation_training as shared

    paths = [Path(__file__).resolve(), Path(inspect.getfile(CompensationControl)),
             ROOT / "src/kernel_analyzer/adamw8bit_error_compensation.py",
             Path(shared.__file__).resolve(), MODEL / "config.json",
             MODEL / "model.safetensors", EVAL_BANK, *bank_paths()]
    if any(not path.is_file() for path in paths):
        raise ValueError("build input banks and verify model inputs before freezing")
    starts = [int(load(path)["start_block"]) for path in bank_paths()]
    if starts != selected_starts():
        raise ValueError("input banks differ from the seeded iid draws")
    protocol = {
        "schema": "adamw8bit-iid-training-confirmation-v1",
        "status": "FROZEN_BEFORE_TRAINING",
        "model": str(MODEL),
        "conditions": list(CONDITIONS),
        "stream_count": STREAMS,
        "steps": STEPS,
        "evaluation_steps": list(EVALUATION_STEPS),
        "evaluation_states": EVALUATION_STATES,
        "evaluation_bank": str(EVAL_BANK),
        "train_banks": [str(path) for path in bank_paths()],
        "stream_sampling": {
            "population": "integer start blocks in [500000,700000) of the fixed cached WikiText tokenization",
            "method": "IID_UNIFORM_WITH_REPLACEMENT",
            "seed": DESIGN_SEED,
            "drawn_starts": starts,
            "duplicates_or_overlaps_are_not_removed": True,
            "development_start_range_is_disjoint": True,
        },
        "independent_unit": "ONE_IID_DRAWN_CONTIGUOUS_1024_STEP_TRAINING_STREAM",
        "pairing": "same checkpoint, tokens, evaluation set, and per-step RNG within stream",
        "optimizer": {"lr": 1e-4, "betas": [0.9, 0.999], "eps": 1e-8,
                      "weight_decay": 0.01},
        "comparison": "identical compensation implementation; only residual read multiplier is OFF or ON",
        "primary_endpoint": "MEAN_FIXED_EVALUATION_LOSS_AT_STEP_1024",
        "primary_contrast": "OFF_MINUS_ON",
        "material_improvement_margin": MATERIAL_MARGIN,
        "primary_rule": (
            "MATERIAL_IMPROVEMENT only if all eight pairs have finite endpoints and the "
            "two-sided 95% paired t interval lies above +0.01"
        ),
        "paired_t_assumptions": "iid start draws and normally distributed paired endpoint differences",
        "numerical_failure_rule": (
            "A nonfinite training loss, parameter, or evaluation makes the final-loss endpoint "
            "undefined for the full set; report failures and do not impute loss"
        ),
        "execution_failure_rule": "retry the same frozen task; retries never increase sample size",
        "data_use": "PROSPECTIVE_IID_TRAINING_STREAM_CONFIRMATION_OF_FROZEN_FULL_COMPENSATION",
        "not_claimed": ["CROSS_CHECKPOINT_GENERALIZATION", "CROSS_MODEL_GENERALIZATION",
                        "UNCONDITIONAL_NONNORMAL_FINITE_SAMPLE_GUARANTEE"],
        "source_sha256": {str(path.resolve()): sha(path) for path in paths},
    }
    save_new(protocol_path, protocol)
    print(json.dumps({"event": "PROTOCOL_FROZEN", "starts": starts}))


def checked() -> dict[str, Any]:
    protocol = load(RESULT_ROOT / "protocol.json")
    for name, expected in protocol["source_sha256"].items():
        path = Path(name)
        if not path.is_file() or sha(path) != expected:
            raise ValueError("frozen dependency changed: " + name)
    return protocol


def finite_model(model) -> bool:
    import torch
    return all(bool(torch.isfinite(parameter).all()) for parameter in model.parameters())


def run_condition(protocol: dict[str, Any], stream: int, condition: str,
                  device: str) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM
    from kernel_analyzer.compensation_control import CompensationControl
    from scripts import run_adamw8bit_error_compensation_training as shared

    seed = 419000 + stream
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
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
    optimizer = CompensationControl(
        model.parameters(), compensation_enabled=condition == "ON", block_size=256, **settings,
    )
    train_rows = load(Path(protocol["train_banks"][stream]))["states"]
    eval_rows = load(EVAL_BANK)["states"][:EVALUATION_STATES]
    evaluations = {"0": shared.evaluate(model, eval_rows, device)}
    losses = []
    torch.cuda.reset_peak_memory_stats(device); started = time.perf_counter(); model.train()
    failure = None
    for index, row in enumerate(train_rows):
        step = index + 1
        torch.manual_seed(1000000 * stream + 419000 + index)
        torch.cuda.manual_seed_all(1000000 * stream + 419000 + index)
        tokens = torch.tensor([row["token_ids"]], dtype=torch.long, device=device)
        optimizer.zero_grad(set_to_none=True)
        loss = model(input_ids=tokens, labels=tokens).loss
        if not torch.isfinite(loss):
            failure = {"kind": "NONFINITE_TRAINING_LOSS", "step": step}; break
        loss.backward(); optimizer.step(); losses.append(float(loss.detach().cpu()))
        if step in protocol["evaluation_steps"]:
            # A full parameter scan is deliberately restricted to the five
            # frozen evaluation checkpoints.  Scanning every parameter after
            # every update would materially change the cost of the experiment.
            if not finite_model(model):
                failure = {"kind": "NONFINITE_PARAMETER", "step": step}; break
            torch.cuda.synchronize()
            values = shared.evaluate(model, eval_rows, device)
            if not all(math.isfinite(value) for value in values):
                failure = {"kind": "NONFINITE_EVALUATION_LOSS", "step": step}; break
            evaluations[str(step)] = values; model.train()
        if step % 128 == 0:
            print(json.dumps({"event": "IID_CONFIRMATION", "stream": stream,
                              "condition": condition, "step": step}), flush=True)
    torch.cuda.synchronize(); elapsed = time.perf_counter() - started
    digest = hashlib.sha256()
    for parameter in model.parameters():
        digest.update(parameter.detach().float().cpu().contiguous().numpy().tobytes())
    result = {
        "schema": "adamw8bit-iid-training-run-v1",
        "status": "NUMERICAL_FAILURE" if failure else "COMPLETE",
        "stream": stream, "condition": condition,
        "training_loss": losses, "evaluation_loss_by_step": evaluations,
        "last_completed_step": len(losses), "failure": failure,
        "elapsed_seconds": elapsed,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "final_parameter_sha256": digest.hexdigest(),
        "all_parameters_finite_at_recording": finite_model(model),
    }
    del optimizer, model; torch.cuda.empty_cache()
    return result


def run(worker: int, workers: int, device: str) -> None:
    protocol = checked()
    tasks = [(stream, condition) for stream in range(STREAMS) for condition in CONDITIONS]
    for index, (stream, condition) in enumerate(tasks):
        if index % workers != worker:
            continue
        output = RESULT_ROOT / "runs" / f"stream_{stream:02d}_{condition}.json"
        if output.exists():
            continue
        try:
            save_new(output, run_condition(protocol, stream, condition, device))
        except Exception as error:
            attempt = RESULT_ROOT / "execution_failures" / (
                f"stream_{stream:02d}_{condition}_{time.time_ns()}.json"
            )
            save_new(attempt, {
                "schema": "adamw8bit-iid-training-execution-failure-v1",
                "status": "EXECUTION_FAILURE", "stream": stream, "condition": condition,
                "error_type": type(error).__name__, "error": str(error),
                "protocol_sha256": sha(RESULT_ROOT / "protocol.json"),
            })
            print(json.dumps({"event": "EXECUTION_FAILURE", "stream": stream,
                              "condition": condition, "error": str(error)}), flush=True)


def interval(values: list[float]) -> list[float]:
    import scipy.stats
    center = math.fsum(values) / len(values)
    variance = math.fsum((value - center) ** 2 for value in values) / (len(values) - 1)
    if variance == 0.0:
        return [center, center]
    half = float(scipy.stats.t.ppf(0.975, len(values) - 1)) * math.sqrt(variance / len(values))
    return [center - half, center + half]


def summarize() -> None:
    protocol = checked(); endpoint = str(STEPS)
    tasks = []; paired = []
    for stream in range(STREAMS):
        records = {}
        for condition in CONDITIONS:
            path = RESULT_ROOT / "runs" / f"stream_{stream:02d}_{condition}.json"
            record = load(path) if path.exists() else None
            records[condition] = record
            tasks.append({"stream": stream, "condition": condition,
                          "status": record["status"] if record else "NOT_COMPLETED"})
        if all(records[name] and records[name]["status"] == "COMPLETE" for name in CONDITIONS):
            means = {}
            for condition in CONDITIONS:
                values = records[condition]["evaluation_loss_by_step"].get(endpoint, [])
                if len(values) != EVALUATION_STATES or not all(math.isfinite(v) for v in values):
                    raise ValueError("a COMPLETE record lacks its finite endpoint")
                means[condition] = math.fsum(values) / len(values)
            paired.append(means["OFF"] - means["ON"])
    counts = dict(Counter(row["status"] for row in tasks))
    all_finite = len(paired) == STREAMS
    all_tasks_terminal = sum(counts.get(name, 0) for name in ("COMPLETE", "NUMERICAL_FAILURE")) == 2 * STREAMS
    primary = {
        "decision": "NOT_ASSESSED_DUE_TO_NUMERICAL_FAILURE" if counts.get("NUMERICAL_FAILURE")
        else "NOT_ASSESSED_INCOMPLETE" if not all_finite else None,
        "paired_finite_count": len(paired),
        "paired_values": paired,
    }
    if len(paired) >= 2:
        ci = interval(paired)
        primary["complete_case_description"] = {
            "conditioning": "BOTH_CONDITIONS_COMPLETED_WITH_FINITE_ENDPOINT",
            "mean": math.fsum(paired) / len(paired), "interval_95": ci,
            "sample_variance": (
                math.fsum((value - math.fsum(paired) / len(paired)) ** 2 for value in paired)
                / (len(paired) - 1)
            ),
            "zero_sample_variance_does_not_prove_zero_population_variance": True,
            "is_unconditional_primary_result": all_finite,
        }
        if all_finite:
            primary["decision"] = "MATERIAL_IMPROVEMENT" if ci[0] > MATERIAL_MARGIN else (
                "DETECTABLE_IMPROVEMENT" if ci[0] > 0.0 else "NOT_CONFIRMED"
            )
    save_new(RESULT_ROOT / "summary.json", {
        "schema": protocol["schema"],
        "status": (
            "COMPLETE" if all_finite else
            "COMPLETE_WITH_NUMERICAL_FAILURES" if all_tasks_terminal else
            "PARTIAL"
        ),
        "task_counts": counts, "tasks": tasks, "primary": primary,
        "statistical_unit": protocol["independent_unit"],
        "data_use": protocol["data_use"], "scope": protocol["not_claimed"],
    })
    print(json.dumps({"task_counts": counts, "primary": primary}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("build-banks", "freeze", "run", "summarize"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--worker", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.action == "build-banks": build_banks()
    elif args.action == "freeze": freeze()
    elif args.action == "run": run(args.worker, args.workers, args.device)
    else: summarize()


if __name__ == "__main__":
    main()
