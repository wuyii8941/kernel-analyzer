#!/usr/bin/env python3
"""Reproduce the original optimizer order before comparing final parameters."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

from kernel_analyzer.training_equivalence import exact_binomial_one_sided_bounds
from scripts.run_optimizer_training_confirmation import (
    CONDITIONS,
    EVAL_BANK,
    MODEL,
    condition_order,
    load,
    make_optimizer,
)


ROOT = Path(__file__).resolve().parents[1]
PRIOR_ROOT = ROOT / "results/property/numerical_coverage_v1/mamba_adamw8bit_training_confirmation_v1"
PRIOR_VERIFICATION = PRIOR_ROOT / "verification.json"
INVALID_V1 = ROOT / "results/property/numerical_coverage_v1/adamw8bit_trajectory_response_audit_v1/invalid_design.json"
TRAIN_BANKS = tuple(ROOT / (
    "results/property/numerical_coverage_v1/"
    f"mamba_adamw8bit_confirm_bank_{index:02d}_v1.json"
) for index in range(8))
CHECKPOINTS = (0, 256, 512, 768, 1024)


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


def freeze(output: Path) -> None:
    if output.exists():
        raise ValueError("freeze requires a new output directory")
    if load(INVALID_V1).get("status") != "INVALID_EXECUTION_DESIGN":
        raise ValueError("the reason for replacing v1 is not recorded")
    prior = load(PRIOR_VERIFICATION)
    if prior.get("status") != "VERIFIED" or prior.get("stream_count") != 8:
        raise ValueError("prior training confirmation is unavailable")
    sources = [
        Path(__file__).resolve(),
        ROOT / "scripts/run_optimizer_training_confirmation.py",
        ROOT / "src/kernel_analyzer/training_equivalence.py",
        PRIOR_ROOT / "protocol.json", PRIOR_VERIFICATION, INVALID_V1,
        EVAL_BANK, MODEL / "config.json", MODEL / "model.safetensors", *TRAIN_BANKS,
    ]
    protocol = {
        "schema": "adamw8bit-trajectory-response-audit-v2",
        "status": "FROZEN_AFTER_V1_EXECUTION_DESIGN_REJECTION",
        "data_use": "RESULT_AWARE_TRAJECTORY_DIAGNOSTIC_NOT_BLIND_CONFIRMATION",
        "question": (
            "After exactly reproducing the original condition execution order, is the "
            "block64 final parameter vector closer to FP32 than block256?"
        ),
        "streams": 8,
        "steps": 1024,
        "evaluation_checkpoints": list(CHECKPOINTS),
        "conditions": list(CONDITIONS),
        "condition_order_by_stream": {
            str(index): list(condition_order(index)) for index in range(8)
        },
        "execution_rule": (
            "one condition trains to completion and is released before the next; order "
            "exactly matches the original frozen training confirmation"
        ),
        "reproduction_gate": (
            "every final 32-state mean evaluation loss must exactly equal the prior "
            "verified value before parameter-distance results are accepted"
        ),
        "primary_diagnostic": (
            "exact sign probability for final full-model parameter distance: "
            "block64-to-FP32 < block256-to-FP32"
        ),
        "training_banks": [str(path) for path in TRAIN_BANKS],
        "evaluation_bank": str(EVAL_BANK),
        "evaluation_states": 32,
        "optimizer": {"lr": 1e-4, "betas": [0.9, 0.999], "eps": 1e-8,
                      "weight_decay": 0.01},
        "not_claimed": ["NEW_TRAINING_OUTCOME", "UNIQUE_LOSS_CAUSE"],
        "source_sha256": {str(path.resolve()): sha(path) for path in sources},
    }
    save_new(output / "protocol.json", protocol)


def verify_protocol(output: Path) -> dict:
    protocol = load(output / "protocol.json")
    if protocol.get("schema") != "adamw8bit-trajectory-response-audit-v2":
        raise ValueError("unexpected protocol")
    for name, expected in protocol["source_sha256"].items():
        path = Path(name)
        if not path.is_file() or sha(path) != expected:
            raise ValueError("frozen source or input changed: " + name)
    return protocol


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


def run_condition(protocol: dict, stream_index: int, condition: str, device: str):
    import torch
    from transformers import AutoModelForCausalLM

    seed = 92000 + stream_index
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
    train_rows = load(TRAIN_BANKS[stream_index])["states"]
    eval_rows = load(EVAL_BANK)["states"][:protocol["evaluation_states"]]
    evaluations = {"0": evaluate(model, eval_rows, device)}
    model.train()
    losses = []
    started = time.perf_counter()
    for index, row in enumerate(train_rows):
        step = index + 1
        step_seed = 1_000_000 * stream_index + 81_000 + index
        torch.manual_seed(step_seed)
        torch.cuda.manual_seed_all(step_seed)
        tokens = torch.tensor([row["token_ids"]], dtype=torch.long, device=device)
        optimizer.zero_grad(set_to_none=True)
        loss = model(input_ids=tokens, labels=tokens).loss
        if not torch.isfinite(loss):
            raise RuntimeError(f"nonfinite loss for {condition} at step {step}")
        loss.backward(); optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if step in CHECKPOINTS:
            evaluations[str(step)] = evaluate(model, eval_rows, device)
            model.train()
        if step % 128 == 0:
            print(json.dumps({"event": "TRAJECTORY_V2_STEP", "stream": stream_index,
                              "condition": condition, "step": step}), flush=True)
    final_parameters = [parameter.detach().float().cpu().clone()
                        for parameter in model.parameters()]
    record = {
        "condition": condition,
        "training_loss": losses,
        "evaluation_loss_by_step": evaluations,
        "elapsed_seconds": time.perf_counter() - started,
    }
    del optimizer, model
    torch.cuda.empty_cache()
    return record, final_parameters


def distance(left, right) -> dict[str, float]:
    difference = math.fsum(float((a.double() - b.double()).square().sum())
                           for a, b in zip(left, right))
    reference = math.fsum(float(b.double().square().sum()) for b in right)
    return {"l2": math.sqrt(difference), "relative_l2": math.sqrt(difference / reference)}


def run_stream(output: Path, stream_index: int, device: str) -> None:
    protocol = verify_protocol(output)
    prior = load(PRIOR_VERIFICATION)["stream_rows"][stream_index]
    records = {}
    parameters = {}
    for condition in condition_order(stream_index):
        records[condition], parameters[condition] = run_condition(
            protocol, stream_index, condition, device
        )
    final_means = {
        condition: sum(records[condition]["evaluation_loss_by_step"]["1024"])
        / len(records[condition]["evaluation_loss_by_step"]["1024"])
        for condition in CONDITIONS
    }
    if final_means != prior["final_mean_evaluation_loss"]:
        raise RuntimeError(
            "final loss reproduction gate failed: "
            + json.dumps({"observed": final_means,
                          "expected": prior["final_mean_evaluation_loss"]}, sort_keys=True)
        )
    reference = parameters["FP32_ADAMW"]
    distances = {
        condition: distance(parameters[condition], reference)
        for condition in ("ADAMW8BIT_BLOCK64", "ADAMW8BIT_BLOCK256")
    }
    save_new(output / "streams" / f"stream-{stream_index:02d}.json", {
        "schema": "adamw8bit-trajectory-response-stream-v2",
        "status": "COMPLETE_REPRODUCED_PRIOR_TRAINING",
        "stream_index": stream_index,
        "condition_execution_order": list(condition_order(stream_index)),
        "final_mean_evaluation_loss": final_means,
        "prior_final_mean_evaluation_loss": prior["final_mean_evaluation_loss"],
        "final_parameter_distance_to_fp32": distances,
        "final_block64_parameter_closer": (
            distances["ADAMW8BIT_BLOCK64"]["relative_l2"]
            < distances["ADAMW8BIT_BLOCK256"]["relative_l2"]
        ),
        "records": records,
    })


def summarize(output: Path, protocol: dict) -> dict:
    rows = [load(output / "streams" / f"stream-{index:02d}.json")
            for index in range(protocol["streams"])]
    successes = sum(row["final_block64_parameter_closer"] for row in rows)
    bounds = exact_binomial_one_sided_bounds(successes, len(rows), alpha=0.05)
    return {
        "schema": "adamw8bit-trajectory-response-summary-v2",
        "status": "COMPLETE_RESULT_AWARE_DIAGNOSTIC",
        "reproduction_gate": "PASSED_ALL_STREAMS",
        "stream_count": len(rows),
        "block64_final_parameter_closer_count": successes,
        "one_sided_probability_bounds": list(bounds),
        "parameter_proximity_result": (
            "CONFIRMED_MORE_OFTEN_THAN_HALF" if bounds[0] > 0.5 else "NOT_CONFIRMED"
        ),
        "mean_final_relative_parameter_distance": {
            condition: math.fsum(
                row["final_parameter_distance_to_fp32"][condition]["relative_l2"]
                for row in rows
            ) / len(rows)
            for condition in ("ADAMW8BIT_BLOCK64", "ADAMW8BIT_BLOCK256")
        },
        "interpretation": (
            "Result-aware parameter-trajectory diagnostic after exact reproduction of "
            "the prior loss endpoint; not a new training confirmation or unique cause."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "run", "summarize"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stream-index", type=int)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(Path("/data1/tzh")):
        parser.error("output must be under /data1/tzh")
    if args.action == "freeze":
        freeze(args.output)
    elif args.action == "run":
        if args.stream_index is None or not 0 <= args.stream_index < 8:
            parser.error("run requires --stream-index in [0, 7]")
        run_stream(args.output, args.stream_index, args.device)
    else:
        save_new(args.output / "summary.json", summarize(args.output, verify_protocol(args.output)))


if __name__ == "__main__":
    main()
