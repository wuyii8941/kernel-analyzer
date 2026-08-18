#!/usr/bin/env python3
"""Confirm discovery-selected structured carriers on 32 disjoint states."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
from pathlib import Path
from typing import Any

if __package__:
    from scripts.long_horizon_trigger import atomic_json, build_model, load_eval_states, load_milestone, run_backward, under_root
else:
    from long_horizon_trigger import atomic_json, build_model, load_eval_states, load_milestone, run_backward, under_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discovery", type=Path, default=Path("results/final/structured_carrier_trigger.json"))
    parser.add_argument("--bank", type=Path, default=Path("results/final/long_horizon_bank.json"))
    parser.add_argument("--model", type=Path, default=Path("/data1/tzh/models/Qwen/Qwen3-1.7B"))
    parser.add_argument("--output", type=Path, default=Path("results/final/structured_carrier_confirmation.json"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bootstrap", type=int, default=10000)
    return parser.parse_args()


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def take_block(value: Any, trigger: dict[str, Any]) -> Any:
    level = trigger["level"]
    if level == "PARAMETER":
        return value.reshape(-1)
    if level == "VECTOR_BLOCK":
        return value[trigger["start"] : trigger["stop"]].reshape(-1)
    if level == "MATRIX_TILE":
        return value[
            trigger["row_start"] : trigger["row_stop"],
            trigger["column_start"] : trigger["column_stop"],
        ].reshape(-1)
    raise ValueError(f"unknown block level: {level}")


def binomial_upper_tail(successes: int, trials: int) -> float:
    return sum(math.comb(trials, value) for value in range(successes, trials + 1)) / (2**trials)


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def main() -> None:
    args = parse_args()
    discovery_path = under_root(args.discovery, "discovery")
    bank_path = under_root(args.bank, "bank")
    model_path = under_root(args.model, "model")
    output_path = under_root(args.output, "output")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HF_HOME", "/data1/tzh/cache/huggingface")
    os.environ.setdefault("HF_DATASETS_CACHE", "/data1/tzh/cache/huggingface/datasets")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/data1/tzh/cache/huggingface/transformers")
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/data1/tzh/cache/huggingface/hub")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/data1/tzh/cache/kernel_analyzer/structured_trigger_compile")

    import torch
    from torch._dynamo.backends.registry import lookup_backend
    from transformers import AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device(args.device)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    discovery = json.loads(discovery_path.read_text())
    bank = json.loads(bank_path.read_text())
    if discovery["status"] != "COMPLETE" or discovery["trigger_count"] < 1:
        raise RuntimeError("discovery artifact has no completed triggers")
    triggers = discovery["triggers"]
    selected_parameters = sorted({row["parameter"] for row in triggers})
    milestones = bank["milestones"]
    by_step = {int(row["step"]): row for row in milestones}
    required_steps = [0, 64, 256, 1024, 2048, 4096]
    if sorted(by_step) != required_steps:
        raise RuntimeError("milestone grid mismatch")

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
    all_states, all_evaluation = load_eval_states(tokenizer, 1024, 40, device)
    calibration_states = all_states[:8]
    confirmation_states = all_states[8:]
    confirmation_evaluation = {
        **all_evaluation,
        "states": 32,
        "offsets": all_evaluation["offsets"][8:],
        "token_sha256": all_evaluation["token_sha256"][8:],
        "disjoint_from_discovery_offsets": all_evaluation["offsets"][:8],
    }
    model = build_model(model_path, device)
    model_names = {name for name, _ in model.named_parameters()}
    if not set(selected_parameters) <= model_names:
        raise RuntimeError("discovery parameter is absent from model")

    class LossStep(torch.nn.Module):
        def __init__(self, subject: Any) -> None:
            super().__init__()
            self.subject = subject

        def forward(self, input_ids: Any, labels: Any) -> Any:
            return self.subject(input_ids=input_ids, labels=labels, use_cache=False, return_dict=False)[0]

    audit = {"backend_compiles": 0, "runtime_invocations": 0, "graph_sha256": []}
    inductor = lookup_backend("inductor")

    def backend(graph_module: Any, example_inputs: list[Any]) -> Any:
        audit["backend_compiles"] += 1
        audit["graph_sha256"].append(hashlib.sha256(graph_module.code.encode()).hexdigest())
        compiled = inductor(graph_module, example_inputs)

        def counted(*values: Any) -> Any:
            audit["runtime_invocations"] += 1
            return compiled(*values)
        return counted

    candidate = torch.compile(LossStep(model), backend=backend, fullgraph=True, dynamic=False)
    model.zero_grad(set_to_none=True)
    warm = candidate(calibration_states[0][0], calibration_states[0][1])
    warm.backward()
    torch.cuda.synchronize(device)
    del warm

    pilot = {name: None for name in selected_parameters}
    for state_id, inputs in enumerate(calibration_states):
        _, reference = run_backward(model, inputs)
        _, observed = run_backward(model, inputs, candidate)
        for name in selected_parameters:
            delta = observed[name].float() - reference[name].float()
            if pilot[name] is None:
                pilot[name] = torch.zeros_like(delta)
            pilot[name].add_(delta)
        print(json.dumps({"calibration_state": state_id}), flush=True)
        del reference, observed
    for value in pilot.values():
        value.div_(len(calibration_states))

    result: dict[str, Any] = {
        "schema": "kernel-analyzer-structured-carrier-confirmation-v1",
        "status": "RUNNING",
        "sources": {"discovery": str(discovery_path), "discovery_sha256": digest(discovery_path), "bank": str(bank_path), "bank_sha256": digest(bank_path)},
        "family_size": len(triggers),
        "selected_parameter_count": len(selected_parameters),
        "calibration": {"checkpoint_step": 0, "state_offsets": all_evaluation["offsets"][:8], "candidate_repeats": 1},
        "confirmation_evaluation": confirmation_evaluation,
        "measurements": [],
        "reference_repeat_checks": [],
        "completed_steps": [],
        "full_gradient_tensors_saved": False,
    }

    measurements: dict[int, dict[int, list[tuple[float, float]]]] = {
        index: {} for index in range(len(triggers))
    }
    for step in required_steps[1:]:
        print(f"confirmation step {step}: loading weights", flush=True)
        load_milestone(model, by_step[step], model_path)
        for state_id, inputs in enumerate(confirmation_states):
            reference_loss, reference = run_backward(model, inputs)
            candidate_loss, observed = run_backward(model, inputs, candidate)
            if state_id == 0:
                repeated_loss, repeated = run_backward(model, inputs)
                repeat_exact = all(
                    not bool(torch.count_nonzero(repeated[name].float() - reference[name].float()))
                    for name in model_names
                )
                result["reference_repeat_checks"].append({
                    "step": step, "state_id": state_id, "loss_delta": repeated_loss - reference_loss,
                    "all_parameter_gradients_exact": repeat_exact,
                })
                del repeated
            deltas = {
                name: observed[name].float() - reference[name].float()
                for name in selected_parameters
            }
            for index, trigger in enumerate(triggers):
                name = trigger["parameter"]
                current = take_block(deltas[name], trigger)
                direction = take_block(pilot[name], trigger)
                dot = float(torch.dot(current, direction))
                direction_norm = float(direction.norm())
                current_norm = float(current.norm())
                projection = dot / direction_norm if direction_norm > 0.0 else float("nan")
                cosine = dot / (direction_norm * current_norm) if direction_norm > 0.0 and current_norm > 0.0 else float("nan")
                measurements[index].setdefault(step, []).append((projection, cosine))
            result["measurements"].append({
                "step": step, "state_id": state_id,
                "reference_loss": reference_loss, "candidate_loss": candidate_loss,
                "loss_delta": candidate_loss - reference_loss,
            })
            print(json.dumps({"step": step, "confirmation_state": state_id, "loss_delta": candidate_loss - reference_loss}), flush=True)
            del reference, observed, deltas
        result["completed_steps"].append(step)
        result["compile_audit"] = audit
        atomic_json(output_path, result)
        gc.collect()

    rows = []
    for index, trigger in enumerate(triggers):
        sign = 1.0 if trigger["direction_relative_to_step0_pilot"] == "POSITIVE" else -1.0
        steps = trigger["best_window"]["steps"]
        checkpoint_rows = []
        per_state = [0.0] * len(confirmation_states)
        finite = True
        for step in steps:
            values = measurements[index][step]
            signed_projections = [sign * value[0] for value in values]
            signed_cosines = [sign * value[1] for value in values]
            finite = finite and all(math.isfinite(value) for value in signed_projections + signed_cosines)
            for state_id, value in enumerate(signed_projections):
                per_state[state_id] += value / len(steps)
            checkpoint_rows.append({
                "step": step,
                "signed_projection_mean": sum(signed_projections) / len(signed_projections),
                "signed_cosine_mean": sum(signed_cosines) / len(signed_cosines),
            })
        successes = sum(value > 0.0 for value in per_state)
        sign_p = binomial_upper_tail(successes, len(per_state))
        generator = random.Random(int.from_bytes(hashlib.sha256(f"confirm:{index}".encode()).digest()[:8], "little"))
        boot = [
            sum(per_state[generator.randrange(len(per_state))] for _ in per_state) / len(per_state)
            for _ in range(args.bootstrap)
        ]
        rows.append({
            "trigger_index": index,
            "trigger": trigger,
            "finite": finite,
            "checkpoint_rows": checkpoint_rows,
            "all_checkpoint_signed_projection_means_positive": all(row["signed_projection_mean"] > 0.0 for row in checkpoint_rows),
            "minimum_checkpoint_signed_mean_cosine": min(row["signed_cosine_mean"] for row in checkpoint_rows),
            "positive_state_count": successes,
            "state_count": len(per_state),
            "one_sided_exact_sign_p": sign_p,
            "state_mean_signed_projection": sum(per_state) / len(per_state),
            "cluster_bootstrap_lower_95": percentile(boot, 0.025),
        })

    # Holm step-down family-wise error control over all frozen discovery arms.
    ordered = sorted(rows, key=lambda row: (row["one_sided_exact_sign_p"], row["trigger_index"]))
    holm_open = True
    for rank, row in enumerate(ordered, start=1):
        threshold = 0.05 / (len(ordered) - rank + 1)
        rejected = holm_open and row["one_sided_exact_sign_p"] <= threshold
        if not rejected:
            holm_open = False
        row["holm_rank"] = rank
        row["holm_threshold"] = threshold
        row["holm_rejected"] = rejected
    for row in rows:
        row["confirmed"] = bool(
            row["finite"]
            and row["all_checkpoint_signed_projection_means_positive"]
            and row["minimum_checkpoint_signed_mean_cosine"] > 0.1
            and row["cluster_bootstrap_lower_95"] > 0.0
            and row["holm_rejected"]
        )

    result["gate"] = {
        "independent_state_count": 32,
        "all_preregistered_checkpoint_means_signed_positive": True,
        "minimum_checkpoint_signed_mean_cosine": 0.1,
        "cluster_bootstrap_lower_95_positive": True,
        "holm_family_wise_alpha": 0.05,
        "one_sided_exact_sign_test": True,
        "causal_intervention_required_after_confirmation": True,
    }
    result["rows"] = rows
    result["confirmed_count"] = sum(row["confirmed"] for row in rows)
    result["confirmed_trigger_indices"] = [row["trigger_index"] for row in rows if row["confirmed"]]
    result["confirmed_parameters"] = sorted({row["trigger"]["parameter"] for row in rows if row["confirmed"]})
    result["natural_case_added"] = False
    result["status"] = "COMPLETE"
    result["result_sha256"] = hashlib.sha256(json.dumps(result, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    atomic_json(output_path, result)
    print(json.dumps({"status": result["status"], "confirmed_count": result["confirmed_count"], "confirmed_parameters": result["confirmed_parameters"]}), flush=True)


if __name__ == "__main__":
    main()
