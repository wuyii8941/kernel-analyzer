#!/usr/bin/env python3
"""Screen all parameter-gradient carriers across frozen long-horizon states.

The primary candidate is a real full-step Inductor program compiled from the
same eager mathematical model.  A fixed step-0/state-0 coordinate direction is
the only pilot.  All later checkpoint/state/repeat measurements are held out.
No full gradient or candidate tensor is written to disk.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any


ROOT = Path("/data1/tzh").resolve()


def under_root(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if ROOT not in (resolved, *resolved.parents):
        raise ValueError(f"{label} must stay under {ROOT}: {resolved}")
    return resolved


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, default=Path("results/final/long_horizon_bank.json"))
    parser.add_argument("--model", type=Path, default=Path("/data1/tzh/models/Qwen/Qwen3-1.7B"))
    parser.add_argument("--output", type=Path, default=Path("results/final/long_horizon_trigger.json"))
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--states", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--summarize-only", action="store_true")
    return parser.parse_args()


def fixed_indices(name: str, size: int, count: int, torch: Any) -> Any:
    count = min(count, size)
    seed = int.from_bytes(hashlib.sha256(f"{name}:trigger".encode()).digest()[:8], "little") % (2**31)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    if size <= 4096:
        return torch.randperm(size, generator=generator)[:count]
    selected: set[int] = set()
    while len(selected) < count:
        draws = torch.randint(0, size, (2 * (count - len(selected)),), generator=generator)
        selected.update(int(value) for value in draws)
    return torch.tensor(sorted(selected)[:count], dtype=torch.long)


def load_eval_states(tokenizer: Any, seq_len: int, count: int, device: Any) -> tuple[list[Any], dict[str, Any]]:
    import torch
    from datasets import load_dataset

    dataset = load_dataset(
        "Salesforce/wikitext", "wikitext-103-raw-v1", split="validation", revision="main",
        download_mode="reuse_dataset_if_exists",
    )
    required = count * seq_len
    tokens: list[int] = []
    documents = 0
    for row in dataset:
        text = str(row["text"]).strip()
        if text:
            documents += 1
            tokens.extend(tokenizer(text, add_special_tokens=False, return_attention_mask=False)["input_ids"])
        if len(tokens) >= required:
            break
    if len(tokens) < required:
        raise RuntimeError("validation stream is too short")
    states = []
    hashes = []
    for state in range(count):
        block = tokens[state * seq_len : (state + 1) * seq_len]
        hashes.append(hashlib.sha256(json.dumps(block, separators=(",", ":")).encode()).hexdigest())
        tensor = torch.tensor([block], dtype=torch.long, device=device)
        # AutoModelForCausalLM shifts once internally.  Pre-shifted labels are
        # forbidden because they caused the superseded short bank's two-token target.
        states.append((tensor, tensor))
    return states, {
        "split": "validation",
        "revision": "main",
        "states": count,
        "seq_len": seq_len,
        "offsets": [state * seq_len for state in range(count)],
        "token_sha256": hashes,
        "documents_consumed": documents,
        "causal_label_alignment": "labels_equal_input_ids; exactly one internal model shift",
    }


def build_model(model_path: Path, device: Any) -> Any:
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, attn_implementation="eager", local_files_only=True
    ).to(device)
    model.config.use_cache = False
    model.eval()
    return model


def load_milestone(model: Any, row: dict[str, Any], model_path: Path) -> None:
    if int(row["step"]) == 0:
        # The model was constructed from the immutable local source checkpoint.
        return
    from safetensors.torch import load_file

    state = load_file(str(under_root(Path(row["path"]), "milestone")), device="cpu")
    incompatible = model.load_state_dict(state, strict=True, assign=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"milestone mismatch: {incompatible}")
    del state


def run_backward(model: Any, inputs: tuple[Any, Any], candidate: Any | None = None) -> tuple[float, dict[str, Any]]:
    model.zero_grad(set_to_none=True)
    input_ids, labels = inputs
    loss = (
        model(input_ids=input_ids, labels=labels, use_cache=False, return_dict=False)[0]
        if candidate is None
        else candidate(input_ids, labels)
    )
    loss.backward()
    gradients = {
        name: parameter.grad.detach().cpu().clone() if parameter.grad is not None else None
        for name, parameter in model.named_parameters()
    }
    return float(loss.detach().cpu()), gradients


def compare_gradients(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    *,
    indices: dict[str, Any],
    pilot: dict[str, Any] | None,
    capture_pilot: bool,
    torch: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    new_pilot: dict[str, Any] = {}
    for name in sorted(set(reference) | set(candidate)):
        left, right = reference.get(name), candidate.get(name)
        if left is None or right is None:
            rows.append({"name": name, "status": "MISSING"})
            continue
        left_f = left.reshape(-1).float()
        right_f = right.reshape(-1).float()
        delta = right_f - left_f
        sample = delta.index_select(0, indices[name])
        sample_norm = float(sample.norm())
        if capture_pilot:
            new_pilot[name] = sample / sample.norm().clamp_min(1e-30)
        basis = new_pilot.get(name) if capture_pilot else (pilot or {}).get(name)
        projection = float(torch.dot(sample, basis)) if basis is not None else None
        cosine = projection / (sample_norm + 1e-30) if projection is not None else None
        delta_l2 = float(delta.norm())
        rows.append({
            "name": name,
            "status": "OK",
            "numel": delta.numel(),
            "delta_l2": delta_l2,
            "delta_rms": delta_l2 / math.sqrt(delta.numel()),
            "delta_max_abs": float(delta.abs().max()),
            "delta_mean": float(delta.mean()),
            "nonzero": int(torch.count_nonzero(delta)),
            "pilot_projection": projection,
            "pilot_cosine": cosine,
            "sample_l2": sample_norm,
        })
    return rows, new_pilot


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize(result: dict[str, Any], bootstrap: int) -> dict[str, Any]:
    import random

    heldout_steps = [64, 256, 1024, 2048, 4096]
    by_name: dict[str, list[dict[str, Any]]] = {}
    for row in result["rows"]:
        if row["checkpoint_step"] == 0:
            continue
        for parameter in row["parameters"]:
            if parameter["status"] == "OK":
                by_name.setdefault(parameter["name"], []).append({
                    "step": row["checkpoint_step"],
                    "state": row["state_id"],
                    "repeat": row["repeat"],
                    "projection": parameter["pilot_projection"],
                    "cosine": parameter["pilot_cosine"],
                    "delta_l2": parameter["delta_l2"],
                })
    summaries = []
    for name, values in by_name.items():
        checkpoint_rows = []
        for step in heldout_steps:
            selected = [row for row in values if row["step"] == step]
            state_means = []
            for state in range(result["evaluation"]["states"]):
                repeats = [row for row in selected if row["state"] == state]
                state_means.append(sum(row["projection"] for row in repeats) / len(repeats))
            checkpoint_rows.append({
                "step": step,
                "mean_projection": sum(state_means) / len(state_means),
                "mean_cosine": sum(row["cosine"] for row in selected) / len(selected),
                "state_mean_projections": state_means,
            })
        positive = [row["mean_projection"] > 0.0 for row in checkpoint_rows]
        longest = current = 0
        for value in positive:
            current = current + 1 if value else 0
            longest = max(longest, current)
        best_window = None
        generator = random.Random(int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "little"))
        for length in (5, 4):
            for start in range(0, len(checkpoint_rows) - length + 1):
                window = checkpoint_rows[start : start + length]
                if not all(row["mean_projection"] > 0.0 for row in window):
                    continue
                samples = []
                for _ in range(bootstrap):
                    state_ids = [generator.randrange(result["evaluation"]["states"]) for _ in range(result["evaluation"]["states"])]
                    samples.append(sum(row["state_mean_projections"][state] for row in window for state in state_ids) / (length * len(state_ids)))
                candidate = {
                    "steps": [row["step"] for row in window],
                    "projection_mean": sum(row["mean_projection"] for row in window) / length,
                    "projection_cluster_bootstrap_lower_95": percentile(samples, 0.025),
                    "minimum_checkpoint_mean_cosine": min(row["mean_cosine"] for row in window),
                }
                if best_window is None or candidate["projection_cluster_bootstrap_lower_95"] > best_window["projection_cluster_bootstrap_lower_95"]:
                    best_window = candidate
        repeat_pairs = {}
        for row in values:
            repeat_pairs.setdefault((row["step"], row["state"]), []).append(row["projection"])
        sign_agreement = sum(
            len(pair) == result["repeats"]
            and all((value > 0.0) == (pair[0] > 0.0) for value in pair[1:])
            for pair in repeat_pairs.values()
        ) / len(repeat_pairs)
        triggered = bool(
            longest >= 4
            and best_window is not None
            and best_window["projection_cluster_bootstrap_lower_95"] > 0.0
            and best_window["minimum_checkpoint_mean_cosine"] > 0.1
            and sign_agreement >= 0.875
        )
        summaries.append({
            "name": name,
            "longest_positive_checkpoint_run": longest,
            "repeat_positive_agreement_fraction": sign_agreement,
            "best_window": best_window,
            "triggered": triggered,
        })
    summaries.sort(key=lambda row: (
        not row["triggered"],
        -(row["best_window"]["projection_cluster_bootstrap_lower_95"] if row["best_window"] else -math.inf),
        row["name"],
    ))
    return {
        "parameters": summaries,
        "triggered_parameters": [row["name"] for row in summaries if row["triggered"]],
        "trigger_count": sum(row["triggered"] for row in summaries),
        "gate": {
            "consecutive_heldout_checkpoints": 4,
            "cluster_bootstrap_lower_bound_positive": True,
            "minimum_mean_sketch_cosine": 0.1,
            "repeat_positive_agreement_fraction": 0.875,
            "causal_intervention_required_after_screen": True,
        },
    }


def main() -> None:
    args = parse_args()
    if args.seq_len != 1024 or args.states < 2 or args.repeats != 2:
        raise ValueError("frozen trigger campaign requires seq1024, at least two states, and exactly two repeats")
    bank_path = under_root(args.bank, "bank")
    model_path = under_root(args.model, "model")
    output_path = under_root(args.output, "output")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.summarize_only:
        result = json.loads(output_path.read_text())
        if result.get("schema") != "kernel-analyzer-long-horizon-trigger-v1":
            raise ValueError("trigger artifact schema mismatch")
        result["summary"] = summarize(result, args.bootstrap)
        result["status"] = "COMPLETE"
        result.pop("result_sha256", None)
        result["result_sha256"] = hashlib.sha256(
            json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        atomic_json(output_path, result)
        print(json.dumps({
            "status": result["status"],
            "trigger_count": result["summary"]["trigger_count"],
            "triggered_parameters": result["summary"]["triggered_parameters"][:20],
        }), flush=True)
        return

    os.environ.setdefault("HF_HOME", "/data1/tzh/cache/huggingface")
    os.environ.setdefault("HF_DATASETS_CACHE", "/data1/tzh/cache/huggingface/datasets")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/data1/tzh/cache/huggingface/transformers")
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/data1/tzh/cache/huggingface/hub")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

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

    bank = json.loads(bank_path.read_text())
    if bank["status"] != "COMPLETE" or bank["completed_step"] != 4096:
        raise RuntimeError("long-horizon bank is incomplete")
    milestones = bank["milestones"]
    if [row["step"] for row in milestones] != [0, 64, 256, 1024, 2048, 4096]:
        raise RuntimeError("milestone grid mismatch")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
    states, evaluation = load_eval_states(tokenizer, args.seq_len, args.states, device)
    model = build_model(model_path, device)
    parameter_names = [name for name, _ in model.named_parameters()]
    parameter_numel = {name: parameter.numel() for name, parameter in model.named_parameters()}
    indices = {name: fixed_indices(name, parameter_numel[name], 256, torch) for name in parameter_names}

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
    # Compile before any measurement.  Warmup values are never used to select
    # a carrier, direction, checkpoint, or state.
    model.zero_grad(set_to_none=True)
    warm = candidate(states[0][0], states[0][1])
    warm.backward()
    torch.cuda.synchronize(device)
    del warm

    result: dict[str, Any] = {
        "schema": "kernel-analyzer-long-horizon-trigger-v1",
        "subject": "Qwen3-1.7B full-step eager versus Inductor",
        "bank": str(bank_path),
        "bank_protocol_sha256": bank["protocol_sha256"],
        "evaluation": evaluation,
        "repeats": args.repeats,
        "pilot": {"checkpoint_step": 0, "state_id": 0, "repeat": 0, "sample_coordinates_candidate_blind": True},
        "candidate": {"backend": "Inductor", "fullgraph": True, "changed_f_b_units": 889},
        "reference": {"backend": "eager", "dtype": "bfloat16", "tf32": False},
        "parameter_denominator": len(parameter_names),
        "rows": [],
        "reference_repeat_rows": [],
        "status": "RUNNING",
        "full_gradient_tensors_saved": False,
    }
    pilot: dict[str, Any] | None = None
    for milestone in milestones:
        step = int(milestone["step"])
        print(f"milestone {step}: loading weights", flush=True)
        load_milestone(model, milestone, model_path)
        reference_by_state = []
        for state_id, inputs in enumerate(states):
            loss0, reference = run_backward(model, inputs)
            loss1, repeated = run_backward(model, inputs)
            repeat_parameters, _ = compare_gradients(
                reference, repeated, indices=indices, pilot=None, capture_pilot=False, torch=torch
            )
            result["reference_repeat_rows"].append({
                "checkpoint_step": step,
                "state_id": state_id,
                "loss_delta": loss1 - loss0,
                "max_parameter_repeat_delta_l2": max(row.get("delta_l2", 0.0) for row in repeat_parameters),
                "all_parameter_repeat_exact": all(row.get("nonzero", 0) == 0 for row in repeat_parameters),
            })
            reference_by_state.append((loss0, reference))
            del repeated, repeat_parameters
        for state_id, inputs in enumerate(states):
            reference_loss, reference = reference_by_state[state_id]
            for repeat in range(args.repeats):
                candidate_loss, observed = run_backward(model, inputs, candidate)
                capture = pilot is None and step == 0 and state_id == 0 and repeat == 0
                parameters, new_pilot = compare_gradients(
                    reference, observed, indices=indices, pilot=pilot, capture_pilot=capture, torch=torch
                )
                if capture:
                    pilot = new_pilot
                result["rows"].append({
                    "checkpoint_step": step,
                    "state_id": state_id,
                    "repeat": repeat,
                    "reference_loss": reference_loss,
                    "candidate_loss": candidate_loss,
                    "loss_delta": candidate_loss - reference_loss,
                    "parameters": parameters,
                })
                print(json.dumps({
                    "step": step, "state": state_id, "repeat": repeat,
                    "loss_delta": candidate_loss - reference_loss,
                    "nonexact_parameters": sum(row.get("nonzero", 0) > 0 for row in parameters),
                }), flush=True)
                del observed, parameters
        del reference_by_state
        gc.collect()
        result["compile_audit"] = audit
        atomic_json(output_path, result)

    result["summary"] = summarize(result, args.bootstrap)
    result["status"] = "COMPLETE"
    result["natural_case_added"] = False
    result["case_claim_boundary"] = (
        "A trigger is only a carrier screen. A natural case is added only after exact-vector confirmation, "
        "single-region reference replacement, and negative repair."
    )
    result["result_sha256"] = hashlib.sha256(
        json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    atomic_json(output_path, result)
    print(json.dumps({
        "status": result["status"],
        "trigger_count": result["summary"]["trigger_count"],
        "triggered_parameters": result["summary"]["triggered_parameters"][:20],
    }), flush=True)


if __name__ == "__main__":
    main()
