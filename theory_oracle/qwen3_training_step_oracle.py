#!/usr/bin/env python
"""Materialized Qwen3 causal-LM/AdamW Training-Step Oracle v0.1."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


SCHEMA_VERSION = "forkcert.qwen3-training-step-oracle.v0.1"


@dataclass
class CompileAudit:
    backend_compiles: int = 0
    runtime_invocations: int = 0
    graph_code_sha256: list[str] = field(default_factory=list)
    graph_node_counts: list[int] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--count", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--initial-step-counter", type=int, default=0)
    parser.add_argument(
        "--candidate-counter-mode", choices=["correct", "stale"], default="correct",
        help="Independent exact-state mutation control for the compiled arm.",
    )
    return parser.parse_args()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_manifest(path: Path) -> dict[str, Any]:
    names = [
        "config.json", "model.safetensors", "tokenizer.json",
        "tokenizer_config.json", "generation_config.json",
    ]
    files = []
    for name in names:
        item = path / name
        if item.is_file():
            files.append(
                {"name": name, "size": item.stat().st_size, "sha256": sha256_file(item)}
            )
    return {"path": str(path.resolve()), "files": files}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row {line_number} is not an object")
            rows.append(value)
    return rows


def make_tracking_backend(torch: Any, audit: CompileAudit) -> Callable[..., Any]:
    from torch._dynamo.backends.registry import lookup_backend

    inductor = lookup_backend("inductor")

    def backend(graph_module: Any, example_inputs: list[Any]) -> Callable[..., Any]:
        audit.backend_compiles += 1
        audit.graph_code_sha256.append(sha256_text(graph_module.code))
        audit.graph_node_counts.append(sum(1 for _ in graph_module.graph.nodes))
        compiled = inductor(graph_module, example_inputs)

        def counted(*args: Any) -> Any:
            audit.runtime_invocations += 1
            return compiled(*args)

        return counted

    return backend


def cpu_model_snapshot(model: Any) -> dict[str, Any]:
    return {
        name: tensor.detach().to(device="cpu", copy=True)
        for name, tensor in model.state_dict().items()
    }


def restore_model(torch: Any, model: Any, baseline: dict[str, Any]) -> None:
    model.load_state_dict(baseline, strict=True)
    current = model.state_dict()
    if current.keys() != baseline.keys():
        raise RuntimeError("model-state structure changed during reset")
    for name, expected in baseline.items():
        observed = current[name]
        if observed.shape != expected.shape or observed.dtype != expected.dtype:
            raise RuntimeError(f"model-state reset structure failed for {name}")
        if not torch.equal(observed.detach().cpu(), expected):
            raise RuntimeError(f"model-state reset value failed for {name}")


def rng_snapshot(torch: Any) -> dict[str, Any]:
    return {
        "cpu": torch.get_rng_state().clone(),
        "cuda": torch.cuda.get_rng_state().clone(),
    }


def restore_rng(torch: Any, snapshot: dict[str, Any]) -> None:
    torch.set_rng_state(snapshot["cpu"])
    torch.cuda.set_rng_state(snapshot["cuda"])


def scalar_json(value: Any) -> Any:
    if isinstance(value, tuple):
        return [scalar_json(item) for item in value]
    if isinstance(value, (bool, int, float, str, type(None))):
        return value
    return repr(value)


def optimizer_options(optimizer: Any) -> list[dict[str, Any]]:
    return [
        {
            key: scalar_json(value)
            for key, value in group.items()
            if key != "params"
        }
        for group in optimizer.param_groups
    ]


def optimizer_snapshot(optimizer: Any, model: Any) -> dict[str, Any]:
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    state: dict[str, dict[str, Any]] = {}
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            name = names[id(parameter)]
            entry = optimizer.state.get(parameter, {})
            state[name] = {
                key: value.detach().to(device="cpu", copy=True)
                if hasattr(value, "detach") else value
                for key, value in entry.items()
            }
    return {"options": optimizer_options(optimizer), "state": state}


def gradient_signature(model: Any) -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "present": parameter.grad is not None,
            "shape": list(parameter.grad.shape) if parameter.grad is not None else None,
            "dtype": str(parameter.grad.dtype) if parameter.grad is not None else None,
        }
        for name, parameter in model.named_parameters()
    ]


def tied_embedding_alias(model: Any) -> dict[str, Any]:
    input_embedding = model.get_input_embeddings()
    output_embedding = model.get_output_embeddings()
    if input_embedding is None or output_embedding is None:
        return {"present": False, "same_object": False, "same_storage": False}
    left = input_embedding.weight
    right = output_embedding.weight
    return {
        "present": True,
        "same_object": left is right,
        "same_storage": left.untyped_storage().data_ptr() == right.untyped_storage().data_ptr(),
        "shape_equal": left.shape == right.shape,
        "dtype_equal": left.dtype == right.dtype,
    }


def finite_tensor(torch: Any, tensor: Any, label: str) -> None:
    if not bool(torch.isfinite(tensor).all().item()):
        raise RuntimeError(f"non-finite value in {label}")


def build_inputs(torch: Any, row: dict[str, Any], max_length: int) -> tuple[Any, Any, int]:
    prompt = [int(value) for value in row.get("prompt_ids", [])]
    response = [int(value) for value in row.get("response_ids", [])]
    if not prompt or not response:
        raise ValueError(f"state {row.get('case_id')} lacks prompt or response IDs")
    tokens = (prompt + response)[:max_length]
    if len(tokens) < 2:
        raise ValueError(f"state {row.get('case_id')} has fewer than two retained tokens")
    retained_prompt = min(len(prompt), len(tokens))
    labels = list(tokens)
    for index in range(retained_prompt):
        labels[index] = -100
    target_count = sum(value != -100 for value in labels[1:])
    if target_count < 1:
        raise ValueError(
            f"state {row.get('case_id')} has no retained response target at max_length={max_length}"
        )
    return (
        torch.tensor([tokens], dtype=torch.long, device="cuda"),
        torch.tensor([labels], dtype=torch.long, device="cuda"),
        target_count,
    )


def make_optimizer(torch: Any, model: Any, learning_rate: float) -> Any:
    return torch.optim.AdamW(
        model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8,
        weight_decay=0.0, amsgrad=False, maximize=False, foreach=False,
        capturable=False, differentiable=False, fused=False,
    )


def selected_decisions(torch: Any, logits: Any, labels: Any) -> list[dict[str, Any]]:
    shifted_logits = logits[:, :-1, :]
    shifted_labels = labels[:, 1:]
    mask = shifted_labels.ne(-100)
    selected_logits = shifted_logits[mask].float()
    targets = shifted_labels[mask]
    top_values, top_indices = selected_logits.topk(k=2, dim=-1)
    positions = mask.nonzero(as_tuple=False)[:, 1] + 1
    return [
        {
            "token_position": int(position),
            "target_token": int(target),
            "top1_token": int(indices[0]),
            "top2_token": int(indices[1]),
            "top1_logit": float(values[0]),
            "top2_logit": float(values[1]),
            "top1_top2_margin": float(values[0] - values[1]),
        }
        for position, target, values, indices in zip(
            positions.detach().cpu().tolist(),
            targets.detach().cpu().tolist(),
            top_values.detach().cpu().tolist(),
            top_indices.detach().cpu().tolist(),
            strict=True,
        )
    ]


def run_arm(
    torch: Any,
    model: Any,
    function: Callable[..., tuple[Any, Any]],
    input_ids: Any,
    labels: Any,
    baseline: dict[str, Any],
    initial_rng: dict[str, Any],
    learning_rate: float,
    path: str,
    audit: CompileAudit,
    initial_step_counter: int,
    candidate_counter_mode: str,
) -> dict[str, Any]:
    restore_model(torch, model, baseline)
    restore_rng(torch, initial_rng)
    model.zero_grad(set_to_none=True)
    optimizer = make_optimizer(torch, model, learning_rate)
    initial_optimizer = optimizer_snapshot(optimizer, model)
    before_invocations = audit.runtime_invocations
    started = time.perf_counter_ns()
    loss, logits = function(input_ids, labels)
    finite_tensor(torch, loss, f"{path} loss")
    finite_tensor(torch, logits, f"{path} logits")
    decisions = selected_decisions(torch, logits, labels)
    loss.backward()
    gradients = gradient_signature(model)
    optimizer.step()
    torch.cuda.synchronize()
    elapsed_ns = time.perf_counter_ns() - started
    next_step_counter = initial_step_counter + 1
    if path == "compiled" and candidate_counter_mode == "stale":
        next_step_counter = initial_step_counter
    result = {
        "path": path,
        "loss": float(loss.detach().item()),
        "decisions": decisions,
        "gradient_signature": gradients,
        "optimizer_initial": initial_optimizer,
        "optimizer_next": optimizer_snapshot(optimizer, model),
        "model_next": cpu_model_snapshot(model),
        "alias_next": tied_embedding_alias(model),
        "rng_next": rng_snapshot(torch),
        "next_step_counter": next_step_counter,
        "compiled_runtime_invocations": audit.runtime_invocations - before_invocations,
        "elapsed_ns": elapsed_ns,
    }
    del logits, loss, optimizer
    model.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()
    return result


def tensor_difference(left: Any, right: Any) -> tuple[float, float]:
    left_flat = left.reshape(-1)
    right_flat = right.reshape(-1)
    square = 0.0
    maximum = 0.0
    chunk_size = 1_048_576
    for start in range(0, left_flat.numel(), chunk_size):
        difference = (
            right_flat[start : start + chunk_size].float()
            - left_flat[start : start + chunk_size].float()
        )
        square += float((difference * difference).sum(dtype=difference.new_tensor(0.0).double().dtype).item())
        if difference.numel():
            maximum = max(maximum, float(difference.abs().max().item()))
    return square, maximum


def compare_tensor_maps(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    if reference.keys() != candidate.keys():
        return {"structure_equal": False, "exact_nonfloating_equal": False}
    square = 0.0
    maximum = 0.0
    exact_nonfloating = True
    changed = []
    for name in reference:
        left = reference[name]
        right = candidate[name]
        if left.shape != right.shape or left.dtype != right.dtype:
            return {
                "structure_equal": False,
                "exact_nonfloating_equal": False,
                "field": name,
            }
        equal = bool(left.equal(right))
        if not equal:
            changed.append(name)
        if left.is_floating_point() or left.is_complex():
            part_square, part_maximum = tensor_difference(left, right)
            square += part_square
            maximum = max(maximum, part_maximum)
        else:
            exact_nonfloating = exact_nonfloating and equal
    return {
        "structure_equal": True,
        "exact_nonfloating_equal": exact_nonfloating,
        "floating_difference_l2": math.sqrt(square),
        "max_abs_floating_delta": maximum,
        "changed_field_count": len(changed),
        "changed_fields_head": changed[:20],
    }


def compare_optimizer(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    options_equal = reference["options"] == candidate["options"]
    left_state = reference["state"]
    right_state = candidate["state"]
    if left_state.keys() != right_state.keys():
        return {
            "structure_equal": False,
            "options_equal": options_equal,
            "exact_control_equal": False,
        }
    square = 0.0
    maximum = 0.0
    exact_control = True
    changed = []
    for parameter_name in left_state:
        left_entry = left_state[parameter_name]
        right_entry = right_state[parameter_name]
        if left_entry.keys() != right_entry.keys():
            return {
                "structure_equal": False,
                "options_equal": options_equal,
                "exact_control_equal": False,
                "parameter": parameter_name,
            }
        for key in left_entry:
            left = left_entry[key]
            right = right_entry[key]
            label = f"{parameter_name}:{key}"
            if hasattr(left, "shape") != hasattr(right, "shape"):
                return {
                    "structure_equal": False,
                    "options_equal": options_equal,
                    "exact_control_equal": False,
                    "field": label,
                }
            if hasattr(left, "shape"):
                if left.shape != right.shape or left.dtype != right.dtype:
                    return {
                        "structure_equal": False,
                        "options_equal": options_equal,
                        "exact_control_equal": False,
                        "field": label,
                    }
                equal = bool(left.equal(right))
                if not equal:
                    changed.append(label)
                if key == "step" or not (left.is_floating_point() or left.is_complex()):
                    exact_control = exact_control and equal
                else:
                    part_square, part_maximum = tensor_difference(left, right)
                    square += part_square
                    maximum = max(maximum, part_maximum)
            else:
                equal = left == right
                exact_control = exact_control and equal
                if not equal:
                    changed.append(label)
    return {
        "structure_equal": True,
        "options_equal": options_equal,
        "exact_control_equal": exact_control,
        "floating_moment_difference_l2": math.sqrt(square),
        "max_abs_floating_moment_delta": maximum,
        "changed_field_count": len(changed),
        "changed_fields_head": changed[:20],
    }


def main() -> None:
    args = parse_args()
    if args.count < 1 or args.repeats < 2 or args.max_length < 2:
        raise ValueError("count must be positive, repeats >= 2 and max_length >= 2")
    out_dir = Path(args.out_dir)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("PYTHONHASHSEED", "0")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import torch
    from transformers import AutoModelForCausalLM

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; refusing CPU fallback for the frozen CUDA contract")
    out_dir.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch._dynamo.reset()
    torch._dynamo.config.suppress_errors = False
    torch._dynamo.config.recompile_limit = 2

    model_path = Path(args.model_path)
    samples_path = Path(args.samples)
    all_rows = read_jsonl(samples_path)
    if args.count > len(all_rows):
        raise ValueError("count exceeds frozen sample bank")
    rows = all_rows[: args.count]
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.float32, attn_implementation="sdpa",
        trust_remote_code=False,
    ).eval().to("cuda")
    model.config.use_cache = False
    initial_alias = tied_embedding_alias(model)
    if not initial_alias.get("same_storage"):
        raise RuntimeError("declared tied input/output embedding alias is absent")

    def core(input_ids: Any, labels: Any) -> tuple[Any, Any]:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(input_ids=input_ids, use_cache=False, return_dict=False)[0]
        shifted_logits = logits[:, :-1, :].float()
        shifted_labels = labels[:, 1:]
        loss = torch.nn.functional.cross_entropy(
            shifted_logits.reshape(-1, shifted_logits.shape[-1]),
            shifted_labels.reshape(-1), ignore_index=-100,
        )
        return loss, logits

    audit = CompileAudit()
    compiled_core = torch.compile(
        core, backend=make_tracking_backend(torch, audit), fullgraph=True, dynamic=False,
    )
    baseline = cpu_model_snapshot(model)
    initial_rng = rng_snapshot(torch)

    warm_input_ids, warm_labels, _ = build_inputs(torch, rows[0], args.max_length)
    restore_model(torch, model, baseline)
    restore_rng(torch, initial_rng)
    model.zero_grad(set_to_none=True)
    warm_loss, warm_logits = compiled_core(warm_input_ids, warm_labels)
    warm_loss.backward()
    torch.cuda.synchronize()
    del warm_loss, warm_logits
    model.zero_grad(set_to_none=True)
    restore_model(torch, model, baseline)
    restore_rng(torch, initial_rng)
    gc.collect()
    torch.cuda.empty_cache()

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "arguments": vars(args),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "cuda_build": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
        },
        "model_artifact": artifact_manifest(model_path),
        "sample_artifact": {
            "path": str(samples_path.resolve()),
            "sha256": sha256_file(samples_path),
            "available_rows": len(all_rows),
        },
        "contract_path": str(
            (Path(__file__).parent / "QWEN3_TRAINING_STEP_CONTRACT_V0_1_2026-07-17.md").resolve()
        ),
        "contract": {
            "subject": "Qwen3-0.6B causal-LM response loss plus new AdamW step",
            "historical_optimizer_replay": False,
            "model_mode": "eval; dropout zero",
            "precision": "float32 weights, CUDA float16 autocast forward",
            "attention": "SDPA math only",
            "optimizer": "AdamW foreach=False fused=False, empty initial state",
            "exact_core": "structure/control/RNG/alias/counter",
            "numerical_transition": "UNINSTANTIATED",
            "impact": "DESCRIPTIVE token greedy disagreement",
            "candidate_counter_mode": args.candidate_counter_mode,
        },
        "initial_alias": initial_alias,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    results = []
    with (out_dir / "states.jsonl").open("w", encoding="utf-8") as handle:
        for ordinal, row in enumerate(rows):
            input_ids, labels, target_count = build_inputs(torch, row, args.max_length)
            for repeat in range(args.repeats):
                reference = run_arm(
                    torch, model, core, input_ids, labels, baseline, initial_rng,
                    args.learning_rate, "eager", audit, args.initial_step_counter,
                    args.candidate_counter_mode,
                )
                candidate = run_arm(
                    torch, model, compiled_core, input_ids, labels, baseline, initial_rng,
                    args.learning_rate, "compiled", audit, args.initial_step_counter,
                    args.candidate_counter_mode,
                )
                model_comparison = compare_tensor_maps(
                    reference["model_next"], candidate["model_next"]
                )
                optimizer_comparison = compare_optimizer(
                    reference["optimizer_next"], candidate["optimizer_next"]
                )
                gradient_structure_equal = (
                    reference["gradient_signature"] == candidate["gradient_signature"]
                )
                optimizer_initial_equal = (
                    reference["optimizer_initial"]["options"]
                    == candidate["optimizer_initial"]["options"]
                    and all(not entry for entry in reference["optimizer_initial"]["state"].values())
                    and all(not entry for entry in candidate["optimizer_initial"]["state"].values())
                )
                rng_equal = bool(
                    torch.equal(reference["rng_next"]["cpu"], candidate["rng_next"]["cpu"])
                    and torch.equal(reference["rng_next"]["cuda"], candidate["rng_next"]["cuda"])
                )
                counter_equal = (
                    reference["next_step_counter"] == candidate["next_step_counter"]
                    == args.initial_step_counter + 1
                )
                alias_equal = (
                    reference["alias_next"] == candidate["alias_next"] == initial_alias
                )
                candidate_identity = candidate["compiled_runtime_invocations"] == 1
                exact_reject = (
                    not model_comparison.get("structure_equal", False)
                    or not model_comparison.get("exact_nonfloating_equal", False)
                    or not optimizer_comparison.get("structure_equal", False)
                    or not optimizer_comparison.get("options_equal", False)
                    or not optimizer_comparison.get("exact_control_equal", False)
                    or not gradient_structure_equal
                    or not optimizer_initial_equal
                    or not rng_equal
                    or not counter_equal
                    or not alias_equal
                )
                disagreement_records = [
                    {"reference": left, "candidate": right}
                    for left, right in zip(
                        reference["decisions"], candidate["decisions"], strict=True
                    )
                    if left["top1_token"] != right["top1_token"]
                ]
                disagreements = len(disagreement_records)
                item = {
                    "state_id": str(row.get("case_id", f"row-{ordinal:06d}")),
                    "ordinal": ordinal,
                    "repeat": repeat,
                    "target_token_count": target_count,
                    "candidate_identity_valid": candidate_identity,
                    "exact_transition_verdict": (
                        "INVALID" if not candidate_identity
                        else "REJECT" if exact_reject else "ACCEPT"
                    ),
                    "numerical_transition_verdict": (
                        "UNINSTANTIATED" if candidate_identity else "INVALID"
                    ),
                    "impact_verdict": "NOT_INSTANTIATED",
                    "loss_signed_delta": candidate["loss"] - reference["loss"],
                    "greedy_token_disagreement_count": disagreements,
                    "greedy_sequence_disagreement": disagreements > 0,
                    "greedy_disagreement_records": disagreement_records,
                    "gradient_structure_equal": gradient_structure_equal,
                    "optimizer_initial_state_equal_and_empty": optimizer_initial_equal,
                    "rng_next_state_equal": rng_equal,
                    "counter_relation_satisfied": counter_equal,
                    "tied_embedding_alias_preserved": alias_equal,
                    "model_next": model_comparison,
                    "optimizer_next": optimizer_comparison,
                }
                results.append(item)
                handle.write(json.dumps(item, sort_keys=True, allow_nan=False) + "\n")
                handle.flush()
                del reference, candidate
                gc.collect()

    primary = [row for row in results if row["repeat"] == 0]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in results:
        grouped.setdefault(row["state_id"], []).append(row)
    repeat_stable = 0
    for state_rows in grouped.values():
        signatures = {
            (
                row["exact_transition_verdict"], row["loss_signed_delta"],
                row["greedy_token_disagreement_count"],
                row["model_next"].get("floating_difference_l2"),
                row["optimizer_next"].get("floating_moment_difference_l2"),
            )
            for row in state_rows
        }
        repeat_stable += int(len(signatures) == 1)
    exact_accept = sum(row["exact_transition_verdict"] == "ACCEPT" for row in primary)
    exact_reject = sum(row["exact_transition_verdict"] == "REJECT" for row in primary)
    invalid = sum(row["exact_transition_verdict"] == "INVALID" for row in primary)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "states": len(primary),
        "rows": len(results),
        "compile_audit": {
            "backend_compiles": audit.backend_compiles,
            "runtime_invocations": audit.runtime_invocations,
            "graph_code_sha256": audit.graph_code_sha256,
            "graph_node_counts": audit.graph_node_counts,
        },
        "candidate_identity_all_valid": all(
            row["candidate_identity_valid"] for row in results
        ),
        "exact_accept_states": exact_accept,
        "exact_reject_states": exact_reject,
        "invalid_states": invalid,
        "repeat_stable_states": repeat_stable,
        "numerical_transition_verdict": "UNINSTANTIATED",
        "impact_verdict": "NOT_INSTANTIATED",
        "greedy_sequence_disagreement_states": sum(
            row["greedy_sequence_disagreement"] for row in primary
        ),
        "greedy_token_disagreements": sum(
            row["greedy_token_disagreement_count"] for row in primary
        ),
        "mean_model_next_difference_l2": sum(
            row["model_next"]["floating_difference_l2"] for row in primary
        ) / len(primary),
        "max_abs_model_next_delta": max(
            row["model_next"]["max_abs_floating_delta"] for row in primary
        ),
        "mean_optimizer_moment_difference_l2": sum(
            row["optimizer_next"]["floating_moment_difference_l2"] for row in primary
        ) / len(primary),
        "max_abs_optimizer_moment_delta": max(
            row["optimizer_next"]["max_abs_floating_moment_delta"] for row in primary
        ),
        "claim": (
            "invalid candidate execution; no transition claim"
            if invalid else
            "covered exact-core violation detected; numerical correctness uninstantiated"
            if exact_reject else
            "covered exact-core conformance plus numerical discrepancy measurement; numerical correctness uninstantiated"
        ),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
