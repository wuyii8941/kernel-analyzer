#!/usr/bin/env python
"""Fail-closed matched-state one-step gradient/SGD transition Oracle."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


SCHEMA_VERSION = "forkcert.matched-transition-oracle.v1"


@dataclass
class CompileAudit:
    backend_compiles: int = 0
    runtime_invocations: int = 0
    graph_code_sha256: list[str] = field(default_factory=list)
    graph_node_counts: list[int] = field(default_factory=list)


@dataclass
class TransitionState:
    state_id: str
    inputs: tuple[Any, ...]
    metadata: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", choices=["bert_sst2", "qwen_causal"], required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--repeats", type=int, required=True)
    parser.add_argument("--sequence-length", type=int, required=True)
    parser.add_argument("--minibatch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260716)
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
    names = ["config.json", "model.safetensors", "pytorch_model.bin", "tokenizer.json", "vocab.txt"]
    files = []
    for name in names:
        item = path / name
        if item.is_file():
            files.append({"name": name, "size": item.stat().st_size, "sha256": sha256_file(item)})
    return {"path": str(path.resolve()), "files": files}


def make_tracking_backend(torch: Any, audit: CompileAudit) -> Callable[..., Any]:
    from torch._dynamo.backends.registry import lookup_backend

    inductor_backend = lookup_backend("inductor")

    def backend(graph_module: Any, example_inputs: list[Any]) -> Callable[..., Any]:
        audit.backend_compiles += 1
        audit.graph_code_sha256.append(sha256_text(graph_module.code))
        audit.graph_node_counts.append(sum(1 for _ in graph_module.graph.nodes))
        compiled = inductor_backend(graph_module, example_inputs)

        def counted(*args: Any) -> Any:
            audit.runtime_invocations += 1
            return compiled(*args)

        return counted

    return backend


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_subject(
    torch: Any, args: argparse.Namespace
) -> tuple[Any, Callable[..., tuple[Any, ...]], list[TransitionState], dict[str, Any]]:
    from datasets import load_from_disk
    from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

    model_path = Path(args.model_path)
    if args.subject == "bert_sst2":
        if args.minibatch_size != 1:
            raise ValueError("BERT contract requires minibatch size 1")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path, dtype=torch.float32)
        model.eval().to("cuda")
        dataset = load_from_disk(args.data_path)
        if args.start != 0 or args.count > len(dataset):
            raise ValueError("saved BERT dataset is already a frozen partition; require start=0")
        rows = dataset.select(range(args.count))
        states = []
        for ordinal, row in enumerate(rows):
            encoded = tokenizer(
                row["sentence"],
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=args.sequence_length,
            )
            states.append(
                TransitionState(
                    state_id=f"sst2-{int(row['idx']):06d}",
                    inputs=(
                        encoded["input_ids"].to("cuda"),
                        encoded["attention_mask"].to("cuda"),
                        torch.tensor([int(row["label"])], dtype=torch.long, device="cuda"),
                    ),
                    metadata={"ordinal": ordinal, "dataset_idx": int(row["idx"]), "label": int(row["label"])},
                )
            )

        def core(input_ids: Any, attention_mask: Any, labels: Any) -> tuple[Any, ...]:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)[0]
            loss = torch.nn.functional.cross_entropy(logits.float(), labels)
            return loss, logits.float()

        return model, core, states, {
            "loss": "two-class cross entropy",
            "parameter_dtype": "float32",
            "autocast_dtype": "float16",
            "dropout": "disabled by eval mode",
            "state_unit": "one SST-2 example",
        }

    if args.minibatch_size != 4:
        raise ValueError("Qwen contract requires minibatch size 4")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float32,
        trust_remote_code=False,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    model.eval().to("cuda")
    rows = read_jsonl(Path(args.data_path))
    stop = args.start + args.count
    if stop > len(rows) or args.count % args.minibatch_size:
        raise ValueError("Qwen requested range must exist and divide into frozen minibatches")
    selected = rows[args.start:stop]
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad_id is None:
        raise ValueError("Qwen tokenizer has no pad/eos token")
    states = []
    for group_start in range(0, len(selected), args.minibatch_size):
        group = selected[group_start : group_start + args.minibatch_size]
        batch_ids = []
        attention = []
        response_masks = []
        case_ids = []
        response_tokens = 0
        for row in group:
            prompt = [int(value) for value in row["prompt_ids"]]
            response = [int(value) for value in row["response_ids"]]
            combined = (prompt + response)[: args.sequence_length]
            actual_length = len(combined)
            padded = combined + [int(pad_id)] * (args.sequence_length - actual_length)
            mask = [1] * actual_length + [0] * (args.sequence_length - actual_length)
            response_mask = [
                1.0 if (position + 1) >= len(prompt) and (position + 1) < actual_length else 0.0
                for position in range(args.sequence_length - 1)
            ]
            response_tokens += int(sum(response_mask))
            batch_ids.append(padded)
            attention.append(mask)
            response_masks.append(response_mask)
            case_ids.append(str(row.get("case_id", f"row-{args.start + group_start}")))
        inputs = (
            torch.tensor(batch_ids, dtype=torch.long, device="cuda"),
            torch.tensor(attention, dtype=torch.long, device="cuda"),
            torch.tensor(response_masks, dtype=torch.float32, device="cuda"),
        )
        states.append(
            TransitionState(
                state_id=f"qwen-group-{args.start + group_start:06d}-{args.start + group_start + len(group) - 1:06d}",
                inputs=inputs,
                metadata={
                    "ordinal": group_start // args.minibatch_size,
                    "source_row_start": args.start + group_start,
                    "source_row_stop": args.start + group_start + len(group),
                    "case_ids": case_ids,
                    "response_tokens": response_tokens,
                },
            )
        )

    def core(input_ids: Any, attention_mask: Any, response_mask: Any) -> tuple[Any, ...]:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)[0]
        shifted = logits[:, :-1, :].float()
        targets = input_ids[:, 1:]
        target_logps = torch.nn.functional.log_softmax(shifted, dim=-1).gather(
            -1, targets.unsqueeze(-1)
        ).squeeze(-1)
        denominator = response_mask.sum()
        mean_target_logp = (target_logps * response_mask).sum() / denominator
        loss = -mean_target_logp
        return loss, mean_target_logp.unsqueeze(0)

    return model, core, states, {
        "loss": "mean teacher-forced response-token negative log-probability",
        "parameter_dtype": "float32",
        "autocast_dtype": "float16",
        "log_softmax_dtype": "float32",
        "dropout": "disabled by eval mode/model configuration",
        "state_unit": "four consecutive rollout sequences",
    }


def finite_tensor(torch: Any, value: Any, label: str) -> None:
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"non-finite tensor in {label}")


def capture_gradient(
    torch: Any,
    parameters: list[tuple[str, Any]],
) -> tuple[list[Any | None], dict[str, Any]]:
    clones: list[Any | None] = []
    sum_value = torch.zeros((), dtype=torch.float64, device="cuda")
    square_value = torch.zeros_like(sum_value)
    max_value = torch.zeros_like(sum_value)
    sample_value = torch.zeros_like(sum_value)
    nonzero = 0
    missing = 0
    for index, (_name, parameter) in enumerate(parameters):
        gradient = parameter.grad
        if gradient is None:
            clones.append(None)
            missing += 1
            continue
        finite_tensor(torch, gradient, "gradient")
        item = gradient.detach().clone()
        clones.append(item)
        flat = item.reshape(-1)
        work = flat.float()
        sum_value += work.sum(dtype=torch.float64)
        square_value += (work * work).sum(dtype=torch.float64)
        max_value = torch.maximum(max_value, work.abs().max().double())
        sample = work[0].double() + work[work.numel() // 2].double() + work[-1].double()
        sample_value += sample * (index + 1)
        nonzero += int(torch.count_nonzero(work).item())
    scalars = {
        "sum": float(sum_value.item()),
        "square_sum": float(square_value.item()),
        "max_abs": float(max_value.item()),
        "weighted_sample": float(sample_value.item()),
        "nonzero": nonzero,
        "missing_parameters": missing,
    }
    scalars["metric_fingerprint_sha256"] = sha256_text(json.dumps(scalars, sort_keys=True))
    return clones, scalars


def execute_path(
    torch: Any,
    model: Any,
    function: Callable[..., tuple[Any, ...]],
    state: TransitionState,
    path: str,
    parameters: list[tuple[str, Any]],
    audit: CompileAudit,
) -> dict[str, Any]:
    model.zero_grad(set_to_none=True)
    before = audit.runtime_invocations
    started = time.perf_counter_ns()
    loss, auxiliary = function(*state.inputs)
    finite_tensor(torch, loss, f"{path} loss")
    finite_tensor(torch, auxiliary, f"{path} auxiliary")
    loss.backward()
    torch.cuda.synchronize()
    elapsed = time.perf_counter_ns() - started
    gradients, fingerprint = capture_gradient(torch, parameters)
    return {
        "path": path,
        "loss": float(loss.detach().item()),
        "auxiliary": auxiliary.detach().float().cpu().tolist(),
        "gradients": gradients,
        "gradient_fingerprint": fingerprint,
        "compiled_runtime_invocations": audit.runtime_invocations - before,
        "elapsed_ns": elapsed,
    }


def gradient_comparison(
    torch: Any,
    parameters: list[tuple[str, Any]],
    reference: dict[str, Any],
    candidate: dict[str, Any],
    learning_rate: float,
    emit_blocks: bool,
    state_id: str,
    repeat: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ref_square = torch.zeros((), dtype=torch.float64, device="cuda")
    alt_square = torch.zeros_like(ref_square)
    diff_square = torch.zeros_like(ref_square)
    dot = torch.zeros_like(ref_square)
    max_abs = torch.zeros_like(ref_square)
    support_disagreement = 0
    exact_equal = True
    block_rows = []
    for index, ((name, parameter), ref_grad, alt_grad) in enumerate(
        zip(parameters, reference["gradients"], candidate["gradients"], strict=True)
    ):
        if ref_grad is None and alt_grad is None:
            if emit_blocks:
                block_rows.append(
                    {
                        "state_id": state_id,
                        "repeat": repeat,
                        "parameter_index": index,
                        "parameter_name": name,
                        "coordinates": parameter.numel(),
                        "reference_gradient_present": False,
                        "candidate_gradient_present": False,
                        "gradient_difference_l2": 0.0,
                        "reference_gradient_l2": 0.0,
                        "candidate_gradient_l2": 0.0,
                        "max_abs_gradient_delta": 0.0,
                        "support_disagreement_coordinates": 0,
                    }
                )
            continue
        if ref_grad is None:
            ref_grad = torch.zeros_like(alt_grad)
        if alt_grad is None:
            alt_grad = torch.zeros_like(ref_grad)
        ref_work = ref_grad.float()
        alt_work = alt_grad.float()
        difference = alt_work - ref_work
        block_ref_square = (ref_work * ref_work).sum(dtype=torch.float64)
        block_alt_square = (alt_work * alt_work).sum(dtype=torch.float64)
        block_diff_square = (difference * difference).sum(dtype=torch.float64)
        block_max = difference.abs().max().double()
        block_support = int(torch.count_nonzero((ref_work == 0) != (alt_work == 0)).item())
        ref_square += block_ref_square
        alt_square += block_alt_square
        diff_square += block_diff_square
        dot += (ref_work * alt_work).sum(dtype=torch.float64)
        max_abs = torch.maximum(max_abs, block_max)
        support_disagreement += block_support
        exact_equal = exact_equal and bool(torch.equal(ref_work, alt_work))
        if emit_blocks:
            block_rows.append(
                {
                    "state_id": state_id,
                    "repeat": repeat,
                    "parameter_index": index,
                    "parameter_name": name,
                    "coordinates": parameter.numel(),
                    "reference_gradient_present": reference["gradients"][index] is not None,
                    "candidate_gradient_present": candidate["gradients"][index] is not None,
                    "gradient_difference_l2": math.sqrt(float(block_diff_square.item())),
                    "reference_gradient_l2": math.sqrt(float(block_ref_square.item())),
                    "candidate_gradient_l2": math.sqrt(float(block_alt_square.item())),
                    "max_abs_gradient_delta": float(block_max.item()),
                    "support_disagreement_coordinates": block_support,
                }
            )
    ref_norm = math.sqrt(float(ref_square.item()))
    alt_norm = math.sqrt(float(alt_square.item()))
    diff_norm = math.sqrt(float(diff_square.item()))
    denominator = ref_norm * alt_norm
    comparison = {
        "reference_gradient_l2": ref_norm,
        "candidate_gradient_l2": alt_norm,
        "gradient_difference_l2": diff_norm,
        "relative_gradient_difference_l2": diff_norm / ref_norm if ref_norm else None,
        "gradient_cosine_similarity": float(dot.item()) / denominator if denominator else None,
        "max_abs_gradient_delta": float(max_abs.item()),
        "gradient_support_disagreement_coordinates": support_disagreement,
        "gradient_exact_equal": exact_equal,
        "reference_update_l2": learning_rate * ref_norm,
        "candidate_update_l2": learning_rate * alt_norm,
        "update_difference_l2": learning_rate * diff_norm,
        "relative_update_difference_l2": diff_norm / ref_norm if ref_norm else None,
        "next_parameter_state_difference_l2": learning_rate * diff_norm,
        "max_abs_next_parameter_delta": learning_rate * float(max_abs.item()),
        "update_is_derived_from_gradient": True,
    }
    return comparison, block_rows


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def state_bootstrap_ci(
    rows: list[dict[str, Any]], field: str, draws: int, seed: int
) -> list[float | None]:
    values = {row["state_id"]: float(row[field]) for row in rows if row["repeat"] == 0}
    keys = sorted(values)
    if not keys:
        return [None, None]
    rng = random.Random(seed)
    estimates = [mean([values[rng.choice(keys)] for _ in keys]) for _ in range(draws)]
    return [quantile(estimates, 0.025), quantile(estimates, 0.975)]


def repeated_variance(rows: list[dict[str, Any]], field: str) -> float:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row["state_id"]].append(float(row[field]))
    values = [statistics.variance(items) if len(items) > 1 else 0.0 for items in grouped.values()]
    return mean(values)


def state_variance(rows: list[dict[str, Any]], field: str) -> float:
    values = [float(row[field]) for row in rows if row["repeat"] == 0]
    return statistics.variance(values) if len(values) > 1 else 0.0


def main() -> None:
    args = parse_args()
    if args.repeats < 2:
        raise ValueError("at least two repeats are required to measure runtime variability")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("PYTHONHASHSEED", "0")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; fail instead of falling back to CPU")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch._dynamo.reset()
    torch._dynamo.config.suppress_errors = False
    torch._dynamo.config.recompile_limit = 2

    model, eager_core, states, subject_metadata = load_subject(torch, args)
    parameters = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("model has no trainable parameters")
    audit = CompileAudit()
    compiled_core = torch.compile(
        eager_core,
        backend=make_tracking_backend(torch, audit),
        fullgraph=True,
        dynamic=False,
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "started_unix": time.time(),
        "arguments": vars(args),
        "subject_metadata": subject_metadata,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_build": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "inductor_cache_dir": os.environ.get("TORCHINDUCTOR_CACHE_DIR"),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        },
        "model_artifact": artifact_manifest(Path(args.model_path)),
        "data_artifact": {
            "path": str(Path(args.data_path).resolve()),
            "file_sha256": sha256_file(Path(args.data_path)) if Path(args.data_path).is_file() else None,
        },
        "state_ids": [state.state_id for state in states],
        "transition_contract": {
            "shared_model_object_between_paths": True,
            "optimizer": "hypothetical full-parameter SGD, no momentum, no weight decay",
            "optimizer_step_applied": False,
            "updates_are_derived_exactly_as_negative_lr_times_gradient": True,
            "state_carry_between_observations": False,
            "claim": "controlled one-step transition sensitivity, not historical optimizer replay",
        },
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )

    # Warm both forward/backward paths before recording. Backward graph
    # compilation must also finish here; later graph compilation is a validity
    # failure rather than silently changing the measured program.
    warm_candidate = execute_path(
        torch, model, compiled_core, states[0], "compiled", parameters, audit
    )
    del warm_candidate
    model.zero_grad(set_to_none=True)
    warm_reference = execute_path(torch, model, eager_core, states[0], "eager", parameters, audit)
    del warm_reference
    model.zero_grad(set_to_none=True)
    warmup_runtime_invocations = audit.runtime_invocations
    warmup_backend_compiles = audit.backend_compiles

    rows: list[dict[str, Any]] = []
    block_rows: list[dict[str, Any]] = []
    baseline_fingerprints: dict[tuple[str, str], dict[str, Any]] = {}
    with (out_dir / "states.jsonl").open("w", encoding="utf-8") as state_handle, (
        out_dir / "blocks.jsonl"
    ).open("w", encoding="utf-8") as block_handle:
        for state_index, state in enumerate(states):
            for repeat in range(args.repeats):
                order = ["eager", "compiled"] if repeat % 2 == 0 else ["compiled", "eager"]
                outputs: dict[str, dict[str, Any]] = {}
                for path in order:
                    function = eager_core if path == "eager" else compiled_core
                    outputs[path] = execute_path(
                        torch, model, function, state, path, parameters, audit
                    )
                comparison, current_blocks = gradient_comparison(
                    torch,
                    parameters,
                    outputs["eager"],
                    outputs["compiled"],
                    args.learning_rate,
                    emit_blocks=repeat == 0,
                    state_id=state.state_id,
                    repeat=repeat,
                )
                for block in current_blocks:
                    block_handle.write(json.dumps(block, sort_keys=True, allow_nan=False) + "\n")
                block_handle.flush()
                block_rows.extend(current_blocks)
                self_details = {}
                for path in ["eager", "compiled"]:
                    key = (state.state_id, path)
                    fingerprint = outputs[path]["gradient_fingerprint"]
                    current = {
                        "loss": outputs[path]["loss"],
                        "auxiliary": outputs[path]["auxiliary"],
                        "gradient_metric_fingerprint": fingerprint["metric_fingerprint_sha256"],
                    }
                    if repeat == 0:
                        baseline_fingerprints[key] = current
                        exact = True
                    else:
                        exact = current == baseline_fingerprints[key]
                    self_details[path] = {"metric_exact": exact, **current}
                row = {
                    "schema_version": SCHEMA_VERSION,
                    "subject": args.subject,
                    "state_id": state.state_id,
                    "state_index": state_index,
                    "state_metadata": state.metadata,
                    "repeat": repeat,
                    "execution_order": order,
                    "reference_loss": outputs["eager"]["loss"],
                    "candidate_loss": outputs["compiled"]["loss"],
                    "loss_signed_delta": outputs["compiled"]["loss"] - outputs["eager"]["loss"],
                    "loss_abs_delta": abs(outputs["compiled"]["loss"] - outputs["eager"]["loss"]),
                    "reference_auxiliary": outputs["eager"]["auxiliary"],
                    "candidate_auxiliary": outputs["compiled"]["auxiliary"],
                    "candidate_compiled_runtime_invocations": outputs["compiled"][
                        "compiled_runtime_invocations"
                    ],
                    "candidate_execution_valid": outputs["compiled"][
                        "compiled_runtime_invocations"
                    ]
                    > 0,
                    "elapsed_ns": {
                        "eager": outputs["eager"]["elapsed_ns"],
                        "compiled": outputs["compiled"]["elapsed_ns"],
                    },
                    "self_pair_details": self_details,
                    "self_pair_metric_exact": all(
                        item["metric_exact"] for item in self_details.values()
                    ),
                    **comparison,
                }
                if args.subject == "bert_sst2":
                    reference_logits = outputs["eager"]["auxiliary"][0]
                    candidate_logits = outputs["compiled"]["auxiliary"][0]
                    label = int(state.metadata["label"])
                    reference_prediction = max(range(len(reference_logits)), key=reference_logits.__getitem__)
                    candidate_prediction = max(range(len(candidate_logits)), key=candidate_logits.__getitem__)
                    row.update(
                        {
                            "reference_prediction": reference_prediction,
                            "candidate_prediction": candidate_prediction,
                            "prediction_disagreement": reference_prediction != candidate_prediction,
                            "correctness_event_disagreement": (reference_prediction == label)
                            != (candidate_prediction == label),
                            "target_logp_signed_delta": None,
                        }
                    )
                else:
                    row.update(
                        {
                            "reference_prediction": None,
                            "candidate_prediction": None,
                            "prediction_disagreement": None,
                            "correctness_event_disagreement": None,
                            "target_logp_signed_delta": -row["loss_signed_delta"],
                        }
                    )
                state_handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                state_handle.flush()
                rows.append(row)
                del outputs
                model.zero_grad(set_to_none=True)

    primary = [row for row in rows if row["repeat"] == 0]
    self_nonzero = [row for row in rows if row["repeat"] > 0 and not row["self_pair_metric_exact"]]
    measurement_compiles = audit.backend_compiles - warmup_backend_compiles
    summary = {
        "schema_version": SCHEMA_VERSION,
        "subject": args.subject,
        "validity": {
            "shared_model_object_between_paths": True,
            "candidate_calls_valid": all(row["candidate_execution_valid"] for row in rows),
            "backend_compiles": audit.backend_compiles,
            "backend_compiles_during_measurement": measurement_compiles,
            "no_graph_proliferation_after_warmup": measurement_compiles == 0,
            "measurement_runtime_invocations": audit.runtime_invocations - warmup_runtime_invocations,
            "warmup_runtime_invocations": warmup_runtime_invocations,
            "graph_code_sha256": audit.graph_code_sha256,
            "graph_node_counts": audit.graph_node_counts,
            "self_pair_metric_nonzero_count": len(self_nonzero),
            "self_pair_metric_nonzero_examples": self_nonzero[:5],
            "finite_gradients": True,
        },
        "sampling": {
            "states": len(states),
            "observations": len(primary),
            "repeats": args.repeats,
            "bootstrap_unit": "matched transition state/minibatch",
        },
        "loss": {
            "mean_signed_delta": mean([row["loss_signed_delta"] for row in primary]),
            "mean_signed_delta_state_bootstrap_95ci": state_bootstrap_ci(
                rows, "loss_signed_delta", args.bootstrap, args.seed + 1
            ),
            "mean_abs_delta": mean([row["loss_abs_delta"] for row in primary]),
            "between_state_heterogeneity_variance": state_variance(rows, "loss_signed_delta"),
            "same_state_repeat_variance": repeated_variance(rows, "loss_signed_delta"),
        },
        "gradient": {
            "mean_reference_l2": mean([row["reference_gradient_l2"] for row in primary]),
            "mean_candidate_l2": mean([row["candidate_gradient_l2"] for row in primary]),
            "mean_difference_l2": mean([row["gradient_difference_l2"] for row in primary]),
            "mean_difference_l2_state_bootstrap_95ci": state_bootstrap_ci(
                rows, "gradient_difference_l2", args.bootstrap, args.seed + 2
            ),
            "max_difference_l2": max(row["gradient_difference_l2"] for row in primary),
            "mean_relative_difference_l2": mean(
                [row["relative_gradient_difference_l2"] for row in primary]
            ),
            "mean_cosine_similarity": mean(
                [row["gradient_cosine_similarity"] for row in primary]
            ),
            "max_abs_coordinate_delta": max(row["max_abs_gradient_delta"] for row in primary),
            "support_disagreement_coordinates": sum(
                row["gradient_support_disagreement_coordinates"] for row in primary
            ),
            "exact_equal_states": sum(row["gradient_exact_equal"] for row in primary),
            "between_state_difference_l2_variance": state_variance(rows, "gradient_difference_l2"),
            "same_state_difference_l2_repeat_variance": repeated_variance(
                rows, "gradient_difference_l2"
            ),
        },
        "transition": {
            "optimizer": "full-parameter SGD, no momentum, no weight decay",
            "learning_rate": args.learning_rate,
            "update_applied": False,
            "mean_update_difference_l2": mean([row["update_difference_l2"] for row in primary]),
            "mean_relative_update_difference_l2": mean(
                [row["relative_update_difference_l2"] for row in primary]
            ),
            "mean_next_parameter_state_difference_l2": mean(
                [row["next_parameter_state_difference_l2"] for row in primary]
            ),
            "max_abs_next_parameter_delta": max(
                row["max_abs_next_parameter_delta"] for row in primary
            ),
            "linear_map_note": "for declared SGD, update and next-state differences are lr times gradient difference",
        },
        "semantic": {
            "prediction_disagreements": sum(
                bool(row["prediction_disagreement"]) for row in primary
                if row["prediction_disagreement"] is not None
            ),
            "correctness_event_disagreements": sum(
                bool(row["correctness_event_disagreement"]) for row in primary
                if row["correctness_event_disagreement"] is not None
            ),
        },
        "claim_scope": {
            "supported": "implementation-relative controlled one-step gradient/SGD transition discrepancy on the named matched-state population",
            "not_supported": [
                "historical optimizer-state replay",
                "long-run training harm",
                "mathematical/compiler correctness failure",
                "generalization outside the named subject/state/protocol",
            ],
        },
    }
    if not summary["validity"]["candidate_calls_valid"]:
        raise RuntimeError("candidate execution validity failed")
    if not summary["validity"]["no_graph_proliferation_after_warmup"]:
        raise RuntimeError("compiled graph proliferation occurred after warmup")
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()

