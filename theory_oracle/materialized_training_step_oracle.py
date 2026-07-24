#!/usr/bin/env python
"""Materialized deterministic BERT/SGD training-step Oracle v0.1."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


SCHEMA_VERSION = "forkcert.training-step-oracle.v0.1"


@dataclass
class CompileAudit:
    backend_compiles: int = 0
    runtime_invocations: int = 0
    graph_code_sha256: list[str] = field(default_factory=list)
    graph_node_counts: list[int] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--initial-step-counter", type=int, default=7)
    parser.add_argument(
        "--candidate-counter-mode", choices=["correct", "stale"], default="correct",
        help="independently labeled exact-state mutation control",
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
    names = ["config.json", "model.safetensors", "pytorch_model.bin", "tokenizer.json", "vocab.txt"]
    files = []
    for name in names:
        item = path / name
        if item.is_file():
            files.append({"name": name, "size": item.stat().st_size, "sha256": sha256_file(item)})
    return {"path": str(path.resolve()), "files": files}


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


def clone_state_dict(model: Any) -> dict[str, Any]:
    return {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}


def restore_state_dict(torch: Any, model: Any, baseline: dict[str, Any]) -> None:
    model.load_state_dict(baseline, strict=True)
    current = model.state_dict()
    if current.keys() != baseline.keys():
        raise RuntimeError("state-dict structure changed during reset")
    for name, expected in baseline.items():
        if not torch.equal(current[name], expected):
            raise RuntimeError(f"initial state reset failed for {name}")


def finite_tensor(torch: Any, value: Any, label: str) -> None:
    if not bool(torch.isfinite(value).all().item()):
        raise RuntimeError(f"non-finite tensor in {label}")


def rng_snapshot(torch: Any) -> dict[str, Any]:
    return {
        "cpu": torch.get_rng_state().clone(),
        "cuda": torch.cuda.get_rng_state().clone(),
    }


def restore_rng(torch: Any, snapshot: dict[str, Any]) -> None:
    torch.set_rng_state(snapshot["cpu"])
    torch.cuda.set_rng_state(snapshot["cuda"])


def optimizer_signature(optimizer: Any) -> dict[str, Any]:
    groups = []
    for group in optimizer.param_groups:
        groups.append(
            {
                key: value
                for key, value in group.items()
                if key != "params" and isinstance(value, (bool, int, float, str, type(None)))
            }
        )
    return {"state_entries": len(optimizer.state), "param_groups": groups}


def gradient_signature(model: Any) -> dict[str, Any]:
    entries = []
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        entries.append(
            {
                "name": name,
                "present": gradient is not None,
                "shape": list(gradient.shape) if gradient is not None else None,
                "dtype": str(gradient.dtype) if gradient is not None else None,
            }
        )
    return {"entries": entries}


def run_arm(
    torch: Any,
    model: Any,
    function: Callable[..., tuple[Any, Any]],
    inputs: tuple[Any, ...],
    baseline: dict[str, Any],
    initial_rng: dict[str, Any],
    learning_rate: float,
    path: str,
    audit: CompileAudit,
    initial_step_counter: int,
    candidate_counter_mode: str,
) -> dict[str, Any]:
    restore_state_dict(torch, model, baseline)
    restore_rng(torch, initial_rng)
    model.zero_grad(set_to_none=True)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=learning_rate, momentum=0.0, weight_decay=0.0
    )
    before_optimizer = optimizer_signature(optimizer)
    before_invocations = audit.runtime_invocations
    started = time.perf_counter_ns()
    loss, logits = function(*inputs)
    finite_tensor(torch, loss, f"{path} loss")
    finite_tensor(torch, logits, f"{path} logits")
    loss.backward()
    gradients = gradient_signature(model)
    optimizer.step()
    torch.cuda.synchronize()
    elapsed_ns = time.perf_counter_ns() - started
    next_step_counter = initial_step_counter + 1
    if path == "compiled" and candidate_counter_mode == "stale":
        next_step_counter = initial_step_counter
    return {
        "path": path,
        "loss": float(loss.detach().item()),
        "prediction": int(logits.detach().argmax(dim=-1).item()),
        "gradient_signature": gradients,
        "optimizer_before": before_optimizer,
        "optimizer_after": optimizer_signature(optimizer),
        "next_state": clone_state_dict(model),
        "next_rng": rng_snapshot(torch),
        "next_step_counter": next_step_counter,
        "compiled_runtime_invocations": audit.runtime_invocations - before_invocations,
        "elapsed_ns": elapsed_ns,
    }


def compare_next_state(
    torch: Any,
    reference: dict[str, Any],
    candidate: dict[str, Any],
    parameter_names: set[str],
) -> dict[str, Any]:
    if reference.keys() != candidate.keys():
        return {"structure_equal": False, "exact_core_reject": True}
    diff_square = torch.zeros((), dtype=torch.float64, device="cuda")
    max_abs = torch.zeros_like(diff_square)
    exact_parameters = True
    exact_buffers = True
    nonfloating_equal = True
    changed_fields = []
    for name in reference:
        left = reference[name]
        right = candidate[name]
        if left.shape != right.shape or left.dtype != right.dtype:
            return {"structure_equal": False, "exact_core_reject": True, "field": name}
        equal = bool(torch.equal(left, right))
        if not equal:
            changed_fields.append(name)
        if name in parameter_names:
            exact_parameters = exact_parameters and equal
        else:
            exact_buffers = exact_buffers and equal
        if left.is_floating_point() or left.is_complex():
            difference = right.double() - left.double()
            diff_square += (difference * difference).sum()
            if difference.numel():
                max_abs = torch.maximum(max_abs, difference.abs().max())
        else:
            nonfloating_equal = nonfloating_equal and equal
    return {
        "structure_equal": True,
        "exact_core_reject": not nonfloating_equal,
        "floating_parameter_exact_equal": exact_parameters,
        "buffer_exact_equal": exact_buffers,
        "nonfloating_state_equal": nonfloating_equal,
        "floating_next_state_difference_l2": math.sqrt(float(diff_square.item())),
        "max_abs_floating_next_state_delta": float(max_abs.item()),
        "changed_field_count": len(changed_fields),
        "changed_fields_head": changed_fields[:20],
    }


def main() -> None:
    args = parse_args()
    if args.count < 1 or args.repeats < 1:
        raise ValueError("count and repeats must be positive")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("PYTHONHASHSEED", "0")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import torch
    from datasets import load_from_disk
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; refusing CPU fallback")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch._dynamo.reset()
    torch._dynamo.config.suppress_errors = False
    torch._dynamo.config.recompile_limit = 2

    model_path = Path(args.model_path)
    data_path = Path(args.data_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, dtype=torch.float32
    ).eval().to("cuda")
    dataset = load_from_disk(data_path)
    if args.count > len(dataset):
        raise ValueError("count exceeds frozen dataset partition")
    rows = dataset.select(range(args.count))

    def core(input_ids: Any, attention_mask: Any, labels: Any) -> tuple[Any, Any]:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(
                input_ids=input_ids, attention_mask=attention_mask, return_dict=False
            )[0]
        return torch.nn.functional.cross_entropy(logits.float(), labels), logits.float()

    audit = CompileAudit()
    compiled_core = torch.compile(
        core,
        backend=make_tracking_backend(torch, audit),
        fullgraph=True,
        dynamic=False,
    )
    baseline = clone_state_dict(model)
    parameter_names = {name for name, _ in model.named_parameters()}
    initial_rng = rng_snapshot(torch)

    first = rows[0]
    warm = tokenizer(
        first["sentence"], return_tensors="pt", truncation=True,
        padding="max_length", max_length=args.sequence_length,
    )
    warm_inputs = (
        warm["input_ids"].to("cuda"),
        warm["attention_mask"].to("cuda"),
        torch.tensor([int(first["label"])], dtype=torch.long, device="cuda"),
    )
    restore_state_dict(torch, model, baseline)
    restore_rng(torch, initial_rng)
    model.zero_grad(set_to_none=True)
    warm_loss, _ = compiled_core(*warm_inputs)
    warm_loss.backward()
    torch.cuda.synchronize()
    restore_state_dict(torch, model, baseline)
    restore_rng(torch, initial_rng)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "arguments": vars(args),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_build": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
        },
        "model_artifact": artifact_manifest(model_path),
        "data_path": str(data_path.resolve()),
        "contract": {
            "subject": "BERT SST-2 deterministic materialized SGD step",
            "optimizer": "torch.optim.SGD, no momentum, no weight decay",
            "grad_scaler": "absent and outside subject",
            "dropout": "disabled by eval mode",
            "exact_core": "state structure, gradient structure, optimizer structure, nonfloating state, coupled RNG",
            "numerical_transition": "UNINSTANTIATED: no independent gradient/update envelope",
            "impact": "loss and prediction disagreement, descriptive",
            "step_counter_relation": "next counter must equal initial counter plus one",
            "candidate_counter_mode": args.candidate_counter_mode,
        },
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    results = []
    with (out_dir / "states.jsonl").open("w", encoding="utf-8") as handle:
        for ordinal, row in enumerate(rows):
            encoded = tokenizer(
                row["sentence"], return_tensors="pt", truncation=True,
                padding="max_length", max_length=args.sequence_length,
            )
            inputs = (
                encoded["input_ids"].to("cuda"),
                encoded["attention_mask"].to("cuda"),
                torch.tensor([int(row["label"])], dtype=torch.long, device="cuda"),
            )
            for repeat in range(args.repeats):
                reference = run_arm(
                    torch, model, core, inputs, baseline, initial_rng,
                    args.learning_rate, "eager", audit,
                    args.initial_step_counter, args.candidate_counter_mode,
                )
                candidate = run_arm(
                    torch, model, compiled_core, inputs, baseline, initial_rng,
                    args.learning_rate, "compiled", audit,
                    args.initial_step_counter, args.candidate_counter_mode,
                )
                next_state = compare_next_state(
                    torch, reference["next_state"], candidate["next_state"], parameter_names
                )
                gradient_structure_equal = (
                    reference["gradient_signature"] == candidate["gradient_signature"]
                )
                optimizer_structure_equal = (
                    reference["optimizer_before"] == candidate["optimizer_before"]
                    and reference["optimizer_after"] == candidate["optimizer_after"]
                )
                rng_equal = bool(
                    torch.equal(reference["next_rng"]["cpu"], candidate["next_rng"]["cpu"])
                    and torch.equal(reference["next_rng"]["cuda"], candidate["next_rng"]["cuda"])
                )
                step_counter_equal = (
                    reference["next_step_counter"] == candidate["next_step_counter"]
                    == args.initial_step_counter + 1
                )
                candidate_identity = candidate["compiled_runtime_invocations"] == 1
                exact_reject = (
                    next_state.get("exact_core_reject", True)
                    or not next_state.get("buffer_exact_equal", False)
                    or not gradient_structure_equal
                    or not optimizer_structure_equal
                    or not rng_equal
                    or not step_counter_equal
                )
                exact_verdict = "REJECT" if exact_reject else "ACCEPT"
                item = {
                    "state_id": f"sst2-{int(row['idx']):06d}",
                    "ordinal": ordinal,
                    "repeat": repeat,
                    "label": int(row["label"]),
                    "candidate_identity_valid": candidate_identity,
                    "exact_transition_verdict": exact_verdict if candidate_identity else "INVALID",
                    "numerical_transition_verdict": "UNINSTANTIATED" if candidate_identity else "INVALID",
                    "impact_verdict": "NOT_INSTANTIATED",
                    "loss_signed_delta": candidate["loss"] - reference["loss"],
                    "prediction_disagreement": candidate["prediction"] != reference["prediction"],
                    "gradient_structure_equal": gradient_structure_equal,
                    "optimizer_structure_equal": optimizer_structure_equal,
                    "rng_next_state_equal": rng_equal,
                    "step_counter_relation_satisfied": step_counter_equal,
                    "reference_next_step_counter": reference["next_step_counter"],
                    "candidate_next_step_counter": candidate["next_step_counter"],
                    "next_state": next_state,
                }
                results.append(item)
                handle.write(json.dumps(item, sort_keys=True, allow_nan=False) + "\n")
                handle.flush()

    restore_state_dict(torch, model, baseline)
    primary = [row for row in results if row["repeat"] == 0]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in results:
        grouped.setdefault(row["state_id"], []).append(row)
    repeat_stable_states = 0
    for rows_for_state in grouped.values():
        signatures = {
            (
                row["exact_transition_verdict"],
                row["numerical_transition_verdict"],
                row["loss_signed_delta"],
                row["prediction_disagreement"],
                row["next_state"]["floating_next_state_difference_l2"],
                row["next_state"]["max_abs_floating_next_state_delta"],
            )
            for row in rows_for_state
        }
        repeat_stable_states += int(len(signatures) == 1)
    exact_accept_states = sum(row["exact_transition_verdict"] == "ACCEPT" for row in primary)
    exact_reject_states = sum(row["exact_transition_verdict"] == "REJECT" for row in primary)
    claim = (
        "covered exact-core transition violation detected; numerical correctness uninstantiated"
        if exact_reject_states
        else "covered exact-core transition conformance plus numerical discrepancy; no numerical correctness verdict"
    )
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
        "candidate_identity_all_valid": all(row["candidate_identity_valid"] for row in results),
        "exact_accept_states": exact_accept_states,
        "exact_reject_states": exact_reject_states,
        "numerical_transition_verdict": "UNINSTANTIATED",
        "prediction_disagreement_states": sum(row["prediction_disagreement"] for row in primary),
        "repeat_stable_states": repeat_stable_states,
        "mean_next_state_difference_l2": sum(
            row["next_state"]["floating_next_state_difference_l2"] for row in primary
        ) / len(primary),
        "max_abs_next_state_delta": max(
            row["next_state"]["max_abs_floating_next_state_delta"] for row in primary
        ),
        "claim": claim,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
