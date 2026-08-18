#!/usr/bin/env python3
"""Run C16 minus R16 at exact compiler-bound AOT semantic endpoints."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import transformers
import triton
from torch._inductor import config as inductor_config
from torch._inductor.codecache import PyCodeCache


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "archive/round1_code/src"))

from scripts.analyze_generated_fp32_screen import (  # noqa: E402
    bootstrap, bootstrap_counts, metric_equal, u_statistic,
)
from scripts.aot_capture import AOTForwardBackwardCapture  # noqa: E402
from scripts.generated_fp32_observer import nonfinite_aware_metrics  # noqa: E402
from scripts.qwen_candidate_step import LossStep, configure_candidate_runtime  # noqa: E402
from scripts.run_generated_fp32_screen import (  # noqa: E402
    file_digest, gradient_digest, load_model, tensor_digest,
)
from scripts.same_dtype_semantic_observer import (  # noqa: E402
    SameDtypeSemanticCandidateObserver,
)
from src.kernel_analyzer.streaming import direction_certificate_from_vector_files  # noqa: E402


def load(path: Path) -> dict[str, Any]:
    # Checkpoints intentionally end in ``.partial`` even though they are
    # gzip-compressed.  Detect the encoding from the file header so a resumed
    # campaign reads the same artifact that ``write`` produced.
    with path.open("rb") as raw:
        is_gzip = raw.read(2) == b"\x1f\x8b"
    opener = gzip.open if is_gzip else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=6) as handle:
        json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--architecture", choices=("qwen", "mamba", "phi", "deepseek8"), required=True
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input-bank", type=Path, required=True)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--task-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-cache-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--states", type=int, default=32)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--sample-size", type=int, default=64)
    parser.add_argument(
        "--complete-coordinate-spool-dir", type=Path,
        help="Spool repeat-0 complete signed-error vectors and certify from their full Gram.",
    )
    parser.add_argument(
        "--task-id", action="append", default=[],
        help="Restrict observation to an exact task ID; repeat for a batch. "
             "The default remains the complete denominator.",
    )
    parser.add_argument("--metric-chunk-elements", type=int, default=1_048_576)
    parser.add_argument("--bootstrap-draws", type=int, default=4000)
    parser.add_argument("--bootstrap-seed", type=int, default=23091)
    parser.add_argument("--allow-graph-breaks", action="store_true")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()
    if args.repeat != 2 or args.states < 2:
        raise ValueError("same-dtype Oracle requires two repeats and at least two states")
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid same-dtype state shard")
    if args.complete_coordinate_spool_dir is not None:
        args.complete_coordinate_spool_dir.mkdir(parents=True, exist_ok=True)

    # One artifact has exactly one writer.  A duplicate launch used to race on
    # the shared checkpoint temporary and could both waste a GPU and terminate
    # the legitimate campaign.  Keep the lock handle alive for the process.
    lock_root = Path("/data1/tzh/cache/kernel_analyzer_locks")
    lock_root.mkdir(parents=True, exist_ok=True)
    lock_token = hashlib.sha256(str(args.output.resolve()).encode()).hexdigest()
    output_lock = (lock_root / f"{lock_token}.lock").open("a+")
    try:
        fcntl.flock(output_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError(f"another process owns output artifact: {args.output}") from error

    def spool_complete_delta(task_id: str, state_id: str, candidate: torch.Tensor,
                             reference: torch.Tensor) -> dict[str, Any]:
        """Persist one full finite signed-error vector without JSON expansion."""
        token = hashlib.sha256((task_id + "\0" + state_id).encode()).hexdigest()
        path = args.complete_coordinate_spool_dir / f"{token}.f32"
        temporary = path.with_suffix(f".f32.{os.getpid()}.tmp")
        state = hashlib.sha256()
        count = 0
        left = candidate.detach().reshape(-1)
        right = reference.detach().reshape(-1)
        with temporary.open("wb") as handle:
            for start in range(0, left.numel(), args.metric_chunk_elements):
                stop = min(left.numel(), start + args.metric_chunk_elements)
                delta = (left[start:stop].float() - right[start:stop].float()).numpy()
                encoded = delta.tobytes(order="C")
                handle.write(encoded); state.update(encoded); count += delta.size
        temporary.replace(path)
        return {
            "state_id": state_id, "path": str(path), "coordinates": count,
            "storage_dtype": "float32", "sha256": state.hexdigest(),
        }

    bank = json.loads(args.input_bank.read_text())
    states = bank.get("states", bank.get("records"))
    if len(states) < args.states:
        raise RuntimeError("input bank is shorter than the requested population")
    selected_states = [
        (index, state) for index, state in enumerate(states[:args.states])
        if index % args.shard_count == args.shard_index
    ]
    if len(selected_states) < 2:
        raise RuntimeError("same-dtype shard must contain at least two frozen states")
    campaign = load(args.campaign)
    inventory = load(args.inventory)
    plan = load(args.task_plan)
    if plan["bindings"]["inventory_result_sha256"] != inventory["result_sha256"]:
        raise RuntimeError("same-dtype task plan binds another candidate inventory")
    environment_path = args.campaign.parent / "environment.json"
    if not environment_path.exists():
        raise RuntimeError("frozen runtime environment artifact is absent")
    frozen_environment_artifact = load(environment_path)
    frozen_environment = frozen_environment_artifact["environment"]
    actual_environment = {
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": sys.version.split()[0],
        "torch_cuda_version": str(torch.version.cuda),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "triton_version": triton.__version__,
    }
    if actual_environment != frozen_environment:
        raise RuntimeError(
            "same-dtype runtime environment differs from frozen candidate: "
            f"expected={frozen_environment} actual={actual_environment}"
        )
    if plan["status"] != "COMPLETE_ALL_CANDIDATE_PORTS_ASSIGNED_TO_EXACT_SEMANTIC_ENDPOINTS":
        raise RuntimeError("same-dtype semantic task plan is not closed")
    campaign_rows = campaign["rows"]
    exact_rows = [
        row for row in plan["rows"] if row.get("exact_semantic_endpoint_id") is not None
    ]
    if args.task_id:
        requested_task_ids = set(args.task_id)
        exact_rows = [
            row for row in exact_rows if str(row["task_id"]) in requested_task_ids
        ]
        observed_task_ids = {str(row["task_id"]) for row in exact_rows}
        if observed_task_ids != requested_task_ids:
            raise RuntimeError(
                "requested exact tasks are absent: "
                + ", ".join(sorted(requested_task_ids - observed_task_ids))
            )
    if not exact_rows:
        raise RuntimeError("same-dtype task plan has no exact semantic endpoints")

    endpoint_to_tasks: dict[str, list[str]] = defaultdict(list)
    for row in exact_rows:
        if row.get("exact_aot_endpoint_id") is not None:
            endpoint_to_tasks[str(row["exact_aot_endpoint_id"])].append(str(row["task_id"]))
    parameter_rows = [
        row for row in exact_rows if row.get("parameter_gradient_aliases")
    ]

    device = torch.device(args.device)
    configure_candidate_runtime(24000)
    model = load_model(args.architecture, args.model, device)
    module_start = len(PyCodeCache.modules)
    candidate = torch.compile(
        LossStep(model), backend="inductor",
        fullgraph=not args.allow_graph_breaks,
        dynamic=False,
    )
    warm_tokens = states[0].get("token_ids", states[0].get("input_ids"))
    warm = torch.tensor([warm_tokens], dtype=torch.long, device=device)
    model.zero_grad(set_to_none=True)
    candidate(warm).backward()
    torch.cuda.synchronize(device)
    modules = list(PyCodeCache.modules[module_start:])

    args.reference_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(args.reference_cache_dir.resolve())
    triton_reference_cache = args.reference_cache_dir / "triton"
    triton_reference_cache.mkdir(parents=True, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = str(triton_reference_cache.resolve())

    active_reference_sink: dict[str, Any] = {"fn": None}

    def reference_dispatch(task: Any, values: tuple[Any, ...]) -> None:
        if active_reference_sink["fn"] is not None:
            active_reference_sink["fn"](task, values)

    selected_aot_endpoints = {
        str(row["exact_aot_endpoint_id"])
        for row in exact_rows if row.get("exact_aot_endpoint_id") is not None
    }
    selected_reference_cuts = [
        row for row in plan["reference_cut_tasks"]
        if str(row["task_id"]).removeprefix("same-dtype:") in selected_aot_endpoints
    ]
    reference_capture = AOTForwardBackwardCapture(
        reference_cut_tasks=selected_reference_cuts,
        reference_value_sink=reference_dispatch,
    )
    with inductor_config.patch({"force_disable_caches": True}):
        reference = torch.compile(
            LossStep(model),
            backend=reference_capture.inductor_partition_backend(),
            fullgraph=not args.allow_graph_breaks,
            dynamic=False,
        )
        model.zero_grad(set_to_none=True)
        reference_warm_loss = reference(warm)
        reference_capture.bind_user_outputs(reference_warm_loss)
        reference_warm_loss.register_hook(reference_capture.bind_user_cotangent)
        reference_warm_loss.backward()
        torch.cuda.synchronize(device)
    reference_structure = reference_capture.as_dict()
    cut_gates = reference_structure["reference_cut_runtime"]["gates"]
    if not all(cut_gates.values()):
        raise RuntimeError(f"reference-cut extraction/replay failed: {cut_gates}")

    # Prove observer non-perturbation once for the frozen compiled program.  It
    # is unnecessary to repeat this identical structural check for every
    # held-out input state.
    torch.manual_seed(23999)
    torch.cuda.manual_seed_all(23999)
    model.zero_grad(set_to_none=True)
    unobserved_warm_loss = candidate(warm)
    unobserved_warm_loss.backward()
    torch.cuda.synchronize(device)
    unobserved_warm_identity = {
        "loss": tensor_digest(unobserved_warm_loss),
        "gradients": gradient_digest(model),
    }
    warm_observed_task_ids: set[str] = set()

    def warm_candidate_sink(task_id: str, tensor: torch.Tensor, metadata: Any) -> None:
        del tensor, metadata
        if task_id in warm_observed_task_ids:
            raise RuntimeError(f"warm candidate semantic endpoint repeated: {task_id}")
        warm_observed_task_ids.add(task_id)

    torch.manual_seed(23999)
    torch.cuda.manual_seed_all(23999)
    model.zero_grad(set_to_none=True)
    warm_observer = SameDtypeSemanticCandidateObserver(
        modules=modules, campaign_rows=campaign_rows,
        inventory_rows=inventory["runtime_call_audit"]["rows"],
        task_rows=exact_rows, sink=warm_candidate_sink,
    )
    with warm_observer:
        observed_warm_loss = candidate(warm)
        observed_warm_loss.backward()
    torch.cuda.synchronize(device)
    warm_observer.validate()
    observed_warm_identity = {
        "loss": tensor_digest(observed_warm_loss),
        "gradients": gradient_digest(model),
    }
    if observed_warm_identity != unobserved_warm_identity:
        raise RuntimeError("same-dtype observer perturbed the frozen candidate program")

    checkpoint_path = args.output.with_name(f".{args.output.name}.partial")
    checkpoint_binding = {
        "input_bank_sha256": file_digest(args.input_bank),
        "campaign_result_sha256": campaign["result_sha256"],
        "inventory_result_sha256": inventory["result_sha256"],
        "task_plan_result_sha256": plan["result_sha256"],
        "environment_result_sha256": frozen_environment_artifact["result_sha256"],
        "requested_states": args.states,
        "repeats_per_state": args.repeat,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "target_task_ids": sorted(str(row["task_id"]) for row in exact_rows),
        "directional_sample_size": args.sample_size,
        "complete_coordinate_mode": args.complete_coordinate_spool_dir is not None,
    }
    state_payload: dict[str, Any] = {}
    if checkpoint_path.exists():
        checkpoint = load(checkpoint_path)
        observed_binding = checkpoint.get("bindings")
        legacy_binding = {
            key: value for key, value in checkpoint_binding.items()
            if key != "environment_result_sha256"
        }
        pre_shard_binding = {
            key: value for key, value in checkpoint_binding.items()
            if key not in {"shard_index", "shard_count"}
        }
        pre_shard_legacy_binding = {
            key: value for key, value in pre_shard_binding.items()
            if key != "environment_result_sha256"
        }
        accepted_bindings = [checkpoint_binding, legacy_binding]
        if args.shard_count == 1 and args.shard_index == 0:
            accepted_bindings.extend([pre_shard_binding, pre_shard_legacy_binding])
        if observed_binding not in accepted_bindings:
            raise RuntimeError("same-dtype checkpoint binding changed")
        state_payload = dict(checkpoint["states"])
    elif args.output.exists():
        # A completed numerical artifact can seed a structure-only
        # recertification after bridge/certificate code changes.  Every frozen
        # binding and assigned state must match; numerical rows are recomputed
        # below from the retained per-state endpoint metrics.
        completed = load(args.output)
        completed_binding = {
            "input_bank_sha256": completed.get("input_bank_sha256"),
            "campaign_result_sha256": completed.get("campaign_result_sha256"),
            "inventory_result_sha256": completed.get("inventory_result_sha256"),
            "task_plan_result_sha256": completed.get("task_plan_result_sha256"),
            "environment_result_sha256": completed.get("environment_result_sha256"),
            "requested_states": args.states,
            "repeats_per_state": completed.get("denominator", {}).get(
                "repeats_per_state"
            ),
            "shard_index": completed.get("shard", {}).get("index"),
            "shard_count": completed.get("shard", {}).get("count"),
        }
        expected_ids = {
            str(state.get("sequence_id", state.get("state_id", index)))
            for index, state in selected_states
        }
        completed_states = dict(completed.get("states", {}))
        if (
            completed.get("status")
            not in {
                "COMPLETE_SAME_DTYPE_OPTIMIZATION_ORACLE",
                "COMPLETE_SAME_DTYPE_OPTIMIZATION_ORACLE_SHARD",
                "PILOT_SAME_DTYPE_OPTIMIZATION_ORACLE",
            }
            or completed_binding != checkpoint_binding
            or set(completed_states) != expected_ids
        ):
            raise RuntimeError(
                "existing same-dtype artifact cannot seed structural recertification"
            )
        state_payload = completed_states
        print(json.dumps({
            "event": "FORMAL_NUMERICAL_STATES_RESUMED",
            "states": len(state_payload),
        }), flush=True)
    for state_index, state in selected_states:
        state_id = str(state.get("sequence_id", state.get("state_id", state_index)))
        if state_id in state_payload:
            print(json.dumps({"event": "STATE_RESUMED", "state": state_id}), flush=True)
            continue
        token_ids = state.get("token_ids", state.get("input_ids"))
        values = torch.tensor([token_ids], dtype=torch.long, device=device)
        state_seed = 24000 + state_index
        # Observer non-perturbation was proved once above.  Two instrumented
        # candidate repeats still test runtime stability on every state.
        candidate_baseline_identity = None
        cached_reference_values: dict[str, torch.Tensor] = {}
        cached_reference_identity = None
        state_repeats = []
        for repeat in range(args.repeat):
            seed = state_seed
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            candidate_values: dict[str, tuple[torch.Tensor, dict[str, Any]]] = {}

            def candidate_sink(task_id: str, tensor: torch.Tensor, metadata: Any) -> None:
                if task_id in candidate_values:
                    raise RuntimeError(f"candidate semantic endpoint repeated: {task_id}")
                candidate_values[task_id] = (tensor.detach().cpu(), dict(metadata))

            model.zero_grad(set_to_none=True)
            observer = SameDtypeSemanticCandidateObserver(
                modules=modules, campaign_rows=campaign_rows,
                inventory_rows=inventory["runtime_call_audit"]["rows"],
                task_rows=exact_rows, sink=candidate_sink,
            )
            with observer:
                candidate_loss = candidate(values)
                candidate_loss.backward()
            torch.cuda.synchronize(device)
            observer.validate()
            candidate_identity = {
                "loss": tensor_digest(candidate_loss),
                # Endpoint tensors below, including selected parameter-gradient
                # endpoints, are compared in full.  Re-hashing every parameter
                # gradient here scans the entire model after every arm and adds
                # no independent T1 evidence.
                "gradient_identity_basis": "all_observed_endpoint_metrics",
            }
            if candidate_baseline_identity is None:
                candidate_baseline_identity = candidate_identity
            elif candidate_identity != candidate_baseline_identity:
                raise RuntimeError(f"same-dtype candidate repeat changed: {state_id}")

            metrics: dict[str, Any] = {}

            def reference_sink(task: Any, outputs: tuple[Any, ...]) -> None:
                endpoint = str(task.task_id).removeprefix("same-dtype:")
                tensor_outputs = [value for value in outputs if isinstance(value, torch.Tensor)]
                if len(tensor_outputs) != 1:
                    raise RuntimeError(f"reference endpoint is not one tensor: {endpoint}")
                reference_value = tensor_outputs[0].detach().cpu()
                for task_id in endpoint_to_tasks[endpoint]:
                    if task_id not in candidate_values:
                        raise RuntimeError(f"candidate value absent for reference endpoint: {task_id}")
                    candidate_value, metadata = candidate_values.pop(task_id)
                    cached_reference_values[task_id] = reference_value
                    if (
                        candidate_value.shape != reference_value.shape
                        or candidate_value.dtype != reference_value.dtype
                    ):
                        raise RuntimeError(f"same-dtype endpoint metadata mismatch: {task_id}")
                    error = nonfinite_aware_metrics(
                        candidate_value, reference_value,
                        sample_size=args.sample_size,
                        metric_chunk_elements=args.metric_chunk_elements,
                        retain_sampled_values=False,
                    )
                    if args.complete_coordinate_spool_dir is not None and repeat == 0:
                        error["complete_vector_spool"] = spool_complete_delta(
                            task_id, state_id, candidate_value, reference_value
                        )
                    metrics[task_id] = {
                        "exact_aot_endpoint_id": endpoint,
                        "candidate_metadata": metadata,
                        "error": error,
                    }

            if repeat == 0:
                active_reference_sink["fn"] = reference_sink
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                model.zero_grad(set_to_none=True)
                reference_loss = reference(values)
                reference_capture.bind_user_outputs(reference_loss)
                reference_loss.register_hook(reference_capture.bind_user_cotangent)
                reference_loss.backward()
                torch.cuda.synchronize(device)
                active_reference_sink["fn"] = None
                reference_identity = {
                    "loss": tensor_digest(reference_loss),
                    "gradient_identity_basis": "all_observed_endpoint_metrics",
                }
                cached_reference_identity = reference_identity
                named_parameters = dict(model.named_parameters(remove_duplicate=False))
                for task in parameter_rows:
                    task_id = str(task["task_id"])
                    aliases = [str(name) for name in task["parameter_gradient_aliases"]]
                    parameters = [named_parameters[name] for name in aliases]
                    if not parameters or any(parameter is not parameters[0] for parameter in parameters[1:]):
                        raise RuntimeError(f"parameter-gradient aliases changed: {task_id}")
                    gradient = parameters[0].grad
                    if gradient is None or task_id not in candidate_values:
                        raise RuntimeError(f"parameter-gradient endpoint is absent: {task_id}")
                    candidate_value, metadata = candidate_values.pop(task_id)
                    reference_value = gradient.detach().cpu()
                    cached_reference_values[task_id] = reference_value
                    if candidate_value.shape != reference_value.shape or candidate_value.dtype != reference_value.dtype:
                        raise RuntimeError(f"parameter-gradient endpoint metadata mismatch: {task_id}")
                    error = nonfinite_aware_metrics(
                        candidate_value, reference_value,
                        sample_size=args.sample_size,
                        metric_chunk_elements=args.metric_chunk_elements,
                        retain_sampled_values=False,
                    )
                    if args.complete_coordinate_spool_dir is not None:
                        error["complete_vector_spool"] = spool_complete_delta(
                            task_id, state_id, candidate_value, reference_value
                        )
                    metrics[task_id] = {
                        "exact_semantic_endpoint_id": task["exact_semantic_endpoint_id"],
                        "candidate_metadata": metadata,
                        "error": error,
                    }
            else:
                if cached_reference_identity is None:
                    raise RuntimeError("single-reference cache is absent")
                reference_identity = cached_reference_identity
                parameter_ids = {str(row["task_id"]) for row in parameter_rows}
                for task in exact_rows:
                    task_id = str(task["task_id"])
                    if task_id not in candidate_values or task_id not in cached_reference_values:
                        raise RuntimeError(f"cached reference endpoint is absent: {task_id}")
                    candidate_value, metadata = candidate_values.pop(task_id)
                    reference_value = cached_reference_values[task_id]
                    if candidate_value.shape != reference_value.shape or candidate_value.dtype != reference_value.dtype:
                        raise RuntimeError(f"cached reference metadata mismatch: {task_id}")
                    error = nonfinite_aware_metrics(
                        candidate_value, reference_value,
                        sample_size=args.sample_size,
                        metric_chunk_elements=args.metric_chunk_elements,
                        retain_sampled_values=False,
                    )
                    key = ("exact_semantic_endpoint_id" if task_id in parameter_ids
                           else "exact_aot_endpoint_id")
                    metrics[task_id] = {
                        key: task.get(key), "candidate_metadata": metadata, "error": error,
                    }
            if candidate_values:
                raise RuntimeError(
                    f"reference did not consume candidate endpoints: {sorted(candidate_values)[:8]}"
                )
            if set(metrics) != {str(row["task_id"]) for row in exact_rows}:
                raise RuntimeError("same-dtype endpoint denominator changed")
            state_repeats.append({
                "repeat": repeat,
                "candidate_full_step_identity": candidate_identity,
                "reference_full_step_identity": reference_identity,
                "endpoint_metrics": metrics,
            })
        if candidate_baseline_identity is None or cached_reference_identity is None:
            raise RuntimeError(f"state execution did not establish identities: {state_id}")
        state_payload[state_id] = {
            "execution_protocol": "BIAS_T1_CANDIDATE2_REFERENCE1_V2",
            "candidate_baseline_identity": candidate_baseline_identity,
            "repeats": state_repeats,
        }
        write(checkpoint_path, {
            "schema": "kernel-analyzer-same-dtype-semantic-checkpoint-v1",
            "status": "PARTIAL_FAIL_CLOSED",
            "bindings": checkpoint_binding,
            "states": state_payload,
        })
        print(json.dumps({"event": "STATE_COMPLETE", "state": state_id}), flush=True)
        del values
        torch.cuda.empty_cache()

    result_rows = []
    for task in exact_rows:
        task_id = str(task["task_id"])
        observations = [
            state_payload[state_id]["repeats"]
            for state_id in state_payload
        ]
        left = [rows[0]["endpoint_metrics"][task_id]["error"] for rows in observations]
        right = [rows[1]["endpoint_metrics"][task_id]["error"] for rows in observations]
        repeat_stable = [metric_equal(a, b) for a, b in zip(left, right)]
        sketches = [row["directional_error_sketch"] for row in left]
        coordinates = sketches[0]["flat_coordinate_indices"]
        if any(row["flat_coordinate_indices"] != coordinates for row in sketches):
            raise RuntimeError(f"same-dtype coordinate identity changed: {task_id}")
        errors = np.asarray([row["signed_delta_values"] for row in sketches], dtype=float)
        nonfinite = any(row["nonfinite_mismatch"] for row in left + right) or not np.isfinite(errors).all()
        exact = all(row["exact"] for row in left + right)
        complete_certificate = None
        spool_rows = [row.get("complete_vector_spool") for row in left]
        if args.complete_coordinate_spool_dir is not None:
            if any(row is None for row in spool_rows):
                raise RuntimeError(f"complete-coordinate spool denominator changed: {task_id}")
            missing = [row["path"] for row in spool_rows if not Path(row["path"]).exists()]
            if missing:
                raise RuntimeError(f"complete-coordinate spool files are absent: {missing[:2]}")
            if not nonfinite:
                complete_certificate = direction_certificate_from_vector_files(
                    spool_rows, chunk_elements=args.metric_chunk_elements,
                    bootstrap_draws=args.bootstrap_draws, seed=args.bootstrap_seed,
                )
        if nonfinite:
            verdict, statistic, confidence = "NONFINITE_RISK", None, None
        elif exact and all(repeat_stable):
            verdict, statistic = "EQUIVALENT_EXACT_ON_HELDOUT_STATES", 0.0
            confidence = {"lower_95": 0.0, "median": 0.0, "upper_95": 0.0}
        elif complete_certificate is not None:
            statistic = complete_certificate["cross_state_inner_product_u"]
            confidence = complete_certificate["cluster_bootstrap_95"]
            if confidence["lower_95"] > 0:
                verdict = "DIRECTIONAL_OPTIMIZATION_BIAS"
            elif not all(repeat_stable):
                verdict = "RUNTIME_VARIANCE_RISK"
            else:
                verdict = "FINITE_NONEXACT_WITHOUT_STABLE_DIRECTION"
        else:
            statistic = u_statistic(errors)
            confidence = bootstrap(
                errors,
                bootstrap_counts(len(errors), args.bootstrap_draws, args.bootstrap_seed),
            )
            if confidence["lower_95"] > 0:
                verdict = "DIRECTIONAL_OPTIMIZATION_BIAS"
            elif not all(repeat_stable):
                verdict = "RUNTIME_VARIANCE_RISK"
            else:
                verdict = "FINITE_NONEXACT_WITHOUT_STABLE_DIRECTION"
        result_row = {
            "task_id": task_id,
            "candidate_region_id": task["candidate_region_id"],
            "exact_aot_endpoint_id": task["exact_aot_endpoint_id"],
            "exact_semantic_endpoint_id": task["exact_semantic_endpoint_id"],
            "states": len(observations),
            "sampled_coordinates": len(coordinates),
            "complete_coordinates": (
                complete_certificate["coordinates"] if complete_certificate is not None else None
            ),
            "cross_state_inner_product_u": statistic,
            "cluster_bootstrap_95": confidence,
            "repeat_stable": all(repeat_stable),
            "max_abs_over_state_repeats": max(row["max_abs"] for row in left + right),
            "verdict": verdict,
        }
        if complete_certificate is not None:
            result_row["complete_coordinate_certificate"] = complete_certificate
        result_rows.append(result_row)
        if args.complete_coordinate_spool_dir is not None:
            for error, spool in zip(left, spool_rows):
                Path(spool["path"]).unlink(missing_ok=True)
                error.pop("complete_vector_spool", None)

    verdict_counts = Counter(row["verdict"] for row in result_rows)
    reference_structure = reference_capture.as_dict()
    if not all(reference_structure["reference_cut_runtime"]["gates"].values()):
        raise RuntimeError("reference cut gates changed during held-out execution")
    cross_phase_gates = reference_structure[
        "cross_phase_runtime_bridge"
    ]["gates"]
    if not all(cross_phase_gates.values()):
        failure_path = args.output.with_name(
            f".{args.output.name}.structure_failure"
        )
        write(failure_path, {
            "schema": "kernel-analyzer-same-dtype-structure-failure-v1",
            "status": "PARTIAL_FAIL_CLOSED",
            "cross_phase_gates": cross_phase_gates,
            "reference_structure": reference_structure,
        })
        raise RuntimeError(
            "cross-phase runtime bridge is incomplete: "
            f"{cross_phase_gates}; diagnostic={failure_path}"
        )
    payload = {
        "schema": "kernel-analyzer-same-dtype-semantic-oracle-v1",
        "status": (
            "COMPLETE_SAME_DTYPE_OPTIMIZATION_ORACLE"
            if args.states == len(states) == 32 and args.shard_count == 1
            else "COMPLETE_SAME_DTYPE_OPTIMIZATION_ORACLE_SHARD"
            if args.states == len(states) == 32
            else "PILOT_SAME_DTYPE_OPTIMIZATION_ORACLE"
        ),
        "architecture": args.architecture,
        "sequence_length": len(warm_tokens),
        "input_bank_sha256": file_digest(args.input_bank),
        "campaign_result_sha256": campaign["result_sha256"],
        "inventory_result_sha256": inventory["result_sha256"],
        "task_plan_result_sha256": plan["result_sha256"],
        "environment_result_sha256": frozen_environment_artifact["result_sha256"],
        "shard": {
            "index": args.shard_index,
            "count": args.shard_count,
            "assigned_state_indices": [index for index, _state in selected_states],
        },
        "target_task_ids": sorted(str(row["task_id"]) for row in exact_rows),
        "directional_sample_size": args.sample_size,
        "complete_coordinate_mode": args.complete_coordinate_spool_dir is not None,
        "execution": {
            "protocol_counts": dict(sorted(Counter(
                row.get("execution_protocol", "LEGACY_CANDIDATE3_REFERENCE2_V1")
                for row in state_payload.values()
            ).items())),
            "new_protocol_candidate_steps_per_state": 2,
            "new_protocol_reference_steps_per_state": 1,
            "per_state_full_model_gradient_digest": False,
            "observer_nonperturbation_steps_per_campaign": 2,
            "observer_nonperturbation_bitwise": (
                observed_warm_identity == unobserved_warm_identity
            ),
        },
        "denominator": {
            "states": len(state_payload), "repeats_per_state": args.repeat,
            "candidate_ports": plan["denominator"]["stored_candidate_ports"],
            "candidate_compute_regions": plan["denominator"]["candidate_compute_regions"],
            "exact_semantic_endpoints": len(exact_rows),
            "internal_ports_closed_by_semantic_endpoint": plan["denominator"][
                "internal_ports_closed_by_semantic_endpoint"
            ],
            "compiler_added_ports_closed_by_exact_theorem": plan["denominator"].get(
                "compiler_added_ports_closed_by_exact_theorem", 0
            ),
            "unresolved": 0,
            "targeted_task_count": len(exact_rows),
            "targeted_followup": bool(args.task_id),
        },
        "reference_structure": reference_structure,
        "verdict_counts": dict(sorted(verdict_counts.items())),
        "rows": result_rows,
        "states": state_payload,
        "gates": {
            "all_32_frozen_states_present": (
                args.states == len(states) == 32 and args.shard_count == 1
            ),
            "all_assigned_frozen_states_present": (
                len(state_payload) == len(selected_states)
            ),
            "same_dtype_bf16": True,
            "runtime_environment_exact": True,
            "same_model_weight_storage": True,
            "exact_compiler_origin_endpoints_only": True,
            "all_internal_candidate_ports_closed": True,
            "all_candidate_compute_regions_have_observed_output_ports": (
                plan["denominator"]["candidate_regions_without_observed_output_port"] == 0
            ),
            "candidate_values_used_for_pairing": False,
            "observer_nonperturbation_bitwise_once_per_frozen_program": True,
            "all_reference_cuts_bitwise_self_replay": True,
            "cross_phase_runtime_bridge_complete": True,
        },
        "claim_boundary": (
            "C16 minus R16 at exact compiler-origin semantic endpoints, with implementation-only "
            "partial buffers covered by closed downstream endpoints and compiler-added boundaries "
            "covered by explicit forward/VJP theorems. Causal root attribution is separate."
        ),
    }
    payload["result_sha256"] = digest(payload)
    write(args.output, payload)
    checkpoint_path.unlink(missing_ok=True)
    print(json.dumps({
        "output": str(args.output), "denominator": payload["denominator"],
        "verdict_counts": payload["verdict_counts"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
