#!/usr/bin/env python
"""Replay one arm of the frozen Qwen3 GRPO step-29 natural transition.

One invocation executes exactly one eager or compiled arm in a fresh process.
It restores the captured model, AdamW, scheduler, GradScaler, RNG and target
minibatch, then performs one real GRPO update.  Cross-arm analysis is performed
by a separate evaluator so that self repeats remain process independent.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import random
import re
import time
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA_VERSION = "forkcert.qwen3-grpo-natural-transition-arm.v0.2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", required=True)
    parser.add_argument("--anchor-states")
    parser.add_argument("--realization-contract")
    parser.add_argument(
        "--instantiate-realization-contract",
        help=(
            "Write a prospective scorer/compiler realization contract and exit "
            "before loss, backward, or optimizer execution. Requires --arm compiled "
            "and forbids both existing anchor sources."
        ),
    )
    parser.add_argument("--arm", choices=["eager", "compiled"], required=True)
    parser.add_argument("--repeat", type=int, required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--save-vectors", action="store_true")
    parser.add_argument(
        "--trace-dir",
        help="Enable TorchInductor debug/provenance artifacts under this directory.",
    )
    parser.add_argument(
        "--graph-manifest-dir",
        help="Write Dynamo/FX node metadata captured by the existing backend wrapper.",
    )
    parser.add_argument(
        "--dump-kernel-inputs",
        help=(
            "Comma-separated generated Triton kernel-name filters. This uses "
            "TorchInductor's launch-tensor dumper and requires a new equivalence gate."
        ),
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(tensor: Any) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(tuple(value.shape)).encode())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def move_tree(value: Any, device: str) -> Any:
    if hasattr(value, "to"):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_tree(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_tree(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_tree(item, device) for item in value)
    return value


def restore_rng(torch: Any, payload: dict[str, Any]) -> None:
    torch.random.set_rng_state(payload["torch"]["cpu"])
    torch.cuda.set_rng_state_all(payload["torch"]["cuda"])
    random.setstate(payload["python"])
    np.random.set_state(payload["numpy"])


def rng_fingerprint(torch: Any) -> dict[str, str]:
    return {
        "torch_cpu": tensor_sha256(torch.random.get_rng_state()),
        "torch_cuda": json_sha256([tensor_sha256(x) for x in torch.cuda.get_rng_state_all()]),
        "python": json_sha256(random.getstate()),
        "numpy": json_sha256(np.random.get_state()),
    }


def named_tensor_hashes(values: Any) -> tuple[dict[str, str], str]:
    rows = {name: tensor_sha256(value) for name, value in values}
    return rows, json_sha256(rows)


def optimizer_tensor_hashes(optimizer: Any, named_parameters: list[tuple[str, Any]]) -> dict[str, Any]:
    reverse = {id(parameter): name for name, parameter in named_parameters}
    rows: dict[str, dict[str, str]] = {}
    scalars: dict[str, dict[str, Any]] = {}
    for parameter, state in optimizer.state.items():
        name = reverse[id(parameter)]
        rows[name] = {}
        scalars[name] = {}
        for key, value in state.items():
            if hasattr(value, "detach"):
                rows[name][str(key)] = tensor_sha256(value)
            else:
                scalars[name][str(key)] = value
    payload = {"tensors": rows, "scalars": scalars}
    payload["sha256"] = json_sha256(payload)
    return payload


def tensor_collection_summary(torch: Any, named_values: list[tuple[str, Any]]) -> dict[str, Any]:
    square = 0.0
    max_abs = 0.0
    finite = True
    nonzero = 0
    hashes: dict[str, str | None] = {}
    per_parameter: list[dict[str, Any]] = []
    for name, value in named_values:
        if value is None:
            hashes[name] = None
            per_parameter.append({"name": name, "present": False})
            continue
        work = value.detach().float()
        current_finite = bool(torch.isfinite(work).all().item())
        finite = finite and current_finite
        current_square = float((work * work).sum(dtype=torch.float64).item())
        current_max = float(work.abs().max().item())
        square += current_square
        max_abs = max(max_abs, current_max)
        current_nonzero = int(torch.count_nonzero(work).item())
        nonzero += current_nonzero
        hashes[name] = tensor_sha256(value)
        per_parameter.append(
            {
                "name": name,
                "present": True,
                "shape": list(value.shape),
                "l2": math.sqrt(current_square),
                "max_abs": current_max,
                "nonzero": current_nonzero,
                "sha256": hashes[name],
            }
        )
    return {
        "l2": math.sqrt(square),
        "max_abs": max_abs,
        "finite": finite,
        "nonzero": nonzero,
        "tensor_hashes_sha256": json_sha256(hashes),
        "per_parameter": per_parameter,
    }


def clip_decisions(torch: Any, logps: Any, inputs: dict[str, Any], epsilon: float) -> Any:
    advantages = inputs["advantages"].unsqueeze(1)
    ratio = torch.exp(logps - inputs["old_per_token_logps"])
    return ((ratio < 1.0 - epsilon) & (advantages < 0)) | (
        (ratio > 1.0 + epsilon) & (advantages > 0)
    )


def grpo_loss(torch: Any, logps: Any, inputs: dict[str, Any], epsilon: float) -> Any:
    advantages = inputs["advantages"].unsqueeze(1)
    mask = inputs["completion_mask"]
    ratio = torch.exp(logps - inputs["old_per_token_logps"])
    clipped = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon)
    per_token = -torch.minimum(ratio * advantages, clipped * advantages)
    return ((per_token * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)).mean()


def make_optimizer(torch: Any, model: Any, snapshot: dict[str, Any]) -> Any:
    from torch import nn
    from transformers.trainer_pt_utils import get_parameter_names

    forbidden = [r"bias", r"layernorm", r"rmsnorm", r"(?:^|\.)norm(?:$|\.)", r"_norm(?:$|\.)"]
    decay = set(get_parameter_names(model, [nn.LayerNorm], forbidden))
    groups = [
        {"params": [p for n, p in model.named_parameters() if n in decay and p.requires_grad], "weight_decay": 0.0},
        {"params": [p for n, p in model.named_parameters() if n not in decay and p.requires_grad], "weight_decay": 0.0},
    ]
    if [len(group["params"]) for group in groups] != [len(group["params"]) for group in snapshot["param_groups"]]:
        raise RuntimeError("optimizer parameter grouping does not reproduce the captured 197/113 grouping")
    optimizer = torch.optim.AdamW(
        groups,
        lr=1e-6,
        betas=(0.9, 0.999),
        eps=1e-8,
        fused=True,
    )
    optimizer.load_state_dict(snapshot)
    return optimizer


def load_anchor(path: Path, step: int) -> dict[str, Any]:
    rows = [row for row in read_jsonl(path) if int(row["optimizer_step"]) == step]
    if len(rows) != 1:
        raise ValueError(f"expected exactly one anchor row for step {step}, found {len(rows)}")
    return rows[0]


def load_realization_contract(path: Path, snapshot_dir: Path, step: int) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != "forkcert.qwen3-transition-realization-contract.v0.1":
        raise ValueError("unsupported realization contract schema")
    if int(value.get("optimizer_step", -1)) != step:
        raise ValueError("realization contract optimizer_step differs from snapshot")
    expected_snapshot = sha256_file(snapshot_dir / "forkcert_transition_snapshot.json")
    if value.get("snapshot_metadata_sha256") != expected_snapshot:
        raise ValueError("realization contract snapshot identity differs from snapshot")
    content = {key: item for key, item in value.items() if key != "contract_sha256"}
    if value.get("contract_sha256") != json_sha256(content):
        raise ValueError("realization contract digest mismatch")
    if value.get("status") != "FROZEN_BEFORE_TRANSITION_ENDPOINT_EXECUTION":
        raise ValueError("realization contract is not frozen and valid")
    if value.get("history_state_preserved") is not True:
        raise ValueError("realization contract history did not preserve state")
    if value.get("contract_state_preserved") is not True:
        raise ValueError("realization contract preflight did not preserve state")
    return value


def compiler_protocol_identity(torch: Any) -> dict[str, Any]:
    return {
        "torch_version": torch.__version__,
        "cuda_build": torch.version.cuda,
        "backend": "inductor",
        "torch_compile_dynamic": "default",
        "torch_compile_fullgraph": False,
        "dynamo_suppress_errors": False,
        "dynamo_recompile_limit": 64,
        "attention_backend": "SDPA_MATH",
        "accelerate_mixed_precision": "fp16",
        "deterministic_algorithms": True,
        "cudnn_benchmark": False,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "history_scope": "all captured optimizer pre-step scorer inputs through target step",
    }


def main() -> None:
    args = parse_args()
    if args.dump_kernel_inputs and args.arm != "compiled":
        raise ValueError("kernel launch-input dumping requires --arm compiled")
    instantiate_contract = args.instantiate_realization_contract is not None
    existing_identity_sources = int(args.anchor_states is not None) + int(
        args.realization_contract is not None
    )
    if instantiate_contract:
        if args.arm != "compiled" or existing_identity_sources != 0:
            raise ValueError(
                "contract instantiation requires --arm compiled and no existing anchor source"
            )
    elif existing_identity_sources != 1:
        raise ValueError(
            "normal transition execution requires exactly one of --anchor-states "
            "or --realization-contract"
        )
    snapshot_dir = Path(args.snapshot_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("PYTHONHASHSEED", "0")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if args.trace_dir:
        os.environ.setdefault("INDUCTOR_PROVENANCE", "1")
    if args.dump_kernel_inputs:
        os.environ["TORCHINDUCTOR_DUMP_LAUNCH_TENSORS"] = "1"
        os.environ["TORCHINDUCTOR_KERNELS_TO_DUMP"] = args.dump_kernel_inputs

    import torch
    from accelerate import Accelerator
    from safetensors.torch import save_file
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from transformers import AutoModelForCausalLM, get_scheduler
    from trl.trainer.grpo_trainer import selective_log_softmax

    dump_hook = None
    if args.dump_kernel_inputs:
        # Kernel objects read the dump flag at construction time, but the
        # launch helper is resolved dynamically.  Suppress history dumps and
        # restore the official helper only for the measured target call.
        import torch._inductor.runtime.triton_heuristics as triton_heuristics

        dump_hook = triton_heuristics._dump_launch_tensors
        triton_heuristics._dump_launch_tensors = lambda *values, **kwargs: None

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one visible CUDA device is required")
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch._dynamo.config.suppress_errors = False
    torch._dynamo.config.recompile_limit = 64
    trace_dir = Path(args.trace_dir).resolve() if args.trace_dir else None
    graph_manifest_dir = (
        Path(args.graph_manifest_dir).resolve() if args.graph_manifest_dir else None
    )
    if trace_dir is not None:
        trace_dir.mkdir(parents=True, exist_ok=False)
        torch._inductor.config.trace.enabled = True
        torch._inductor.config.trace.debug_dir = str(trace_dir)
    if graph_manifest_dir is not None:
        graph_manifest_dir.mkdir(parents=True, exist_ok=False)

    metadata = json.loads((snapshot_dir / "forkcert_transition_snapshot.json").read_text())
    if metadata["schema_version"] != "forkcert.full-pre-minibatch-transition-state.v0.1":
        raise ValueError("unsupported snapshot schema")
    step = int(metadata["optimizer_step"])
    anchor = load_anchor(Path(args.anchor_states), step) if args.anchor_states else None
    realization_contract = (
        load_realization_contract(Path(args.realization_contract), snapshot_dir, step)
        if args.realization_contract
        else None
    )
    if realization_contract is not None:
        current_protocol = compiler_protocol_identity(torch)
        if json_sha256(current_protocol) != realization_contract.get("compiler_config_digest"):
            raise ValueError("current compiler protocol differs from realization contract")
    target_path = Path(metadata["target_minibatch_path"])
    if not target_path.is_file():
        target_path = snapshot_dir / "compiler_history" / target_path.name
    if realization_contract is not None and realization_contract.get(
        "target_minibatch_sha256"
    ) != sha256_file(target_path):
        raise ValueError("realization contract target minibatch differs from snapshot")
    inputs_cpu = torch.load(target_path, map_location="cpu", weights_only=False)
    inputs = move_tree(inputs_cpu, "cuda")

    model = AutoModelForCausalLM.from_pretrained(
        snapshot_dir,
        dtype=torch.float32,
        trust_remote_code=False,
        attn_implementation="sdpa",
        local_files_only=True,
    ).to("cuda")
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.train()
    accelerator = Accelerator(mixed_precision="fp16")
    model = accelerator.prepare_model(model)
    if not (hasattr(model, "_original_forward") and hasattr(model.forward, "__wrapped__")):
        raise RuntimeError("failed to reproduce the captured Accelerate FP16 forward wrapper")
    named_parameters = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    named_buffers = list(model.named_buffers())

    optimizer_snapshot = torch.load(snapshot_dir / "optimizer.pt", map_location="cpu", weights_only=False)
    optimizer = make_optimizer(torch, model, optimizer_snapshot)
    del optimizer_snapshot
    training_horizon = int(metadata.get("training_horizon_optimizer_steps", 30))
    if training_horizon <= step:
        raise ValueError(
            f"snapshot training horizon must exceed pre-transition step: horizon={training_horizon}, step={step}"
        )
    scheduler = get_scheduler(
        "linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=training_horizon
    )
    scheduler.load_state_dict(torch.load(snapshot_dir / "scheduler.pt", map_location="cpu", weights_only=False))
    scaler = accelerator.scaler
    if scaler is None:
        raise RuntimeError("Accelerate did not create the captured native FP16 GradScaler")
    scaler.load_state_dict(torch.load(snapshot_dir / "scaler.pt", map_location="cpu", weights_only=False))
    saved_rng = torch.load(snapshot_dir / "rng_state.pth", map_location="cpu", weights_only=False)
    gc.collect()

    pre_parameter_hashes, pre_parameter_digest = named_tensor_hashes(named_parameters)
    pre_buffer_hashes, pre_buffer_digest = named_tensor_hashes(named_buffers)
    pre_optimizer = optimizer_tensor_hashes(optimizer, named_parameters)
    pre_scheduler = scheduler.state_dict()
    pre_scaler = scaler.state_dict()

    compile_audit = {
        "backend_compiles": 0,
        "runtime_invocations": 0,
        "graph_code_sha256": [],
        "graph_node_counts": [],
        "graph_manifests": [],
    }
    if args.arm == "compiled":
        from torch._dynamo.backends.registry import lookup_backend

        inductor = lookup_backend("inductor")

        def backend(graph_module: Any, example_inputs: list[Any]):
            compile_audit["backend_compiles"] += 1
            graph_hash = hashlib.sha256(graph_module.code.encode()).hexdigest()
            nodes = list(graph_module.graph.nodes)
            compile_audit["graph_code_sha256"].append(graph_hash)
            compile_audit["graph_node_counts"].append(len(nodes))
            if graph_manifest_dir is not None:
                manifest = {
                    "compile_index": compile_audit["backend_compiles"] - 1,
                    "graph_code_sha256": graph_hash,
                    "graph_node_count": len(nodes),
                    "nodes": [
                        {
                            "name": node.name,
                            "op": node.op,
                            "target": str(node.target),
                            "nn_module_stack": str(node.meta.get("nn_module_stack")),
                            "source_fn_stack": str(node.meta.get("source_fn_stack")),
                            "stack_trace": node.meta.get("stack_trace"),
                            "original_aten": str(node.meta.get("original_aten")),
                        }
                        for node in nodes
                    ],
                }
                manifest_path = graph_manifest_dir / (
                    f"graph_{manifest['compile_index']:02d}_{graph_hash}.json"
                )
                manifest_path.write_text(
                    json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                compile_audit["graph_manifests"].append(
                    {
                        "path": str(manifest_path),
                        "sha256": sha256_file(manifest_path),
                        "graph_code_sha256": graph_hash,
                    }
                )
            compiled = inductor(graph_module, example_inputs)

            def counted(*values: Any):
                compile_audit["runtime_invocations"] += 1
                return compiled(*values)

            return counted

        execution_model = torch.compile(model, backend=backend)
        if [id(p) for p in execution_model.parameters()] != [id(p) for p in model.parameters()]:
            raise RuntimeError("compiled wrapper does not share underlying model parameter objects")
    else:
        execution_model = model

    def score(target_model: Any, packed_inputs: dict[str, Any]) -> Any:
        prompt_ids = packed_inputs["prompt_ids"]
        completion_ids = packed_inputs["completion_ids"]
        attention_mask = torch.cat([packed_inputs["prompt_mask"], packed_inputs["completion_mask"]], dim=1)
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        with sdpa_kernel(SDPBackend.MATH):
            outputs = target_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                logits_to_keep=completion_ids.size(1) + 1,
                use_cache=False,
            )
            logits = outputs.logits[:, :-1, :]
            logits = logits[:, -completion_ids.size(1) :, :]
            values = selective_log_softmax(logits, completion_ids)
        return values

    history_valid = True
    history_runtime_invocations = 0
    if args.arm == "compiled":
        state_before_history = {
            "parameters": pre_parameter_digest,
            "buffers": pre_buffer_digest,
            "optimizer": pre_optimizer["sha256"],
            "scheduler": json_sha256(pre_scheduler),
            "scaler": json_sha256(pre_scaler),
        }
        restore_rng(torch, saved_rng)
        for record in metadata["compiler_history"]:
            history_path = Path(record["path"])
            if not history_path.is_file():
                history_path = snapshot_dir / "compiler_history" / history_path.name
            history_inputs = move_tree(torch.load(history_path, map_location="cpu", weights_only=False), "cuda")
            before = int(compile_audit["runtime_invocations"])
            history_output = score(execution_model, history_inputs)
            history_runtime_invocations += int(compile_audit["runtime_invocations"]) - before
            del history_output, history_inputs
            gc.collect()
        restore_rng(torch, saved_rng)
        _, after_parameter_digest = named_tensor_hashes(named_parameters)
        _, after_buffer_digest = named_tensor_hashes(named_buffers)
        state_after_history = {
            "parameters": after_parameter_digest,
            "buffers": after_buffer_digest,
            "optimizer": optimizer_tensor_hashes(optimizer, named_parameters)["sha256"],
            "scheduler": json_sha256(scheduler.state_dict()),
            "scaler": json_sha256(scaler.state_dict()),
        }
        history_valid = state_before_history == state_after_history
        def ordered_unique(values: list[tuple[str, int]]) -> list[tuple[str, int]]:
            result: list[tuple[str, int]] = []
            for value in values:
                if value not in result:
                    result.append(value)
            return result

        actual_family = ordered_unique(
            list(zip(compile_audit["graph_code_sha256"], compile_audit["graph_node_counts"], strict=True))
        )
        if anchor is not None:
            expected_family = ordered_unique(
                list(
                    zip(
                        anchor["compile_audit"]["graph_code_sha256_so_far"],
                        anchor["compile_audit"]["graph_node_counts_so_far"],
                        strict=True,
                    )
                )
            )
        elif realization_contract is not None:
            expected_family = [
                (row["graph_code_sha256"], int(row["graph_node_count"]))
                for row in realization_contract["candidate_ordered_unique_graph_family"]
            ]
        else:
            # Prospective instantiation defines this realization before any
            # transition endpoint is executed.  Subsequent arms must match the
            # emitted contract; this first pass is not an effect measurement.
            expected_family = actual_family
        history_graph_identity_exact = actual_family == expected_family
        if not history_valid:
            raise RuntimeError("candidate compiler-history replay mutated non-implementation state")
        if not history_graph_identity_exact:
            diagnostic = {
                "schema_version": SCHEMA_VERSION,
                "valid": False,
                "verdict": "INVALID_GRAPH_HISTORY_IDENTITY",
                "actual_graph_code_sha256": compile_audit["graph_code_sha256"],
                "expected_graph_code_sha256": [value[0] for value in expected_family],
                "actual_graph_node_counts": compile_audit["graph_node_counts"],
                "expected_graph_node_counts": [value[1] for value in expected_family],
                "actual_backend_compiles": compile_audit["backend_compiles"],
                "history_runtime_invocations": history_runtime_invocations,
                "actual_ordered_unique_graph_family": actual_family,
                "expected_ordered_unique_graph_family": expected_family,
            }
            (out_dir / "invalid_diagnostic.json").write_text(
                json.dumps(diagnostic, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            raise RuntimeError("candidate compiler-history graph identity differs from the captured treatment")
    else:
        history_graph_identity_exact = True

    if dump_hook is not None:
        import torch._inductor.runtime.triton_heuristics as triton_heuristics

        triton_heuristics._dump_launch_tensors = dump_hook
    restore_rng(torch, saved_rng)
    rng_before_transition = rng_fingerprint(torch)
    optimizer.zero_grad(set_to_none=True)
    before_call = int(compile_audit["runtime_invocations"])
    before_measured_compiles = int(compile_audit["backend_compiles"])
    started = time.perf_counter_ns()
    logps = score(execution_model, inputs)
    measured_invocations = int(compile_audit["runtime_invocations"]) - before_call
    measured_backend_compiles = int(compile_audit["backend_compiles"]) - before_measured_compiles
    scorer_hash = tensor_sha256(logps)
    if anchor is not None:
        expected_hash = anchor["ref_first_sha256"] if args.arm == "eager" else anchor["alt_first_sha256"]
    elif realization_contract is not None:
        expected_hash = realization_contract[
            "reference_scorer_sha256" if args.arm == "eager" else "candidate_scorer_sha256"
        ]
    else:
        expected_hash = scorer_hash
    scorer_anchor_exact = scorer_hash == expected_hash

    if instantiate_contract:
        # The eager scorer is measured only to define the paired scorer
        # identity.  No loss, backward, clipping, or optimizer transition has
        # occurred, so this artifact cannot be mistaken for a U/T endpoint.
        restore_rng(torch, saved_rng)
        with torch.no_grad():
            reference_logps = score(model, inputs)
        reference_hash = tensor_sha256(reference_logps)
        restore_rng(torch, saved_rng)
        _, contract_parameter_digest = named_tensor_hashes(named_parameters)
        _, contract_buffer_digest = named_tensor_hashes(named_buffers)
        contract_state_preserved = (
            contract_parameter_digest == pre_parameter_digest
            and contract_buffer_digest == pre_buffer_digest
            and optimizer_tensor_hashes(optimizer, named_parameters)["sha256"]
            == pre_optimizer["sha256"]
            and json_sha256(scheduler.state_dict()) == json_sha256(pre_scheduler)
            and json_sha256(scaler.state_dict()) == json_sha256(pre_scaler)
            and rng_fingerprint(torch) == rng_before_transition
        )
        compiler_protocol = compiler_protocol_identity(torch)
        family_rows = [
            {"graph_code_sha256": graph_hash, "graph_node_count": node_count}
            for graph_hash, node_count in actual_family
        ]
        contract = {
            "schema_version": "forkcert.qwen3-transition-realization-contract.v0.1",
            "status": "FROZEN_BEFORE_TRANSITION_ENDPOINT_EXECUTION",
            "snapshot_metadata_sha256": sha256_file(
                snapshot_dir / "forkcert_transition_snapshot.json"
            ),
            "optimizer_step": step,
            "target_minibatch_sha256": sha256_file(target_path),
            "compiler_protocol": compiler_protocol,
            "compiler_config_digest": json_sha256(compiler_protocol),
            "candidate_ordered_unique_graph_family": family_rows,
            "graph_family_digest": json_sha256(family_rows),
            "reference_scorer_sha256": reference_hash,
            "candidate_scorer_sha256": scorer_hash,
            "history_records": len(metadata["compiler_history"]),
            "history_state_preserved": history_valid,
            "contract_state_preserved": contract_state_preserved,
            "candidate_measured_runtime_invocations": measured_invocations,
            "candidate_measured_backend_compiles": measured_backend_compiles,
            "nonclaims": [
                "this contract defines a prospective implementation treatment, not an effect",
                "scorer discrepancy is not a correctness verdict",
                "no loss, backward, clipping, or optimizer endpoint was executed",
            ],
        }
        contract["contract_sha256"] = json_sha256(contract)
        if not (
            history_valid
            and contract_state_preserved
            and measured_invocations > 0
            and measured_backend_compiles == 0
        ):
            contract["status"] = "INVALID"
            contract["contract_sha256"] = json_sha256(
                {key: item for key, item in contract.items() if key != "contract_sha256"}
            )
        contract_path = Path(args.instantiate_realization_contract).resolve()
        contract_path.parent.mkdir(parents=True, exist_ok=True)
        contract_path.write_text(
            json.dumps(contract, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "status": contract["status"],
                    "contract": str(contract_path),
                    "contract_sha256": contract["contract_sha256"],
                    "graph_family_digest": contract["graph_family_digest"],
                    "reference_scorer_sha256": reference_hash,
                    "candidate_scorer_sha256": scorer_hash,
                },
                indent=2,
            )
        )
        if contract["status"] == "INVALID":
            raise SystemExit(2)
        return
    decisions = clip_decisions(torch, logps.detach(), inputs, epsilon=0.2)
    loss = grpo_loss(torch, logps, inputs, epsilon=0.2)
    loss_value = float(loss.detach().item())

    scale_before = float(scaler.get_scale())
    scaler.scale(loss).backward()
    scaled_gradients = [(name, parameter.grad) for name, parameter in named_parameters]
    scaled_summary = tensor_collection_summary(torch, scaled_gradients)
    scaler.unscale_(optimizer)
    unscaled_gradients = [(name, parameter.grad) for name, parameter in named_parameters]
    unscaled_summary = tensor_collection_summary(torch, unscaled_gradients)
    reference_grad_norm = torch.nn.utils.clip_grad_norm_([p for _, p in named_parameters], max_norm=1.0)
    clipped_gradients = [(name, parameter.grad) for name, parameter in named_parameters]
    clipped_summary = tensor_collection_summary(torch, clipped_gradients)
    gradient_clip_triggered = bool(float(reference_grad_norm.item()) > 1.0)

    pre_parameters_cpu = {name: parameter.detach().cpu().clone() for name, parameter in named_parameters}
    scaler.step(optimizer)
    scaler.update()
    scale_after = float(scaler.get_scale())
    optimizer_step_skipped = scale_after < scale_before
    if not optimizer_step_skipped:
        scheduler.step()
    torch.cuda.synchronize()
    elapsed_ns = time.perf_counter_ns() - started

    update_tensors: dict[str, Any] = {}
    update_named: list[tuple[str, Any]] = []
    for name, parameter in named_parameters:
        update = parameter.detach().cpu() - pre_parameters_cpu[name]
        update_tensors[name] = update.contiguous()
        update_named.append((name, update))
    update_summary = tensor_collection_summary(torch, update_named)
    del update_named, pre_parameters_cpu

    post_parameter_hashes, post_parameter_digest = named_tensor_hashes(named_parameters)
    post_buffer_hashes, post_buffer_digest = named_tensor_hashes(named_buffers)
    post_optimizer = optimizer_tensor_hashes(optimizer, named_parameters)
    post_scheduler = scheduler.state_dict()
    post_scaler = scaler.state_dict()
    rng_after_transition = rng_fingerprint(torch)

    if args.save_vectors:
        clipped_cpu = {
            name: gradient.detach().cpu().contiguous()
            for name, gradient in clipped_gradients
            if gradient is not None
        }
        save_file(clipped_cpu, out_dir / "clipped_gradients.safetensors")
        save_file(update_tensors, out_dir / "parameter_updates.safetensors")
        vector_artifacts = {
            "clipped_gradients": {
                "path": str((out_dir / "clipped_gradients.safetensors").resolve()),
                "sha256": sha256_file(out_dir / "clipped_gradients.safetensors"),
            },
            "parameter_updates": {
                "path": str((out_dir / "parameter_updates.safetensors").resolve()),
                "sha256": sha256_file(out_dir / "parameter_updates.safetensors"),
            },
        }
    else:
        vector_artifacts = None

    candidate_identity_valid = args.arm == "eager" or (
        measured_invocations > 0 and measured_backend_compiles == 0 and history_graph_identity_exact
    )
    valid = all(
        [
            history_valid,
            scorer_anchor_exact,
            candidate_identity_valid,
            scaled_summary["finite"],
            unscaled_summary["finite"],
            clipped_summary["finite"],
            update_summary["finite"],
        ]
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "valid": valid,
        "arm": args.arm,
        "repeat": args.repeat,
        "snapshot": {
            "path": str(snapshot_dir),
            "metadata_sha256": sha256_file(snapshot_dir / "forkcert_transition_snapshot.json"),
            "optimizer_step": step,
            "training_horizon_optimizer_steps": training_horizon,
            "target_minibatch_sha256": sha256_file(target_path),
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_build": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "attention_backend": "MATH",
            "autocast_dtype": "float16",
        },
        "anchors": {
            "expected_scorer_sha256": expected_hash,
            "observed_scorer_sha256": scorer_hash,
            "scorer_anchor_exact": scorer_anchor_exact,
            "loss_is_reconstructed_from_anchored_scorer_and_frozen_inputs": True,
            "identity_source": (
                "legacy_online_anchor"
                if anchor is not None
                else "prospective_realization_contract"
            ),
            "realization_contract_sha256": (
                realization_contract["contract_sha256"]
                if realization_contract is not None
                else None
            ),
        },
        "realization": {
            "compiler_config_digest": (
                realization_contract["compiler_config_digest"]
                if realization_contract is not None
                else json_sha256(
                    {
                        "legacy_anchor": str(Path(args.anchor_states).resolve()),
                        "torch_version": torch.__version__,
                    }
                )
            ),
            "graph_family_digest": (
                realization_contract["graph_family_digest"]
                if realization_contract is not None
                else json_sha256(
                    [
                        {"graph_code_sha256": value[0], "graph_node_count": value[1]}
                        for value in expected_family
                    ]
                    if args.arm == "compiled"
                    else []
                )
            ),
        },
        "compiler": {
            **compile_audit,
            "history_records": len(metadata["compiler_history"]),
            "history_runtime_invocations": history_runtime_invocations,
            "history_state_preserved": history_valid,
            "history_graph_identity_exact": history_graph_identity_exact,
            "history_identity_scope": "ordered_unique_graph_family_not_historical_compile_event_lineage",
            "measured_runtime_invocations": measured_invocations,
            "measured_backend_compiles": measured_backend_compiles,
            "candidate_identity_valid": candidate_identity_valid,
            "wrapper_shares_optimizer_parameters": args.arm == "eager"
            or [id(p) for p in execution_model.parameters()] == [id(p) for p in model.parameters()],
        },
        "observability": {
            "requested": (
                trace_dir is not None
                or graph_manifest_dir is not None
                or args.dump_kernel_inputs is not None
            ),
            "trace_dir": str(trace_dir) if trace_dir is not None else None,
            "graph_manifest_dir": (
                str(graph_manifest_dir) if graph_manifest_dir is not None else None
            ),
            "observer_claim": "debug/provenance capture only; equivalence requires an external baseline gate",
            "kernel_input_dump_filters": (
                [value for value in args.dump_kernel_inputs.split(",") if value]
                if args.dump_kernel_inputs
                else []
            ),
        },
        "pre_state": {
            "parameter_hashes_sha256": json_sha256(pre_parameter_hashes),
            "parameter_digest": pre_parameter_digest,
            "buffer_hashes_sha256": json_sha256(pre_buffer_hashes),
            "buffer_digest": pre_buffer_digest,
            "optimizer_digest": pre_optimizer["sha256"],
            "scheduler_digest": json_sha256(pre_scheduler),
            "scaler_digest": json_sha256(pre_scaler),
            "rng": rng_before_transition,
        },
        "continuous": {
            "scorer_logps": logps.detach().float().cpu().tolist(),
            "loss": loss_value,
            "scaled_gradient": scaled_summary,
            "unscaled_gradient": unscaled_summary,
            "clipped_gradient": clipped_summary,
            "pre_clip_gradient_norm": float(reference_grad_norm.item()),
            "parameter_update": update_summary,
        },
        "semantic": {
            "clip_decisions": decisions.detach().cpu().tolist(),
            "clip_count": int(decisions.sum().item()),
            "gradient_clip_triggered": gradient_clip_triggered,
            "amp_scale_before": scale_before,
            "amp_scale_after": scale_after,
            "optimizer_step_skipped": optimizer_step_skipped,
            "nonfinite_gradient": not unscaled_summary["finite"],
            "nonfinite_update": not update_summary["finite"],
        },
        "post_state": {
            "parameter_hashes_sha256": json_sha256(post_parameter_hashes),
            "parameter_digest": post_parameter_digest,
            "buffer_hashes_sha256": json_sha256(post_buffer_hashes),
            "buffer_digest": post_buffer_digest,
            "optimizer_digest": post_optimizer["sha256"],
            "scheduler_digest": json_sha256(post_scheduler),
            "scaler_digest": json_sha256(post_scaler),
            "rng": rng_after_transition,
        },
        "vector_artifacts": vector_artifacts,
        "elapsed_ns": elapsed_ns,
        "verdict": "VALID" if valid else "INVALID",
        "nonclaims": [
            "implementation-relative transition impact is not numerical correctness",
            "one selected state does not estimate population prevalence",
            "scorer-forward treatment is not operator attribution",
            "one-step impact does not imply long-run training harm",
        ],
    }
    if trace_dir is not None:
        trace_files = []
        for path in sorted(item for item in trace_dir.rglob("*") if item.is_file()):
            trace_files.append(
                {
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                    "kind": path.name,
                }
            )
        result["observability"]["trace_files"] = trace_files
        result["observability"]["provenance_mapping_count"] = sum(
            row["kind"] == "inductor_provenance_tracking_node_mappings.json"
            for row in trace_files
        )
        result["observability"]["generated_code_count"] = sum(
            row["kind"] == "output_code.py" for row in trace_files
        )
    if args.dump_kernel_inputs:
        from torch._inductor.codecache import cache_dir

        filters = [value for value in args.dump_kernel_inputs.split(",") if value]
        dump_files = []
        cache_root = Path(cache_dir())
        for path in sorted(item for item in cache_root.rglob("tensor_*.pt") if item.is_file()):
            if not any(value in path.parent.name for value in filters):
                continue
            dump_files.append(
                {
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                    "kernel_run_dir": path.parent.name,
                }
            )
        result["observability"]["kernel_input_dumps"] = dump_files
        result["observability"]["kernel_input_dump_capture_valid"] = bool(dump_files)
    (out_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"valid": valid, "arm": args.arm, "repeat": args.repeat, "loss": loss_value,
                      "scorer_anchor_exact": scorer_anchor_exact,
                      "gradient_l2": clipped_summary["l2"], "update_l2": update_summary["l2"],
                      "post_parameter_digest": post_parameter_digest}, indent=2))
    if not valid:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
