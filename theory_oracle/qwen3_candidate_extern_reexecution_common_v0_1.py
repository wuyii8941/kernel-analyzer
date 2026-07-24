#!/usr/bin/env python
"""Fail-closed runner for candidate external GEMM calls re-executed via eager ATen."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

from qwen3_lm_head_operator_pilot_v0_1 import (
    CANDIDATE_ANCHOR,
    EAGER_ANCHOR,
    metrics,
    move_tree,
    tensor_sha256,
)
from qwen3_original_compiled_inventory_v0_1 import EXPECTED_FAMILY, ordered_unique


def tensor_metadata(value: Any) -> dict[str, Any]:
    return {
        "shape": list(value.shape),
        "stride": list(value.stride()),
        "dtype": str(value.dtype),
        "device": str(value.device),
    }


def run_experiment(config: dict[str, Any]) -> None:
    parser = argparse.ArgumentParser(description=str(config["description"]))
    parser.add_argument("--snapshot-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    snapshot_dir = Path(args.snapshot_dir).resolve()
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError(out)

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("PYTHONHASHSEED", "0")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import torch
    from accelerate import Accelerator
    from torch._inductor.select_algorithm import extern_kernels
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from transformers import AutoModelForCausalLM
    from trl.trainer.grpo_trainer import selective_log_softmax

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one visible CUDA device is required")
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch._dynamo.config.suppress_errors = False
    torch._dynamo.config.recompile_limit = 64

    metadata = json.loads((snapshot_dir / "forkcert_transition_snapshot.json").read_text())
    model = AutoModelForCausalLM.from_pretrained(
        snapshot_dir, dtype=torch.float32, attn_implementation="sdpa", local_files_only=True
    ).to("cuda")
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.train()
    wrapped = Accelerator(mixed_precision="fp16").prepare_model(model)

    from torch._dynamo.backends.registry import lookup_backend

    inductor = lookup_backend("inductor")
    audit: dict[str, Any] = {
        "backend_compiles": 0,
        "runtime_invocations": 0,
        "graph_hashes": [],
        "graph_nodes": [],
    }
    artifacts: list[Any] = []

    def backend(graph_module: Any, example_inputs: list[Any]) -> Any:
        audit["backend_compiles"] += 1
        audit["graph_hashes"].append(hashlib.sha256(graph_module.code.encode()).hexdigest())
        audit["graph_nodes"].append(sum(1 for _ in graph_module.graph.nodes))
        artifact = inductor(graph_module, example_inputs)
        artifacts.append(artifact)

        def counted(*values: Any) -> Any:
            audit["runtime_invocations"] += 1
            return artifact(*values)

        return counted

    candidate = torch.compile(wrapped, backend=backend)

    def score(callable_model: Any, value: dict[str, Any]) -> Any:
        completion_ids = value["completion_ids"]
        input_ids = torch.cat([value["prompt_ids"], completion_ids], dim=1)
        attention_mask = torch.cat([value["prompt_mask"], value["completion_mask"]], dim=1)
        with sdpa_kernel(SDPBackend.MATH):
            outputs = callable_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                logits_to_keep=completion_ids.size(1) + 1,
                use_cache=False,
            )
            logits = outputs.logits[:, :-1, :]
            return selective_log_softmax(logits[:, -completion_ids.size(1) :, :], completion_ids)

    for record in metadata["compiler_history"]:
        path = Path(record["path"])
        if not path.is_file():
            path = snapshot_dir / "compiler_history" / path.name
        history = move_tree(torch.load(path, map_location="cpu", weights_only=False), "cuda")
        value = score(candidate, history)
        del value, history
        gc.collect()

    target_path = Path(metadata["target_minibatch_path"])
    if not target_path.is_file():
        target_path = snapshot_dir / "compiler_history" / target_path.name
    inputs = move_tree(torch.load(target_path, map_location="cpu", weights_only=False), "cuda")

    def repeated(callable_model: Any) -> tuple[list[Any], list[str]]:
        values: list[Any] = []
        hashes: list[str] = []
        for _ in range(2):
            output = score(callable_model, inputs)
            detached = output.detach().float().cpu()
            values.append(detached)
            hashes.append(tensor_sha256(detached))
            del output
            gc.collect()
        return values, hashes

    eager_values, eager_hashes = repeated(wrapped)
    candidate_values, candidate_hashes = repeated(candidate)

    generated_modules: list[str] = []
    extern_identity: dict[str, bool] = {}
    for module_name, module in list(sys.modules.items()):
        if module is None or not module_name.startswith("torch._inductor.runtime.compile_tasks."):
            continue
        if hasattr(module, "extern_kernels"):
            generated_modules.append(module_name)
            extern_identity[module_name] = getattr(module, "extern_kernels") is extern_kernels

    operation = str(config["operation"])
    eager_operation = getattr(torch, operation)
    original = getattr(extern_kernels, operation)

    class ExternProxy:
        def __init__(self, selected: int):
            self.selected = selected
            self.calls = 0
            self.repairs = 0
            self.selected_metadata: list[dict[str, Any]] = []

        def __call__(self, *values: Any, **kwargs: Any) -> Any:
            index = self.calls
            self.calls += 1
            if index != self.selected:
                return original(*values, **kwargs)
            self.repairs += 1
            output = kwargs.get("out")
            if len(values) != 2 or output is None:
                raise RuntimeError(f"unexpected extern {operation} signature")
            self.selected_metadata.append(
                {
                    "left": tensor_metadata(values[0]),
                    "right": tensor_metadata(values[1]),
                    "out": tensor_metadata(output),
                }
            )
            return eager_operation(values[0], values[1], out=output)

    eager = eager_values[0]
    candidate_value = candidate_values[0]
    eager_candidate_l2 = float(torch.linalg.vector_norm(candidate_value - eager).item())
    compile_count_before = audit["backend_compiles"]
    repairs: dict[str, Any] = {}
    for selection in config["selections"]:
        selected = int(selection["index"])
        proxy = ExternProxy(selected)
        setattr(extern_kernels, operation, proxy)
        values: list[Any] = []
        hashes: list[str] = []
        call_records: list[dict[str, Any]] = []
        try:
            for _ in range(2):
                proxy.calls = 0
                proxy.repairs = 0
                proxy.selected_metadata = []
                output = score(candidate, inputs)
                detached = output.detach().float().cpu()
                values.append(detached)
                hashes.append(tensor_sha256(detached))
                call_records.append(
                    {
                        "calls": proxy.calls,
                        "repairs": proxy.repairs,
                        "metadata": proxy.selected_metadata,
                    }
                )
                del output
                gc.collect()
        finally:
            setattr(extern_kernels, operation, original)

        repaired = values[0]
        repaired_l2 = float(torch.linalg.vector_norm(repaired - eager).item())
        repair_vector = repaired - candidate_value
        target_vector = eager - candidate_value
        norm_product = float(
            torch.linalg.vector_norm(repair_vector).item()
            * torch.linalg.vector_norm(target_vector).item()
        )
        cosine = (
            float(torch.dot(repair_vector.flatten(), target_vector.flatten()).item() / norm_product)
            if norm_product
            else None
        )
        repairs[str(selected)] = {
            "role": selection["role"],
            "sha256": hashes,
            "repeat_exact": hashes[0] == hashes[1],
            "repeat_max_abs": float((values[1] - repaired).abs().max().item()),
            "call_records": call_records,
            "candidate_to_repair": metrics(torch, candidate_value, repaired),
            "eager_to_repair": metrics(torch, eager, repaired),
            "direction": {
                "eager_candidate_l2": eager_candidate_l2,
                "eager_repair_l2": repaired_l2,
                "l2_distance_change": repaired_l2 - eager_candidate_l2,
                "fractional_l2_reduction": (eager_candidate_l2 - repaired_l2)
                / eager_candidate_l2,
                "cosine_repair_with_candidate_to_eager": cosine,
            },
        }

    _, restored_hashes = repeated(candidate)
    compile_count_after = audit["backend_compiles"]
    actual_family = ordered_unique(list(zip(audit["graph_hashes"], audit["graph_nodes"], strict=True)))
    expected_calls = int(config["expected_family_calls"])
    gates = {
        "graph_family_exact": actual_family == EXPECTED_FAMILY,
        "eager_anchor_exact": eager_hashes == [EAGER_ANCHOR, EAGER_ANCHOR],
        "candidate_anchor_exact": candidate_hashes == [CANDIDATE_ANCHOR, CANDIDATE_ANCHOR],
        "generated_modules_resolved": bool(generated_modules),
        "generated_modules_share_extern_object": bool(extern_identity)
        and all(extern_identity.values()),
        "all_repair_repeats_exact": all(row["repeat_exact"] for row in repairs.values()),
        "all_selected_calls_reexecuted_once": all(
            record["calls"] == expected_calls
            and record["repairs"] == 1
            and len(record["metadata"]) == 1
            for row in repairs.values()
            for record in row["call_records"]
        ),
        "no_backend_recompile_during_repairs": compile_count_before == compile_count_after,
        "extern_restoration_exact": getattr(extern_kernels, operation) is original,
        "candidate_restoration_exact": restored_hashes == [CANDIDATE_ANCHOR, CANDIDATE_ANCHOR],
    }
    payload = {
        "schema_version": config["schema_version"],
        "status": config["valid_status"] if all(gates.values()) else "INVALID_TREATMENT",
        "state": "heldout-transport-B-step29",
        "kernel_family": f"extern:{operation}",
        "operation": operation,
        "expected_family_calls": expected_calls,
        "selected_call_indices": [int(row["index"]) for row in config["selections"]],
        "selected_roles": {str(row["index"]): row["role"] for row in config["selections"]},
        "gates": gates,
        "anchors": {
            "eager": eager_hashes,
            "candidate": candidate_hashes,
            "restored": restored_hashes,
        },
        "eager_to_candidate": metrics(torch, eager, candidate_value),
        "repairs": repairs,
        "generated_modules": sorted(generated_modules),
        "generated_module_extern_identity": extern_identity,
        "actual_graph_family": actual_family,
        "compile_audit": audit,
        "compile_counts": {"before_repairs": compile_count_before, "after_repairs": compile_count_after},
        "claim_limits": list(config["claim_limits"]),
    }
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "gates": gates,
                "effects": {
                    key: {
                        "role": row["role"],
                        "candidate_to_repair": row["candidate_to_repair"],
                        "direction": row["direction"],
                    }
                    for key, row in repairs.items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
