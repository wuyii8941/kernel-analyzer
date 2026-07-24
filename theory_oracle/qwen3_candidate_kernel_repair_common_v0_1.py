#!/usr/bin/env python
"""Common fail-closed runner for original-candidate generated-kernel repairs."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable

from qwen3_candidate_kernel15_repair_v0_1 import resolve_generated_modules
from qwen3_lm_head_operator_pilot_v0_1 import (
    CANDIDATE_ANCHOR,
    EAGER_ANCHOR,
    metrics,
    move_tree,
    tensor_sha256,
)
from qwen3_original_compiled_inventory_v0_1 import EXPECTED_FAMILY, ordered_unique


Repair = Callable[[Any, tuple[Any, ...]], None]


def run_experiment(config: dict[str, Any], repair: Repair) -> None:
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

    kernel = str(config["kernel_family"])
    modules: dict[str, Any] = {}
    for artifact in artifacts:
        resolved, _ = resolve_generated_modules(artifact, kernel)
        modules.update(resolved)
    process_module_matches: list[str] = []
    for module_name, module in list(sys.modules.items()):
        if module is None or not module_name.startswith("torch._inductor.runtime.compile_tasks."):
            continue
        if hasattr(module, kernel):
            modules[module_name] = module
            process_module_matches.append(module_name)

    class KernelProxy:
        def __init__(self, original: Any, selected: int):
            self.original = original
            self.selected = selected
            self.calls = 0
            self.repairs = 0

        def run(self, *values: Any, **kwargs: Any) -> Any:
            index = self.calls
            self.calls += 1
            if index != self.selected:
                return self.original.run(*values, **kwargs)
            self.repairs += 1
            repair(torch, values)
            return None

    eager = eager_values[0]
    candidate_value = candidate_values[0]
    eager_candidate_l2 = float(torch.linalg.vector_norm(candidate_value - eager).item())
    compile_count_before = audit["backend_compiles"]
    repairs: dict[str, Any] = {}
    for selected in config["selected_call_indices"]:
        originals: dict[str, Any] = {}
        proxies: dict[str, KernelProxy] = {}
        for module_name, module in modules.items():
            originals[module_name] = getattr(module, kernel)
            proxies[module_name] = KernelProxy(originals[module_name], int(selected))
            setattr(module, kernel, proxies[module_name])
        values: list[Any] = []
        hashes: list[str] = []
        call_records: list[dict[str, Any]] = []
        try:
            for _ in range(2):
                for proxy in proxies.values():
                    proxy.calls = 0
                    proxy.repairs = 0
                output = score(candidate, inputs)
                detached = output.detach().float().cpu()
                values.append(detached)
                hashes.append(tensor_sha256(detached))
                call_records.append(
                    {name: {"calls": proxy.calls, "repairs": proxy.repairs} for name, proxy in proxies.items()}
                )
                del output
                gc.collect()
        finally:
            for module_name, module in modules.items():
                setattr(module, kernel, originals[module_name])

        repaired = values[0]
        repaired_l2 = float(torch.linalg.vector_norm(repaired - eager).item())
        repair_vector = repaired - candidate_value
        target_vector = eager - candidate_value
        norm_product = float(torch.linalg.vector_norm(repair_vector).item() * torch.linalg.vector_norm(target_vector).item())
        cosine = float(torch.dot(repair_vector.flatten(), target_vector.flatten()).item() / norm_product) if norm_product else None
        repairs[str(selected)] = {
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
                "fractional_l2_reduction": (eager_candidate_l2 - repaired_l2) / eager_candidate_l2,
                "cosine_repair_with_candidate_to_eager": cosine,
            },
        }

    _, restored_hashes = repeated(candidate)
    compile_count_after = audit["backend_compiles"]
    actual_family = ordered_unique(list(zip(audit["graph_hashes"], audit["graph_nodes"], strict=True)))
    expected_calls = int(config["expected_family_calls"])

    def selected_once(record: dict[str, Any]) -> bool:
        active = [row for row in record.values() if row["calls"] > 0]
        return len(active) == 1 and active[0] == {"calls": expected_calls, "repairs": 1}

    gates = {
        "graph_family_exact": actual_family == EXPECTED_FAMILY,
        "eager_anchor_exact": eager_hashes == [EAGER_ANCHOR, EAGER_ANCHOR],
        "candidate_anchor_exact": candidate_hashes == [CANDIDATE_ANCHOR, CANDIDATE_ANCHOR],
        "generated_target_module_resolved": bool(modules),
        "all_repair_repeats_exact": all(row["repeat_exact"] for row in repairs.values()),
        "all_selected_calls_repaired_once": all(
            selected_once(record) for row in repairs.values() for record in row["call_records"]
        ),
        "no_backend_recompile_during_repairs": compile_count_before == compile_count_after,
        "kernel_restoration_exact": restored_hashes == [CANDIDATE_ANCHOR, CANDIDATE_ANCHOR],
    }
    payload = {
        "schema_version": config["schema_version"],
        "status": config["valid_status"] if all(gates.values()) else "INVALID_TREATMENT",
        "state": "heldout-transport-B-step29",
        "kernel_family": kernel,
        "expected_family_calls": expected_calls,
        "selected_call_indices": list(config["selected_call_indices"]),
        "gates": gates,
        "anchors": {"eager": eager_hashes, "candidate": candidate_hashes, "restored": restored_hashes},
        "eager_to_candidate": metrics(torch, eager, candidate_value),
        "repairs": repairs,
        "process_module_matches": sorted(process_module_matches),
        "actual_graph_family": actual_family,
        "compile_audit": audit,
        "compile_counts": {"before_repairs": compile_count_before, "after_repairs": compile_count_after},
        "claim_limits": list(config["claim_limits"]),
    }
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": payload["status"], "gates": gates, "effects": {key: {"candidate_to_repair": row["candidate_to_repair"], "direction": row["direction"]} for key, row in repairs.items()}}, indent=2, sort_keys=True))

