#!/usr/bin/env python
"""Fail-closed shared-compile runner for separate singleton kernel treatments."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable

from qwen3_lm_head_operator_pilot_v0_1 import CANDIDATE_ANCHOR, EAGER_ANCHOR, metrics, move_tree, tensor_sha256
from qwen3_original_compiled_inventory_v0_1 import EXPECTED_FAMILY, ordered_unique


Repair = Callable[[Any, tuple[Any, ...]], None]


def run_campaign(treatments: list[dict[str, Any]], campaign: dict[str, Any]) -> None:
    parser = argparse.ArgumentParser(description=str(campaign["description"]))
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
    model = AutoModelForCausalLM.from_pretrained(snapshot_dir, dtype=torch.float32, attn_implementation="sdpa", local_files_only=True).to("cuda")
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.train()
    wrapped = Accelerator(mixed_precision="fp16").prepare_model(model)

    from torch._dynamo.backends.registry import lookup_backend
    inductor = lookup_backend("inductor")
    compile_audit: dict[str, Any] = {"backend_compiles": 0, "runtime_invocations": 0, "graph_hashes": [], "graph_nodes": []}

    def backend(graph_module: Any, example_inputs: list[Any]) -> Any:
        compile_audit["backend_compiles"] += 1
        compile_audit["graph_hashes"].append(hashlib.sha256(graph_module.code.encode()).hexdigest())
        compile_audit["graph_nodes"].append(sum(1 for _ in graph_module.graph.nodes))
        artifact = inductor(graph_module, example_inputs)

        def counted(*values: Any) -> Any:
            compile_audit["runtime_invocations"] += 1
            return artifact(*values)

        return counted

    candidate = torch.compile(wrapped, backend=backend)

    def score(callable_model: Any, value: dict[str, Any]) -> Any:
        completion_ids = value["completion_ids"]
        input_ids = torch.cat([value["prompt_ids"], completion_ids], dim=1)
        attention_mask = torch.cat([value["prompt_mask"], value["completion_mask"]], dim=1)
        with sdpa_kernel(SDPBackend.MATH):
            outputs = callable_model(input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=completion_ids.size(1) + 1, use_cache=False)
            logits = outputs.logits[:, :-1, :]
            return selective_log_softmax(logits[:, -completion_ids.size(1):, :], completion_ids)

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

    def repeated(callable_model: Any, repeats: int = 2) -> tuple[list[Any], list[str]]:
        values, hashes = [], []
        for _ in range(repeats):
            value = score(callable_model, inputs)
            detached = value.detach().float().cpu()
            values.append(detached)
            hashes.append(tensor_sha256(detached))
            del value
            gc.collect()
        return values, hashes

    eager_values, eager_hashes = repeated(wrapped)
    candidate_values, candidate_hashes = repeated(candidate)
    eager, baseline = eager_values[0], candidate_values[0]
    base_l2 = float(torch.linalg.vector_norm(baseline - eager).item())
    graph_family = ordered_unique(list(zip(compile_audit["graph_hashes"], compile_audit["graph_nodes"], strict=True)))
    global_gates = {
        "graph_family_exact": graph_family == EXPECTED_FAMILY,
        "eager_anchor_exact": eager_hashes == [EAGER_ANCHOR, EAGER_ANCHOR],
        "candidate_anchor_exact": candidate_hashes == [CANDIDATE_ANCHOR, CANDIDATE_ANCHOR],
    }

    generated_modules = {
        name: module
        for name, module in list(sys.modules.items())
        if module is not None and name.startswith("torch._inductor.runtime.compile_tasks.")
    }
    family_results: dict[str, Any] = {}
    for treatment in treatments:
        kernel = str(treatment["kernel_family"])
        modules = {name: module for name, module in generated_modules.items() if hasattr(module, kernel)}

        class Proxy:
            def __init__(self, original: Any):
                self.original = original
                self.calls = 0
                self.repairs = 0

            def run(self, *values: Any, **kwargs: Any) -> Any:
                self.calls += 1
                self.repairs += 1
                treatment["repair"](torch, values)
                return None

        originals, proxies = {}, {}
        for name, module in modules.items():
            originals[name] = getattr(module, kernel)
            proxies[name] = Proxy(originals[name])
            setattr(module, kernel, proxies[name])
        compile_before = compile_audit["backend_compiles"]
        values, hashes, call_records = [], [], []
        failure = None
        try:
            for _ in range(2):
                for proxy in proxies.values():
                    proxy.calls = 0
                    proxy.repairs = 0
                value = score(candidate, inputs)
                detached = value.detach().float().cpu()
                values.append(detached)
                hashes.append(tensor_sha256(detached))
                call_records.append({name: {"calls": proxy.calls, "repairs": proxy.repairs} for name, proxy in proxies.items()})
                del value
                gc.collect()
        except Exception as error:  # retained as explicit invalid treatment
            failure = {"type": type(error).__name__, "message": str(error)}
        finally:
            for name, module in modules.items():
                setattr(module, kernel, originals[name])

        restored_values, restored_hashes = repeated(candidate, repeats=1)
        del restored_values
        active_records_valid = bool(call_records) and all(
            len([row for row in record.values() if row["calls"] > 0]) == 1
            and [row for row in record.values() if row["calls"] > 0][0] == {"calls": 1, "repairs": 1}
            for record in call_records
        )
        family_gates = {
            "module_resolved": bool(modules),
            "two_repaired_outputs": len(values) == 2,
            "repeat_exact": len(hashes) == 2 and hashes[0] == hashes[1],
            "one_call_one_repair_per_run": len(call_records) == 2 and active_records_valid,
            "no_backend_recompile": compile_audit["backend_compiles"] == compile_before,
            "candidate_restored": restored_hashes == [CANDIDATE_ANCHOR],
            "no_runtime_exception": failure is None,
        }
        record: dict[str, Any] = {
            "status": treatment["valid_status"] if all(family_gates.values()) else "INVALID_TREATMENT",
            "kernel_family": kernel,
            "gates": family_gates,
            "failure": failure,
            "module_matches": sorted(modules),
            "sha256": hashes,
            "call_records": call_records,
            "restored_sha256": restored_hashes,
            "claim_limits": treatment["claim_limits"],
        }
        if len(values) == 2:
            repaired = values[0]
            repaired_l2 = float(torch.linalg.vector_norm(repaired - eager).item())
            repair_vector, target_vector = repaired - baseline, eager - baseline
            norm_product = float(torch.linalg.vector_norm(repair_vector).item() * torch.linalg.vector_norm(target_vector).item())
            record["candidate_to_repair"] = metrics(torch, baseline, repaired)
            record["eager_to_repair"] = metrics(torch, eager, repaired)
            record["direction"] = {
                "eager_candidate_l2": base_l2,
                "eager_repair_l2": repaired_l2,
                "l2_distance_change": repaired_l2 - base_l2,
                "fractional_l2_reduction": (base_l2 - repaired_l2) / base_l2,
                "cosine_repair_with_candidate_to_eager": float(torch.dot(repair_vector.flatten(), target_vector.flatten()).item() / norm_product) if norm_product else None,
            }
        family_results[kernel] = record

    _, final_hashes = repeated(candidate)
    global_gates["final_candidate_restored"] = final_hashes == [CANDIDATE_ANCHOR, CANDIDATE_ANCHOR]
    global_gates["no_campaign_backend_recompile"] = compile_audit["backend_compiles"] == len(EXPECTED_FAMILY)
    payload = {
        "schema_version": campaign["schema_version"],
        "status": "VALID_FAIL_CLOSED_CAMPAIGN" if all(global_gates.values()) else "INVALID_CAMPAIGN",
        "state": "heldout-transport-B-step29",
        "global_gates": global_gates,
        "anchors": {"eager": eager_hashes, "candidate": candidate_hashes, "final": final_hashes},
        "eager_to_candidate": metrics(torch, eager, baseline),
        "actual_graph_family": graph_family,
        "families": family_results,
        "compile_audit": compile_audit,
    }
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": payload["status"], "global_gates": global_gates, "families": {key: {"status": row["status"], "gates": row["gates"], "effect": row.get("candidate_to_repair"), "failure": row["failure"]} for key, row in family_results.items()}}, indent=2, sort_keys=True))

