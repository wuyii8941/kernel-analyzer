#!/usr/bin/env python
"""Live original-candidate production/mediation test for the final-RMSNorm kernel group."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import qwen3_grpo_natural_transition_v0_2 as natural
from forkcert.operator_evidence import production_mediation_interpretation, tensor_fingerprint
from qwen3_candidate_kernel15_repair_v0_1 import resolve_generated_modules


def metrics(torch: Any, left: Any, right: Any) -> dict[str, Any]:
    delta = right.double() - left.double()
    return {
        "mean_signed": float(delta.mean().item()),
        "mean_abs": float(delta.abs().mean().item()),
        "max_abs": float(delta.abs().max().item()),
        "l2": float(torch.linalg.vector_norm(delta).item()),
        "nonzero": int(torch.count_nonzero(delta).item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--observability-gate", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    inventory = json.loads(Path(args.inventory).read_text())
    gate = json.loads(Path(args.observability_gate).read_text())
    if gate.get("forward_kernel_inventory_eligible") is not True:
        raise RuntimeError("forward provenance inventory is not eligible")
    kernel_name = manifest["pointwise_kernel"]
    provenance_rows = [
        row for row in inventory["kernels"] if row["generated_symbol"] == kernel_name
    ]
    if len(provenance_rows) != 1:
        raise RuntimeError(f"expected one provenance row for {kernel_name}")

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

    snapshot = Path(manifest["snapshot_dir"])
    metadata = json.loads((snapshot / "forkcert_transition_snapshot.json").read_text())
    contract = natural.load_realization_contract(
        Path(manifest["realization_contract"]), snapshot, int(metadata["optimizer_step"])
    )
    model = AutoModelForCausalLM.from_pretrained(
        snapshot, dtype=torch.float32, attn_implementation="sdpa", local_files_only=True
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
    artifacts = []

    def backend(graph_module: Any, example_inputs: list[Any]) -> Any:
        audit["backend_compiles"] += 1
        audit["graph_hashes"].append(
            hashlib.sha256(graph_module.code.encode()).hexdigest()
        )
        audit["graph_nodes"].append(sum(1 for _ in graph_module.graph.nodes))
        artifact = inductor(graph_module, example_inputs)
        artifacts.append(artifact)

        def counted(*values: Any) -> Any:
            audit["runtime_invocations"] += 1
            return artifact(*values)

        return counted

    candidate = torch.compile(wrapped, backend=backend)

    def score(callable_model: Any, value: dict[str, Any]) -> Any:
        completion = value["completion_ids"]
        input_ids = torch.cat([value["prompt_ids"], completion], dim=1)
        attention_mask = torch.cat(
            [value["prompt_mask"], value["completion_mask"]], dim=1
        )
        with sdpa_kernel(SDPBackend.MATH):
            outputs = callable_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                logits_to_keep=completion.size(1) + 1,
                use_cache=False,
            )
            logits = outputs.logits[:, :-1, :]
            return selective_log_softmax(logits[:, -completion.size(1) :, :], completion)

    for record in metadata["compiler_history"]:
        path = Path(record["path"])
        if not path.is_file():
            path = snapshot / "compiler_history" / path.name
        history = natural.move_tree(
            torch.load(path, map_location="cpu", weights_only=False), "cuda"
        )
        value = score(candidate, history)
        del value, history
        gc.collect()

    target_path = Path(metadata["target_minibatch_path"])
    if not target_path.is_file():
        target_path = snapshot / "compiler_history" / target_path.name
    inputs = natural.move_tree(
        torch.load(target_path, map_location="cpu", weights_only=False), "cuda"
    )

    def repeated(callable_model: Any) -> tuple[list[Any], list[str]]:
        values = []
        hashes = []
        for _ in range(2):
            current = score(callable_model, inputs).detach().float().cpu()
            values.append(current)
            hashes.append(natural.tensor_sha256(current))
            gc.collect()
        return values, hashes

    eager_values, eager_hashes = repeated(wrapped)
    candidate_values, candidate_hashes = repeated(candidate)

    modules: dict[str, Any] = {}
    resolution_records = []
    for artifact in artifacts:
        resolved, observations = resolve_generated_modules(artifact, kernel_name)
        modules.update(resolved)
        resolution_records.append(
            {
                "artifact_type": f"{type(artifact).__module__}.{type(artifact).__name__}",
                "resolved_modules": sorted(resolved),
                "observations": observations,
            }
        )
    for module_name, module in list(sys.modules.items()):
        if (
            module is not None
            and module_name.startswith("torch._inductor.runtime.compile_tasks.")
            and hasattr(module, kernel_name)
        ):
            modules[module_name] = module
    if not modules:
        raise RuntimeError("failed to resolve live original generated kernel")

    class KernelProxy:
        def __init__(self, original: Any, repair: bool):
            self.original = original
            self.repair = repair
            self.calls = 0
            self.records = []

        def run(self, *values: Any, **kwargs: Any) -> Any:
            self.calls += 1
            result = self.original.run(*values, **kwargs)
            torch.cuda.synchronize()
            weight, component0, component1, component2, reciprocal_rms, output = values[:6]
            with torch.no_grad():
                original_output = output.detach().clone()
                logical_shape = component0.shape
                hidden = component0.float() + component1.reshape(logical_shape).float()
                hidden = hidden + component2.reshape(logical_shape).float()
                reference_rms = torch.rsqrt(
                    hidden.square().mean(dim=-1, keepdim=True)
                    + float(manifest["epsilon"])
                )
                reference_output = (
                    weight.float() * hidden * reference_rms
                )[:, -output.shape[1] :, :].to(output.dtype).contiguous()
                record = {
                    "input_fingerprints": [
                        tensor_fingerprint(item)
                        for item in (weight, component0, component1, component2)
                    ],
                    "compiled_reduction_to_reference_reduction": metrics(
                        torch, reference_rms, reciprocal_rms.reshape(reference_rms.shape)
                    ),
                    "compiled_to_reference_output": metrics(
                        torch, reference_output, original_output
                    ),
                    "compiled_output_sha256": natural.tensor_sha256(original_output),
                    "reference_output_sha256": natural.tensor_sha256(reference_output),
                }
                if self.repair:
                    output.copy_(reference_output)
            self.records.append(record)
            return result

    def proxy_arm(repair: bool) -> dict[str, Any]:
        originals = {}
        proxies = {}
        for module_name, module in modules.items():
            originals[module_name] = getattr(module, kernel_name)
            proxies[module_name] = KernelProxy(originals[module_name], repair)
            setattr(module, kernel_name, proxies[module_name])
        values = []
        hashes = []
        records = []
        calls = []
        try:
            for _ in range(2):
                for proxy in proxies.values():
                    proxy.calls = 0
                    proxy.records = []
                current = score(candidate, inputs).detach().float().cpu()
                values.append(current)
                hashes.append(natural.tensor_sha256(current))
                active = [proxy for proxy in proxies.values() if proxy.calls]
                calls.append([proxy.calls for proxy in active])
                records.append([record for proxy in active for record in proxy.records])
                gc.collect()
        finally:
            for module_name, module in modules.items():
                setattr(module, kernel_name, originals[module_name])
        return {
            "values": values,
            "hashes": hashes,
            "repeat_exact": hashes[0] == hashes[1],
            "active_call_counts": calls,
            "local_records": records,
        }

    compiles_before = audit["backend_compiles"]
    noop = proxy_arm(repair=False)
    repair = proxy_arm(repair=True)
    compiles_after = audit["backend_compiles"]
    _, restored_hashes = repeated(candidate)

    expected_family = [
        (row["graph_code_sha256"], int(row["graph_node_count"]))
        for row in contract["candidate_ordered_unique_graph_family"]
    ]
    actual_family = []
    for row in zip(audit["graph_hashes"], audit["graph_nodes"], strict=True):
        if row not in actual_family:
            actual_family.append(row)

    candidate_value = candidate_values[0]
    repair_value = repair["values"][0]
    eager_value = eager_values[0]
    candidate_decisions = natural.clip_decisions(torch, candidate_value, natural.move_tree(inputs, "cpu"), 0.2)
    repair_decisions = natural.clip_decisions(torch, repair_value, natural.move_tree(inputs, "cpu"), 0.2)
    upward = (~candidate_decisions) & repair_decisions
    downward = candidate_decisions & (~repair_decisions)
    local_records = repair["local_records"]
    local_repeat_exact = (
        len(local_records) == 2
        and len(local_records[0]) == len(local_records[1]) == 1
        and local_records[0][0] == local_records[1][0]
    )
    production_observed = (
        local_repeat_exact
        and local_records[0][0]["compiled_to_reference_output"]["nonzero"] > 0
    )
    mediation_observed = repair["hashes"][0] != candidate_hashes[0]
    gates = {
        "graph_family_exact": actual_family == expected_family,
        "eager_anchor_exact": eager_hashes
        == [contract["reference_scorer_sha256"]] * 2,
        "candidate_anchor_exact": candidate_hashes
        == [contract["candidate_scorer_sha256"]] * 2,
        "live_generated_kernel_resolved": bool(modules),
        "noop_proxy_exact": noop["hashes"] == candidate_hashes,
        "noop_and_repair_each_call_once": noop["active_call_counts"]
        == repair["active_call_counts"]
        == [[1], [1]],
        "local_production_repeat_exact": local_repeat_exact,
        "repair_repeat_exact": repair["repeat_exact"],
        "no_backend_recompile": compiles_before == compiles_after,
        "kernel_restoration_exact": restored_hashes == candidate_hashes,
    }
    valid = all(gates.values())
    count = candidate_decisions.numel()
    payload = {
        "schema_version": "forkcert.qwen3-step236-live-original-kernel-group.v0.1",
        "status": "VALID" if valid else "INVALID",
        "valid": valid,
        "case_id": manifest["case_id"],
        "state_id": manifest["state_id"],
        "kernel_group": {
            "intercepted_generated_kernel": kernel_name,
            "semantic_scope": "fused final residual additions plus final RMSNorm output",
            "provenance": provenance_rows[0],
        },
        "gates": gates,
        "anchors": {
            "eager": eager_hashes,
            "candidate": candidate_hashes,
            "noop": noop["hashes"],
            "repair": repair["hashes"],
            "restored": restored_hashes,
        },
        "same_input_production": {
            "observed": production_observed,
            "repeat_exact": local_repeat_exact,
            "records": local_records,
        },
        "fixed_original_suffix_mediation": {
            "observed_continuous": mediation_observed,
            "candidate_to_repair": metrics(torch, candidate_value, repair_value),
            "eager_to_candidate": metrics(torch, eager_value, candidate_value),
            "eager_to_repair": metrics(torch, eager_value, repair_value),
            "off_to_on": int(upward.sum().item()),
            "on_to_off": int(downward.sum().item()),
            "semantic_disagreement": float((upward.sum() + downward.sum()).item() / count),
        },
        "interpretation": production_mediation_interpretation(
            production_observed, mediation_observed
        ),
        "compile_audit": audit,
        "resolution_records": resolution_records,
        "claim_scope": "one original generated kernel-group invocation at one forward matched state",
        "limitations": [
            "intervention identifies the fused generated group, not an individual constituent ATen op",
            "added synchronization and reference launches are controlled by an exact no-op proxy arm",
            "forward evidence does not license backward/update attribution",
            "eager expression is a baseline, not independent mathematical truth",
            "one state does not estimate population prevalence",
        ],
    }
    out.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({
        "valid": valid,
        "gates": gates,
        "production_observed": production_observed,
        "mediation_observed": mediation_observed,
        "candidate_to_repair": payload["fixed_original_suffix_mediation"]["candidate_to_repair"],
        "semantic_disagreement": payload["fixed_original_suffix_mediation"]["semantic_disagreement"],
    }, indent=2, sort_keys=True))
    raise SystemExit(0 if valid else 2)


if __name__ == "__main__":
    main()
