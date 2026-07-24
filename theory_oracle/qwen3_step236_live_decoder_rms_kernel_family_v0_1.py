#!/usr/bin/env python
"""Production/mediation map for selected calls of an original decoder RMS kernel family."""

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


def endpoint_records(
    torch: Any,
    logps: Any,
    inputs: dict[str, Any],
    coordinates: list[list[int]],
    epsilon: float,
) -> list[dict[str, Any]]:
    records = []
    for sample, token in coordinates:
        logp = logps[sample, token]
        old_logp = inputs["old_per_token_logps"][sample, token]
        advantage = inputs["advantages"][sample]
        ratio = torch.exp(logp - old_logp)
        if float(advantage.item()) > 0:
            boundary = 1.0 + epsilon
            signed_margin = ratio - boundary
        elif float(advantage.item()) < 0:
            boundary = 1.0 - epsilon
            signed_margin = boundary - ratio
        else:
            boundary = None
            signed_margin = None
        records.append(
            {
                "sample": sample,
                "token": token,
                "logp": float(logp.item()),
                "old_logp": float(old_logp.item()),
                "advantage": float(advantage.item()),
                "ratio": float(ratio.item()),
                "boundary": boundary,
                "clip_signed_margin": (
                    None if signed_margin is None else float(signed_margin.item())
                ),
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--observability-gate", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    manifest_path = Path(args.manifest).resolve()
    inventory_path = Path(args.inventory).resolve()
    gate_path = Path(args.observability_gate).resolve()
    manifest = json.loads(manifest_path.read_text())
    inventory = json.loads(inventory_path.read_text())
    gate = json.loads(gate_path.read_text())
    if gate.get("forward_kernel_inventory_eligible") is not True:
        raise RuntimeError("forward observability/provenance gate is not eligible")
    kernel_name = manifest["generated_kernel"]
    provenance_rows = [
        row for row in inventory["kernels"] if row["generated_symbol"] == kernel_name
    ]
    expected_calls = int(manifest["expected_runtime_calls"])
    if len(provenance_rows) != expected_calls:
        raise RuntimeError(
            f"inventory has {len(provenance_rows)} rows for {kernel_name}, expected {expected_calls}"
        )

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
        completion = value["completion_ids"]
        input_ids = torch.cat([value["prompt_ids"], completion], dim=1)
        attention_mask = torch.cat([value["prompt_mask"], value["completion_mask"]], dim=1)
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
    cpu_inputs = natural.move_tree(inputs, "cpu")

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
        raise RuntimeError("failed to resolve live original generated kernel family")

    class KernelProxy:
        def __init__(self, original: Any, selected: frozenset[int], repair: bool):
            self.original = original
            self.selected = selected
            self.repair = repair
            self.calls = 0
            self.repairs = 0
            self.records: list[dict[str, Any]] = []

        def run(self, *values: Any, **kwargs: Any) -> Any:
            call_index = self.calls
            self.calls += 1
            if call_index not in self.selected:
                return self.original.run(*values, **kwargs)
            before = [tensor_fingerprint(item) for item in values[:4]]
            result = self.original.run(*values, **kwargs)
            torch.cuda.synchronize()
            component0, component1, component2, weight, hidden_output, norm_output = values[:6]
            after = [tensor_fingerprint(item) for item in values[:4]]
            with torch.no_grad():
                compiled_hidden = hidden_output.detach().clone()
                compiled_norm = norm_output.detach().clone()
                logical_shape = hidden_output.shape
                reference_hidden = (
                    component0.reshape(logical_shape).float()
                    + component1.reshape(logical_shape).float()
                ) + component2.reshape(logical_shape).float()
                reference_rms = torch.rsqrt(
                    reference_hidden.square().mean(dim=-1, keepdim=True)
                    + float(manifest["epsilon"])
                )
                reference_norm = (
                    weight.reshape((1,) * (reference_hidden.ndim - 1) + (-1,)).float()
                    * reference_hidden
                    * reference_rms
                ).to(norm_output.dtype)
                record = {
                    "call_index": call_index,
                    "input_fingerprints_before": before,
                    "input_fingerprints_after": after,
                    "inputs_unchanged_by_generated_kernel": before == after,
                    "compiled_to_reference_hidden": metrics(
                        torch, reference_hidden, compiled_hidden
                    ),
                    "compiled_to_reference_norm": metrics(
                        torch, reference_norm, compiled_norm
                    ),
                    "compiled_hidden_sha256": natural.tensor_sha256(compiled_hidden),
                    "reference_hidden_sha256": natural.tensor_sha256(reference_hidden),
                    "compiled_norm_sha256": natural.tensor_sha256(compiled_norm),
                    "reference_norm_sha256": natural.tensor_sha256(reference_norm),
                }
                if self.repair:
                    hidden_output.copy_(reference_hidden)
                    norm_output.copy_(reference_norm)
                    self.repairs += 1
            self.records.append(record)
            return result

    def proxy_arm(selected: frozenset[int], repair: bool) -> dict[str, Any]:
        originals = {}
        proxies = {}
        for module_name, module in modules.items():
            originals[module_name] = getattr(module, kernel_name)
            proxies[module_name] = KernelProxy(originals[module_name], selected, repair)
            setattr(module, kernel_name, proxies[module_name])
        values = []
        hashes = []
        call_records = []
        local_records = []
        try:
            for _ in range(2):
                for proxy in proxies.values():
                    proxy.calls = 0
                    proxy.repairs = 0
                    proxy.records = []
                current = score(candidate, inputs).detach().float().cpu()
                values.append(current)
                hashes.append(natural.tensor_sha256(current))
                active = [proxy for proxy in proxies.values() if proxy.calls]
                call_records.append(
                    [
                        {"calls": proxy.calls, "repairs": proxy.repairs}
                        for proxy in active
                    ]
                )
                local_records.append(
                    [record for proxy in active for record in proxy.records]
                )
                gc.collect()
        finally:
            for module_name, module in modules.items():
                setattr(module, kernel_name, originals[module_name])
        return {
            "values": values,
            "hashes": hashes,
            "repeat_exact": hashes[0] == hashes[1],
            "call_records": call_records,
            "local_records": local_records,
        }

    compiles_before = audit["backend_compiles"]
    candidate_value = candidate_values[0]
    eager_value = eager_values[0]
    candidate_decisions = natural.clip_decisions(torch, candidate_value, cpu_inputs, 0.2)
    eager_decisions = natural.clip_decisions(torch, eager_value, cpu_inputs, 0.2)
    fork_coordinates = (
        torch.nonzero(candidate_decisions != eager_decisions, as_tuple=False)
        .tolist()
    )
    configured_sets = manifest.get("selected_call_sets")
    if configured_sets is None:
        configured_sets = {
            str(index): [int(index)] for index in manifest["selected_call_indices"]
        }
    callset_mode = any(len(indices) > 1 for indices in configured_sets.values())
    selected_results: dict[str, Any] = {}
    for treatment_id, indices in configured_sets.items():
        selected = frozenset(int(index) for index in indices)
        if not selected or min(selected) < 0 or max(selected) >= expected_calls:
            raise ValueError(f"invalid selected call set {treatment_id}: {sorted(selected)}")
        noop = proxy_arm(selected, repair=False)
        repair = proxy_arm(selected, repair=True)
        repair_value = repair["values"][0]
        repair_decisions = natural.clip_decisions(torch, repair_value, cpu_inputs, 0.2)
        upward = (~candidate_decisions) & repair_decisions
        downward = candidate_decisions & (~repair_decisions)
        records = repair["local_records"]
        production_repeat_exact = (
            len(records) == 2
            and len(records[0]) == len(records[1]) == len(selected)
            and records[0] == records[1]
        )
        production_observed = (
            production_repeat_exact
            and any(
                record["compiled_to_reference_hidden"]["nonzero"] > 0
                or record["compiled_to_reference_norm"]["nonzero"] > 0
                for record in records[0]
            )
        )
        continuous_mediation = repair["hashes"][0] != candidate_hashes[0]
        selected_results[str(treatment_id)] = {
            "selected_call_indices": sorted(selected),
            "noop": {
                "hashes": noop["hashes"],
                "repeat_exact": noop["repeat_exact"],
                "call_records": noop["call_records"],
                "local_records": noop["local_records"],
            },
            "repair": {
                "hashes": repair["hashes"],
                "repeat_exact": repair["repeat_exact"],
                "call_records": repair["call_records"],
            },
            "same_input_production": {
                "observed": production_observed,
                "repeat_exact": production_repeat_exact,
                "records": records,
            },
            "fixed_original_suffix_mediation": {
                "observed_continuous": continuous_mediation,
                "candidate_to_repair": metrics(torch, candidate_value, repair_value),
                "eager_to_candidate": metrics(torch, eager_value, candidate_value),
                "eager_to_repair": metrics(torch, eager_value, repair_value),
                "off_to_on": int(upward.sum().item()),
                "on_to_off": int(downward.sum().item()),
                "semantic_disagreement": float(
                    (upward.sum() + downward.sum()).item() / candidate_decisions.numel()
                ),
                "fork_coordinate_endpoints": endpoint_records(
                    torch, repair_value, cpu_inputs, fork_coordinates, 0.2
                ),
            },
            "interpretation": production_mediation_interpretation(
                production_observed, continuous_mediation
            ),
        }
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

    def arm_counts_exact(row: dict[str, Any], repair_count: int) -> bool:
        return all(
            call_record == [{"calls": expected_calls, "repairs": repair_count}]
            for call_record in row["call_records"]
        )

    gates = {
        "graph_family_exact": actual_family == expected_family,
        "eager_anchor_exact": eager_hashes == [contract["reference_scorer_sha256"]] * 2,
        "candidate_anchor_exact": candidate_hashes
        == [contract["candidate_scorer_sha256"]] * 2,
        "live_generated_kernel_resolved": bool(modules),
        "all_noop_proxies_exact": all(
            row["noop"]["hashes"] == candidate_hashes
            for row in selected_results.values()
        ),
        "all_noop_call_counts_exact": all(
            arm_counts_exact(row["noop"], 0) for row in selected_results.values()
        ),
        "all_repair_call_counts_exact": all(
            arm_counts_exact(row["repair"], len(row["selected_call_indices"]))
            for row in selected_results.values()
        ),
        "all_local_inputs_unchanged": all(
            record["inputs_unchanged_by_generated_kernel"]
            for row in selected_results.values()
            for repeat in row["same_input_production"]["records"]
            for record in repeat
        ),
        "all_local_production_repeats_exact": all(
            row["same_input_production"]["repeat_exact"]
            for row in selected_results.values()
        ),
        "all_repair_repeats_exact": all(
            row["repair"]["repeat_exact"] for row in selected_results.values()
        ),
        "no_backend_recompile": compiles_before == compiles_after,
        "kernel_restoration_exact": restored_hashes == candidate_hashes,
    }
    valid = all(gates.values())
    payload = {
        "schema_version": "forkcert.qwen3-step236-live-decoder-rms-kernel-family.v0.1",
        "status": "VALID" if valid else "INVALID",
        "valid": valid,
        "case_id": manifest["case_id"],
        "state_id": manifest["state_id"],
        "kernel_family": {
            "generated_kernel": kernel_name,
            "expected_runtime_calls": expected_calls,
            "selected_call_sets": {
                key: [int(index) for index in indices]
                for key, indices in configured_sets.items()
            },
            "semantic_scope": "fused decoder residual additions plus next-layer input RMSNorm",
            "provenance_rows": provenance_rows,
        },
        "gates": gates,
        "anchors": {
            "eager": eager_hashes,
            "candidate": candidate_hashes,
            "restored": restored_hashes,
        },
        "baseline_endpoint": {
            "fork_coordinates": fork_coordinates,
            "semantic_disagreement": float(
                (candidate_decisions != eager_decisions).sum().item()
                / candidate_decisions.numel()
            ),
            "eager": endpoint_records(torch, eager_value, cpu_inputs, fork_coordinates, 0.2),
            "candidate": endpoint_records(
                torch, candidate_value, cpu_inputs, fork_coordinates, 0.2
            ),
        },
        "selected_calls": selected_results,
        "compile_audit": audit,
        "resolution_records": resolution_records,
        "claim_scope": (
            "sequential call-set mediation in one original generated fused-kernel family at one forward matched state"
            if callset_mode
            else "selected singleton invocations of one original generated fused-kernel family at one forward matched state"
        ),
        "limitations": [
            "the generated family fuses residual additions, RMS reduction, normalization, cast, and views",
            "provenance does not uniquely identify one constituent ATen op",
            "added synchronization, fingerprints, and reference launches are controlled by exact no-op proxy arms",
            "forward evidence does not license backward/update attribution",
            "the declared ATen expression and eager are baselines, not independent mathematical truth",
            "one state does not estimate population prevalence",
            "within a multi-call treatment, only the first selected call necessarily receives its unmodified original-candidate boundary input; later selected calls may receive propagated repaired inputs",
        ],
        "artifact_inputs": {
            "manifest": str(manifest_path),
            "inventory": str(inventory_path),
            "observability_gate": str(gate_path),
        },
    }
    out.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "valid": valid,
                "gates": gates,
                "selected_calls": {
                    key: {
                        "production": row["same_input_production"]["observed"],
                        "continuous_mediation": row["fixed_original_suffix_mediation"][
                            "observed_continuous"
                        ],
                        "semantic_disagreement": row[
                            "fixed_original_suffix_mediation"
                        ]["semantic_disagreement"],
                        "candidate_to_repair": row[
                            "fixed_original_suffix_mediation"
                        ]["candidate_to_repair"],
                    }
                    for key, row in selected_results.items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    raise SystemExit(0 if valid else 2)


if __name__ == "__main__":
    main()
