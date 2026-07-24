#!/usr/bin/env python
"""Inject aligned eager layer boundaries into the unchanged original compiled suffix."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import qwen3_grpo_natural_transition_v0_2 as natural
from forkcert.operator_evidence import tensor_fingerprint
from qwen3_candidate_kernel15_repair_v0_1 import resolve_generated_modules
from qwen3_step236_live_decoder_rms_kernel_family_v0_1 import endpoint_records, metrics


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
    expected_calls = int(manifest["expected_runtime_calls"])
    provenance_rows = [
        row for row in inventory["kernels"] if row["generated_symbol"] == kernel_name
    ]
    if len(provenance_rows) != expected_calls:
        raise RuntimeError("inventory/runtime call-family cardinality contract failed")

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
    accelerator = Accelerator(mixed_precision="fp16")
    wrapped = accelerator.prepare_model(model)
    raw = accelerator.unwrap_model(wrapped)
    layers = raw.model.layers
    if len(layers) != expected_calls + 1:
        raise RuntimeError(f"expected {expected_calls + 1} decoder layers, found {len(layers)}")

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

    target_path = Path(metadata["target_minibatch_path"])
    if not target_path.is_file():
        target_path = snapshot / "compiler_history" / target_path.name
    inputs = natural.move_tree(
        torch.load(target_path, map_location="cpu", weights_only=False), "cuda"
    )
    cpu_inputs = natural.move_tree(inputs, "cpu")

    def capture_eager_once() -> tuple[Any, list[Any], list[Any]]:
        residuals: list[Any | None] = [None] * expected_calls
        next_norms: list[Any | None] = [None] * expected_calls
        handles = []

        def residual_hook(index: int):
            def hook(_module: Any, _inputs: Any, output: Any) -> None:
                value = output[0] if isinstance(output, tuple) else output
                residuals[index] = value.detach().clone()

            return hook

        def norm_hook(index: int):
            def hook(_module: Any, _inputs: Any, output: Any) -> None:
                next_norms[index] = output.detach().clone()

            return hook

        for index in range(expected_calls):
            handles.append(layers[index].register_forward_hook(residual_hook(index)))
            handles.append(
                layers[index + 1].input_layernorm.register_forward_hook(norm_hook(index))
            )
        try:
            scorer = score(wrapped, inputs).detach().float().cpu()
        finally:
            for handle in handles:
                handle.remove()
        if any(value is None for value in residuals + next_norms):
            raise RuntimeError("eager boundary hooks did not capture every aligned boundary")
        return scorer, list(residuals), list(next_norms)

    eager_score_1, eager_residuals, eager_norms = capture_eager_once()
    eager_score_2, eager_residuals_2, eager_norms_2 = capture_eager_once()
    eager_hashes = [
        natural.tensor_sha256(eager_score_1),
        natural.tensor_sha256(eager_score_2),
    ]
    eager_boundary_hashes = []
    for index in range(expected_calls):
        eager_boundary_hashes.append(
            {
                "call_index": index,
                "residual": [
                    natural.tensor_sha256(eager_residuals[index]),
                    natural.tensor_sha256(eager_residuals_2[index]),
                ],
                "next_input_norm": [
                    natural.tensor_sha256(eager_norms[index]),
                    natural.tensor_sha256(eager_norms_2[index]),
                ],
            }
        )
    del eager_residuals_2, eager_norms_2
    gc.collect()

    subblock_layer_indices = [
        int(index) for index in manifest.get("subblock_layer_slices", [])
    ]
    kernel_variant_names = [
        str(name) for name in manifest.get("kernel_op_variants", [])
    ]
    live_kernel_variant_names = [
        str(name) for name in manifest.get("live_kernel_variants", [])
    ]
    allowed_kernel_variants = {
        "sum_fp32",
        "high_precision",
        "input_fp16",
        "rsqrt_fp64",
        "reduce_fp64",
        "weight_fp16",
        "reference_reduce",
    }
    unknown_kernel_variants = set(kernel_variant_names) - allowed_kernel_variants
    if unknown_kernel_variants:
        raise ValueError(f"unknown kernel operation variants: {sorted(unknown_kernel_variants)}")
    if set(live_kernel_variant_names) - {"split_reduction"}:
        raise ValueError(
            "unknown live kernel variants: "
            f"{sorted(set(live_kernel_variant_names) - {'split_reduction'})}"
        )
    expected_subblock_modes = [
        ("noop", 0, 0, 0),
        ("compiled_attention", 1, 0, 0),
        ("eager_attention", 1, 1, 0),
        ("kernel_reference", 1, 1, 0),
        ("eager_block", 1, 1, 1),
    ] + [
        (f"kernel_variant:{name}", 1, 1, 0)
        for name in kernel_variant_names
    ] + [
        (f"live_kernel_variant:{name}", 1, 0, 0)
        for name in live_kernel_variant_names
    ]

    def capture_eager_subblocks_once() -> tuple[Any, dict[int, dict[str, Any]]]:
        captures: dict[int, dict[str, Any]] = {
            index: {} for index in subblock_layer_indices
        }
        handles = []

        def attention_hook(index: int):
            def hook(_module: Any, _inputs: Any, output: Any) -> None:
                value = output[0] if isinstance(output, tuple) else output
                captures[index]["attention_output"] = value.detach().clone()

            return hook

        def post_norm_pre_hook(index: int):
            def hook(_module: Any, inputs_: Any) -> None:
                captures[index]["post_attention_residual"] = inputs_[0].detach().clone()

            return hook

        def post_norm_hook(index: int):
            def hook(_module: Any, _inputs: Any, output: Any) -> None:
                captures[index]["post_attention_norm"] = output.detach().clone()

            return hook

        for index in subblock_layer_indices:
            handles.append(layers[index].self_attn.register_forward_hook(attention_hook(index)))
            handles.append(
                layers[index].post_attention_layernorm.register_forward_pre_hook(
                    post_norm_pre_hook(index)
                )
            )
            handles.append(
                layers[index].post_attention_layernorm.register_forward_hook(
                    post_norm_hook(index)
                )
            )
        try:
            scorer = score(wrapped, inputs).detach().float().cpu()
        finally:
            for handle in handles:
                handle.remove()
        for index, record in captures.items():
            missing = {
                "attention_output",
                "post_attention_residual",
                "post_attention_norm",
            } - set(record)
            if missing:
                raise RuntimeError(f"eager subblock capture {index} missing {sorted(missing)}")
        return scorer, captures

    eager_subblock_captures: dict[int, dict[str, Any]] = {}
    eager_subblock_hashes: dict[str, Any] = {}
    if subblock_layer_indices:
        subblock_score_1, eager_subblock_captures = capture_eager_subblocks_once()
        subblock_score_2, eager_subblock_captures_2 = capture_eager_subblocks_once()
        if [natural.tensor_sha256(subblock_score_1), natural.tensor_sha256(subblock_score_2)] != eager_hashes:
            raise RuntimeError("subblock hooks changed eager scorer anchor")
        for index in subblock_layer_indices:
            eager_subblock_hashes[str(index)] = {
                name: [
                    natural.tensor_sha256(eager_subblock_captures[index][name]),
                    natural.tensor_sha256(eager_subblock_captures_2[index][name]),
                ]
                for name in (
                    "attention_output",
                    "post_attention_residual",
                    "post_attention_norm",
                )
            }
        del subblock_score_1, subblock_score_2, eager_subblock_captures_2
        gc.collect()

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

    def repeated_candidate() -> tuple[list[Any], list[str]]:
        values, hashes = [], []
        for _ in range(2):
            current = score(candidate, inputs).detach().float().cpu()
            values.append(current)
            hashes.append(natural.tensor_sha256(current))
            gc.collect()
        return values, hashes

    candidate_values, candidate_hashes = repeated_candidate()
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
    intermediate_kernel_name = manifest.get("intermediate_generated_kernel")
    intermediate_modules: dict[str, Any] = {}
    intermediate_live_module_names: list[str] = []
    intermediate_provenance_rows: list[dict[str, Any]] = []
    if subblock_layer_indices:
        if not intermediate_kernel_name:
            raise ValueError("subblock slices require intermediate_generated_kernel")
        intermediate_provenance_rows = [
            row
            for row in inventory["kernels"]
            if row["generated_symbol"] == intermediate_kernel_name
        ]
        expected_intermediate_calls = len(layers)
        if len(intermediate_provenance_rows) != expected_intermediate_calls:
            raise RuntimeError(
                f"intermediate inventory has {len(intermediate_provenance_rows)} rows, expected {expected_intermediate_calls}"
            )
        for artifact in artifacts:
            resolved, observations = resolve_generated_modules(
                artifact, intermediate_kernel_name
            )
            intermediate_modules.update(resolved)
            resolution_records.append(
                {
                    "artifact_type": f"{type(artifact).__module__}.{type(artifact).__name__}",
                    "kernel": intermediate_kernel_name,
                    "resolved_modules": sorted(resolved),
                    "observations": observations,
                }
            )
        for module_name, module in list(sys.modules.items()):
            if (
                module is not None
                and module_name.startswith("torch._inductor.runtime.compile_tasks.")
                and hasattr(module, intermediate_kernel_name)
            ):
                intermediate_modules[module_name] = module
        intermediate_live_module_names = sorted(intermediate_modules)
        resolution_records.append(
            {
                "kernel": intermediate_kernel_name,
                "source": "live_compile_tasks_sys_modules",
                "resolved_modules": intermediate_live_module_names,
            }
        )
        if not intermediate_modules:
            raise RuntimeError("failed to resolve live intermediate generated kernel family")

    live_kernel_variant_kernels: dict[str, Any] = {}
    live_kernel_variant_metadata: dict[str, dict[str, Any]] = {}
    if live_kernel_variant_names:
        if not intermediate_provenance_rows:
            raise RuntimeError("live kernel variants require intermediate provenance")
        source_path = Path(intermediate_provenance_rows[0]["output_code_path"])
        source = source_path.read_text()
        old_reduction = "tmp8 = tl.sum(tmp7, 1)[:, None].to(tl.float32)"
        if source.count(old_reduction) != 1:
            raise RuntimeError("expected exactly one target reduction in generated code")
        for variant_name in live_kernel_variant_names:
            if variant_name != "split_reduction":
                raise AssertionError(variant_name)
            modified = source.replace(
                old_reduction,
                "tmp7_lo = tl.where(r0_index < 512, tmp7, 0)\n"
                "    tmp7_hi = tl.where(r0_index >= 512, tmp7, 0)\n"
                "    tmp8 = (tl.sum(tmp7_lo, 1) + tl.sum(tmp7_hi, 1))[:, None].to(tl.float32)",
            )
            temp_dir = Path(tempfile.mkdtemp(prefix="forkcert_live_kernel_"))
            modified_path = temp_dir / source_path.name
            modified_path.write_text(modified)
            module_name = f"forkcert_live_kernel_{variant_name}_{os.getpid()}"
            spec = importlib.util.spec_from_file_location(module_name, modified_path)
            if spec is None or spec.loader is None:
                raise RuntimeError(f"cannot load modified generated code: {modified_path}")
            modified_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(modified_module)
            live_kernel_variant_kernels[variant_name] = getattr(
                modified_module, intermediate_kernel_name
            )
            live_kernel_variant_metadata[variant_name] = {
                "source_output_code_path": str(source_path),
                "source_output_code_sha256": hashlib.sha256(source.encode()).hexdigest(),
                "modified_output_code_sha256": hashlib.sha256(modified.encode()).hexdigest(),
                "modification": "replace tl.sum over 1024 reduction with two 512-element sums",
                "modified_output_code_path": str(modified_path),
            }

    class BoundaryProxy:
        def __init__(self, original: Any, selected: int, intervene: bool):
            self.original = original
            self.selected = selected
            self.intervene = intervene
            self.calls = 0
            self.interventions = 0
            self.records: list[dict[str, Any]] = []

        def run(self, *values: Any, **kwargs: Any) -> Any:
            call_index = self.calls
            self.calls += 1
            if call_index != self.selected:
                return self.original.run(*values, **kwargs)
            result = self.original.run(*values, **kwargs)
            torch.cuda.synchronize()
            _component0, _component1, _component2, weight, residual, next_norm = values[:6]
            eager_residual = eager_residuals[call_index]
            eager_norm = eager_norms[call_index]
            transported_eager_norm = eager_norm.to(dtype=next_norm.dtype)
            expected_weight = layers[call_index + 1].input_layernorm.weight
            residual_stride_before = tuple(residual.stride())
            norm_stride_before = tuple(next_norm.stride())
            record = {
                "call_index": call_index,
                "weight_storage_identity": weight.data_ptr() == expected_weight.data_ptr(),
                "weight_exact": tensor_fingerprint(weight) == tensor_fingerprint(expected_weight),
                "residual_value_transport_contract": (
                    residual.shape == eager_residual.shape
                    and residual.dtype == eager_residual.dtype
                    and residual.device == eager_residual.device
                ),
                "norm_value_transport_contract": (
                    next_norm.shape == transported_eager_norm.shape
                    and next_norm.dtype == transported_eager_norm.dtype
                    and next_norm.device == transported_eager_norm.device
                ),
                "eager_residual_dtype": str(eager_residual.dtype),
                "compiled_residual_dtype": str(residual.dtype),
                "eager_norm_module_output_dtype": str(eager_norm.dtype),
                "compiled_norm_consumer_input_dtype": str(next_norm.dtype),
                "declared_norm_transport": f"{eager_norm.dtype} -> {next_norm.dtype}",
                "eager_residual_stride": list(eager_residual.stride()),
                "compiled_residual_stride": list(residual.stride()),
                "eager_norm_stride": list(eager_norm.stride()),
                "compiled_norm_stride": list(next_norm.stride()),
                "compiled_to_eager_residual": metrics(torch, eager_residual, residual),
                "compiled_to_transported_eager_next_norm": metrics(
                    torch, transported_eager_norm, next_norm
                ),
                "compiled_residual_sha256": natural.tensor_sha256(residual),
                "eager_residual_sha256": natural.tensor_sha256(eager_residual),
                "compiled_norm_sha256": natural.tensor_sha256(next_norm),
                "transported_eager_norm_sha256": natural.tensor_sha256(
                    transported_eager_norm
                ),
            }
            if self.intervene:
                residual.copy_(eager_residual)
                next_norm.copy_(transported_eager_norm)
                self.interventions += 1
            record["destination_layout_preserved"] = (
                tuple(residual.stride()) == residual_stride_before
                and tuple(next_norm.stride()) == norm_stride_before
            )
            self.records.append(record)
            return result

    def arm(selected: int, intervene: bool) -> dict[str, Any]:
        originals, proxies = {}, {}
        for module_name, module in modules.items():
            originals[module_name] = getattr(module, kernel_name)
            proxies[module_name] = BoundaryProxy(originals[module_name], selected, intervene)
            setattr(module, kernel_name, proxies[module_name])
        values, hashes, calls, records = [], [], [], []
        try:
            for _ in range(2):
                for proxy in proxies.values():
                    proxy.calls = 0
                    proxy.interventions = 0
                    proxy.records = []
                current = score(candidate, inputs).detach().float().cpu()
                values.append(current)
                hashes.append(natural.tensor_sha256(current))
                active = [proxy for proxy in proxies.values() if proxy.calls]
                calls.append(
                    [
                        {"calls": proxy.calls, "interventions": proxy.interventions}
                        for proxy in active
                    ]
                )
                records.append([record for proxy in active for record in proxy.records])
                gc.collect()
        finally:
            for module_name, module in modules.items():
                setattr(module, kernel_name, originals[module_name])
        return {
            "values": values,
            "hashes": hashes,
            "repeat_exact": hashes[0] == hashes[1],
            "call_records": calls,
            "boundary_records": records,
        }

    class ContextualLayerProxy:
        def __init__(self, original: Any, layer_index: int, mode: str):
            self.original = original
            self.layer_index = layer_index
            self.entry_call = layer_index - 1
            self.exit_call = layer_index
            self.mode = mode
            self.calls = 0
            self.entry_injections = 0
            self.exit_injections = 0
            self.records: list[dict[str, Any]] = []

        def run(self, *values: Any, **kwargs: Any) -> Any:
            call_index = self.calls
            self.calls += 1
            if call_index not in (self.entry_call, self.exit_call):
                return self.original.run(*values, **kwargs)
            result = self.original.run(*values, **kwargs)
            torch.cuda.synchronize()
            _component0, _component1, _component2, weight, residual, next_norm = values[:6]
            eager_residual = eager_residuals[call_index]
            eager_norm = eager_norms[call_index].to(dtype=next_norm.dtype)
            expected_weight = layers[call_index + 1].input_layernorm.weight
            residual_stride = tuple(residual.stride())
            norm_stride = tuple(next_norm.stride())
            residual_stride = tuple(residual.stride())
            norm_stride = tuple(next_norm.stride())
            record = {
                "call_index": call_index,
                "role": "entry" if call_index == self.entry_call else "exit",
                "weight_storage_identity": weight.data_ptr() == expected_weight.data_ptr(),
                "weight_exact": tensor_fingerprint(weight) == tensor_fingerprint(expected_weight),
                "residual_value_transport_contract": (
                    residual.shape == eager_residual.shape
                    and residual.dtype == eager_residual.dtype
                    and residual.device == eager_residual.device
                ),
                "norm_value_transport_contract": (
                    next_norm.shape == eager_norm.shape
                    and next_norm.dtype == eager_norm.dtype
                    and next_norm.device == eager_norm.device
                ),
                "compiled_to_eager_residual": metrics(torch, eager_residual, residual),
                "compiled_to_transported_eager_next_norm": metrics(
                    torch, eager_norm, next_norm
                ),
                "compiled_residual_sha256": natural.tensor_sha256(residual),
                "eager_residual_sha256": natural.tensor_sha256(eager_residual),
                "compiled_norm_sha256": natural.tensor_sha256(next_norm),
                "transported_eager_norm_sha256": natural.tensor_sha256(eager_norm),
            }
            inject_entry = call_index == self.entry_call and self.mode != "noop"
            inject_exit = call_index == self.exit_call and self.mode == "eager_block"
            if inject_entry or inject_exit:
                residual.copy_(eager_residual)
                next_norm.copy_(eager_norm)
                if inject_entry:
                    self.entry_injections += 1
                else:
                    self.exit_injections += 1
            record["injected"] = inject_entry or inject_exit
            record["post_residual_sha256"] = natural.tensor_sha256(residual)
            record["post_norm_sha256"] = natural.tensor_sha256(next_norm)
            record["destination_layout_preserved"] = (
                tuple(residual.stride()) == residual_stride
                and tuple(next_norm.stride()) == norm_stride
            )
            self.records.append(record)
            return result

    def contextual_arm(layer_index: int, mode: str) -> dict[str, Any]:
        originals, proxies = {}, {}
        for module_name, module in modules.items():
            originals[module_name] = getattr(module, kernel_name)
            proxies[module_name] = ContextualLayerProxy(
                originals[module_name], layer_index, mode
            )
            setattr(module, kernel_name, proxies[module_name])
        values, hashes, calls, records = [], [], [], []
        try:
            for _ in range(2):
                for proxy in proxies.values():
                    proxy.calls = 0
                    proxy.entry_injections = 0
                    proxy.exit_injections = 0
                    proxy.records = []
                current = score(candidate, inputs).detach().float().cpu()
                values.append(current)
                hashes.append(natural.tensor_sha256(current))
                active = [proxy for proxy in proxies.values() if proxy.calls]
                calls.append(
                    [
                        {
                            "calls": proxy.calls,
                            "entry_injections": proxy.entry_injections,
                            "exit_injections": proxy.exit_injections,
                        }
                        for proxy in active
                    ]
                )
                records.append([record for proxy in active for record in proxy.records])
                gc.collect()
        finally:
            for module_name, module in modules.items():
                setattr(module, kernel_name, originals[module_name])
        return {
            "values": values,
            "hashes": hashes,
            "repeat_exact": hashes[0] == hashes[1],
            "call_records": calls,
            "boundary_records": records,
        }

    class SubblockBoundaryProxy:
        def __init__(self, original: Any, layer_index: int, mode: str):
            self.original = original
            self.layer_index = layer_index
            self.entry_call = layer_index - 1
            self.exit_call = layer_index
            self.mode = mode
            self.calls = 0
            self.entry_injections = 0
            self.exit_injections = 0
            self.records: list[dict[str, Any]] = []

        def run(self, *values: Any, **kwargs: Any) -> Any:
            call_index = self.calls
            self.calls += 1
            if call_index not in (self.entry_call, self.exit_call):
                return self.original.run(*values, **kwargs)
            result = self.original.run(*values, **kwargs)
            torch.cuda.synchronize()
            _component0, _component1, _component2, weight, residual, next_norm = values[:6]
            eager_residual = eager_residuals[call_index]
            eager_norm = eager_norms[call_index].to(dtype=next_norm.dtype)
            expected_weight = layers[call_index + 1].input_layernorm.weight
            residual_stride = tuple(residual.stride())
            norm_stride = tuple(next_norm.stride())
            record = {
                "call_index": call_index,
                "role": "entry" if call_index == self.entry_call else "exit",
                "weight_storage_identity": weight.data_ptr() == expected_weight.data_ptr(),
                "weight_exact": tensor_fingerprint(weight) == tensor_fingerprint(expected_weight),
                "compiled_to_eager_residual": metrics(torch, eager_residual, residual),
                "compiled_to_transported_eager_next_norm": metrics(
                    torch, eager_norm, next_norm
                ),
                "compiled_residual_sha256": natural.tensor_sha256(residual),
                "compiled_norm_sha256": natural.tensor_sha256(next_norm),
                "eager_residual_sha256": natural.tensor_sha256(eager_residual),
                "transported_eager_norm_sha256": natural.tensor_sha256(eager_norm),
            }
            inject_entry = call_index == self.entry_call and self.mode != "noop"
            inject_exit = call_index == self.exit_call and self.mode == "eager_block"
            if inject_entry or inject_exit:
                residual.copy_(eager_residual)
                next_norm.copy_(eager_norm)
                if inject_entry:
                    self.entry_injections += 1
                else:
                    self.exit_injections += 1
            record["injected"] = inject_entry or inject_exit
            record["post_residual_sha256"] = natural.tensor_sha256(residual)
            record["post_norm_sha256"] = natural.tensor_sha256(next_norm)
            record["destination_layout_preserved"] = (
                tuple(residual.stride()) == residual_stride
                and tuple(next_norm.stride()) == norm_stride
            )
            self.records.append(record)
            return result

    class SubblockIntermediateProxy:
        def __init__(self, original: Any, layer_index: int, mode: str, module_name: str):
            self.original = original
            self.layer_index = layer_index
            self.mode = mode
            self.module_name = module_name
            self.calls = 0
            self.injections = 0
            self.records: list[dict[str, Any]] = []

        def run(self, *values: Any, **kwargs: Any) -> Any:
            call_index = self.calls
            self.calls += 1
            if call_index != self.layer_index:
                return self.original.run(*values, **kwargs)
            live_variant_name = (
                self.mode.split(":", 1)[1]
                if self.mode.startswith("live_kernel_variant:")
                else None
            )
            if live_variant_name is None:
                result = self.original.run(*values, **kwargs)
            else:
                result = live_kernel_variant_kernels[live_variant_name].run(
                    *values, **kwargs
                )
            torch.cuda.synchronize()
            residual_input, attention_output, weight, post_norm = values[:4]
            eager = eager_subblock_captures[call_index]
            eager_attention_module = eager["attention_output"]
            if eager_attention_module.numel() != attention_output.numel():
                raise RuntimeError("attention boundary element count differs")
            eager_attention = eager_attention_module.to(
                dtype=attention_output.dtype
            ).reshape(attention_output.shape)
            eager_post_residual = eager["post_attention_residual"]
            eager_post_norm = eager["post_attention_norm"].to(dtype=post_norm.dtype)
            expected_weight = layers[call_index].post_attention_layernorm.weight
            compiled_post_residual = residual_input.float() + attention_output.reshape(
                residual_input.shape
            ).float()
            # Kernel-local production control: replay the eager RMSNorm on the
            # *same* residual and attention-output tensors consumed by this
            # generated kernel.  The separately captured eager post-attention
            # output is useful for region analysis, but it is not a same-input
            # kernel reference because eager attention may already differ.
            with torch.no_grad():
                same_input_reference_post_norm = layers[
                    call_index
                ].post_attention_layernorm(
                    compiled_post_residual.to(dtype=residual_input.dtype)
                ).to(dtype=post_norm.dtype)
                variant_outputs: dict[str, Any] = {}
                weight_broadcast = weight.float().reshape(
                    (1,) * (compiled_post_residual.ndim - 1) + (weight.numel(),)
                )
                for variant_name in kernel_variant_names:
                    if variant_name == "sum_fp32":
                        variance = (
                            compiled_post_residual.square().sum(
                                dim=-1, keepdim=True
                            )
                            / compiled_post_residual.shape[-1]
                        )
                        normalized = compiled_post_residual * torch.rsqrt(
                            variance + 1e-6
                        )
                    elif variant_name == "high_precision":
                        high = compiled_post_residual.double()
                        variance = high.square().mean(dim=-1, keepdim=True)
                        normalized = (
                            high * torch.rsqrt(variance + 1e-6)
                        ).float()
                    elif variant_name == "input_fp16":
                        low = compiled_post_residual.to(torch.float16).float()
                        variance = low.square().mean(dim=-1, keepdim=True)
                        normalized = low * torch.rsqrt(variance + 1e-6)
                    elif variant_name == "rsqrt_fp64":
                        variance = compiled_post_residual.square().mean(
                            dim=-1, keepdim=True
                        )
                        inverse = torch.rsqrt((variance + 1e-6).double()).float()
                        normalized = compiled_post_residual * inverse
                    elif variant_name == "reduce_fp64":
                        # Change only the reduction accumulator.  The input,
                        # epsilon, rsqrt, normalization, weight and store
                        # stages remain at the baseline dtypes.
                        variance = (
                            compiled_post_residual.double().square().mean(
                                dim=-1, keepdim=True
                            )
                        ).float()
                        normalized = compiled_post_residual * torch.rsqrt(
                            variance + 1e-6
                        )
                    elif variant_name == "weight_fp16":
                        # Change only the weight representation at the
                        # multiply stage; the residual, reduction, rsqrt and
                        # output store remain unchanged.
                        variance = compiled_post_residual.square().mean(
                            dim=-1, keepdim=True
                        )
                        normalized = compiled_post_residual * torch.rsqrt(
                            variance + 1e-6
                        )
                        weight_broadcast = weight.to(torch.float16).float().reshape(
                            (1,) * (compiled_post_residual.ndim - 1)
                            + (weight.numel(),)
                        )
                    elif variant_name == "reference_reduce":
                        # Keep the compiled input, rsqrt, multiply and store,
                        # but use the reference RMSNorm reduction expression
                        # only.  This is the narrowest available reduction
                        # counterfactual without editing the generated kernel.
                        variance = compiled_post_residual.pow(2).mean(
                            dim=-1, keepdim=True
                        )
                        normalized = compiled_post_residual * torch.rsqrt(
                            variance + 1e-6
                        )
                    else:  # guarded by allowed_kernel_variants above
                        raise AssertionError(variant_name)
                    variant_outputs[variant_name] = (
                        weight_broadcast * normalized
                    ).to(dtype=post_norm.dtype)
            attention_stride = tuple(attention_output.stride())
            norm_stride = tuple(post_norm.stride())
            record = {
                "call_index": call_index,
                "live_module_name": self.module_name,
                "weight_storage_identity": weight.data_ptr() == expected_weight.data_ptr(),
                "weight_exact": tensor_fingerprint(weight) == tensor_fingerprint(expected_weight),
                "attention_transport_contract": (
                    attention_output.shape == eager_attention.shape
                    and attention_output.dtype == eager_attention.dtype
                    and attention_output.device == eager_attention.device
                ),
                "attention_module_shape": list(eager_attention_module.shape),
                "attention_generated_abi_shape": list(attention_output.shape),
                "declared_attention_view_transport": "reshape preserving contiguous logical order",
                "post_norm_transport_contract": (
                    post_norm.shape == eager_post_norm.shape
                    and post_norm.dtype == eager_post_norm.dtype
                    and post_norm.device == eager_post_norm.device
                ),
                "compiled_to_eager_attention_output": metrics(
                    torch, eager_attention, attention_output
                ),
                "compiled_to_eager_post_attention_residual": metrics(
                    torch, eager_post_residual, compiled_post_residual
                ),
                "compiled_to_transported_eager_post_norm": metrics(
                    torch, eager_post_norm, post_norm
                ),
                "same_input_kernel_post_norm": metrics(
                    torch, same_input_reference_post_norm, post_norm
                ),
                "same_input_kernel_post_norm_production": (
                    metrics(torch, same_input_reference_post_norm, post_norm)[
                        "nonzero"
                    ]
                    > 0
                ),
                "same_input_kernel_reference_post_norm_sha256": natural.tensor_sha256(
                    same_input_reference_post_norm
                ),
                "same_input_kernel_input_residual_sha256": natural.tensor_sha256(
                    residual_input
                ),
                "same_input_kernel_input_attention_sha256": natural.tensor_sha256(
                    attention_output
                ),
                "compiled_attention_sha256": natural.tensor_sha256(attention_output),
                "eager_attention_sha256": natural.tensor_sha256(eager_attention),
                "compiled_post_residual_sha256": natural.tensor_sha256(
                    compiled_post_residual
                ),
                "eager_post_residual_sha256": natural.tensor_sha256(
                    eager_post_residual
                ),
                "compiled_post_norm_sha256": natural.tensor_sha256(post_norm),
                "transported_eager_post_norm_sha256": natural.tensor_sha256(
                    eager_post_norm
                ),
                "kernel_op_variant_metrics": {
                    name: metrics(torch, value, post_norm)
                    for name, value in variant_outputs.items()
                },
                "kernel_op_variant_sha256": {
                    name: natural.tensor_sha256(value)
                    for name, value in variant_outputs.items()
                },
                "live_kernel_variant": live_variant_name,
                "live_kernel_variant_metadata": (
                    live_kernel_variant_metadata.get(live_variant_name)
                    if live_variant_name is not None
                    else None
                ),
            }
            inject_attention = self.mode in ("eager_attention", "eager_block")
            inject_kernel_post_norm = self.mode in (
                "eager_attention",
                "eager_block",
                "kernel_reference",
            ) or self.mode.startswith("kernel_variant:")
            if inject_attention:
                attention_output.copy_(eager_attention)
            if inject_kernel_post_norm:
                post_norm.copy_(
                    same_input_reference_post_norm
                    if self.mode == "kernel_reference"
                    else (
                        variant_outputs[self.mode.split(":", 1)[1]]
                        if self.mode.startswith("kernel_variant:")
                        else eager_post_norm
                    )
                )
                self.injections += 1
            record["injected"] = inject_attention or inject_kernel_post_norm
            record["injected_attention"] = inject_attention
            record["injected_kernel_post_norm"] = inject_kernel_post_norm
            record["post_attention_sha256"] = natural.tensor_sha256(attention_output)
            record["post_norm_sha256"] = natural.tensor_sha256(post_norm)
            record["destination_layout_preserved"] = (
                tuple(attention_output.stride()) == attention_stride
                and tuple(post_norm.stride()) == norm_stride
            )
            self.records.append(record)
            return result

    def subblock_arm(layer_index: int, mode: str) -> dict[str, Any]:
        boundary_originals, boundary_proxies = {}, {}
        intermediate_originals, intermediate_proxies = {}, {}
        for module_name, module in modules.items():
            boundary_originals[module_name] = getattr(module, kernel_name)
            boundary_proxies[module_name] = SubblockBoundaryProxy(
                boundary_originals[module_name], layer_index, mode
            )
            setattr(module, kernel_name, boundary_proxies[module_name])
        for module_name, module in intermediate_modules.items():
            intermediate_originals[module_name] = getattr(
                module, intermediate_kernel_name
            )
            intermediate_proxies[module_name] = SubblockIntermediateProxy(
                intermediate_originals[module_name], layer_index, mode, module_name
            )
            setattr(
                module,
                intermediate_kernel_name,
                intermediate_proxies[module_name],
            )
        values, hashes, call_records, boundary_records, intermediate_records = (
            [],
            [],
            [],
            [],
            [],
        )
        try:
            for _ in range(2):
                for proxy in boundary_proxies.values():
                    proxy.calls = 0
                    proxy.entry_injections = 0
                    proxy.exit_injections = 0
                    proxy.records = []
                for proxy in intermediate_proxies.values():
                    proxy.calls = 0
                    proxy.injections = 0
                    proxy.records = []
                current = score(candidate, inputs).detach().float().cpu()
                values.append(current)
                hashes.append(natural.tensor_sha256(current))
                active_boundary = [
                    proxy for proxy in boundary_proxies.values() if proxy.calls
                ]
                active_intermediate = [
                    proxy for proxy in intermediate_proxies.values() if proxy.calls
                ]
                call_records.append(
                    {
                        "boundary": [
                            {
                                "calls": proxy.calls,
                                "entry_injections": proxy.entry_injections,
                                "exit_injections": proxy.exit_injections,
                            }
                            for proxy in active_boundary
                        ],
                        "intermediate": [
                            {"calls": proxy.calls, "injections": proxy.injections}
                            for proxy in active_intermediate
                        ],
                    }
                )
                boundary_records.append(
                    [record for proxy in active_boundary for record in proxy.records]
                )
                intermediate_records.append(
                    [
                        record
                        for proxy in active_intermediate
                        for record in proxy.records
                    ]
                )
                gc.collect()
        finally:
            for module_name, module in modules.items():
                setattr(module, kernel_name, boundary_originals[module_name])
            for module_name, module in intermediate_modules.items():
                setattr(
                    module,
                    intermediate_kernel_name,
                    intermediate_originals[module_name],
                )
        return {
            "values": values,
            "hashes": hashes,
            "repeat_exact": hashes[0] == hashes[1],
            "call_records": call_records,
            "boundary_records": boundary_records,
            "intermediate_records": intermediate_records,
        }

    candidate_value = candidate_values[0]
    eager_value = eager_score_1
    candidate_decisions = natural.clip_decisions(torch, candidate_value, cpu_inputs, 0.2)
    eager_decisions = natural.clip_decisions(torch, eager_value, cpu_inputs, 0.2)
    fork_coordinates = (
        torch.nonzero(candidate_decisions != eager_decisions, as_tuple=False).tolist()
    )
    compiles_before = audit["backend_compiles"]
    treatments = {}
    for selected in manifest["selected_call_indices"]:
        selected = int(selected)
        noop = arm(selected, intervene=False)
        intervention = arm(selected, intervene=True)
        intervention_value = intervention["values"][0]
        intervention_decisions = natural.clip_decisions(
            torch, intervention_value, cpu_inputs, 0.2
        )
        upward = (~candidate_decisions) & intervention_decisions
        downward = candidate_decisions & (~intervention_decisions)
        records = intervention["boundary_records"]
        record_repeat_exact = (
            len(records) == 2
            and len(records[0]) == len(records[1]) == 1
            and records[0][0] == records[1][0]
        )
        treatments[str(selected)] = {
            "aligned_boundary": {
                "residual": f"eager model.layers.{selected} output",
                "next_input_norm": f"eager model.layers.{selected + 1}.input_layernorm output",
            },
            "noop": {
                "hashes": noop["hashes"],
                "repeat_exact": noop["repeat_exact"],
                "call_records": noop["call_records"],
                "boundary_records": noop["boundary_records"],
            },
            "intervention": {
                "hashes": intervention["hashes"],
                "repeat_exact": intervention["repeat_exact"],
                "call_records": intervention["call_records"],
                "boundary_records": records,
                "boundary_record_repeat_exact": record_repeat_exact,
            },
            "fixed_original_suffix_mediation": {
                "observed_continuous": intervention["hashes"][0]
                != candidate_hashes[0],
                "candidate_to_intervention": metrics(
                    torch, candidate_value, intervention_value
                ),
                "eager_to_candidate": metrics(torch, eager_value, candidate_value),
                "eager_to_intervention": metrics(
                    torch, eager_value, intervention_value
                ),
                "off_to_on": int(upward.sum().item()),
                "on_to_off": int(downward.sum().item()),
                "semantic_disagreement": float(
                    (upward.sum() + downward.sum()).item()
                    / candidate_decisions.numel()
                ),
                "fork_coordinate_endpoints": endpoint_records(
                    torch, intervention_value, cpu_inputs, fork_coordinates, 0.2
                ),
            },
        }
    contextual_slices = {}
    boundary_map = None
    if manifest.get("boundary_map_evidence"):
        boundary_map = json.loads(Path(manifest["boundary_map_evidence"]).read_text())
    for layer_index in manifest.get("contextual_layer_slices", []):
        layer_index = int(layer_index)
        if layer_index <= 0 or layer_index >= expected_calls:
            raise ValueError(f"contextual layer {layer_index} lacks aligned entry/exit calls")
        noop = contextual_arm(layer_index, "noop")
        compiled_block = contextual_arm(layer_index, "compiled_block")
        eager_block = contextual_arm(layer_index, "eager_block")
        compiled_value = compiled_block["values"][0]
        eager_block_value = eager_block["values"][0]
        compiled_decisions = natural.clip_decisions(
            torch, compiled_value, cpu_inputs, 0.2
        )
        eager_block_decisions = natural.clip_decisions(
            torch, eager_block_value, cpu_inputs, 0.2
        )
        upward = (~compiled_decisions) & eager_block_decisions
        downward = compiled_decisions & (~eager_block_decisions)
        compiled_records = compiled_block["boundary_records"]
        eager_records = eager_block["boundary_records"]
        compiled_exit = compiled_records[0][1]
        def pre_repair_signature(record: dict[str, Any]) -> dict[str, Any]:
            return {
                key: value
                for key, value in record.items()
                if key not in {"injected", "post_residual_sha256", "post_norm_sha256"}
            }
        local_production = (
            compiled_exit["compiled_to_eager_residual"]["nonzero"] > 0
            or compiled_exit["compiled_to_transported_eager_next_norm"]["nonzero"] > 0
        )
        entry_id, exit_id = str(layer_index - 1), str(layer_index)
        expected_compiled_hashes = (
            None
            if boundary_map is None
            else boundary_map["treatments"][entry_id]["intervention"]["hashes"]
        )
        expected_eager_hashes = (
            None
            if boundary_map is None
            else boundary_map["treatments"][exit_id]["intervention"]["hashes"]
        )
        contextual_slices[str(layer_index)] = {
            "subject": f"model.layers.{layer_index} composite decoder block",
            "entry_boundary_call": layer_index - 1,
            "exit_boundary_call": layer_index,
            "noop": {key: value for key, value in noop.items() if key != "values"},
            "compiled_block": {
                **{key: value for key, value in compiled_block.items() if key != "values"},
                "matches_independent_entry_boundary_treatment": (
                    expected_compiled_hashes is not None
                    and compiled_block["hashes"] == expected_compiled_hashes
                ),
            },
            "eager_block": {
                **{key: value for key, value in eager_block.items() if key != "values"},
                "matches_independent_exit_boundary_treatment": (
                    expected_eager_hashes is not None
                    and eager_block["hashes"] == expected_eager_hashes
                ),
            },
            "same_eager_input_composite_layer_production": {
                "observed": local_production,
                "compiled_exit_record": compiled_exit,
                "compiled_exit_records_repeat_exact": (
                    len(compiled_records) == 2
                    and len(compiled_records[0]) == len(compiled_records[1]) == 2
                    and compiled_records[0] == compiled_records[1]
                ),
                "compiled_and_eager_arms_pre_repair_exit_exact": (
                    len(eager_records) == 2
                    and len(eager_records[0]) == len(eager_records[1]) == 2
                    and pre_repair_signature(eager_records[0][1])
                    == pre_repair_signature(compiled_records[0][1])
                    and pre_repair_signature(eager_records[1][1])
                    == pre_repair_signature(compiled_records[1][1])
                ),
            },
            "fixed_original_suffix_layer_mediation": {
                "observed_continuous": compiled_block["hashes"][0]
                != eager_block["hashes"][0],
                "compiled_to_eager_block": metrics(
                    torch, compiled_value, eager_block_value
                ),
                "off_to_on": int(upward.sum().item()),
                "on_to_off": int(downward.sum().item()),
                "semantic_disagreement": float(
                    (upward.sum() + downward.sum()).item()
                    / compiled_decisions.numel()
                ),
                "compiled_block_fork_coordinate_endpoints": endpoint_records(
                    torch, compiled_value, cpu_inputs, fork_coordinates, 0.2
                ),
                "eager_block_fork_coordinate_endpoints": endpoint_records(
                    torch, eager_block_value, cpu_inputs, fork_coordinates, 0.2
                ),
            },
        }
    subblock_slices = {}
    for layer_index in subblock_layer_indices:
        subblock_modes = [mode for mode, *_ in expected_subblock_modes]
        arms = {
            mode: subblock_arm(layer_index, mode)
            for mode in subblock_modes
        }
        compiled_attention_value = arms["compiled_attention"]["values"][0]
        eager_attention_value = arms["eager_attention"]["values"][0]
        kernel_reference_value = arms["kernel_reference"]["values"][0]
        eager_block_value = arms["eager_block"]["values"][0]
        compiled_attention_decisions = natural.clip_decisions(
            torch, compiled_attention_value, cpu_inputs, 0.2
        )
        eager_attention_decisions = natural.clip_decisions(
            torch, eager_attention_value, cpu_inputs, 0.2
        )
        kernel_reference_decisions = natural.clip_decisions(
            torch, kernel_reference_value, cpu_inputs, 0.2
        )
        eager_block_decisions = natural.clip_decisions(
            torch, eager_block_value, cpu_inputs, 0.2
        )
        kernel_variant_values = {
            name: arms[f"kernel_variant:{name}"]["values"][0]
            for name in kernel_variant_names
        }
        kernel_variant_decisions = {
            name: natural.clip_decisions(torch, value, cpu_inputs, 0.2)
            for name, value in kernel_variant_values.items()
        }
        live_kernel_variant_values = {
            name: arms[f"live_kernel_variant:{name}"]["values"][0]
            for name in live_kernel_variant_names
        }
        live_kernel_variant_decisions = {
            name: natural.clip_decisions(torch, value, cpu_inputs, 0.2)
            for name, value in live_kernel_variant_values.items()
        }

        def semantic_contrast(left: Any, right: Any) -> dict[str, Any]:
            upward = (~left) & right
            downward = left & (~right)
            return {
                "off_to_on": int(upward.sum().item()),
                "on_to_off": int(downward.sum().item()),
                "semantic_disagreement": float(
                    (upward.sum() + downward.sum()).item() / left.numel()
                ),
            }

        def pre_repair_signature(record: dict[str, Any]) -> dict[str, Any]:
            return {
                key: value
                for key, value in record.items()
                if key
                not in {
                    "injected",
                    "injected_attention",
                    "injected_kernel_post_norm",
                    "post_attention_sha256",
                    "post_norm_sha256",
                    "post_residual_sha256",
                }
            }

        compiled_mid = arms["compiled_attention"]["intermediate_records"]
        eager_mid = arms["eager_attention"]["intermediate_records"]
        eager_block_mid = arms["eager_block"]["intermediate_records"]
        eager_attention_exit = arms["eager_attention"]["boundary_records"]
        eager_block_exit = arms["eager_block"]["boundary_records"]
        attention_record = compiled_mid[0][0]
        mlp_exit_record = eager_attention_exit[0][1]
        attention_production = any(
            attention_record[key]["nonzero"] > 0
            for key in (
                "compiled_to_eager_attention_output",
                "compiled_to_eager_post_attention_residual",
                "compiled_to_transported_eager_post_norm",
            )
        )
        mlp_production = (
            mlp_exit_record["compiled_to_eager_residual"]["nonzero"] > 0
            or mlp_exit_record["compiled_to_transported_eager_next_norm"][
                "nonzero"
            ]
            > 0
        )
        kernel_local_record = compiled_mid[0][0]
        entry_id, exit_id = str(layer_index - 1), str(layer_index)
        expected_entry_hashes = (
            None
            if boundary_map is None
            else boundary_map["treatments"][entry_id]["intervention"]["hashes"]
        )
        expected_exit_hashes = (
            None
            if boundary_map is None
            else boundary_map["treatments"][exit_id]["intervention"]["hashes"]
        )
        subblock_slices[str(layer_index)] = {
            "subject": f"model.layers.{layer_index} attention/MLP contextual split",
            "entry_boundary_call": layer_index - 1,
            "intermediate_kernel_call": layer_index,
            "exit_boundary_call": layer_index,
            "arms": {
                mode: {
                    **{key: value for key, value in arm_value.items() if key != "values"},
                    **(
                        {
                            "matches_independent_entry_boundary_treatment": arm_value[
                                "hashes"
                            ]
                            == expected_entry_hashes
                        }
                        if mode == "compiled_attention"
                        else {
                            "matches_independent_exit_boundary_treatment": arm_value[
                                "hashes"
                            ]
                            == expected_exit_hashes
                        }
                        if mode == "eager_block"
                        else {}
                    ),
                }
                for mode, arm_value in arms.items()
            },
            "same_eager_input_attention_region_production": {
                "observed": attention_production,
                "compiled_attention_record": attention_record,
                "records_repeat_exact": (
                    len(compiled_mid) == 2
                    and len(compiled_mid[0]) == len(compiled_mid[1]) == 1
                    and compiled_mid[0] == compiled_mid[1]
                ),
                "compiled_and_eager_attention_arms_pre_repair_exact": (
                    len(eager_mid) == len(eager_block_mid) == 2
                    and all(len(row) == 1 for row in eager_mid + eager_block_mid)
                    and pre_repair_signature(eager_mid[0][0])
                    == pre_repair_signature(compiled_mid[0][0])
                    and pre_repair_signature(eager_mid[1][0])
                    == pre_repair_signature(compiled_mid[1][0])
                    and pre_repair_signature(eager_block_mid[0][0])
                    == pre_repair_signature(compiled_mid[0][0])
                    and pre_repair_signature(eager_block_mid[1][0])
                    == pre_repair_signature(compiled_mid[1][0])
                ),
            },
            "same_input_intermediate_kernel_production": {
                "generated_kernel": intermediate_kernel_name,
                "call_index": layer_index,
                "observed": bool(
                    kernel_local_record["same_input_kernel_post_norm_production"]
                ),
                "record": kernel_local_record,
            },
            "same_eager_attention_input_mlp_region_production": {
                "observed": mlp_production,
                "compiled_mlp_exit_record": mlp_exit_record,
                "records_repeat_exact": (
                    len(eager_attention_exit) == len(eager_block_exit) == 2
                    and all(len(row) == 2 for row in eager_attention_exit + eager_block_exit)
                    and eager_attention_exit[0] == eager_attention_exit[1]
                ),
                "eager_attention_and_eager_block_arms_pre_repair_exit_exact": (
                    pre_repair_signature(eager_attention_exit[0][1])
                    == pre_repair_signature(eager_block_exit[0][1])
                    and pre_repair_signature(eager_attention_exit[1][1])
                    == pre_repair_signature(eager_block_exit[1][1])
                ),
            },
            "fixed_original_suffix_attention_region_mediation": {
                "observed_continuous": arms["compiled_attention"]["hashes"][0]
                != arms["eager_attention"]["hashes"][0],
                "compiled_to_eager_attention": metrics(
                    torch, compiled_attention_value, eager_attention_value
                ),
                **semantic_contrast(
                    compiled_attention_decisions, eager_attention_decisions
                ),
                "compiled_attention_fork_coordinate_endpoints": endpoint_records(
                    torch,
                    compiled_attention_value,
                    cpu_inputs,
                    fork_coordinates,
                    0.2,
                ),
                "eager_attention_fork_coordinate_endpoints": endpoint_records(
                    torch,
                    eager_attention_value,
                    cpu_inputs,
                    fork_coordinates,
                    0.2,
                ),
            },
            "fixed_original_suffix_mlp_region_mediation": {
                "observed_continuous": arms["eager_attention"]["hashes"][0]
                != arms["eager_block"]["hashes"][0],
                "compiled_to_eager_mlp": metrics(
                    torch, eager_attention_value, eager_block_value
                ),
                **semantic_contrast(eager_attention_decisions, eager_block_decisions),
                "compiled_mlp_fork_coordinate_endpoints": endpoint_records(
                    torch,
                    eager_attention_value,
                    cpu_inputs,
                    fork_coordinates,
                    0.2,
                ),
                "eager_mlp_fork_coordinate_endpoints": endpoint_records(
                    torch, eager_block_value, cpu_inputs, fork_coordinates, 0.2
                ),
            },
            "fixed_original_suffix_kernel_mediation": {
                "observed_continuous": arms["compiled_attention"]["hashes"][0]
                != arms["kernel_reference"]["hashes"][0],
                "compiled_to_kernel_reference": metrics(
                    torch, compiled_attention_value, kernel_reference_value
                ),
                **semantic_contrast(
                    compiled_attention_decisions, kernel_reference_decisions
                ),
                "compiled_kernel_fork_coordinate_endpoints": endpoint_records(
                    torch,
                    compiled_attention_value,
                    cpu_inputs,
                    fork_coordinates,
                    0.2,
                ),
                "kernel_reference_fork_coordinate_endpoints": endpoint_records(
                    torch,
                    kernel_reference_value,
                    cpu_inputs,
                    fork_coordinates,
                    0.2,
                ),
            },
            "kernel_op_variant_mediation": {
                name: {
                    "observed_continuous": arms["compiled_attention"]["hashes"][0]
                    != arms[f"kernel_variant:{name}"]["hashes"][0],
                    "compiled_to_variant": metrics(
                        torch, compiled_attention_value, kernel_variant_values[name]
                    ),
                    **semantic_contrast(
                        compiled_attention_decisions,
                        kernel_variant_decisions[name],
                    ),
                    "compiled_fork_coordinate_endpoints": endpoint_records(
                        torch,
                        compiled_attention_value,
                        cpu_inputs,
                        fork_coordinates,
                        0.2,
                    ),
                    "variant_fork_coordinate_endpoints": endpoint_records(
                        torch,
                        kernel_variant_values[name],
                        cpu_inputs,
                        fork_coordinates,
                        0.2,
                    ),
                }
                for name in kernel_variant_names
            },
            "live_kernel_variant_mediation": {
                name: {
                    "observed_continuous": arms["compiled_attention"]["hashes"][0]
                    != arms[f"live_kernel_variant:{name}"]["hashes"][0],
                    "compiled_to_live_variant": metrics(
                        torch,
                        compiled_attention_value,
                        live_kernel_variant_values[name],
                    ),
                    **semantic_contrast(
                        compiled_attention_decisions,
                        live_kernel_variant_decisions[name],
                    ),
                    "live_variant_metadata": (
                        arms[f"live_kernel_variant:{name}"]["intermediate_records"][0][0]
                        .get("live_kernel_variant_metadata")
                    ),
                }
                for name in live_kernel_variant_names
            },
        }
    compiles_after = audit["backend_compiles"]
    _, restored_hashes = repeated_candidate()

    expected_family = [
        (row["graph_code_sha256"], int(row["graph_node_count"]))
        for row in contract["candidate_ordered_unique_graph_family"]
    ]
    actual_family = []
    for row in zip(audit["graph_hashes"], audit["graph_nodes"], strict=True):
        if row not in actual_family:
            actual_family.append(row)

    def exact_calls(row: dict[str, Any], interventions: int) -> bool:
        return all(
            record == [{"calls": expected_calls, "interventions": interventions}]
            for record in row["call_records"]
        )

    all_records = [
        record
        for treatment in treatments.values()
        for repeat in treatment["intervention"]["boundary_records"]
        for record in repeat
    ]
    contextual_records = [
        record
        for row in contextual_slices.values()
        for arm_name in ("compiled_block", "eager_block")
        for repeat in row[arm_name]["boundary_records"]
        for record in repeat
    ]
    subblock_boundary_records = [
        record
        for row in subblock_slices.values()
        for arm in row["arms"].values()
        for repeat in arm["boundary_records"]
        for record in repeat
    ]
    subblock_intermediate_records = [
        record
        for row in subblock_slices.values()
        for arm in row["arms"].values()
        for repeat in arm["intermediate_records"]
        for record in repeat
    ]
    gates = {
        "graph_family_exact": actual_family == expected_family,
        "eager_anchor_exact": eager_hashes == [contract["reference_scorer_sha256"]] * 2,
        "candidate_anchor_exact": candidate_hashes
        == [contract["candidate_scorer_sha256"]] * 2,
        "eager_boundary_capture_repeat_exact": all(
            row["residual"][0] == row["residual"][1]
            and row["next_input_norm"][0] == row["next_input_norm"][1]
            for row in eager_boundary_hashes
        ),
        "live_generated_kernel_resolved": bool(modules),
        "all_noop_exact": all(
            row["noop"]["hashes"] == candidate_hashes for row in treatments.values()
        ),
        "all_noop_call_counts_exact": all(
            exact_calls(row["noop"], 0) for row in treatments.values()
        ),
        "all_intervention_call_counts_exact": all(
            exact_calls(row["intervention"], 1) for row in treatments.values()
        ),
        "all_boundary_records_repeat_exact": all(
            row["intervention"]["boundary_record_repeat_exact"]
            for row in treatments.values()
        ),
        "all_weight_storage_mappings_exact": all(
            record["weight_storage_identity"] and record["weight_exact"]
            for record in all_records
        ),
        "all_boundary_value_transport_contracts_exact": all(
            record["residual_value_transport_contract"]
            and record["norm_value_transport_contract"]
            and record["destination_layout_preserved"]
            for record in all_records
        ),
        "all_intervention_repeats_exact": all(
            row["intervention"]["repeat_exact"] for row in treatments.values()
        ),
        "all_contextual_noops_exact": all(
            row["noop"]["hashes"] == candidate_hashes
            for row in contextual_slices.values()
        ),
        "all_contextual_arms_repeat_exact": all(
            row[arm_name]["repeat_exact"]
            for row in contextual_slices.values()
            for arm_name in ("compiled_block", "eager_block")
        ),
        "all_contextual_entry_exit_counts_exact": all(
            all(
                record == [
                    {
                        "calls": expected_calls,
                        "entry_injections": entry_count,
                        "exit_injections": exit_count,
                    }
                ]
                for record in row[arm_name]["call_records"]
            )
            for row in contextual_slices.values()
            for arm_name, entry_count, exit_count in (
                ("noop", 0, 0),
                ("compiled_block", 1, 0),
                ("eager_block", 1, 1),
            )
        ),
        "all_contextual_boundary_contracts_exact": all(
            record["weight_storage_identity"]
            and record["weight_exact"]
            and record["residual_value_transport_contract"]
            and record["norm_value_transport_contract"]
            and record["destination_layout_preserved"]
            for record in contextual_records
        ),
        "all_contextual_same_input_exit_records_exact": all(
            row["same_eager_input_composite_layer_production"][
                "compiled_exit_records_repeat_exact"
            ]
            and row["same_eager_input_composite_layer_production"][
                "compiled_and_eager_arms_pre_repair_exit_exact"
            ]
            for row in contextual_slices.values()
        ),
        "all_contextual_cross_artifact_transports_exact": all(
            row["compiled_block"][
                "matches_independent_entry_boundary_treatment"
            ]
            and row["eager_block"][
                "matches_independent_exit_boundary_treatment"
            ]
            for row in contextual_slices.values()
        ),
        "eager_subblock_capture_repeat_exact": all(
            hashes[0] == hashes[1]
            for row in eager_subblock_hashes.values()
            for hashes in row.values()
        ),
        "all_subblock_noops_exact": all(
            row["arms"]["noop"]["hashes"] == candidate_hashes
            for row in subblock_slices.values()
        ),
        "all_subblock_arms_repeat_exact": all(
            arm["repeat_exact"]
            for row in subblock_slices.values()
            for arm in row["arms"].values()
        ),
        "all_subblock_call_counts_exact": all(
            all(
                record["boundary"]
                == [
                    {
                        "calls": expected_calls,
                        "entry_injections": entry_count,
                        "exit_injections": exit_count,
                    }
                ]
                and record["intermediate"]
                == [{"calls": len(layers), "injections": intermediate_count}]
                for record in row["arms"][mode]["call_records"]
            )
            for row in subblock_slices.values()
            for mode, entry_count, intermediate_count, exit_count
            in expected_subblock_modes
        ),
        "all_subblock_weight_mappings_exact": all(
            record["weight_storage_identity"] and record["weight_exact"]
            for record in subblock_boundary_records + subblock_intermediate_records
        ),
        "all_subblock_transport_layout_contracts_exact": all(
            record["destination_layout_preserved"]
            for record in subblock_boundary_records + subblock_intermediate_records
        )
        and all(
            record["attention_transport_contract"]
            and record["post_norm_transport_contract"]
            for record in subblock_intermediate_records
        ),
        "all_subblock_production_records_exact": all(
            row["same_eager_input_attention_region_production"][
                "records_repeat_exact"
            ]
            and row["same_eager_input_attention_region_production"][
                "compiled_and_eager_attention_arms_pre_repair_exact"
            ]
            and row["same_eager_attention_input_mlp_region_production"][
                "records_repeat_exact"
            ]
            and row["same_eager_attention_input_mlp_region_production"][
                "eager_attention_and_eager_block_arms_pre_repair_exit_exact"
            ]
            for row in subblock_slices.values()
        ),
        "all_subblock_kernel_same_input_records_exact": all(
            row["same_input_intermediate_kernel_production"]["record"]
            == next(
                record
                for record in row["arms"]["compiled_attention"][
                    "intermediate_records"
                ][1]
            )
            for row in subblock_slices.values()
        ),
        "all_subblock_kernel_op_variant_records_exact": all(
            all(
                name in record.get("kernel_op_variant_metrics", {})
                and name in record.get("kernel_op_variant_sha256", {})
                for name in kernel_variant_names
            )
            for row in subblock_slices.values()
            for repeat in row["arms"]["compiled_attention"]["intermediate_records"]
            for record in repeat
        ),
        "all_subblock_live_kernel_variant_records_exact": all(
            all(
                record.get("live_kernel_variant") == name
                and record.get("live_kernel_variant_metadata", {}).get(
                    "source_output_code_sha256"
                )
                for repeat in row["arms"][f"live_kernel_variant:{name}"][
                    "intermediate_records"
                ]
                for record in repeat
            )
            for row in subblock_slices.values()
            for name in live_kernel_variant_names
        ),
        "all_subblock_cross_artifact_transports_exact": all(
            row["arms"]["compiled_attention"][
                "matches_independent_entry_boundary_treatment"
            ]
            and row["arms"]["eager_block"][
                "matches_independent_exit_boundary_treatment"
            ]
            for row in subblock_slices.values()
        ),
        "no_backend_recompile": compiles_before == compiles_after,
        "kernel_restoration_exact": restored_hashes == candidate_hashes,
    }
    valid = all(gates.values())
    payload = {
        "schema_version": "forkcert.qwen3-step236-eager-layer-boundary-mediation.v0.1",
        "status": "VALID" if valid else "INVALID",
        "valid": valid,
        "case_id": manifest["case_id"],
        "state_id": manifest["state_id"],
        "gates": gates,
        "anchors": {
            "eager": eager_hashes,
            "candidate": candidate_hashes,
            "restored": restored_hashes,
        },
        "baseline_endpoint": {
            "fork_coordinates": fork_coordinates,
            "eager": endpoint_records(torch, eager_value, cpu_inputs, fork_coordinates, 0.2),
            "candidate": endpoint_records(
                torch, candidate_value, cpu_inputs, fork_coordinates, 0.2
            ),
        },
        "eager_boundary_hashes": eager_boundary_hashes,
        "kernel_family": {
            "generated_kernel": kernel_name,
            "provenance_rows": provenance_rows,
            "mapping_rule": manifest["mapping_rule"],
            "intermediate_generated_kernel": intermediate_kernel_name,
            "intermediate_provenance_rows": intermediate_provenance_rows,
            "intermediate_live_module_names": intermediate_live_module_names,
            "intermediate_runtime_order": [
                {
                    "call_index": index,
                    "kernel_id": row["kernel_id"],
                    "launch_index": row.get("launch_index"),
                }
                for index, row in enumerate(intermediate_provenance_rows)
            ],
        },
        "treatments": treatments,
        "contextual_layer_slices": contextual_slices,
        "subblock_layer_slices": subblock_slices,
        "eager_subblock_hashes": eager_subblock_hashes,
        "compile_audit": audit,
        "resolution_records": resolution_records,
        "claim_scope": (
            "aligned eager attention/MLP boundary substitution into the unchanged "
            "original compiled suffix"
            if subblock_slices
            else "aligned eager layer-boundary substitution into the unchanged "
            "original compiled suffix at selected boundaries"
        ),
        "limitations": [
            "this is endpoint mediation, not same-input discrepancy production",
            "the substituted boundary removes all upstream implementation differences at that boundary",
            "the eager input_layernorm module output is explicitly transported to the fp16 dtype consumed by the original compiled suffix; copy_ preserves the original destination stride/layout",
            "a positive result localizes an influential prefix but does not identify a unique source op",
            "a negative result can depend on suffix context and interaction",
            "forward evidence does not license backward/update attribution",
            "eager is a comparison baseline, not an independent truth source",
            "a contextual composite-layer effect does not identify a constituent attention, MLP, normalization, or generated-kernel root cause",
            "attention and MLP contextual regions are composite dataflow regions; each still contains multiple ATen ops and generated kernels",
            "the kernel-only repair applies to the declared post-norm output of one generated kernel call; it does not prove that the kernel is the first or unique discrepancy source",
        ],
        "artifact_inputs": {
            "manifest": str(manifest_path),
            "inventory": str(inventory_path),
            "observability_gate": str(gate_path),
            # This input is required by the historical singleton-boundary study,
            # but deliberately not by the contextual subblock study.  Preserve
            # the distinction in the artifact rather than fabricating a link.
            "singleton_production_evidence": manifest.get(
                "singleton_production_evidence", "not_applicable_for_subblock_slice"
            ),
        },
    }
    out.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "valid": valid,
                "gates": gates,
                "treatments": {
                    key: row["fixed_original_suffix_mediation"]
                    for key, row in treatments.items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    raise SystemExit(0 if valid else 2)


if __name__ == "__main__":
    main()
