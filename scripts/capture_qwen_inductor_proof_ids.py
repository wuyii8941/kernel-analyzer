#!/usr/bin/env python3
"""Propagate stable AOT proof IDs into a real Qwen Inductor compilation."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch._inductor import config as inductor_config
from torch._inductor.compile_fx import compile_fx, compile_fx_inner
from transformers import AutoModelForCausalLM, MambaForCausalLM
from transformers.models.mamba import modeling_mamba

from qwen_candidate_step import LossStep
from scripts.aot_capture import _input_edges, _jsonable, _tensor_meta
from scripts.inductor_buffer_origins import InductorBufferOriginRecorder


ROOT = Path(__file__).resolve().parents[1]


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def gradient_digest(model: torch.nn.Module) -> str:
    combined = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        combined.update(name.encode())
        if parameter.grad is None:
            combined.update(b"NONE")
        else:
            value = parameter.grad.detach().contiguous().cpu()
            combined.update(value.view(torch.uint8).numpy().tobytes())
    return combined.hexdigest()


def load_model(architecture: str, model_path: Path, device: str) -> torch.nn.Module:
    if architecture == "mamba":
        modeling_mamba.selective_scan_fn = None
        modeling_mamba.mamba_inner_fn = None
        modeling_mamba.selective_state_update = None
        model = MambaForCausalLM.from_pretrained(
            model_path, dtype=torch.bfloat16, local_files_only=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            attn_implementation="eager",
            local_files_only=True,
        )
    model = model.to(device).train()
    model.config.use_cache = False
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--architecture",
        choices=("qwen", "mamba", "moe", "phi", "deepseek8"),
        default="qwen",
    )
    parser.add_argument("--model", type=Path, default=Path("/data1/tzh/models/Qwen/Qwen3-1.7B"))
    parser.add_argument("--input-bank", type=Path, required=True)
    parser.add_argument("--state", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--preserve-aot-aten",
        action="store_true",
        help="Disable AOT decomposition so proof IDs start at canonical ATen nodes.",
    )
    parser.add_argument("--allow-graph-breaks", action="store_true")
    parser.add_argument(
        "--no-proof-node-renaming", action="store_true",
        help="Capture an unmodified standard Inductor schedule; proof IDs are not claimed to propagate.",
    )
    args = parser.parse_args()

    if args.trace_dir.exists() and any(args.trace_dir.iterdir()):
        raise RuntimeError("trace directory must be absent or empty")
    args.trace_dir.mkdir(parents=True, exist_ok=True)
    bank = json.loads(args.input_bank.read_text())
    records = bank.get("states", bank.get("records"))
    record = records[args.state]
    token_ids = record.get("token_ids", record.get("input_ids"))

    torch.manual_seed(24000 + args.state)
    torch.cuda.manual_seed_all(24000 + args.state)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cudnn.allow_tf32 = False
    torch._dynamo.config.suppress_errors = False
    inductor_config.trace.enabled = True
    inductor_config.trace.debug_dir = str(args.trace_dir.resolve())
    inductor_config.trace.save_real_tensors = False
    inductor_config.trace.fx_graph = True
    inductor_config.trace.fx_graph_transformed = True
    inductor_config.trace.ir_pre_fusion = True
    inductor_config.trace.ir_post_fusion = True
    inductor_config.trace.output_code = True
    inductor_config.trace.provenance_tracking_level = 1

    model = load_model(args.architecture, args.model, args.device)
    inputs = torch.tensor([token_ids], dtype=torch.long, device=args.device)

    proof_rows: list[dict[str, Any]] = []
    standard_aot_graphs: list[dict[str, Any]] = []

    def tagged_inner_compile(
        graph_module: torch.fx.GraphModule,
        example_inputs: list[Any],
        compile_region_name: str | None = None,
        **kwargs: Any,
    ):
        phase = "BACKWARD" if kwargs.get("is_backward", False) else "FORWARD"
        graph_index = sum(row["phase"] == phase for row in proof_rows)
        graph_code_before_tags = graph_module.code
        standard_nodes = []
        for ordinal, node in enumerate(graph_module.graph.nodes):
            standard_nodes.append({
                "phase": phase,
                "ordinal": ordinal,
                "name": node.name,
                "op": node.op,
                "target": str(node.target),
                "arguments": _jsonable({"args": node.args, "kwargs": node.kwargs}),
                "input_nodes": sorted(value.name for value in node.all_input_nodes),
                "input_edges": list(_input_edges(node)),
                "users": sorted(user.name for user in node.users),
                "tensor_meta": _tensor_meta(node.meta.get("tensor_meta")),
                "nn_module_stack": _jsonable(node.meta.get("nn_module_stack")),
                "source_fn_stack": _jsonable(node.meta.get("source_fn_stack")),
                "seq_nr": int(node.meta["seq_nr"]) if node.meta.get("seq_nr") is not None else None,
                "fwd_source_fn_stack": _jsonable(node.meta.get("fwd_source_fn_stack")),
                "fwd_nn_module_stack": _jsonable(node.meta.get("fwd_nn_module_stack")),
                "original_aten": _jsonable(node.meta.get("original_aten")),
                "from_node": _jsonable(node.meta.get("from_node")),
                "is_gradient_acc": bool(node.meta.get("is_gradient_acc", False)),
                "partitioner_tag": _jsonable(node.meta.get("partitioner_tag")),
                "stack_trace": node.meta.get("stack_trace"),
            })
        standard_aot_graphs.append({
            "phase": phase,
            "graph_index": graph_index,
            "code_sha256": hashlib.sha256(graph_code_before_tags.encode()).hexdigest(),
            "input_count": len(example_inputs),
            "node_count": len(standard_nodes),
            "call_function_count": sum(row["op"] == "call_function" for row in standard_nodes),
            "nodes": standard_nodes,
        })
        call_index = 0
        graph_rows = []
        for node in graph_module.graph.nodes:
            if node.op != "call_function":
                continue
            original_name = node.name
            proof_id = f"{phase.lower()}:graph{graph_index}:{original_name}"
            safe = re.sub(r"[^a-zA-Z0-9_]", "_", original_name)
            tagged_name = (
                original_name if args.no_proof_node_renaming
                else (
                    f"ka_{phase[0].lower()}_g{graph_index}_"
                    f"{call_index:05d}_{safe}"
                )
            )
            if not args.no_proof_node_renaming:
                node.name = tagged_name
                node.meta["kernel_analyzer_proof_id"] = proof_id
            graph_rows.append({
                "proof_id": proof_id,
                "tagged_fx_name": tagged_name,
                "target": str(node.target),
                "input_tagged_fx_names": sorted(
                    value.name for value in node.all_input_nodes
                ),
                "seq_nr": node.meta.get("seq_nr"),
                "original_aten": str(node.meta.get("original_aten")),
                "from_node": str(node.meta.get("from_node")),
                "source_fn_stack": str(node.meta.get("source_fn_stack")),
                "fwd_source_fn_stack": str(node.meta.get("fwd_source_fn_stack")),
            })
            call_index += 1
        graph_module.graph.lint()
        graph_module.recompile()
        proof_rows.append({
            "phase": phase,
            "graph_index": graph_index,
            "call_function_nodes": len(graph_rows),
            "rows": graph_rows,
            "tagged_graph_code_sha256": hashlib.sha256(graph_module.code.encode()).hexdigest(),
        })
        return compile_fx_inner(
            graph_module,
            example_inputs,
            compile_region_name=compile_region_name,
            **kwargs,
        )

    def backend(graph_module: torch.fx.GraphModule, example_inputs: list[Any]):
        return compile_fx(
            graph_module,
            example_inputs,
            inner_compile=tagged_inner_compile,
            decompositions={} if args.preserve_aot_aten else None,
        )
    candidate = torch.compile(
        LossStep(model),
        backend="inductor" if args.no_proof_node_renaming else backend,
        fullgraph=not args.allow_graph_breaks,
        dynamic=False,
    )
    runs = []
    with InductorBufferOriginRecorder() as origin_recorder:
        for repeat in range(2):
            execution_seed = 24000 + args.state
            torch.manual_seed(execution_seed)
            torch.cuda.manual_seed_all(execution_seed)
            model.zero_grad(set_to_none=True)
            loss = candidate(inputs)
            loss.backward()
            torch.cuda.synchronize(torch.device(args.device))
            runs.append({
                "repeat": repeat,
                "loss": float(loss.detach()),
                "gradient_digest": gradient_digest(model),
            })
    buffer_origin_certificate = origin_recorder.certificate()

    trace_files = []
    tagged_occurrences = Counter()
    all_tags = {
        row["tagged_fx_name"]
        for graph in proof_rows
        for row in graph["rows"]
    }
    proof_row_count = sum(len(graph["rows"]) for graph in proof_rows)
    if not args.no_proof_node_renaming and len(all_tags) != proof_row_count:
        raise RuntimeError(
            "proof-tagged FX names are not globally unique across AOT segments"
        )
    for path in sorted(args.trace_dir.rglob("*")):
        if not path.is_file():
            continue
        text = path.read_text(errors="ignore")
        # Extract tags once per file.  Testing every tag against every large
        # trace file is pathologically expensive on unfolded recurrent graphs.
        present = [] if args.no_proof_node_renaming else sorted(
            set(re.findall(
                r"ka_[fb]_(?:g\d+_)?\d{4,}_[A-Za-z0-9_]+", text
            )) & all_tags
        )
        for tag in present:
            tagged_occurrences[tag] += 1
        trace_files.append({
            "path": str(path.relative_to(args.trace_dir)),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "proof_tags_present": len(present),
        })
    proof_tag_count = len(all_tags)
    observed_tag_count = len(tagged_occurrences)
    standard_capture = {
        "schema_version": "kernel-analyzer-standard-aot-capture-v1",
        "graphs": standard_aot_graphs,
    }
    standard_capture["capture_sha256"] = digest(standard_capture)
    payload = {
        "schema": "kernel-analyzer-architecture-inductor-proof-id-capture-v1",
        "status": (
            "COMPLETE_STANDARD_UNMODIFIED_SCHEDULE_CAPTURE"
            if args.no_proof_node_renaming
            else "COMPLETE_PROOF_ID_PROPAGATION_CAPTURE"
        ),
        "architecture": args.architecture,
        "model": str(args.model.resolve()),
        "input": {"state": args.state, "sequence_length": len(token_ids), "token_ids_sha256": digest(token_ids)},
        "preserve_aot_aten": args.preserve_aot_aten,
        "allow_graph_breaks": args.allow_graph_breaks,
        "standard_aot_capture": standard_capture,
        "proof_node_renaming": not args.no_proof_node_renaming,
        "runs": runs,
        "repeat_stable": runs[0] == {**runs[1], "repeat": 0},
        "proof_graphs": proof_rows,
        "inductor_buffer_origins": buffer_origin_certificate,
        "proof_tag_summary": {
            "aot_call_function_nodes": proof_tag_count,
            "tags_observed_in_inductor_trace": observed_tag_count,
            "tags_not_observed": (
                None if args.no_proof_node_renaming
                else proof_tag_count - observed_tag_count
            ),
        },
        "trace_files": trace_files,
        "trace_dir": str(args.trace_dir.resolve()),
        "claim_boundary": (
            "This captures an unmodified standard candidate schedule and deterministic execution; "
            "it does not claim proof-ID propagation."
            if args.no_proof_node_renaming else
            "This captures compiler-carried proof IDs and deterministic candidate execution. "
            "Only IDs actually present in lowered/generated artifacts may be promoted to exact bindings."
        ),
    }
    payload["result_sha256"] = digest(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, sort_keys=True) + "\n").encode()
    if args.output.suffix == ".gz":
        with gzip.open(args.output, "wb", compresslevel=6) as handle:
            handle.write(encoded)
    else:
        args.output.write_bytes(encoded)
    print(json.dumps({
        "output": str(args.output.resolve().relative_to(ROOT)),
        "status": payload["status"],
        "repeat_stable": payload["repeat_stable"],
        "proof_tag_summary": payload["proof_tag_summary"],
        "trace_files": len(trace_files),
        "result_sha256": payload["result_sha256"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
