#!/usr/bin/env python
"""Collect a hash-gated operator/kernel inventory of the original Qwen3 candidate."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from qwen3_lm_head_operator_pilot_v0_1 import CANDIDATE_ANCHOR, move_tree, tensor_sha256


EXPECTED_FAMILY = [
    ("31ec1dd1b3689460c96b5b7882e5cbcb3a33dea614ef07111b47f48373caef04", 455),
    ("ee4053cb35f6351f6303e6b9922ccf0fa2189246fc5bcbee31d4793241164e5b", 457),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ordered_unique(values: list[tuple[str, int]]) -> list[tuple[str, int]]:
    result: list[tuple[str, int]] = []
    for value in values:
        if value not in result:
            result.append(value)
    return result


def metadata_string(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): metadata_string(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [metadata_string(item) for item in value]
    return str(value)


def main() -> None:
    args = parse_args()
    snapshot_dir = Path(args.snapshot_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=False)
    trace_dir = out_dir / "inductor_trace"
    graph_dir = out_dir / "dynamo_graphs"
    graph_dir.mkdir()
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
    torch._inductor.config.trace.enabled = True
    torch._inductor.config.trace.debug_dir = str(trace_dir)
    torch._inductor.config.trace.save_real_tensors = False
    torch._inductor.config.trace.fx_graph = True
    torch._inductor.config.trace.fx_graph_transformed = True
    torch._inductor.config.trace.ir_pre_fusion = True
    torch._inductor.config.trace.ir_post_fusion = True
    torch._inductor.config.trace.output_code = True

    metadata = json.loads((snapshot_dir / "forkcert_transition_snapshot.json").read_text())
    model = AutoModelForCausalLM.from_pretrained(
        snapshot_dir,
        dtype=torch.float32,
        attn_implementation="sdpa",
        local_files_only=True,
    ).to("cuda")
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.train()
    accelerator = Accelerator(mixed_precision="fp16")
    wrapped = accelerator.prepare_model(model)
    if not (hasattr(wrapped, "_original_forward") and hasattr(wrapped.forward, "__wrapped__")):
        raise RuntimeError("Accelerate FP16 forward wrapper was not reproduced")

    audit: dict[str, Any] = {
        "backend_compiles": 0,
        "runtime_invocations": 0,
        "graph_hashes": [],
        "graph_nodes": [],
        "graphs": [],
    }
    from torch._dynamo.backends.registry import lookup_backend

    inductor = lookup_backend("inductor")

    def backend(graph_module: Any, example_inputs: list[Any]):
        index = int(audit["backend_compiles"])
        code = graph_module.code
        code_hash = hashlib.sha256(code.encode()).hexdigest()
        nodes = []
        for node in graph_module.graph.nodes:
            nodes.append(
                {
                    "name": node.name,
                    "op": node.op,
                    "target": str(node.target),
                    "users": sorted(user.name for user in node.users),
                    "nn_module_stack": metadata_string(node.meta.get("nn_module_stack")),
                    "source_fn_stack": metadata_string(node.meta.get("source_fn_stack")),
                    "stack_trace": node.meta.get("stack_trace"),
                    "tensor_meta": metadata_string(node.meta.get("tensor_meta")),
                }
            )
        (graph_dir / f"graph_{index:02d}_{code_hash}.py").write_text(code)
        (graph_dir / f"graph_{index:02d}_{code_hash}_nodes.json").write_text(
            json.dumps(nodes, indent=2, sort_keys=True) + "\n"
        )
        audit["backend_compiles"] += 1
        audit["graph_hashes"].append(code_hash)
        audit["graph_nodes"].append(len(nodes))
        audit["graphs"].append({"index": index, "sha256": code_hash, "node_count": len(nodes)})
        compiled = inductor(graph_module, example_inputs)

        def counted(*values: Any):
            audit["runtime_invocations"] += 1
            return compiled(*values)

        return counted

    candidate = torch.compile(wrapped, backend=backend)

    def score(value: dict[str, Any]) -> Any:
        completion_ids = value["completion_ids"]
        input_ids = torch.cat([value["prompt_ids"], completion_ids], dim=1)
        attention_mask = torch.cat([value["prompt_mask"], value["completion_mask"]], dim=1)
        with sdpa_kernel(SDPBackend.MATH):
            outputs = candidate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                logits_to_keep=completion_ids.size(1) + 1,
                use_cache=False,
            )
            logits = outputs.logits[:, :-1, :]
            logits = logits[:, -completion_ids.size(1) :, :]
            return selective_log_softmax(logits, completion_ids)

    for record in metadata["compiler_history"]:
        path = Path(record["path"])
        if not path.is_file():
            path = snapshot_dir / "compiler_history" / path.name
        history = move_tree(torch.load(path, map_location="cpu", weights_only=False), "cuda")
        value = score(history)
        del value, history
        gc.collect()
    target = Path(metadata["target_minibatch_path"])
    if not target.is_file():
        target = snapshot_dir / "compiler_history" / target.name
    inputs = move_tree(torch.load(target, map_location="cpu", weights_only=False), "cuda")
    scorer_hashes = []
    for _ in range(2):
        value = score(inputs)
        scorer_hashes.append(tensor_sha256(value.detach().float().cpu()))
        del value
        gc.collect()

    actual_family = ordered_unique(list(zip(audit["graph_hashes"], audit["graph_nodes"], strict=True)))
    trace_files = []
    if trace_dir.exists():
        for path in sorted(item for item in trace_dir.rglob("*") if item.is_file()):
            trace_files.append(
                {
                    "path": str(path.relative_to(out_dir)),
                    "size": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
            )
    real_tensor_like = [row for row in trace_files if "real_tensor" in row["path"].lower()]
    gates = {
        "graph_family_exact": actual_family == EXPECTED_FAMILY,
        "target_scorer_anchor_exact": scorer_hashes == [CANDIDATE_ANCHOR, CANDIDATE_ANCHOR],
        "target_repeats_exact": len(set(scorer_hashes)) == 1,
        "real_tensor_trace_absent": not real_tensor_like,
    }
    payload = {
        "schema_version": "forkcert.qwen3-original-compiled-inventory.v0.1",
        "status": "VALID_ORIGINAL_CANDIDATE_INVENTORY" if all(gates.values()) else "INVALID_OBSERVER_OR_REALIZATION",
        "state": "heldout-transport-B-step29",
        "expected_graph_family": EXPECTED_FAMILY,
        "actual_graph_family": actual_family,
        "expected_scorer_sha256": CANDIDATE_ANCHOR,
        "observed_scorer_sha256": scorer_hashes,
        "gates": gates,
        "compile_audit": audit,
        "trace": {
            "save_real_tensors": False,
            "file_count": len(trace_files),
            "total_bytes": sum(row["size"] for row in trace_files),
            "files": trace_files,
        },
        "claim": "descriptive op/kernel inventory only; no causal attribution",
    }
    (out_dir / "result.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in payload.items() if key != "trace"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
