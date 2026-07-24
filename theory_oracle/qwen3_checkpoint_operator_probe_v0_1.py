#!/usr/bin/env python
"""Run a small, auditable eager/compiled probe on a real Qwen3 checkpoint.

This is deliberately a scale-up gate, not an operator ranking tool.  It uses a
fixed input and one immutable model state, compares the same tensor endpoints,
and records compiler graph metadata without naming a preferred operation.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def tensor_hash(value: Any) -> str:
    import torch

    return hashlib.sha256(value.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def tensor_record(value: Any) -> dict[str, Any]:
    value = value.detach()
    return {
        "sha256": tensor_hash(value),
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "stride": list(value.stride()),
        "device": str(value.device),
    }


def max_abs(a: Any, b: Any) -> float:
    import torch

    return float((a.detach().float() - b.detach().float()).abs().max().item())


def endpoint_record(ref: Any, cand: Any) -> dict[str, Any]:
    import torch

    delta = cand.detach().float() - ref.detach().float()
    return {
        "reference": tensor_record(ref),
        "candidate": tensor_record(cand),
        "exact": bool(torch.equal(ref, cand)),
        "finite": bool(torch.isfinite(delta).all().item()),
        "max_abs": float(delta.abs().max().item()),
        "mean_abs": float(delta.abs().mean().item()),
        "l2": float(torch.linalg.vector_norm(delta).item()),
        "disagreement_count": int((delta != 0).sum().item()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--layers", default="0,14,27")
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--backend", choices=("inductor", "aot_eager"), default="inductor")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    snapshot = Path(args.snapshot_dir).resolve()
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError(out)
    layers = [int(item) for item in args.layers.split(",") if item.strip()]

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("PYTHONHASHSEED", "0")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import torch
    from transformers import AutoModelForCausalLM

    result: dict[str, Any] = {
        "schema_version": "forkcert.qwen3-checkpoint-operator-probe.v0.1",
        "snapshot_dir": str(snapshot),
        "requested_layers": layers,
        "requested_seq_len": args.seq_len,
        "requested_dtype": args.dtype,
        "requested_backend": args.backend,
        "status": "UNSET",
    }
    if args.device == "cuda" and (not torch.cuda.is_available() or torch.cuda.device_count() != 1):
        result["status"] = "INVALID_NO_SINGLE_CUDA_DEVICE"
        out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        raise RuntimeError(result["status"])

    torch.use_deterministic_algorithms(True, warn_only=True)
    if args.device == "cuda":
        torch.backends.cudnn.benchmark = False
    torch._dynamo.config.suppress_errors = False
    torch._dynamo.config.recompile_limit = 64
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        snapshot,
        dtype=dtype,
        attn_implementation="sdpa",
        local_files_only=True,
    ).to(device)
    model.config.use_cache = False
    model.eval()

    vocab_size = int(model.config.vocab_size)
    # No tokenizer or sampling is involved: this makes the state an explicit
    # tensor artifact and avoids introducing algorithmic RNG into the gate.
    input_ids = (torch.arange(args.seq_len, device=device, dtype=torch.long)[None, :] * 17 + 11) % vocab_size

    class TensorForward(torch.nn.Module):
        def __init__(self, wrapped: Any, selected: list[int]):
            super().__init__()
            self.wrapped = wrapped
            self.selected = selected

        def forward(self, tokens: Any) -> tuple[Any, ...]:
            output = self.wrapped(
                input_ids=tokens,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden = output.hidden_states
            # hidden_states[0] is embedding output; layer i is at i+1.
            return (output.logits, *(hidden[index + 1] for index in self.selected))

    wrapper = TensorForward(model, layers).to(device)
    with torch.inference_mode():
        eager_values = tuple(value.detach() for value in wrapper(input_ids))
    if args.device == "cuda":
        torch.cuda.synchronize()

    graph_dir = out.parent / (out.stem + "_graphs")
    graph_dir.mkdir(parents=True, exist_ok=False)
    compile_audit: dict[str, Any] = {"graphs": [], "runtime_invocations": 0}
    from torch._dynamo.backends.registry import lookup_backend

    selected_backend = lookup_backend(args.backend)

    def backend(graph_module: Any, example_inputs: list[Any]) -> Any:
        index = len(compile_audit["graphs"])
        code = graph_module.code
        code_hash = hashlib.sha256(code.encode()).hexdigest()
        (graph_dir / f"graph_{index:03d}_{code_hash}.py").write_text(code)
        nodes = []
        for node in graph_module.graph.nodes:
            nodes.append(
                {
                    "name": node.name,
                    "op": node.op,
                    "target": str(node.target),
                    "users": sorted(user.name for user in node.users),
                    "nn_module_stack": str(node.meta.get("nn_module_stack")),
                    "source_fn_stack": str(node.meta.get("source_fn_stack")),
                    "tensor_meta": str(node.meta.get("tensor_meta")),
                }
            )
        (graph_dir / f"graph_{index:03d}_{code_hash}_nodes.json").write_text(
            json.dumps(nodes, indent=2, sort_keys=True) + "\n"
        )
        compile_audit["graphs"].append(
            {"index": index, "sha256": code_hash, "node_count": len(nodes)}
        )
        compiled = selected_backend(graph_module, example_inputs)

        def counted(*values: Any) -> Any:
            compile_audit["runtime_invocations"] += 1
            return compiled(*values)

        return counted

    try:
        compiled = torch.compile(wrapper, backend=backend, dynamic=False)
        with torch.inference_mode():
            compiled_values_1 = tuple(value.detach() for value in compiled(input_ids))
            if args.device == "cuda":
                torch.cuda.synchronize()
            compiled_values_2 = tuple(value.detach() for value in compiled(input_ids))
            if args.device == "cuda":
                torch.cuda.synchronize()
    except Exception as exc:  # fail closed and retain the scale-up artifact
        result.update(
            {
                "status": "INDETERMINATE_COMPILE_FAILURE",
                "compile_error": repr(exc),
                "compile_audit": compile_audit,
                "input": tensor_record(input_ids),
            }
        )
        out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        raise

    names = ["logits", *(f"layer_{index}" for index in layers)]
    endpoints = {
        name: endpoint_record(ref, cand)
        for name, ref, cand in zip(names, eager_values, compiled_values_1)
    }
    repeatability = {
        name: endpoint_record(first, second)
        for name, first, second in zip(names, compiled_values_1, compiled_values_2)
    }
    changed = [name for name in names if not endpoints[name]["exact"]]
    result.update(
        {
            "status": "VALID_SCALE_UP_PROBE",
            "environment": {
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "device": args.device,
                "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
                "capability": list(torch.cuda.get_device_capability(0)) if args.device == "cuda" else None,
            },
            "model_config": {
                "hidden_size": int(model.config.hidden_size),
                "layers": int(model.config.num_hidden_layers),
                "heads": int(model.config.num_attention_heads),
                "kv_heads": int(model.config.num_key_value_heads),
                "vocab_size": vocab_size,
            },
            "input": tensor_record(input_ids),
            "endpoints": endpoints,
            "compiled_repeatability": repeatability,
            "changed_endpoints": changed,
            "compile_audit": compile_audit,
            "provenance_limit": "FX graph/source metadata only; no operator root-cause claim",
        }
    )
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    del compiled, wrapper, model, eager_values, compiled_values_1, compiled_values_2
    gc.collect()
    if args.device == "cuda":
        torch.cuda.empty_cache()
    print(json.dumps({"status": result["status"], "changed_endpoints": changed, "graphs": len(compile_audit["graphs"])}, sort_keys=True))


if __name__ == "__main__":
    main()
