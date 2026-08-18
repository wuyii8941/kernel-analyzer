#!/usr/bin/env python3
"""Capture dtype-specific generated Triton topology without tensor values.

This is a boundary artifact for configurations whose symbols do not map to the
BF16 semantic campaign.  It records only generated symbol names, invocation
order, pointer signatures and tensor metadata; it never reads tensor values or
assigns a correctness mechanism.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path("/data1/tzh").resolve()
REPO = Path(__file__).resolve().parents[1]
OLD_SRC = REPO / "archive" / "round1_code" / "src"
if str(OLD_SRC) not in sys.path:
    sys.path.insert(0, str(OLD_SRC))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def under_root(path: Path, label: str) -> Path:
    value = path.expanduser().resolve()
    if ROOT not in (value, *value.parents):
        raise ValueError(f"{label} must stay under {ROOT}: {value}")
    return value


def _runtime_signature(kernel: Any) -> list[tuple[str, Any]]:
    return [
        (str(name), value)
        for name, value in kernel.triton_meta["signature"].items()
        if value != "constexpr"
    ]


def discover_all_triton_symbols(modules: list[Any]) -> list[tuple[str, Any]]:
    found: dict[str, Any] = {}
    for module in modules:
        for symbol, value in vars(module).items():
            if (
                symbol.startswith("triton_")
                and callable(getattr(value, "run", None))
                and hasattr(value, "triton_meta")
            ):
                found[symbol] = value
    return sorted(found.items())


def _tensor_meta(value: Any) -> dict[str, Any]:
    import torch

    if not isinstance(value, torch.Tensor):
        return {"kind": type(value).__name__}
    return {
        "kind": "tensor",
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "stride": list(value.stride()),
        "numel": int(value.numel()),
        "device": str(value.device),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank-manifest", type=Path, default=Path("results/final/natural_bank.json"))
    parser.add_argument("--model", type=Path, default=Path("/data1/tzh/models/Qwen/Qwen3-1.7B"))
    parser.add_argument("--seq-len", type=int, choices=(64, 128, 256), required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), required=True)
    parser.add_argument("--tf32", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.bank_manifest = under_root(args.bank_manifest, "bank manifest")
    args.model = under_root(args.model, "model")
    args.output = under_root(args.output, "output")
    if args.tf32 and args.dtype != "fp32":
        raise ValueError("--tf32 requires --dtype fp32")

    os.environ.setdefault("HF_HOME", "/data1/tzh/cache/huggingface")
    os.environ.setdefault("HF_DATASETS_CACHE", "/data1/tzh/cache/huggingface/datasets")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/data1/tzh/cache/huggingface/transformers")
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/data1/tzh/cache/huggingface/hub")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    import torch
    from torch._dynamo.backends.registry import lookup_backend
    from torch._inductor.codecache import PyCodeCache
    from scripts.checkpoint_inductor import build_model, load_natural_validation

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device(args.device)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cuda.matmul.allow_tf32 = bool(args.tf32)
    torch.set_float32_matmul_precision("high" if args.tf32 else "highest")
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cudnn.allow_tf32 = bool(args.tf32)
    torch.backends.cudnn.benchmark = False

    bank = json.loads(args.bank_manifest.read_text())
    checkpoint_row = next((row for row in bank["checkpoints"] if int(row["step"]) == args.step), None)
    if checkpoint_row is None:
        raise RuntimeError(f"checkpoint step {args.step} is absent from bank")
    checkpoint = under_root(Path(checkpoint_row["path"]), "checkpoint")
    tokenizer_mod = __import__("transformers", fromlist=["AutoTokenizer"])
    tokenizer = tokenizer_mod.AutoTokenizer.from_pretrained(args.model, local_files_only=True, use_fast=True)
    eval_args = SimpleNamespace(
        dataset="Salesforce/wikitext",
        dataset_config="wikitext-103-raw-v1",
        eval_split="validation",
        eval_offset=0,
        seq_len=args.seq_len,
        device=device,
    )
    inputs, eval_protocol = load_natural_validation(tokenizer, eval_args)
    model = build_model(args.model, checkpoint, "eager", dtype, device)
    model.config.use_cache = False

    class LossStep(torch.nn.Module):
        def __init__(self, subject: torch.nn.Module) -> None:
            super().__init__()
            self.subject = subject

        def forward(self, input_ids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
            return self.subject(input_ids=input_ids, labels=labels, use_cache=False, return_dict=False)[0]

    audit = {"backend_compiles": 0, "runtime_invocations": 0, "graph_hashes": []}
    inductor = lookup_backend("inductor")

    def backend(graph_module: Any, example_inputs: list[Any]) -> Any:
        audit["backend_compiles"] += 1
        audit["graph_hashes"].append(hashlib.sha256(graph_module.code.encode()).hexdigest())
        compiled = inductor(graph_module, example_inputs)

        def counted(*values: Any) -> Any:
            audit["runtime_invocations"] += 1
            return compiled(*values)

        return counted

    module_start = len(PyCodeCache.modules)
    candidate = torch.compile(LossStep(model), backend=backend, fullgraph=True, dynamic=False)
    input_ids, labels = inputs
    model.zero_grad(set_to_none=True)
    warm_loss = candidate(input_ids, labels)
    warm_loss.backward()
    torch.cuda.synchronize(device)
    kernels = discover_all_triton_symbols(list(PyCodeCache.modules[module_start:]))
    kernel_by_symbol = dict(kernels)
    rows_by_symbol: dict[str, list[dict[str, Any]]] = {symbol: [] for symbol, _ in kernels}
    invocation_order: list[dict[str, Any]] = []
    restores: list[tuple[Any, Any]] = []
    for symbol, kernel in kernels:
        original = kernel.run
        pointer_names = [name for name, annotation in _runtime_signature(kernel) if str(annotation).startswith("*")]
        meta = getattr(kernel, "inductor_meta", {})
        mutated = sorted(str(value) for value in meta.get("mutated_arg_names", [])) if isinstance(meta, dict) else []

        def wrapped(*values: Any, _symbol=symbol, _kernel=kernel, _original=original, _pointer_names=pointer_names, _mutated=mutated, **kwargs: Any) -> Any:
            invocation_index = len(rows_by_symbol[_symbol])
            tensor_values = [value for value in values if isinstance(value, torch.Tensor)]
            if len(tensor_values) != len(_pointer_names):
                raise RuntimeError(f"pointer signature mismatch for {_symbol}")
            pointers = dict(zip(_pointer_names, tensor_values))
            pointer_meta = {name: _tensor_meta(value) for name, value in pointers.items()}
            result = _original(*values, **kwargs)
            row = {
                "symbol": _symbol,
                "invocation_index": invocation_index,
                "pointer_names": list(_pointer_names),
                "mutated_arg_names": list(_mutated),
                "pointer_metadata": pointer_meta,
            }
            rows_by_symbol[_symbol].append(row)
            invocation_order.append({"symbol": _symbol, "invocation_index": invocation_index})
            return result

        kernel.run = wrapped
        restores.append((kernel, original))

    model.zero_grad(set_to_none=True)
    loss = candidate(input_ids, labels)
    loss.backward()
    torch.cuda.synchronize(device)
    counts = {symbol: len(rows) for symbol, rows in rows_by_symbol.items()}
    output = {
        "schema": "kernel-analyzer-dtype-generated-topology-v1",
        "subject": "Qwen3-1.7B dtype-specific generated Triton topology",
        "dtype": args.dtype,
        "tf32": bool(args.tf32),
        "seq_len": args.seq_len,
        "checkpoint_step": args.step,
        "checkpoint_parameter_sha256": checkpoint_row["parameter_sha256"],
        "evaluation": eval_protocol,
        "compile_audit": audit,
        "warmed_symbol_count": len(kernels),
        "runtime_symbol_count": sum(bool(count) for count in counts.values()),
        "runtime_invocation_count": sum(counts.values()),
        "symbol_invocation_counts": dict(sorted(counts.items())),
        "rows": [row for symbol in sorted(rows_by_symbol) for row in rows_by_symbol[symbol]],
        "all_warmed_symbols_observed": all(counts.values()),
        "candidate_values_used_to_select_or_classify": False,
        "boundary": "Metadata-only topology census. It does not assign a semantic reference, implementation mechanism, or correctness verdict.",
    }
    output["result_sha256"] = hashlib.sha256(json.dumps(output, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "dtype": args.dtype, "seq_len": args.seq_len, "symbols": len(kernels), "invocations": sum(counts.values())}, sort_keys=True))
    for kernel, original in restores:
        kernel.run = original
    del candidate, model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
