#!/usr/bin/env python
"""Run a generic Qwen3-shaped higher-order-gradient case in one torch env."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def tensor_hash(value: Any) -> str:
    return hashlib.sha256(value.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def scalar_endpoint(value: Any, x: Any) -> dict[str, Any]:
    import torch

    result: dict[str, Any] = {
        "numeric_value": float(value.detach().cpu().item()),
        "requires_grad": bool(value.requires_grad),
        "has_grad_fn": value.grad_fn is not None,
        "numeric_sha256": tensor_hash(value),
    }
    try:
        value.backward()
        result["backward_succeeds"] = True
    except Exception as exc:
        result["backward_succeeds"] = False
        result["backward_error"] = f"{type(exc).__name__}: {exc}"
    return result


def stable_target(target: Any) -> str:
    """Avoid process-address strings in FX operation artifacts."""
    packet = getattr(target, "_overloadpacket", target)
    name = getattr(packet, "__name__", None)
    if name:
        return str(name)
    return str(target).split(" at 0x", 1)[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--mode", choices=("eager", "compiled"), required=True)
    parser.add_argument("--backend", choices=("eager", "aot_eager", "inductor"), default="aot_eager")
    parser.add_argument("--projection", choices=("mm", "linear"), default="mm")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import torch

    case = args.case_dir.resolve()
    out = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((case / "case_manifest.json").read_text())
    x0 = torch.load(case / manifest["artifacts"]["inputs"], map_location="cpu", weights_only=True)
    w0 = torch.load(case / manifest["artifacts"]["weights"], map_location="cpu", weights_only=True)

    def projection_endpoint(x: Any, weight: Any) -> Any:
        # This is the Qwen3 query-projection algebra expressed as mm.  The
        # operation name is not used as a locator heuristic; it is simply the
        # declared subject implementation.
        if args.projection == "mm":
            projected = torch.mm(x, weight)
        else:
            projected = torch.nn.functional.linear(x, weight.t().contiguous())
        grad_x, = torch.autograd.grad(projected.sum(), x, create_graph=True)
        return grad_x.sum()

    audit: dict[str, Any] = {"graphs": [], "runtime_invocations": 0}
    if args.mode == "eager":
        x = x0.clone().requires_grad_(True)
        weight = w0.clone().requires_grad_(True)
        value = projection_endpoint(x, weight)
    else:
        from torch._dynamo.backends.registry import lookup_backend

        selected = lookup_backend(args.backend)

        def backend(graph_module: Any, example_inputs: list[Any]) -> Any:
            code = graph_module.code
            code_hash = hashlib.sha256(code.encode()).hexdigest()
            nodes = [
                {"index": index, "op": node.op, "target": stable_target(node.target), "name": node.name}
                for index, node in enumerate(graph_module.graph.nodes)
            ]
            audit["graphs"].append({"sha256": code_hash, "nodes": nodes, "node_count": len(nodes)})
            compiled = selected(graph_module, example_inputs)

            def counted(*values: Any) -> Any:
                audit["runtime_invocations"] += 1
                return compiled(*values)

            return counted

        compiled = torch.compile(projection_endpoint, backend=backend)
        x = x0.clone().requires_grad_(True)
        weight = w0.clone().requires_grad_(True)
        value = compiled(x, weight)
    result = {
        "schema_version": "forkcert.qwen3-compiler-grad-case-run.v0.1",
        "case_id": manifest["case_id"],
        "mode": args.mode,
        "backend": args.backend if args.mode == "compiled" else "eager",
        "projection": args.projection,
        "torch": torch.__version__,
        "input": {"shape": list(x.shape), "dtype": str(x.dtype), "sha256": tensor_hash(x)},
        "endpoint": scalar_endpoint(value, x),
        "compile_audit": audit,
    }
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"mode": args.mode, "endpoint": result["endpoint"], "graphs": len(audit["graphs"])}, sort_keys=True))


if __name__ == "__main__":
    main()
