#!/usr/bin/env python
"""Capture auditable FX-to-generated-Triton evidence for one GPU witness.

This script is intentionally case-local.  It does not inspect issue/patch
metadata and does not identify a source expression; it only records compiler
artifacts actually emitted for the same declared numerical contract.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    import torch
    import torch.fx

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.set_default_device("cuda")

    def model(x, scale):
        max_scaled = x * scale
        return torch.exp(max_scaled - x * scale)

    x = torch.tensor(1134139801600.0, dtype=torch.float32)
    scale = torch.tensor(0.180336877703666687, dtype=torch.float32)
    gm = torch.fx.symbolic_trace(model)
    fx_nodes = [
        {"name": node.name, "op": node.op, "target": str(node.target)}
        for node in gm.graph.nodes
    ]

    torch._dynamo.reset()
    compiled = torch.compile(model, backend="inductor", fullgraph=True)
    values = [float(compiled(x.clone(), scale.clone()).detach().cpu().item()) for _ in range(2)]

    root = Path(os.environ.get("TORCH_COMPILE_DEBUG_DIR", "torch_compile_debug"))
    outputs = sorted(root.rglob("output_code.py"))
    symbols = []
    for code in outputs:
        text = code.read_text(errors="replace")
        # Inductor 2.2 gives the actually invoked generated kernel its stable
        # provenance name through an ``async_compile.triton`` assignment.
        emitted = sorted(set(re.findall(
            r"^([A-Za-z0-9_]+)\s*=\s*async_compile\.triton\(", text,
            flags=re.MULTILINE,
        )))
        symbols.append({
            "path": str(code), "sha256": sha256(code), "triton_symbols": emitted,
            "contains_exp": "exp(" in text, "contains_mul": "*" in text,
        })
    artifact = {
        "schema_version": "forkcert.provenance-capture.v0.1",
        "case_id": "pytorch_fma_context_gpu",
        "environment": {"torch": torch.__version__, "cuda": torch.version.cuda,
                        "gpu": torch.cuda.get_device_name(0)},
        "declared_contract": "exp((x*scale)-(x*scale)) is finite one for the bound inputs",
        "inductor_values": values,
        "repeatable": values[0] == values[1],
        "fx_nodes": fx_nodes,
        "generated_artifacts": symbols,
        "provenance_level": "FX_TO_COMPILER_EMITTED_TRITON_SYMBOLS",
        "not_claimed": [
            "a one-to-one FX-to-kernel mapping", "same-input isolated-op production",
            "root expression", "source line", "merged-patch agreement",
        ],
    }
    output = Path("results/historical_blind/fma_context_122260_v0_1")
    output.mkdir(parents=True, exist_ok=True)
    (output / "post_certificate_provenance.json").write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"inductor_values": values, "artifacts": len(symbols), "symbols": symbols}, indent=2))


if __name__ == "__main__":
    main()
