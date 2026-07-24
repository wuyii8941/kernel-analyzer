#!/usr/bin/env python
"""Post-certificate provenance replay for the frozen #115260 witness.

It captures compiler artifacts without inferring a kernel relation from names.
The report distinguishes files merely co-produced by compilation from explicit
origin/fusion evidence visible inside an artifact.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    import torch

    torch.manual_seed(115260)
    p0 = torch.tensor([1.0879], dtype=torch.float16)
    args = (
        torch.randn((17, 5, 1, 7), dtype=torch.float16) * 0.1 + 5,
        torch.randn((17, 5, 1, 7), dtype=torch.float16) * 0.1 + 5,
        torch.randn((17, 5, 11, 7), dtype=torch.float16) * 0.1 + 5,
        torch.randn((17, 5, 1, 7), dtype=torch.float16) * 0.1 + 5,
        torch.tensor(4.39, dtype=torch.float16),
    )

    def hidden(*values):
        cat = torch.cat((values[3], values[2], values[1], values[0]), dim=2)
        mul = torch.mul(cat, torch.max(values[4], p0))
        return torch.tan(mul)

    def exposed_mul(*values):
        cat = torch.cat((values[3], values[2], values[1], values[0]), dim=2)
        mul = torch.mul(cat, torch.max(values[4], p0))
        return mul, torch.tan(mul)

    torch._dynamo.reset()
    base = torch.compile(hidden, backend="inductor", fullgraph=True)
    torch._dynamo.reset()
    exposed = torch.compile(exposed_mul, backend="inductor", fullgraph=True)
    baseline = base(*[x.clone() for x in args])
    _, final = exposed(*[x.clone() for x in args])
    max_abs = float((baseline - final).abs().max().item())

    debug_root = Path(os.environ.get("TORCH_COMPILE_DEBUG_DIR", "torch_compile_debug"))
    artifact_rows = []
    if debug_root.exists():
        for path in sorted(debug_root.rglob("*")):
            if not path.is_file():
                continue
            # Keep source/IR artifacts only; raw binaries are opaque here.
            if path.suffix not in {".py", ".txt", ".log", ".json", ".cpp"}:
                continue
            text = path.read_text(errors="replace")
            kernel_symbols = []
            if path.name == "output_code.py":
                kernel_symbols = re.findall(
                    r"^([A-Za-z0-9_]*fused[A-Za-z0-9_]*)\s*=\s*async_compile\.cpp",
                    text,
                    flags=re.MULTILINE,
                )
            artifact_rows.append({
                "path": str(path.relative_to(debug_root)), "sha256": digest(path),
                "bytes": path.stat().st_size,
                "contains_mul": "mul" in text,
                "contains_tan": "tan" in text,
                "contains_fusion_marker": "fused" in text or "Fusion" in text,
                "generated_cpp_kernel_symbols": kernel_symbols,
            })
    explicit = [
        row for row in artifact_rows
        if any("mul" in symbol and "tan" in symbol for symbol in row["generated_cpp_kernel_symbols"])
    ]
    record = {
        "schema_version": "forkcert.inductor-provenance-replay.v0.1",
        "case": "pytorch_115260_tan_output_context",
        "role": "post_certificate_provenance_replay",
        "environment": {"torch": torch.__version__, "device": "cpu"},
        "endpoint": {"contextual_max_abs": max_abs, "violation_reproduced": max_abs > 0.0},
        "debug_root": str(debug_root), "artifacts": artifact_rows,
        "explicit_mul_tan_artifacts": explicit,
        "allowed_claim": (
            "FX_TO_FUSED_CPP_KERNEL_SYMBOL_EVIDENCE" if explicit
            else "FX_CANDIDATE_ONLY_NO_GENERATED_ARTIFACT_MAPPING"
        ),
        "not_claimed": ["unique kernel cause", "source line", "patch agreement from filename alone"],
    }
    out = Path("results/historical_blind/tan_output_context_115260_v0_1")
    out.mkdir(parents=True, exist_ok=True)
    (out / "post_certificate_provenance.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"endpoint": record["endpoint"], "artifact_count": len(artifact_rows),
                      "explicit_count": len(explicit), "claim": record["allowed_claim"]}, indent=2))


if __name__ == "__main__":
    main()
