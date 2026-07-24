#!/usr/bin/env python
"""Summarize kernel families in the valid Qwen3 original-candidate trace."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


BOUNDARY_KERNEL = "triton_per_fused__to_copy__unsafe_view_add_mean_mul_pow_rsqrt_15"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    inventory = Path(args.inventory_dir).resolve()
    result = json.loads((inventory / "result.json").read_text())
    if result["status"] != "VALID_ORIGINAL_CANDIDATE_KERNEL_INVENTORY":
        raise ValueError("inventory is not valid")
    forward_dirs = sorted((inventory / "inductor_trace/torchinductor").glob("*forward*"))
    if len(forward_dirs) != 2:
        raise ValueError(f"expected two forward trace directories, found {len(forward_dirs)}")
    # The second specialization is the dynamic target family used at step 29.
    target_dir = forward_dirs[1]
    code_path = target_dir / "output_code.py"
    code = code_path.read_text()

    source_pattern = re.compile(
        r"# Topologically Sorted Source Nodes: \[(.*?)\], Original ATen: \[(.*?)\]"
    )
    definition_pattern = re.compile(
        r"^(triton_[A-Za-z0-9_]+) = async_compile\.triton"
    )
    definitions = {}
    pending_source = None
    for line in code.splitlines():
        source_match = source_pattern.search(line)
        if source_match:
            pending_source = source_match.groups()
            continue
        definition_match = definition_pattern.match(line)
        if definition_match and pending_source is not None:
            source_nodes, aten_nodes = pending_source
            definitions[definition_match.group(1)] = {
                "source_nodes": [item.strip() for item in source_nodes.split(",") if item.strip()],
                "original_aten": [item.strip() for item in aten_nodes.split(",") if item.strip()],
            }
            pending_source = None
    call_counts = {
        name: len(re.findall(rf"\b{re.escape(name)}\.run\(", code))
        for name in definitions
    }
    families = []
    for name, record in definitions.items():
        families.append(
            {
                "name": name,
                "call_count": call_counts[name],
                "source_nodes": record["source_nodes"],
                "original_aten": record["original_aten"],
            }
        )
    families.sort(key=lambda row: (-row["call_count"], row["name"]))
    extern_counts = {}
    for name in re.findall(r"extern_kernels\.([A-Za-z0-9_]+)\(", code):
        extern_counts[name] = extern_counts.get(name, 0) + 1
    boundary = next((row for row in families if row["name"] == BOUNDARY_KERNEL), None)
    gates = {
        "target_dynamic_forward_selected": "model__1_forward" in target_dir.name,
        "custom_kernel_definitions_found": bool(families),
        "boundary_kernel_found": boundary is not None,
        "boundary_kernel_called_27_times": boundary is not None and boundary["call_count"] == 27,
        "boundary_kernel_has_reduction_chain": boundary is not None
        and {"aten.add", "aten.pow", "aten.mean", "aten.rsqrt", "aten.mul"}.issubset(
            set(boundary["original_aten"])
        ),
    }
    payload = {
        "schema_version": "forkcert.qwen3-original-compiled-kernel-summary.v0.2",
        "status": "VALID_DESCRIPTIVE_SUMMARY" if all(gates.values()) else "INVALID_SUMMARY",
        "source_inventory": str(inventory),
        "source_forward_trace": str(target_dir.relative_to(inventory)),
        "gates": gates,
        "custom_kernel_family_count": len(families),
        "custom_kernel_call_count": sum(row["call_count"] for row in families),
        "extern_kernel_call_counts": dict(sorted(extern_counts.items())),
        "boundary_kernel_family": boundary,
        "kernel_families": families,
        "interpretation": {
            "supported": "the original candidate contains a repeated fused cross-layer kernel family",
            "not_supported": "no constituent ATen operator is identified as a discrepancy root cause",
        },
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in payload.items() if key != "kernel_families"}, indent=2, sort_keys=True))
    raise SystemExit(0 if payload["status"] == "VALID_DESCRIPTIVE_SUMMARY" else 1)


if __name__ == "__main__":
    main()
