#!/usr/bin/env python
"""Build the generated-kernel denominator for the audited Qwen3 backward trace."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory-dir", required=True)
    parser.add_argument("--coverage-json", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    inventory = Path(args.inventory_dir).resolve()
    result = json.loads((inventory / "result.json").read_text())
    coverage = json.loads(Path(args.coverage_json).read_text())
    if result["status"] != "VALID_ORIGINAL_CANDIDATE_KERNEL_INVENTORY":
        raise ValueError("source inventory is not valid")

    backward_dirs = sorted((inventory / "inductor_trace/torchinductor").glob("*backward*"))
    if len(backward_dirs) != 1:
        raise ValueError(f"expected one backward trace directory, found {len(backward_dirs)}")
    target_dir = backward_dirs[0]
    code_path = target_dir / "output_code.py"
    code = code_path.read_text()

    source_pattern = re.compile(
        r"# Topologically Sorted Source Nodes: \[(.*?)\], Original ATen: \[(.*?)\]"
    )
    definition_pattern = re.compile(r"^(triton_[A-Za-z0-9_]+) = async_compile\.triton")
    definitions: dict[str, dict[str, object]] = {}
    pending_source: tuple[str, str] | None = None
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

    families = []
    for name, record in definitions.items():
        families.append(
            {
                "name": name,
                "call_count": len(re.findall(rf"\b{re.escape(name)}\.run\(", code)),
                **record,
            }
        )
    families.sort(key=lambda row: (-int(row["call_count"]), str(row["name"])))
    extern_counts: dict[str, int] = {}
    for name in re.findall(r"extern_kernels\.([A-Za-z0-9_]+)\(", code):
        extern_counts[name] = extern_counts.get(name, 0) + 1

    gates = {
        "target_backward_selected": "model__1_backward" in target_dir.name,
        "custom_kernel_family_count_exact": len(families) == 39,
        "all_custom_families_called": bool(families)
        and all(int(row["call_count"]) > 0 for row in families),
        "external_family_counts_exact": extern_counts == {"bmm": 168, "mm": 563},
        "source_backward_atomic_denominator_exact": coverage["metrics"][
            "backward_atomic_calls_observed"
        ]
        == 9471
        and coverage["metrics"]["backward_atomic_target_types_observed"] == 40,
    }
    payload = {
        "schema_version": "forkcert.qwen3-backward-generated-kernel-summary.v0.1",
        "status": "VALID_BACKWARD_GENERATED_KERNEL_SUMMARY" if all(gates.values()) else "INVALID_SUMMARY",
        "source_inventory": str(inventory),
        "source_coverage": str(Path(args.coverage_json).resolve()),
        "source_backward_trace": str(target_dir.relative_to(inventory)),
        "gates": gates,
        "custom_kernel_family_count": len(families),
        "custom_kernel_call_site_count": sum(int(row["call_count"]) for row in families),
        "extern_kernel_call_site_counts": dict(sorted(extern_counts.items())),
        "generated_treatment_family_count": len(families) + len(extern_counts),
        "kernel_families": families,
        "claim_limits": [
            "descriptive generated-treatment denominator only",
            "counts are static generated-code call sites; the source inventory did not execute backward",
            "same-name family is not yet a validated cross-call equivalence class",
            "no repair, injection, causal attribution, population transport or correctness credit",
        ],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError(out)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {key: value for key, value in payload.items() if key != "kernel_families"},
            indent=2,
            sort_keys=True,
        )
    )
    if payload["status"] != "VALID_BACKWARD_GENERATED_KERNEL_SUMMARY":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
