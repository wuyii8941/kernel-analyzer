#!/usr/bin/env python3
"""Transfer the frozen mechanism labels to a shape-specific inventory.

The transfer is deliberately conservative: only phase, kind, normalized
kernel stem, and numeric ordinal must match.  Rows that appear/disappear when
the shape changes remain ``UNRESOLVED_SHAPE_TRANSFER`` and cannot enter the
changed-site denominator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path


ALLOWED = {
    "EXPLICIT_FP32_REDUCTION_SCHEDULE_DIFFERENCE",
    "MATERIALIZATION_OR_ROUNDING_SCHEDULE_INTERVENTION",
    "SAME_PRECISION_GENERATED_SCHEDULE_DIFFERENCE",
}


def stem(symbol: str) -> str:
    return re.sub(r"_\d+$", "", symbol)


def ordinal(symbol: str) -> int:
    match = re.search(r"_(\d+)$", symbol)
    return int(match.group(1)) if match else -1


def key(row: dict) -> tuple[str, str, str]:
    return str(row.get("phase")), str(row.get("kind")), stem(str(row.get("symbol") or row.get("region")))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=Path("results/final/implementation_atlas.json"))
    p.add_argument("--inventory", type=Path, required=True)
    p.add_argument("--seq-len", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    source = json.loads(args.source.read_text())
    inventory = json.loads(args.inventory.read_text())
    target_rows = inventory["inventory"]["regions"]

    source_groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    target_groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in source["rows"]:
        source_groups[key(row)].append(row)
    for row in target_rows:
        target_groups[key(row)].append(row)
    for rows in source_groups.values():
        rows.sort(key=lambda row: ordinal(str(row.get("symbol") or row.get("region"))))
    for rows in target_groups.values():
        rows.sort(key=lambda row: ordinal(str(row["symbol"])))

    output_rows = []
    unresolved = []
    for group_key, rows in target_groups.items():
        source_rows = source_groups.get(group_key, [])
        for index, target in enumerate(rows):
            if index >= len(source_rows):
                output_rows.append({
                    "region": target["region_id"],
                    "phase": target["phase"],
                    "kind": target["kind"],
                    "symbol": target["symbol"],
                    "mechanism": "UNRESOLVED_SHAPE_TRANSFER",
                    "implementation_changed": False,
                    "transfer_status": "UNRESOLVED_SHAPE_TRANSFER",
                })
                unresolved.append(target["region_id"])
                continue
            source_row = source_rows[index]
            mechanism = str(source_row.get("mechanism"))
            changed = mechanism in ALLOWED
            output_rows.append({
                "region": target["region_id"],
                "phase": target["phase"],
                "kind": target["kind"],
                "symbol": target["symbol"],
                "source_region": source_row.get("region"),
                "source_symbol": source_row.get("symbol"),
                "mechanism": mechanism,
                "implementation_changed": changed,
                "transfer_status": "STEM_PHASE_KIND_ORDINAL_TRANSFER",
            })
    source_unmatched = [group for group in source_groups if group not in target_groups or len(source_groups[group]) > len(target_groups[group])]
    changed_rows = [row for row in output_rows if row["implementation_changed"]]
    output = {
        "schema": "kernel-analyzer-shape-implementation-atlas-v1",
        "subject": "Qwen3-1.7B shape-specific generated implementation mapping",
        "seq_len": args.seq_len,
        "source_atlas": str(args.source),
        "source_atlas_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "inventory": str(args.inventory),
        "inventory_sha256": hashlib.sha256(args.inventory.read_bytes()).hexdigest(),
        "denominator": {
            "target_generated_sites": len(target_rows),
            "mapped_sites": len(output_rows) - len(unresolved),
            "unresolved_sites": len(unresolved),
            "changed_sites": len(changed_rows),
            "source_unmatched_groups": len(source_unmatched),
        },
        "gates": {
            "all_target_sites_retained": len(output_rows) == len(target_rows),
            "changed_rows_have_explicit_source_binding": all(row.get("source_region") is not None for row in changed_rows),
            "candidate_values_used_to_map": False,
            "shape_transfer_is_exact_mechanism_proof": False,
        },
        "rows": output_rows,
        "unresolved_region_ids": unresolved,
        "boundary": "Mechanism labels are transferred from the seq64 frozen census by structural kernel identity; this is not an independent shape-specific mechanism proof. Unresolved rows remain outside the changed-site denominator.",
    }
    output["result_sha256"] = hashlib.sha256(json.dumps(output, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), **output["denominator"]}, sort_keys=True))


if __name__ == "__main__":
    main()
