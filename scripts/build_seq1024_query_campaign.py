#!/usr/bin/env python3
"""Bind seq1024 query-norm backward invocations by exact source anchors."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


NEW_SYMBOL = (
    "triton_red_fused__to_copy__unsafe_view_add_bmm_cat_clone_cos_div_expand_"
    "mul_neg_pow_sin_slice_slice_backward_sum_transpose_unsqueeze_view_14"
)
OLD_SYMBOL = (
    "triton_per_fused__to_copy__unsafe_view_add_bmm_cat_clone_cos_div_expand_"
    "mul_neg_pow_sin_slice_slice_backward_sum_transpose_unsqueeze_view_10"
)
ANCHOR = re.compile(r"^(?:linear|view|hidden_states|rsqrt)_\d+$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", type=Path, default=Path(
        "archive/round1_raw/training_semantic_oracle/qwen3_1p7b/full_step_inventory/"
        "triton_online_reference_campaign_v1.json"
    ))
    parser.add_argument("--cache", type=Path, default=Path("/data1/tzh/cache/kernel_analyzer/long_causal_inductor"))
    parser.add_argument("--output", type=Path, default=Path("results/final/seq1024_query_campaign.json"))
    return parser.parse_args()


def source_anchors(values: list[str]) -> tuple[str, ...]:
    return tuple(sorted(value for value in values if ANCHOR.match(value)))


def main() -> None:
    args = parse_args()
    canonical = json.loads(args.canonical.read_text())
    source_rows = [row for row in canonical["rows"] if row["symbol"] == OLD_SYMBOL]
    if len(source_rows) != 28:
        raise RuntimeError(f"expected 28 canonical query invocations, got {len(source_rows)}")

    occurrences = []
    for path in args.cache.rglob("*.py"):
        text = path.read_text(errors="ignore")
        if text.count(f"{NEW_SYMBOL}.run(") < 28:
            continue
        lines = text.splitlines()
        for index, line in enumerate(lines):
            if f"{NEW_SYMBOL}.run(" not in line:
                continue
            comment = next(
                (lines[prior] for prior in range(index - 1, max(-1, index - 5), -1)
                 if "Topologically Sorted Source Nodes:" in lines[prior]),
                None,
            )
            if comment is None:
                raise RuntimeError(f"missing source-node comment before {path}:{index + 1}")
            match = re.search(r"Topologically Sorted Source Nodes: (\[.*?\]), Original ATen:", comment)
            if match is None:
                raise RuntimeError(f"cannot parse source-node comment: {comment}")
            nodes = [value.strip() for value in match.group(1)[1:-1].split(",") if value.strip()]
            occurrences.append({
                "path": str(path),
                "line": index + 1,
                "source_nodes": nodes,
                "anchors": source_anchors(nodes),
            })
        break
    if len(occurrences) != 28:
        raise RuntimeError(f"expected 28 seq1024 query invocations, got {len(occurrences)}")

    by_anchor: dict[tuple[str, ...], list[dict]] = {}
    for occurrence in occurrences:
        by_anchor.setdefault(occurrence["anchors"], []).append(occurrence)
    rows = []
    used_lines = set()
    for source in source_rows:
        anchors = source_anchors(source["source_nodes"])
        matches = by_anchor.get(anchors, [])
        if len(matches) != 1:
            raise RuntimeError(f"non-unique source-anchor binding for {source['region_id']}: {anchors} -> {len(matches)}")
        occurrence = matches[0]
        key = (occurrence["path"], occurrence["line"])
        if key in used_lines:
            raise RuntimeError(f"seq1024 invocation reused: {key}")
        used_lines.add(key)
        row = dict(source)
        row.update({
            "boundary_capture_mode": "SEQ1024_EXACT_SOURCE_ANCHORS_AND_RUNTIME_POINTERS",
            "symbol": NEW_SYMBOL,
            "reference_symbol": OLD_SYMBOL,
            "canonical_reference_symbol": OLD_SYMBOL,
            "input_names": ["in_ptr0", "in_ptr1", "in_ptr2", "in_ptr3", "in_ptr4"],
            "output_names": ["out_ptr0", "out_ptr1", "out_ptr3"],
            "prelaunch_clone_names": [],
            "seq1024_source_path": occurrence["path"],
            "seq1024_source_line": occurrence["line"],
            "seq1024_source_nodes": occurrence["source_nodes"],
            "source_anchor_binding": list(anchors),
        })
        rows.append(row)
    output = {
        "schema": "kernel-analyzer-seq1024-query-campaign-v1",
        "subject": "Qwen3-1.7B seq1024 query rotary RMSNorm backward",
        "canonical_campaign": str(args.canonical),
        "canonical_sha256": hashlib.sha256(args.canonical.read_bytes()).hexdigest(),
        "new_symbol": NEW_SYMBOL,
        "reference_symbol": OLD_SYMBOL,
        "rows": rows,
        "denominator": {"canonical_invocations": 28, "exact_source_anchor_bindings": len(rows), "unresolved": 28 - len(rows)},
        "gates": {
            "all_invocations_uniquely_bound": len(rows) == 28 and len(used_lines) == 28,
            "candidate_values_used_to_bind": False,
            "three_output_topology_explicit": all(row["output_names"] == ["out_ptr0", "out_ptr1", "out_ptr3"] for row in rows),
        },
        "boundary": "Shape-specific source and runtime-pointer binding; no result tensor was read to choose an invocation.",
    }
    output["result_sha256"] = hashlib.sha256(
        json.dumps(output, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), **output["denominator"]}, sort_keys=True))


if __name__ == "__main__":
    main()
