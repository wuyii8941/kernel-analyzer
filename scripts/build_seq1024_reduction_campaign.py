#!/usr/bin/env python3
"""Bind seq1024 query/key norm reduction closures by exact source nodes."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


SYMBOLS = {
    "triton_red_fused_sum_14": "triton_red_fused_sum_15",
    "triton_per_fused_sum_18": "triton_red_fused_sum_16",
    "triton_red_fused__to_copy__unsafe_view_mul_sum_transpose_view_11": (
        "triton_red_fused__to_copy__unsafe_view_mul_sum_transpose_view_10"
    ),
    "triton_per_fused__to_copy__unsafe_view_mul_sum_transpose_view_15": (
        "triton_per_fused__to_copy__unsafe_view_mul_sum_transpose_view_11"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", type=Path, default=Path(
        "archive/round1_raw/training_semantic_oracle/qwen3_1p7b/full_step_inventory/"
        "triton_online_reference_campaign_v1.json"
    ))
    parser.add_argument(
        "--cache", type=Path,
        default=Path("/data1/tzh/cache/kernel_analyzer/long_causal_inductor"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/final/seq1024_reduction_campaign.json"),
    )
    return parser.parse_args()


def parse_nodes(comment: str) -> tuple[str, ...]:
    match = re.search(
        r"Topologically Sorted Source Nodes: (\[.*?\]), Original ATen:",
        comment,
    )
    if match is None:
        raise RuntimeError(f"cannot parse source-node comment: {comment}")
    return tuple(
        value.strip()
        for value in match.group(1)[1:-1].split(",")
        if value.strip()
    )


def find_compiled_module(cache: Path) -> Path:
    candidates = []
    for path in cache.rglob("*.py"):
        text = path.read_text(errors="ignore")
        counts = {new: text.count(f"{new}.run(") for new in SYMBOLS.values()}
        if all(count == 28 for count in counts.values()):
            candidates.append(path)
    if len(candidates) != 1:
        raise RuntimeError(f"expected one compiled module, got {len(candidates)}")
    return candidates[0]


def main() -> None:
    args = parse_args()
    canonical = json.loads(args.canonical.read_text())
    canonical_rows = [row for row in canonical["rows"] if row["symbol"] in SYMBOLS]
    if len(canonical_rows) != 112:
        raise RuntimeError(f"expected 112 canonical closure rows, got {len(canonical_rows)}")

    module = find_compiled_module(args.cache)
    lines = module.read_text(errors="ignore").splitlines()
    occurrences: dict[str, dict[tuple[str, ...], dict]] = {
        symbol: {} for symbol in SYMBOLS.values()
    }
    for index, line in enumerate(lines):
        for symbol in occurrences:
            if f"{symbol}.run(" not in line:
                continue
            comment = next(
                (
                    lines[prior]
                    for prior in range(index - 1, max(-1, index - 5), -1)
                    if "Topologically Sorted Source Nodes:" in lines[prior]
                ),
                None,
            )
            if comment is None:
                raise RuntimeError(f"missing source-node comment before {module}:{index + 1}")
            nodes = parse_nodes(comment)
            if nodes in occurrences[symbol]:
                raise RuntimeError(f"duplicate source nodes for {symbol}: {nodes}")
            occurrences[symbol][nodes] = {
                "path": str(module), "line": index + 1, "source_nodes": list(nodes)
            }

    rows = []
    used = set()
    for source in canonical_rows:
        old_symbol = str(source["symbol"])
        new_symbol = SYMBOLS[old_symbol]
        nodes = tuple(source["source_nodes"])
        occurrence = occurrences[new_symbol].get(nodes)
        if occurrence is None:
            raise RuntimeError(
                f"no exact seq1024 source-node binding for {source['region_id']}: {nodes}"
            )
        key = (new_symbol, occurrence["line"])
        if key in used:
            raise RuntimeError(f"seq1024 invocation reused: {key}")
        used.add(key)
        row = dict(source)
        row.update({
            "boundary_capture_mode": "SEQ1024_EXACT_SOURCE_NODES_AND_RUNTIME_POINTERS",
            "symbol": new_symbol,
            "reference_symbol": old_symbol,
            "canonical_reference_symbol": old_symbol,
            "input_names": list(source["input_names"]),
            "output_names": list(source["output_names"]),
            "prelaunch_clone_names": [],
            "seq1024_source_path": occurrence["path"],
            "seq1024_source_line": occurrence["line"],
            "seq1024_source_nodes": occurrence["source_nodes"],
        })
        rows.append(row)

    output = {
        "schema": "kernel-analyzer-seq1024-reduction-campaign-v1",
        "subject": "Qwen3-1.7B seq1024 query/key norm backward reduction closures",
        "canonical_campaign": str(args.canonical),
        "canonical_sha256": hashlib.sha256(args.canonical.read_bytes()).hexdigest(),
        "compiled_module": str(module),
        "symbol_mapping": SYMBOLS,
        "rows": rows,
        "denominator": {
            "canonical_invocations": 112,
            "exact_source_node_bindings": len(rows),
            "unresolved": 112 - len(rows),
        },
        "gates": {
            "all_invocations_uniquely_bound": len(rows) == 112 and len(used) == 112,
            "candidate_values_used_to_bind": False,
            "partial_final_pairs_complete": all(
                sum(row["reference_symbol"] == symbol for row in rows) == 28
                for symbol in SYMBOLS
            ),
        },
        "boundary": (
            "Shape-specific exact source-node and runtime-pointer binding; no result tensor "
            "was read to choose an invocation."
        ),
    }
    output["result_sha256"] = hashlib.sha256(
        json.dumps(output, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), **output["denominator"]}, sort_keys=True))


if __name__ == "__main__":
    main()
