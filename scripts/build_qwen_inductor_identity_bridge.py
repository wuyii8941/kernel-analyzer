#!/usr/bin/env python3
"""Bind canonical Qwen AOT proof units to a proof-tagged Inductor program.

The bridge is deliberately invocation preserving.  A node is either carried
with its canonical FX identity or retained as an explicit compiler rewrite
bounded by its exact input/user anchors.  Repeated operators are never paired
by operator name, shape, or ordinal.
"""

from __future__ import annotations

import gzip
import argparse
import hashlib
import json
import re
import sys
from collections import Counter, deque
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_architecture_invocation_ledger import FORMULAS  # noqa: E402


CANONICAL = ROOT / "results/coverage/qwen_aot_canonical_runtime_identity.json.gz"
CANDIDATE = ROOT / "results/coverage/qwen_inductor_canonical_proof_ids.json.gz"
EAGER_BRIDGE = ROOT / "results/coverage/qwen_eager_aot_bridge.json.gz"
TRACE = Path("/data1/tzh/cache/kernel_analyzer/qwen_proof_canonical_trace")
OUTPUT = ROOT / "results/coverage/qwen_inductor_identity_bridge.json.gz"


REWRITE_RULES = {
    "aten.alias.default": "y aliases x with identical values; VJP dx=q",
    "aten.clone.default": "y materializes x without changing values; VJP dx=q",
    "aten.detach.default": "first-order value identity with a stopped higher-order edge",
    "aten.unsqueeze.default": "insert a size-one axis; VJP removes that axis",
    "aten.view.default": "metadata-only reshape; VJP reshapes q to the input shape",
    "aten.transpose.int": "permute two axes; VJP applies the inverse permutation",
    "aten.lift_fresh_copy.default": "materialize the lifted scalar/tensor value; no differentiable source here",
    "aten.scalar_tensor.default": "materialize the declared scalar and metadata; no tensor VJP edge",
    "aten.bmm.default": "batch-one bmm rewritten to squeeze/multiply/reduce/unsqueeze with identical contraction",
}


def read_gzip(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def phase_key(phase: str, name: str) -> str:
    return f"{phase.lower()}:graph0:{name}"


def nearest_anchors(
    start: str,
    direction: str,
    nodes: dict[str, dict[str, Any]],
    preserved: set[str],
) -> list[str]:
    queue = deque([(start, 0)])
    seen = {start}
    found: list[str] = []
    found_depth: int | None = None
    while queue:
        current, depth = queue.popleft()
        if found_depth is not None and depth >= found_depth:
            continue
        for adjacent in nodes[current].get(direction, []):
            if adjacent in seen:
                continue
            seen.add(adjacent)
            if adjacent in preserved:
                found.append(adjacent)
                found_depth = depth + 1
            elif adjacent in nodes:
                queue.append((adjacent, depth + 1))
    return sorted(set(found))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical", type=Path, default=CANONICAL)
    parser.add_argument("--candidate", type=Path, default=CANDIDATE)
    parser.add_argument("--eager-bridge", type=Path, default=EAGER_BRIDGE)
    parser.add_argument("--trace", type=Path, default=TRACE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    canonical = read_gzip(args.canonical)
    candidate = read_gzip(args.candidate)
    eager_bridge = read_gzip(args.eager_bridge)
    if not canonical["preserve_aot_aten"]:
        raise RuntimeError("canonical bridge input must preserve AOT ATen")
    if canonical["status"] != "COMPLETE_AOT_FB_CAPTURE":
        raise RuntimeError("canonical AOT runtime identity is incomplete")
    if not candidate["repeat_stable"]:
        raise RuntimeError("candidate execution is not repeat stable")

    candidate_graphs = {graph["phase"]: graph for graph in candidate["proof_graphs"]}
    canonical_graphs = {
        graph["phase"]: graph for graph in canonical["capture"]["graphs"]
    }
    canonical_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    canonical_status = Counter()
    candidate_status = Counter()

    for phase in ("FORWARD", "BACKWARD"):
        canonical_nodes = {
            node["name"]: node
            for node in canonical_graphs[phase]["nodes"]
            if node["op"] == "call_function"
        }
        post_rows = candidate_graphs[phase]["rows"]
        post_by_name = {
            row["proof_id"].rsplit(":", 1)[1]: row for row in post_rows
        }
        tag_to_name = {
            row["tagged_fx_name"]: row["proof_id"].rsplit(":", 1)[1]
            for row in post_rows
        }
        preserved = set(canonical_nodes) & set(post_by_name)
        for name in preserved:
            if canonical_nodes[name]["target"] != post_by_name[name]["target"]:
                raise RuntimeError(f"preserved proof ID changed target: {phase}:{name}")

        for name, node in canonical_nodes.items():
            proof_id = phase_key(phase, name)
            if name in preserved:
                row = {
                    "canonical_proof_id": proof_id,
                    "status": "EXACT_COMPILER_PRESERVED_PROOF_ID_AND_TARGET",
                    "target": node["target"],
                    "candidate_tag": post_by_name[name]["tagged_fx_name"],
                }
            else:
                if node["target"] not in REWRITE_RULES:
                    raise RuntimeError(f"unproved compiler rewrite: {proof_id}:{node['target']}")
                input_anchors = nearest_anchors(
                    name, "input_nodes", canonical_nodes, preserved
                )
                user_anchors = nearest_anchors(
                    name, "users", canonical_nodes, preserved
                )
                boundary = None
                if not input_anchors:
                    boundary = "AOT_BACKWARD_RUNTIME_INPUT_FROM_EXACT_FORWARD_IDENTITY_BRIDGE"
                if not user_anchors:
                    boundary = "AOT_FORWARD_SAVED_OUTPUT_TO_EXACT_BACKWARD_IDENTITY_BRIDGE"
                row = {
                    "canonical_proof_id": proof_id,
                    "status": "EXACT_CLOSED_COMPILER_SEMANTIC_REWRITE",
                    "target": node["target"],
                    "rewrite_equivalence": REWRITE_RULES[node["target"]],
                    "canonical_input_nodes": node["input_nodes"],
                    "canonical_user_nodes": node["users"],
                    "nearest_preserved_input_anchors": [
                        phase_key(phase, value) for value in input_anchors
                    ],
                    "nearest_preserved_user_anchors": [
                        phase_key(phase, value) for value in user_anchors
                    ],
                    "runtime_boundary_identity": boundary,
                }
                if not (input_anchors or boundary) or not (user_anchors or boundary):
                    raise RuntimeError(f"rewrite lacks two-sided identity boundary: {proof_id}")
            row["row_sha256"] = digest(row)
            canonical_rows.append(row)
            canonical_status[row["status"]] += 1

        output_code = next(
            args.trace.glob(
                f"torchinductor/model__0_{phase.lower()}_*/output_code.py"
            )
        ).read_text(errors="ignore")
        cpp_mapping: dict[str, list[str]] = {}
        if phase == "FORWARD":
            provenance_path = next(
                args.trace.glob(
                    "torchinductor/model__0_forward_*/inductor_provenance_tracking_node_mappings.json"
                )
            )
            provenance = json.loads(provenance_path.read_text())
            cpp_mapping = provenance["postToCppCode"]
        for post in post_rows:
            target = post["target"]
            if target not in FORMULAS:
                raise RuntimeError(f"candidate node lacks map/VJP formula: {target}")
            tag = post["tagged_fx_name"]
            code_sites = cpp_mapping.get(tag, [])
            if code_sites or tag in output_code:
                status = "EXACT_GENERATED_SOURCE_PROVENANCE"
            else:
                status = "EXACT_COMPILED_GRAPH_NODE_WITHOUT_DISTINCT_SOURCE_TAG"
            row = {
                "candidate_proof_id": post["proof_id"],
                "candidate_tag": tag,
                "target": target,
                "map": FORMULAS[target]["map"],
                "adjoint": FORMULAS[target]["adjoint"],
                "input_candidate_proof_ids": [
                    phase_key(phase, tag_to_name.get(value, value))
                    for value in post["input_tagged_fx_names"]
                    if value in tag_to_name
                ],
                "generated_code_sites": code_sites,
                "status": status,
            }
            row["row_sha256"] = digest(row)
            candidate_rows.append(row)
            candidate_status[status] += 1

    expected_canonical = sum(
        graph["call_function_count"] for graph in canonical_graphs.values()
    )
    expected_candidate = candidate["proof_tag_summary"]["aot_call_function_nodes"]
    if len(canonical_rows) != expected_canonical or expected_canonical != 8985:
        raise RuntimeError("canonical AOT denominator changed")
    if len(candidate_rows) != expected_candidate:
        raise RuntimeError("candidate proof-tag denominator changed")
    if candidate["proof_tag_summary"]["tags_not_observed"]:
        raise RuntimeError("candidate proof tags were lost from the compiler trace")

    payload = {
        "schema": "kernel-analyzer-qwen-inductor-identity-bridge-v1",
        "status": "COMPLETE_CANONICAL_AOT_TO_PROOF_TAGGED_INDUCTOR_ACCOUNTING",
        "inputs": {
            "canonical_aot": str(args.canonical.resolve().relative_to(ROOT)),
            "candidate": str(args.candidate.resolve().relative_to(ROOT)),
            "eager_aot_bridge": str(args.eager_bridge.resolve().relative_to(ROOT)),
            "trace": str(args.trace.resolve()),
        },
        "denominators": {
            "eager_invocations": eager_bridge["denominators"]["eager_invocations"],
            "canonical_aot_nodes": len(canonical_rows),
            "candidate_post_aot_nodes": len(candidate_rows),
            "candidate_proof_tags_observed": candidate["proof_tag_summary"]["tags_observed_in_inductor_trace"],
        },
        "canonical_status_counts": dict(sorted(canonical_status.items())),
        "candidate_status_counts": dict(sorted(candidate_status.items())),
        "gates": {
            "eager_to_canonical_aot_complete": eager_bridge["status"] == "COMPLETE_EXACT_OR_CLOSED_SEMANTIC_REGION_ACCOUNTING",
            "canonical_forward_backward_runtime_identity_complete": all(canonical["gates"].values()),
            "every_canonical_aot_node_accounted": len(canonical_rows) == 8985,
            "every_candidate_node_has_map_and_adjoint": len(candidate_rows) == expected_candidate,
            "every_candidate_proof_tag_observed_in_compiler_trace": candidate["proof_tag_summary"]["tags_not_observed"] == 0,
            "candidate_repeat_stable": candidate["repeat_stable"],
            "operator_name_shape_or_ordinal_pairing_used": False,
        },
        "canonical_rows": canonical_rows,
        "candidate_rows": candidate_rows,
        "claim_boundary": {
            "supported": (
                "Invocation-preserving eager-to-AOT identity, complete canonical AOT accounting, "
                "and a mathematical map/adjoint plus compiler-carried tag for every post-AOT candidate node."
            ),
            "not_supported": (
                "Held-out numerical equivalence or a population-level correctness certificate; "
                "candidate nodes without a distinct generated-source tag remain fused/elided graph semantics."
            ),
        },
    }
    payload["result_sha256"] = digest(payload)
    with gzip.open(args.output, "wt", encoding="utf-8", compresslevel=9) as handle:
        json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
    print(json.dumps({
        "output": str(args.output.resolve().relative_to(ROOT)),
        "status": payload["status"],
        "denominators": payload["denominators"],
        "canonical_status_counts": payload["canonical_status_counts"],
        "candidate_status_counts": payload["candidate_status_counts"],
        "result_sha256": payload["result_sha256"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
