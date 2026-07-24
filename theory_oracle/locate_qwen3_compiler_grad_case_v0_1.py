#!/usr/bin/env python
"""Bug-agnostic Oracle comparison for the Qwen3 compiler-gradient case."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--stage-control", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    reference = json.loads(args.reference.read_text())
    candidate = json.loads(args.candidate.read_text())
    stage_control = json.loads(args.stage_control.read_text()) if args.stage_control else None
    ref = reference["endpoint"]
    cand = candidate["endpoint"]
    semantic_fields = ["requires_grad", "has_grad_fn", "backward_succeeds"]
    changed_fields = [field for field in semantic_fields if ref.get(field) != cand.get(field)]
    numeric_equal = ref.get("numeric_sha256") == cand.get("numeric_sha256")
    graph_nodes: list[dict[str, Any]] = []
    for graph in candidate.get("compile_audit", {}).get("graphs", []):
        graph_nodes.extend(graph.get("nodes", []))
    stage_control_preserves = bool(
        stage_control is not None
        and all(stage_control["endpoint"].get(field) == ref.get(field) for field in semantic_fields)
    )
    backend_specific = bool(stage_control is not None and stage_control_preserves and changed_fields)
    report = {
        "schema_version": "forkcert.qwen3-compiler-grad-case-blind-locator.v0.1",
        "case_id": candidate["case_id"],
        "oracle": {
            "numeric_equal": numeric_equal,
            "semantic_fields_changed": changed_fields,
            "semantic_disagreement": bool(changed_fields),
        },
        "stage_control": {
            "present": stage_control is not None,
            "control_preserves_reference_contract": stage_control_preserves,
            "candidate_differs_while_control_does_not": backend_specific,
        },
        "candidate_operation_inventory": graph_nodes,
        "claim": {
            "level": "BACKEND_SPECIFIC_STAGE_CANDIDATE" if backend_specific else ("SEMANTIC_OPERATION_CANDIDATE" if changed_fields else "NO_OBSERVED_SEMANTIC_DISCREPANCY"),
            "statement": "generic endpoint, operation inventory and an independent backend control identify a backend-specific stage candidate; no bug metadata or unique root cause was used",
        },
        "nonclaims": [
            "no issue, PR, version or root-cause metadata was consumed",
            "numeric equality does not imply higher-order semantic equivalence",
            "an operation inventory is not proof of a unique compiler pass fault",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
