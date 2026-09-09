#!/usr/bin/env python3
"""Group uncovered Triton definitions for semantic review without outcomes."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from kernel_analyzer.semantic_family_candidates import suggest_family


def group(document):
    groups = defaultdict(list)
    clusters = []
    for cluster in document["clusters"]:
        suggestion = suggest_family(cluster.get("compiler_original_aten", []))
        row = {
            "representative_symbol": cluster["representative_symbol"],
            "definition_count": cluster["definition_count"],
            "source_file_count": cluster["source_file_count"],
            "compiler_original_aten": cluster.get("compiler_original_aten", []),
            "compiler_module_paths": cluster.get("compiler_module_paths", []),
            "member_binding_status_counts": cluster.get(
                "member_binding_status_counts", {}),
            **suggestion,
        }
        clusters.append(row)
        groups[suggestion["suggested_family"]].append(row)
    summaries = []
    for family, rows in groups.items():
        summaries.append({
            "suggested_family": family,
            "definition_count": sum(row["definition_count"] for row in rows),
            "structural_cluster_count": len(rows),
            "family_already_in_reporting_catalogue": all(
                row["family_already_in_reporting_catalogue"] for row in rows),
            "suggestion_basis_counts": dict(Counter(
                row["suggestion_basis"] for row in rows)),
            "review_status": "REQUIRES_HUMAN_REVIEW",
        })
    summaries.sort(key=lambda row: (-row["definition_count"], row["suggested_family"]))
    return {
        "schema": "uncovered-triton-semantic-candidate-groups-v1",
        "selection_uses_numerical_results": False,
        "scope": "UNSUPPORTED_SAVED_TRITON_DEFINITIONS_ONLY",
        "warning": (
            "Suggestions organize review and do not establish mathematical "
            "semantics, a reference implementation, runtime support, bias, or "
            "an independent mechanism."
        ),
        "definition_count": sum(row["definition_count"] for row in clusters),
        "structural_cluster_count": len(clusters),
        "candidate_family_count": len(summaries),
        "families": summaries,
        "clusters": clusters,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--semantic-queue", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(Path("/data1/tzh")):
        parser.error("Choose a new output under /data1/tzh")
    raw = args.semantic_queue.read_bytes()
    document = json.loads(raw)
    if document.get("schema") != "uncovered-triton-semantic-review-queue-compact-v1":
        raise ValueError("Compact semantic review queue required")
    result = group(document)
    result["input_sha256"] = hashlib.sha256(raw).hexdigest()
    result["source_sha256"] = hashlib.sha256(
        Path(__file__).read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({
        "definitions": result["definition_count"],
        "clusters": result["structural_cluster_count"],
        "families": result["families"],
    }))


if __name__ == "__main__":
    main()
