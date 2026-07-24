#!/usr/bin/env python
"""Build an auditable many-to-many FX/ATen-to-kernel inventory from a gated trace."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def kernel_kind(name: str) -> str:
    if name.startswith("triton_"):
        return "triton"
    if name.startswith("extern_kernels."):
        return "external"
    if name.startswith("cpp_"):
        return "cpp"
    return "other"


def select_graph_manifest(
    pre_names: set[str], manifests: list[dict[str, Any]]
) -> tuple[dict[str, Any] | None, int]:
    best = None
    best_overlap = -1
    for manifest in manifests:
        names = {node["name"] for node in manifest.get("nodes", [])}
        overlap = len(pre_names & names)
        if overlap > best_overlap:
            best = manifest
            best_overlap = overlap
    return best, max(best_overlap, 0)


def build_inventory(
    traced: dict[str, Any], gate: dict[str, Any]
) -> dict[str, Any]:
    if gate.get("forward_kernel_inventory_eligible") is not True:
        raise ValueError("whole-model forward observability gate is not eligible")
    manifests = []
    for row in traced.get("compiler", {}).get("graph_manifests", []):
        path = Path(row["path"])
        if not path.is_file() or sha256_file(path) != row["sha256"]:
            raise ValueError(f"invalid graph manifest artifact: {path}")
        manifests.append(json.loads(path.read_text()))

    mapping_rows = [
        row
        for row in traced.get("observability", {}).get("trace_files", [])
        if row.get("kind") == "inductor_provenance_tracking_node_mappings.json"
        and "forward" in Path(row["path"]).parent.name
    ]
    kernels = []
    stage_rows = []
    for mapping_row in mapping_rows:
        mapping_path = Path(mapping_row["path"])
        if not mapping_path.is_file() or sha256_file(mapping_path) != mapping_row["sha256"]:
            raise ValueError(f"invalid provenance mapping artifact: {mapping_path}")
        mapping = json.loads(mapping_path.read_text())
        post_to_pre = mapping.get("postToPre", {})
        all_pre = {
            pre
            for post_nodes in mapping.get("cppCodeToPost", {}).values()
            for post in post_nodes
            for pre in post_to_pre.get(post, [])
        }
        graph_manifest, overlap = select_graph_manifest(all_pre, manifests)
        node_meta = {
            node["name"]: node for node in (graph_manifest or {}).get("nodes", [])
        }
        output_code = mapping_path.with_name("output_code.py")
        output_code_sha = sha256_file(output_code) if output_code.is_file() else None
        stage_id = mapping_path.parent.name
        stage_kernel_start = len(kernels)
        for call_site, post_nodes in mapping.get("cppCodeToPost", {}).items():
            kernel_name, _, launch_index = call_site.rpartition(":")
            pre_nodes = sorted(
                {
                    pre
                    for post in post_nodes
                    for pre in post_to_pre.get(post, [])
                }
            )
            metadata = [node_meta[name] for name in pre_nodes if name in node_meta]
            kernels.append(
                {
                    "kernel_id": f"{stage_id}::{call_site}",
                    "stage_id": stage_id,
                    "launch_index": int(launch_index) if launch_index.isdigit() else None,
                    "generated_symbol": kernel_name,
                    "kind": kernel_kind(kernel_name),
                    "post_fusion_nodes": sorted(post_nodes),
                    "pre_fusion_nodes": pre_nodes,
                    "fx_node_metadata": metadata,
                    "module_paths": sorted(
                        {
                            row["nn_module_stack"]
                            for row in metadata
                            if row.get("nn_module_stack") not in {None, "None"}
                        }
                    ),
                    "source_functions": sorted(
                        {
                            row["source_fn_stack"]
                            for row in metadata
                            if row.get("source_fn_stack") not in {None, "None"}
                        }
                    ),
                    "output_code_path": str(output_code) if output_code.is_file() else None,
                    "output_code_sha256": output_code_sha,
                    "mapping_path": str(mapping_path),
                    "mapping_sha256": mapping_row["sha256"],
                    "claim": "provenance relation only; not discrepancy production or endpoint mediation",
                }
            )
        stage_rows.append(
            {
                "stage_id": stage_id,
                "mapping_path": str(mapping_path),
                "output_code_path": str(output_code) if output_code.is_file() else None,
                "matched_graph_code_sha256": (
                    graph_manifest.get("graph_code_sha256") if graph_manifest else None
                ),
                "pre_node_overlap": overlap,
                "kernel_count": len(kernels) - stage_kernel_start,
            }
        )
    kernels_with_pre_nodes = sum(bool(row["pre_fusion_nodes"]) for row in kernels)
    kernels_with_fx_metadata = sum(bool(row["fx_node_metadata"]) for row in kernels)
    return {
        "schema_version": "forkcert.inductor-kernel-inventory.v0.1",
        "status": "VALID" if kernels else "INVALID_EMPTY",
        "stages": stage_rows,
        "kernels": kernels,
        "summary": {
            "stage_count": len(stage_rows),
            "kernel_count": len(kernels),
            "kernels_with_pre_fusion_nodes": kernels_with_pre_nodes,
            "kernels_with_fx_node_metadata": kernels_with_fx_metadata,
            "kinds": {
                kind: sum(row["kind"] == kind for row in kernels)
                for kind in ("triton", "external", "cpp", "other")
            },
        },
        "limitations": [
            "module/operator/kernel provenance is many-to-many",
            "provenance does not establish numerical discrepancy production",
            "inventory does not establish endpoint mediation",
            "opaque external kernels are not decomposed",
            "the current gate licenses forward kernels only, not backward/update kernels",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traced", required=True)
    parser.add_argument("--gate", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    traced_path = Path(args.traced).resolve()
    gate_path = Path(args.gate).resolve()
    result = build_inventory(
        json.loads(traced_path.read_text()), json.loads(gate_path.read_text())
    )
    result["traced_result"] = {"path": str(traced_path), "sha256": sha256_file(traced_path)}
    result["observability_gate"] = {"path": str(gate_path), "sha256": sha256_file(gate_path)}
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    raise SystemExit(0 if result["status"] == "VALID" else 2)


if __name__ == "__main__":
    main()
