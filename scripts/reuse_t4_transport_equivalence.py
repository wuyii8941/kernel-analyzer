#!/usr/bin/env python3
"""Certify a T4 disposition through an exact value-preserving AOT path.

This avoids rerunning a paired trajectory when repairing an upstream endpoint
is mathematically identical to repairing a downstream endpoint after only
indexing/copy operations, and the complete T3 carrier vectors are identical.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ALLOWED = {
    "aten.unsqueeze.default", "aten.expand.default", "aten.clone.default",
    "aten.view.default", "aten._unsafe_view.default", "aten.permute.default",
    "aten.transpose.int",
}


def load(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def canonical(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def only_path(nodes: dict[str, dict], source: str, target: str) -> list[str]:
    paths: list[list[str]] = []

    def visit(name: str, path: list[str]) -> None:
        if name == target:
            paths.append(path)
            return
        for user in nodes[name].get("users", []):
            if user not in path:
                visit(user, path + [user])

    visit(source, [source])
    if len(paths) != 1:
        raise RuntimeError(f"expected one AOT path, found {len(paths)}")
    path = paths[0]
    for name in path[1:]:
        if nodes[name]["target"] not in ALLOWED:
            raise RuntimeError(f"non-transport node in path: {name} {nodes[name]['target']}")
        if nodes[name].get("input_nodes") != [path[path.index(name) - 1]]:
            raise RuntimeError(f"transport node has additional tensor inputs: {name}")
    return path


def find_t3(cell: str, task_id: str) -> tuple[Path, dict]:
    matches = []
    for path in sorted((ROOT / "results/coverage/cases/carrier" / cell).glob("*.json.gz")):
        value = load(path)
        if value.get("task_id") == task_id:
            matches.append((path, value))
    if len(matches) != 1:
        raise RuntimeError(f"T3 {task_id}: expected one artifact, found {len(matches)}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", required=True)
    parser.add_argument("--source-task", required=True)
    parser.add_argument("--target-task", required=True)
    args = parser.parse_args()

    release = ROOT / "results/coverage/runtime_releases" / args.cell
    source_t3_path, source_t3 = find_t3(args.cell, args.source_task)
    target_t3_path, target_t3 = find_t3(args.cell, args.target_task)
    if source_t3["carrier_parameter"] != target_t3["carrier_parameter"]:
        raise RuntimeError("carrier parameters differ")
    source_vectors = [(r["state_id"], r["vector_sha256"]) for r in source_t3["records"]]
    target_vectors = [(r["state_id"], r["vector_sha256"]) for r in target_t3["records"]]
    if len(source_vectors) != 32 or source_vectors != target_vectors:
        raise RuntimeError("complete T3 carrier vector sequences are not identical")

    capture_path = release / "default_aot_capture_raw.json.gz"
    capture = load(capture_path)
    nodes = {row["name"]: row for row in capture["capture"]["graphs"][0]["nodes"]}
    source_node = source_t3["exact_aot_endpoint_id"].rsplit(":", 1)[-1]
    target_node = target_t3["exact_aot_endpoint_id"].rsplit(":", 1)[-1]
    # Either orientation is accepted, but the reused target repair must be the
    # upstream endpoint so its value flows through the certified transport.
    path = only_path(nodes, target_node, source_node)

    trajectory_dir = ROOT / "results/coverage/cases/trajectory" / args.cell
    source_matches = []
    for candidate in trajectory_dir.glob("*.json.gz"):
        value = load(candidate)
        if value.get("task_id") == args.source_task:
            source_matches.append((candidate, value))
    if len(source_matches) != 1:
        raise RuntimeError("source T4 artifact is not unique")
    source_path, source = source_matches[0]
    carrier = target_t3["carrier_parameter"]
    token = hashlib.sha256((args.target_task + "\0" + carrier).encode()).hexdigest()[:16]
    output = trajectory_dir / f"{token}.json.gz"
    if output.exists():
        raise RuntimeError(f"target T4 artifact already exists: {output}")

    payload = {
        "schema": "kernel-analyzer-t4-exact-transport-reuse-v1",
        "status": source["status"],
        "task_id": args.target_task,
        "exact_aot_endpoint_id": target_t3["exact_aot_endpoint_id"],
        "carrier_parameter": carrier,
        "steps": 32,
        "steps_completed": 32,
        "directional_projection_checkpoints": source["directional_projection_checkpoints"],
        "directional_projections": source["directional_projections"],
        "gates": dict(source["gates"]),
        "terminal_failure": source.get("terminal_failure"),
        "source_trajectory": {
            "task_id": args.source_task,
            "path": str(source_path.relative_to(ROOT)),
            "file_sha256": digest(source_path),
            "result_sha256": source["result_sha256"],
        },
        "equivalence_certificate": {
            "kind": "EXACT_AOT_VALUE_PRESERVING_TRANSPORT",
            "aot_path": path,
            "aot_targets": [nodes[name]["target"] for name in path[1:]],
            "capture_path": str(capture_path.relative_to(ROOT)),
            "capture_file_sha256": digest(capture_path),
            "all_32_t3_carrier_vectors_identical": True,
            "source_t3_result_sha256": source_t3["result_sha256"],
            "target_t3_result_sha256": target_t3["result_sha256"],
            "proof": (
                "Replacing the upstream target endpoint by its reference and then applying "
                "only the listed exact index/copy operations yields exactly the reference "
                "source endpoint. Therefore both repairs present the same downstream tensor "
                "to the remaining graph at every weight state; deterministic paired updates "
                "and the T4 disposition are identical."
            ),
        },
        "bindings": {
            "source_t3_file_sha256": digest(source_t3_path),
            "target_t3_file_sha256": digest(target_t3_path),
        },
        "claim_boundary": (
            "T4 is disposition-equivalent to the bound source trajectory through an exact "
            "AOT transport path; this does not merge the two endpoint-level T1 dispositions."
        ),
    }
    payload["result_sha256"] = canonical(payload)
    with gzip.open(output, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
    print(json.dumps({"output": str(output), "status": payload["status"], "path": path}))


if __name__ == "__main__":
    main()
