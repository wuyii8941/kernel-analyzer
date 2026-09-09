#!/usr/bin/env python3
"""Recover protocol-pinned Python text from prior content-addressed snapshots.

This never reconstructs source from a diff and never substitutes current code.
Every recovered text must already exist in another ``source_snapshot.json``
and match the exact SHA-256 recorded before the capture started.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _sha_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def recover(protocol_path: Path, search_root: Path, output: Path, provenance_output: Path) -> dict:
    if output.exists() or provenance_output.exists():
        raise ValueError("recovery outputs must be new")
    protocol = json.loads(protocol_path.read_text())
    expected = {
        name: digest
        for name, digest in protocol.get("source_sha256", {}).items()
        if Path(name).suffix == ".py"
    }
    if not expected:
        raise ValueError("protocol has no pinned Python sources")
    candidates: dict[str, list[tuple[str, str]]] = {digest: [] for digest in set(expected.values())}
    for snapshot_path in search_root.rglob("source_snapshot.json"):
        if snapshot_path.resolve() == output.resolve():
            continue
        try:
            snapshot = json.loads(snapshot_path.read_text())
        except (OSError, ValueError, TypeError):
            continue
        if not isinstance(snapshot, dict):
            continue
        for original_name, text in snapshot.items():
            if not isinstance(text, str):
                continue
            digest = _sha_text(text)
            if digest in candidates:
                candidates[digest].append((str(snapshot_path), original_name))

    recovered = {}
    evidence = []
    for name, digest in expected.items():
        current = Path(name)
        if current.exists() and hashlib.sha256(current.read_bytes()).hexdigest() == digest:
            text = current.read_text()
            source_snapshot = None
            stored_name = str(current)
            recovery_method = "CURRENT_FILE_STILL_MATCHES_PROTOCOL_DIGEST"
        else:
            matches = candidates.get(digest, [])
            if not matches:
                raise ValueError(f"no current or prior snapshot contains required source digest: {name} {digest}")
            source_snapshot, stored_name = sorted(matches)[0]
            text = json.loads(Path(source_snapshot).read_text())[stored_name]
            recovery_method = "PRIOR_CONTENT_ADDRESSED_SOURCE_SNAPSHOT"
        if _sha_text(text) != digest:
            raise AssertionError("content-addressed lookup changed while reading")
        recovered[name] = text
        evidence.append(
            {
                "required_path": name,
                "required_sha256": digest,
                "recovered_from_snapshot": source_snapshot,
                "stored_path_in_snapshot": stored_name,
                "recovery_method": recovery_method,
            }
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(recovered, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    report = {
        "schema": "content-addressed-source-snapshot-recovery-v1",
        "status": "RECOVERED_EXACT_PROTOCOL_PINNED_SOURCE_TEXT",
        "protocol": str(protocol_path),
        "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        "snapshot": str(output),
        "snapshot_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "sources": evidence,
        "limitations": (
            "This proves byte identity with protocol-pinned source and a prior saved snapshot; "
            "it does not independently prove GPU execution identity or reference semantics."
        ),
    }
    with provenance_output.open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--search-root", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provenance-output", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.output, args.provenance_output):
        if not path.resolve().is_relative_to(Path("/data1/tzh")):
            parser.error("outputs must remain below /data1/tzh")
    report = recover(args.protocol, args.search_root, args.output, args.provenance_output)
    print(json.dumps({"status": report["status"], "sources": len(report["sources"])}))


if __name__ == "__main__":
    main()
