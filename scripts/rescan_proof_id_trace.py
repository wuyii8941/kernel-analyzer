#!/usr/bin/env python3
"""Re-index an existing proof-ID artifact against a completed trace tree."""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import re
from typing import Any


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def load(path: Path) -> dict[str, Any]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def write(path: Path, payload: dict[str, Any]) -> None:
    encoded = (json.dumps(payload, sort_keys=True) + "\n").encode()
    if path.suffix == ".gz":
        with gzip.open(path, "wb", compresslevel=6) as handle:
            handle.write(encoded)
    else:
        path.write_bytes(encoded)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = load(args.artifact)
    all_tags = {
        row["tagged_fx_name"]
        for graph in payload["proof_graphs"]
        for row in graph["rows"]
    }
    occurrences: Counter[str] = Counter()
    files = []
    for path in sorted(args.trace_dir.rglob("*")):
        if not path.is_file():
            continue
        data = path.read_bytes()
        text = data.decode(errors="ignore")
        present = sorted(
            set(re.findall(r"ka_[fb]_\d{4,}_[A-Za-z0-9_]+", text)) & all_tags
        )
        occurrences.update(present)
        files.append({
            "path": str(path.relative_to(args.trace_dir)),
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "proof_tags_present": len(present),
        })
    payload["trace_files"] = files
    payload["trace_dir"] = str(args.trace_dir.resolve())
    payload["proof_tag_summary"] = {
        "aot_call_function_nodes": len(all_tags),
        "tags_observed_in_inductor_trace": len(occurrences),
        "tags_not_observed": len(all_tags) - len(occurrences),
    }
    payload.pop("result_sha256", None)
    payload["result_sha256"] = digest(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write(args.output, payload)
    print(json.dumps({
        "output": str(args.output),
        "trace_files": len(files),
        "proof_tag_summary": payload["proof_tag_summary"],
        "result_sha256": payload["result_sha256"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
