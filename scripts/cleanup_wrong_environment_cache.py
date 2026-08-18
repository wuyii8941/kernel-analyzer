#!/usr/bin/env python3
"""Remove the identified PyTorch-2.10 Mamba compile from the nightly cache."""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path


ROOT = Path("/data1/tzh/cache/torchinductor/frozen/mamba_seq128_r1")
OUTPUT = Path("/data1/tzh/kernel-analyzer/results/coverage/wrong_environment_cache_cleanup.json")
START = datetime.fromisoformat("2026-08-11T01:35:00").timestamp()
STOP = datetime.fromisoformat("2026-08-11T02:00:00").timestamp()
REQUIRED = {
    "zx/czx2mjbxl566ihcv4by35c3ahcmhmm5evdftww2ayjssaps4t7rs.py":
        "fc789b4f1753168cb7cdb9ba355013a66dade8c46cb05d2bfa817ae90cb6908a",
    "yf/cyfqus7ilaxqag3midwq5svvu3prdpglp5bdkoullwgw54yp3cgn.py":
        "2159c64caa03c4cc45b278f7918f3a7f2aca0a4bbeea946180ff3bbd970ce5cb",
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if ROOT.resolve() != Path("/data1/tzh/cache/torchinductor/frozen/mamba_seq128_r1"):
        raise RuntimeError("cleanup root drift")
    for relative, expected in REQUIRED.items():
        path = ROOT / relative
        if not path.is_file() or sha(path) != expected:
            raise RuntimeError("wrong-environment identity not present: " + relative)
    rows = []
    for path in sorted(ROOT.rglob("*")):
        if path.is_file() and START <= path.stat().st_mtime < STOP:
            rows.append({
                "path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
                "sha256": sha(path),
            })
    if not {row["path"] for row in rows}.issuperset(REQUIRED):
        raise RuntimeError("cleanup window does not contain both identified wrappers")
    payload = {
        "schema": "kernel-analyzer-wrong-environment-cache-cleanup-v1",
        "status": "FILES_MANIFESTED_BEFORE_DELETE",
        "cause": "PyTorch 2.10 compile attempted against PyTorch nightly frozen release",
        "root": str(ROOT), "window": [START, STOP], "files": rows,
        "file_count": len(rows), "bytes_removed": sum(row["bytes"] for row in rows),
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    for row in rows:
        (ROOT / row["path"]).unlink()
    for path in sorted((p for p in ROOT.rglob("*") if p.is_dir()), reverse=True):
        try:
            path.rmdir()
        except OSError:
            pass
    payload["status"] = "COMPLETE_RECOVERABLE_ONLY_FROM_RECOMPILATION"
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"files": len(rows), "bytes_removed": payload["bytes_removed"]}))


if __name__ == "__main__":
    main()
