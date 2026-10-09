#!/usr/bin/env python3
"""DSL v2 increment 12: write the gfx9xx TTGIR of every unique dumped official TTIR (official AMD stages up to TTGIR,
scripts/dsl_v2/amd_stages.py) as <out>/<arch>/<sha16>/kernel.ttgir, for name coverage
(scripts/dsl_v2/main_ttir_coverage.py --glob "*/*.ttgir") and for syntax samples.  Nothing runs on an AMD device.

    python scripts/dsl_v2/amd_ttgir_dump.py --dump .cache/dsl_v2/w0_dump --archs gfx942 gfx950 --out DIR
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _one(job):
    arch, path, out = job
    from amd_stages import amd_ttgir
    try:
        text = amd_ttgir(path, arch)
    except Exception as exc:  # noqa: BLE001 -- recorded
        return arch, path, f"{type(exc).__name__}: {exc}"[:200]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(text)
    return arch, path, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", type=Path, required=True)
    ap.add_argument("--archs", nargs="+", default=["gfx942", "gfx950"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    ttirs = {}
    for p in sorted(a.dump.rglob("*.ttir")):
        ttirs.setdefault(hashlib.sha256(p.read_bytes()).hexdigest(), p)
    jobs = [(arch, str(p), str(a.out / arch / d[:16] / "kernel.ttgir")) for arch in a.archs for d, p in ttirs.items()]
    from concurrent.futures import ProcessPoolExecutor
    failures = []
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for arch, path, err in pool.map(_one, jobs, chunksize=4):
            if err:
                failures.append({"arch": arch, "ttir": path, "error": err})
    (a.out / "receipt.json").write_text(json.dumps({"unique_ttir": len(ttirs), "archs": a.archs,
                                                    "failures": failures}, indent=1) + "\n")
    print(len(ttirs), "unique TTIR;", len(failures), "failures")


if __name__ == "__main__":
    main()
