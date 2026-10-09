#!/usr/bin/env python3
"""W0 observed inventory (rc3 04): operations that actually appear in compilations of the built official Triton.

    stage 1 (run in the official env, GPU): run official tutorials and a subset of official unit tests with
             TRITON_KERNEL_DUMP=1 / TRITON_DUMP_DIR=<dump>;
    stage 2 (this script, ``collect``): recompile every dumped TTIR for further NVIDIA targets without running it
             (sm_90, sm_100), parse the operation names of every TTIR / TTGIR, and write the observed inventory.

Not the reviewer's corpus B: official tests and tutorials only; observation is not exhaustive by nature.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import sys
from pathlib import Path

OP = re.compile(r'(?:^|=\s*|\s)"?([a-z_][a-z_0-9]*\.[a-z_][a-z_0-9.]*)"?(?=[\s(<{:]|$)', re.M)
DIALECTS = {"tt", "ttg", "ttng", "amdg", "nvws", "nvg", "gluon", "tti", "proton", "proton_gpu", "arith", "math", "scf",
            "cf", "gpu", "llvm", "nvvm", "rocdl", "ub", "builtin"}


def ops_in(text: str) -> set:
    out = set()
    for line in text.splitlines():
        line = line.split("loc(")[0]
        for m in OP.finditer(line):
            name = m.group(1)
            if re.match(r"\s*=", line[m.end():]):  # an attribute key (tt.divisibility = 16, ttg.target = ...)
                continue
            if name.split(".")[0] in DIALECTS and not name.endswith((".h", ".py")):
                out.add(name)
    return out


def _recompile(job):
    """(target, TTIR path) -> (target, ops of the TTGIR, error class or None); runs in a worker process."""
    t, path = job
    from triton.backends.compiler import GPUTarget
    import triton
    backend, arch = t.split(":")
    try:
        if backend == "hip":   # DSL v2 increment 12: the official AMD stages up to TTGIR (scripts/dsl_v2/amd_stages.py)
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from amd_stages import amd_ttgir
            return t, tuple(sorted(ops_in(amd_ttgir(path, arch)))), None
        ck = triton.compile(path, target=GPUTarget(backend, int(arch) if arch.isdigit() else arch,
                                                   32 if backend == "cuda" else 64))
    except Exception as exc:  # noqa: BLE001 -- recorded: a kernel that does not compile for that target
        return t, (), type(exc).__name__
    return t, tuple(sorted(ops_in(ck.asm.get("ttgir", "")))), None


def collect(dump: Path, targets: list, out: Path, source_files: list, workers: int = 16):
    import triton
    src = {}
    for f in source_files:
        d = json.loads(Path(f).read_text())
        for r in d["records"]:
            src.setdefault(r["id"], r["contract_hash"])
        commit, profile = d["source_commit"], d["profile_id"]
    seen = collections.defaultdict(lambda: collections.Counter())
    ttirs = {}
    for p in sorted(dump.rglob("*")):
        if p.suffix in (".ttir", ".ttgir") and p.is_file():
            text = p.read_text(errors="replace")
            for o in ops_in(text):
                seen[o][f"sm_86:{p.suffix[1:]}"] += 1
            if p.suffix == ".ttir":
                ttirs.setdefault(hashlib.sha256(text.encode()).hexdigest(), p)
    failures = collections.Counter()
    jobs = [(t, str(p)) for t in targets for p in ttirs.values()]
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for t, ops, err in pool.map(_recompile, jobs, chunksize=4):
            if err:
                failures[f"{t}:{err}"] += 1
                continue
            for o in ops:
                seen[o][f"{t}:ttgir"] += 1
    records = [{"id": o, "contract_hash": src.get(o, "not-in-source"), "observations": dict(c)}
               for o, c in sorted(seen.items())]
    doc = {"kind": "observed", "profile_id": profile, "source_commit": commit,
           "producer": "scripts/dsl_v2/w0_observed.py (dumps of official tutorials and unit tests on sm_86, recompiled for "
                       + ", ".join(targets) + ")",
           "receipt": {"dump_dir": str(dump), "unique_ttir": len(ttirs), "recompile_failures": dict(failures),
                       "triton": triton.__version__},
           "complete_for_profile": False,
           "completeness_note": "observation of official tutorials and official unit tests (compile_only, core, "
                                "tensor_descriptor, conversions, random, standard, libdevice); not corpus B",
           "records": records}
    out.write_text(json.dumps(doc, indent=1) + "\n")
    print(len(ttirs), "unique TTIR;", len(records), "observed ops;", dict(failures))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", type=Path, required=True)
    ap.add_argument("--targets", nargs="*", default=["cuda:90", "cuda:100"])
    ap.add_argument("--sources", nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    if a.out.exists():
        sys.exit(f"refusing to overwrite {a.out}")
    collect(a.dump, a.targets, a.out, a.sources, a.workers)


if __name__ == "__main__":
    main()
