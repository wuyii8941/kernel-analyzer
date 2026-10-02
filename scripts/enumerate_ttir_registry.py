"""Enumerate the operations registered in the locked Triton build.

Candidate names are the ``dialect.op`` string literals in ``libtriton.so``.
Each candidate is parsed in generic form inside a context with Triton's
dialects loaded: MLIR reports "unregistered operation" for names that are not
operations (attribute names, other strings); any other diagnostic means the
operation is registered.  The result is the version-locked registry that the
mapping table must cover exhaustively.

    python scripts/enumerate_ttir_registry.py
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "reference_eval" / "ttir_op_registry.json"
DIALECTS = ("tt", "arith", "math", "scf", "cf", "gpu", "ub")


def _strings(path: str) -> set:
    data = Path(path).read_bytes()
    pattern = re.compile(rb"(?<![A-Za-z0-9_.])((?:" + b"|".join(d.encode() for d in DIALECTS) + rb")\.[a-z][a-z0-9_.]*)\x00")
    return {m.group(1).decode() for m in pattern.finditer(data)}


def _probe(ctx, ir, name: str, workdir: str) -> str:
    path = os.path.join(workdir, "probe.mlir")
    Path(path).write_text('module {\n  "%s"() : () -> ()\n}\n' % name)
    log = os.path.join(workdir, "stderr.txt")
    saved = os.dup(2)
    with open(log, "w") as handle:
        os.dup2(handle.fileno(), 2)
        try:
            ir.parse_mlir_module(path, ctx)
            outcome = "parsed"
        except Exception:
            outcome = "error"
        finally:
            os.dup2(saved, 2)
            os.close(saved)
    message = Path(log).read_text()
    if "unregistered operation" in message:
        return "unregistered"
    return "registered" if outcome == "parsed" or message else "unknown"


def main():
    import argparse

    import triton

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    out = parser.parse_args().out
    from triton._C.libtriton import ir

    lib = sorted(glob.glob(os.path.join(os.path.dirname(triton.__file__), "_C", "libtriton*.so")))[0]
    candidates = sorted(_strings(lib))
    ctx = ir.context()
    ir.load_dialects(ctx)
    registered, rejected_names = [], []
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as workdir:
        for name in candidates:
            status = _probe(ctx, ir, name, workdir)
            (registered if status == "registered" else rejected_names).append(name)
    by_dialect = {d: [n for n in registered if n.split(".")[0] == d] for d in DIALECTS}
    record = {
        "schema": "kernel-analyzer-ttir-op-registry-v1",
        "triton": triton.__version__,
        "libtriton": os.path.basename(lib),
        "libtriton_sha256": hashlib.sha256(Path(lib).read_bytes()).hexdigest(),
        "method": "generic-form parse probe of dialect.op string literals in libtriton",
        "counts": {d: len(v) for d, v in by_dialect.items()},
        "operations": by_dialect,
        "non_operation_strings": rejected_names,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record["counts"]), "->", out)


if __name__ == "__main__":
    main()
