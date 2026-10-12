"""RQ1 step 1: compile every non-structural F1 instance for sm_86 and record how Triton 3.6.0 lowers it.

Per instance: compile status (the first error line on failure, with the stage that raised it), and the PTX opcodes of
the kernel minus those of an empty carrier kernel of the same signature (the lowering of the instance itself).
Output: results/lowering.json.  Compilation needs no GPU; workers run in parallel processes.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
import sys
import traceback
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import space  # noqa: E402
import ttir_build as tb  # noqa: E402

WORK = HERE.parents[1] / ".cache/composition_study/ttir"

_ADDR = re.compile(r"^(ld|st|mov|cvta|add\.s64|mul\.wide|setp\.lt\.s32|mad\.lo|and\.b32|shl|or\.b32|bar|ret|@|selp\.b32"
                   r"|mov\.u32|add\.s32|mul\.lo\.s32)")


def ptx_opcodes(ptx: str) -> collections.Counter:
    c = collections.Counter()
    for line in ptx.splitlines():
        s = line.split("//")[0].strip()
        if not s or s.startswith((".", "$", "{", "}")) or s.endswith(":"):
            continue
        tok = s.split()
        if tok[0].startswith("@"):
            tok = tok[1:]
        if tok:
            c[tok[0].rstrip(";")] += 1
    return c


def compile_text(text: str, options: dict):
    import triton
    from triton.backends.compiler import GPUTarget
    WORK.mkdir(parents=True, exist_ok=True)
    path = WORK / f"m_{hashlib.sha256((text + repr(sorted(options.items()))).encode()).hexdigest()[:20]}.ttir"
    path.write_text(text)
    return triton.compile(str(path), target=GPUTarget("cuda", 86, 32), options=options)


def carrier(ins, outs, launch):
    """The same loads and stores with the operation replaced by an exact carrier (for the opcode difference)."""
    if launch == "dot":
        return None
    return tb.elementwise_module(ins, [], [], name="carrier")


def one(inst):
    rec = {"op": inst[0], "attr": inst[1], "sig": inst[2], "kind": space.kind(inst)}
    try:
        built = tb.module_for(inst)
    except tb.Unbuildable as e:
        rec.update(status="separate template", reason=str(e))
        return rec
    if built is None:
        rec.update(status="structural")
        return rec
    text, ins, outs, launch = built
    rec["ttir_sha256"] = hashlib.sha256(text.encode()).hexdigest()
    try:
        k = compile_text(text, {"num_warps": 4})
    except Exception as e:  # the compiler's own rejection is a result, not a harness error
        msg = "".join(traceback.format_exception_only(type(e), e)).strip().splitlines()
        rec.update(status="not compiled", error_type=type(e).__name__,
                   error=[l for l in msg if l.strip()][:3])
        return rec
    ops = ptx_opcodes(k.asm["ptx"])
    base = collections.Counter()
    ct = carrier(ins, outs, launch)
    if ct is not None:
        try:
            base = ptx_opcodes(compile_text(ct, {"num_warps": 4}).asm["ptx"])
        except Exception:
            base = collections.Counter()
    diff = {o: n - base.get(o, 0) for o, n in ops.items() if n - base.get(o, 0) > 0}
    rec.update(status="compiled", ptx_opcodes=dict(sorted(diff.items())),
               libdevice_calls=sorted(set(re.findall(r"__nv_\w+", k.asm["llir"]))))
    return rec


def isolated(inst, timeout=300):
    """Compile in a child process: the MLIR parser aborts the process on some inputs."""
    cmd = [sys.executable, "-I", str(Path(__file__).resolve()), "--one", json.dumps(list(inst))]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"op": inst[0], "attr": inst[1], "sig": inst[2], "kind": space.kind(inst), "status": "not compiled",
                "error_type": "timeout", "error": [f"compile exceeded {timeout} s"]}
    lines = [l for l in p.stdout.splitlines() if l.startswith("{")]
    if p.returncode == 0 and lines:
        return json.loads(lines[-1])
    err = [l for l in p.stderr.splitlines() if l.strip()]
    keep = [l for l in err if "error" in l.lower() or "Assertion" in l or "LLVM ERROR" in l] or err[-3:]
    return {"op": inst[0], "attr": inst[1], "sig": inst[2], "kind": space.kind(inst), "status": "not compiled",
            "error_type": f"process exit {p.returncode}", "error": keep[:4]}


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--one":
        print(json.dumps(one(tuple(json.loads(sys.argv[2])))))
        return
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "results/lowering.json"))
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    insts = [i for i in space.F1 if (not a.only or a.only in i[0])]
    if a.limit:
        insts = insts[:a.limit]
    with ThreadPoolExecutor(a.workers) as ex:
        recs = list(ex.map(isolated, insts))
    st = collections.Counter(r["status"] for r in recs)
    out = {"schema": "composition-study-lowering-v1", "triton": __import__("triton").__version__, "target": "cuda sm_86",
           "options": {"num_warps": 4}, "instances": len(recs), "status": dict(st), "records": recs}
    Path(a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(dict(st))


if __name__ == "__main__":
    main()
