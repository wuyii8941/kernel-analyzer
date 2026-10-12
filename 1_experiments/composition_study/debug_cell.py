"""Debug helper: run one cell and print the elements where the emulator and the device differ (options and block
from the CELL_OPTS environment variable as JSON, else the defaults)."""
import hashlib
import json
import sys

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "/data1/tzh/kernel-analyzer/2_tool/src")
import catalog as C  # noqa: E402
import cells as CL  # noqa: E402
import forms as FM  # noqa: E402
import primitives as P  # noqa: E402
from kernel_analyzer.reference_eval import emulate as E  # noqa: E402
from kernel_analyzer.reference_eval.capture import TritonLaunchRecorder  # noqa: E402

pid, form, param = sys.argv[1], sys.argv[2], sys.argv[3]
limit = int(sys.argv[4]) if len(sys.argv) > 4 else 8
prims = {p.pid: p for p in P.all_prims()}
p = prims[pid]
import os
full = json.loads(os.environ["CELL_OPTS"]) if os.environ.get("CELL_OPTS") else dict(CL.DEFAULT_OPTIONS)
opts = dict(full)
block = opts.pop("block", 128)
cell = FM.build(p, form, param, block=block)
k = C.compile_text(cell.text, opts)
seed = CL.cell_id(pid, form, param, full)
rng = np.random.default_rng(int.from_bytes(hashlib.sha256(seed.encode()).digest()[:8], "little"))
inp = CL.make_inputs(cell, p, rng)
args = [inp[n] for n, _, _ in cell.inputs] + [inp[n] for n, _, _ in cell.outputs]
with TritonLaunchRecorder() as R:
    k[(cell.grid, 1, 1)](*args)
res, emu = E.emulate(R.launches[0], ungrouped=[], swapped=[])
ins = {n: inp[n].double().cpu().numpy() if inp[n].is_floating_point() else inp[n].cpu().numpy() for n, _, _ in cell.inputs}
shown = 0
for name, o in E._outputs(res, R.launches[0]).items():
    written, lo, hi, st, actual, elem, bits = o
    m = E._compare(*o)
    bad = np.where(m["emulated"] & ~m["bit_equal"] & ~m["nan_both"])[0]
    print(name, elem, "mismatches", len(bad), "of", int(m["emulated"].sum()))
    for i in bad[:limit]:
        print("  idx", i, "inputs", {n: v[i] for n, v in ins.items() if i < len(v)}, "emulated", lo[i], "device", actual[i])
for ptr, b in res.buffers.items():
    if b.kind == "i" and b.written.any():
        point = b.written & (b.st == 0)
        bad = np.where(point & (b.lo != b.actual_after))[0]
        print(b.name, "int mismatches", len(bad))
        for i in bad[:limit]:
            print("  idx", i, "inputs", {n: v[i] for n, v in ins.items() if i < len(v)}, "emulated", b.lo[i], "device", b.actual_after[i])
print("ptx:", sorted({o for o in __import__("compile_sweep").ptx_opcodes(k.asm["ptx"]) if any(s in o for s in ("rz", "rm", "rp", "rn", "ftz", "rpi", "rmi", "cvt", "add.f", "div", "sqrt", "rcp"))})[:20])
if "--trace" in sys.argv:
    print("trace:", dict(emu.trace))
    print("sass choices:", emu.sass_choices, "ptx choices:", emu.ptx_choices, "ungrouped:", emu.ungrouped)
