"""Measured bias K - G per program (no prediction, no attribution): how much the bias varies between the programs of
one launch, i.e. how far a sample of programs can stand for the whole kernel.
Writes .cache/flash_attention_case/per_program_<case>[_<part>].json"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "2_tool/src"))
sys.path.insert(0, str(HERE))
OUT = ROOT / ".cache/flash_attention_case"

from audit_fa import CASES  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--part", type=int, default=0, help="this process evaluates programs with index %% parts == part")
    ap.add_argument("--parts", type=int, default=1)
    a = ap.parse_args()
    import importlib

    import numpy as np
    import torch
    import triton

    from fa_case import make_inputs
    from kernel_analyzer.reference_eval import bias_predict as BP
    from kernel_analyzer.reference_eval import intervals as iv
    from kernel_analyzer.reference_eval.capture import TritonLaunchRecorder
    from kernel_analyzer.reference_eval.ttir_eval import ST_OK, KernelReferenceEvaluator, NumericMode
    from kernel_analyzer.reference_eval.ttir_parser import parse_ttir
    mod_name, dtype, N, block_n, causal = CASES[a.case]
    kernel = getattr(importlib.import_module(mod_name.split(".")[0]), mod_name.split(".")[1])
    Z, H, D, block_m = 1, 2, 64, 128
    q, k, v = make_inputs(getattr(torch, dtype), Z, H, N, D, 0)
    o = torch.zeros_like(q)
    M = torch.zeros(Z, H, N, device="cuda")
    triton.set_allocator(lambda size, align, stream: torch.empty(size, dtype=torch.int8, device="cuda"))
    grid = (triton.cdiv(N, block_m), Z * H, 1)
    with TritonLaunchRecorder() as R:
        kernel[grid](1.0 / math.sqrt(D), M, Z, H, q, k, v, o, N_CTX=N, HEAD_DIM=D, BLOCK_M=block_m, BLOCK_N=block_n,
                     FP8_OUTPUT=False, STAGE=3 if causal else 1, warp_specialize=False, IS_HOPPER=False,
                     num_stages=2, num_warps=4)
        torch.cuda.synchronize()
    L = R.launches[0]
    module = parse_ttir(L.asm["ttir"])
    module.source_text = L.asm["ttir"]
    ev = KernelReferenceEvaluator(module, mode=NumericMode.NUMERICAL_DIFFERENCE, ttgir=L.asm["ttgir"])
    iv.elementary_bounds = BP._float64_elementary(iv.elementary_bounds)
    rows = []
    for i in range(grid[0] * grid[1]):
        if i % a.parts != a.part:
            continue
        prog = (i % grid[0], i // grid[0], 0)
        res = ev.evaluate(L, programs=[prog])
        row = {"program": prog}
        for buf in res.buffers.values():
            if buf.kind != "f" or not buf.written.any():
                continue
            ok = buf.written & (buf.st == ST_OK) & (buf.actual_after_st == 0) & np.isfinite(buf.actual_after)
            g = 0.5 * (buf.lo[ok] + buf.hi[ok])
            e = buf.actual_after[ok] - g
            row[buf.name] = {"num": float((e * g).sum()), "den": float((g * g).sum()),
                             "alignment": float((e * g).sum() / (g * g).sum())}
        rows.append(row)
        print(prog, {n: round(r["alignment"], 6) for n, r in row.items() if n != "program"}, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    suffix = f"_{a.part}" if a.parts > 1 else ""
    (OUT / f"per_program_{a.case}{suffix}.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
