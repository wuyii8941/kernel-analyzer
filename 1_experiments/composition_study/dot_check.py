"""RQ3 check: the tensor-core rule as the tool implements it (kernel_analyzer.reference_eval.tensor_core) against the
device on a tutorial-style Triton matmul (blocked K loop, several BLOCK_K and num_warps), bit for bit, on held-out
random inputs of realistic sizes."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "2_tool/src"))
from kernel_analyzer.reference_eval import tensor_core as TC  # noqa: E402


def matmul_kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def mm(a_ptr, b_ptr, c_ptr, M, N, K, sam, sak, sbk, sbn, scm, scn, IP: tl.constexpr, BM: tl.constexpr,
           BN: tl.constexpr, BK: tl.constexpr):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        rm = pid_m * BM + tl.arange(0, BM)
        rn = pid_n * BN + tl.arange(0, BN)
        rk = tl.arange(0, BK)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, K, BK):
            a = tl.load(a_ptr + rm[:, None] * sam + (k0 + rk)[None, :] * sak)
            b = tl.load(b_ptr + (k0 + rk)[:, None] * sbk + rn[None, :] * sbn)
            acc = tl.dot(a, b, acc, input_precision=IP)
        tl.store(c_ptr + rm[:, None] * scm + rn[None, :] * scn, acc)
    return mm


def run(A, B, fmt, BM, BN, BK, nw):
    import torch
    mm = matmul_kernel()
    td = {"tf32": torch.float32, "bf16": torch.bfloat16, "f16": torch.float16}[fmt]
    a = torch.from_numpy(A).to(td).cuda()
    b = torch.from_numpy(B).to(td).cuda()
    M, K = A.shape
    N = B.shape[1]
    c = torch.empty(M, N, device="cuda", dtype=torch.float32)
    mm[(M // BM, N // BN)](a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0),
                           c.stride(1), IP="tf32", BM=BM, BN=BN, BK=BK, num_warps=nw)
    torch.cuda.synchronize()
    return c.double().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "results/dot_check.json"))
    a = ap.parse_args()
    import torch
    rng = np.random.default_rng(7)
    out = []
    for fmt in ("tf32", "bf16", "f16"):
        for (M, N, K) in ((64, 64, 256), (128, 128, 1024), (64, 64, 4864)):
            for (BM, BN, BK, nw) in ((32, 32, 32, 4), (64, 64, 64, 4), (32, 64, 128, 8)):
                if M % BM or N % BN or K % BK:
                    continue
                A = rng.standard_normal((M, K))
                B = rng.standard_normal((K, N)) * 0.05
                td = {"tf32": torch.float32, "bf16": torch.bfloat16, "f16": torch.float16}[fmt]
                A = torch.from_numpy(A).to(td).double().numpy()
                B = torch.from_numpy(B).to(td).double().numpy()
                dev = run(A, B, fmt, BM, BN, BK, nw)
                model, covered = TC.mma_chain(A, B, np.zeros((M, N)), fmt)
                exact = A @ B          # float64 reference for the shrink only
                same = int((dev == model).sum())
                e = dev - exact
                align = float((e * exact).sum() / (exact * exact).sum())
                e_m = model - exact
                rec = {"fmt": fmt, "M": M, "N": N, "K": K, "BM": BM, "BN": BN, "BK": BK, "num_warps": nw,
                       "elements": M * N, "covered": int(covered.sum()), "bit_identical": same,
                       "alignment_device": align,
                       "alignment_model": float((e_m * exact).sum() / (exact * exact).sum())}
                out.append(rec)
                print(rec, flush=True)
    Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
