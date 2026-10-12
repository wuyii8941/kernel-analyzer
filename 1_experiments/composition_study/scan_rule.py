"""RQ3: the combination order of tt.scan on the locked build (Triton 3.6.0, sm_86), for a 1-D scan over a blocked
layout, checked bit for bit against the device.

The order follows the lowering (lib/Conversion/TritonGPUToLLVM/ScanOpToLLVM.cpp, read from the official source and
then tested here -- the device decides):
  1. inside a thread, each chunk of ``spt`` contiguous elements is scanned sequentially: acc = acc (+) x;
  2. the last element of each chunk takes a Hillis-Steele scan over the lanes: for d = 1, 2, 4, ...:
     L = shfl_up(L, d) (+) L on lanes >= d;
  3a. one warp along the axis: chunk r > 0 starts from the total of the previous chunks: L = T (+) L; every other
      element of the chunk gets the inclusive value of the previous lane (lane 0: T) on its left: x = P (+) x;
  3b. several warps: the warp totals of each chunk go through shared memory and are folded left to right, starting
      from the total of all previous chunks; warp w's last elements get that prefix on the left, then step 3a's update
      of the other elements with the previous lane's value.
(+) is the combine region evaluated in the element type with round to nearest.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "2_tool/src"))


def rn(x, fmt):
    import torch
    if fmt == "f32":
        return np.float32(x).astype(np.float64)
    if fmt == "f16":
        return np.float16(x).astype(np.float64)
    if fmt == "bf16":
        return torch.from_numpy(np.asarray(x, np.float64)).float().to(torch.bfloat16).double().numpy()
    raise ValueError(fmt)


def combine(a, b, op, fmt):
    """RN(a op b) in fmt, elementwise (float64 holds every product / sum of two fp32 values exactly, so one rounding
    to fmt is the IEEE result; for bf16 the fp32 intermediate is exact as well)."""
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    if op == "add":
        v = a + b
    elif op == "mul":
        v = a * b
    elif op == "max":
        v = np.maximum(a, b)
    else:
        raise ValueError(op)
    if fmt == "bf16":
        v = np.float32(v).astype(np.float64)       # exact here; then one rounding to bf16
    return rn(v, fmt)


def scan_model(x, spt, warps, op, fmt, lanes=32):
    """Inclusive scan of the 1-D array x (length n = chunks * spt * lanes * warps) under the lowering order."""
    n = len(x)
    cover = spt * lanes * warps
    assert n % cover == 0, (n, cover)
    R = n // cover
    v = np.array(x, np.float64).reshape(R, warps, lanes, spt)     # element e = ((r*W + w)*32 + l)*spt + j
    # 1. sequential inside each chunk
    for j in range(1, spt):
        v[..., j] = combine(v[..., j - 1], v[..., j], op, fmt)
    # 2. Hillis-Steele over lanes on the last element of each chunk
    L = v[..., spt - 1].copy()                                     # (R, W, lanes)
    d = 1
    while d <= lanes // 2:
        sh = np.concatenate([np.zeros_like(L[..., :d]), L[..., :-d]], axis=-1)
        new = combine(sh, L, op, fmt)
        L = np.where(np.arange(lanes) >= d, new, L)
        d *= 2
    out = v.copy()
    if warps == 1:
        T = None
        for r in range(R):
            Lr = L[r, 0]
            if r > 0:
                Lr = combine(np.full(lanes, T), Lr, op, fmt)
            prev = np.concatenate([[np.nan], Lr[:-1]])             # shfl_up by one lane
            acc_for_lane0 = T if r > 0 else None
            for j in range(spt - 1):
                col = v[r, 0, :, j]
                if r == 0:
                    upd = combine(prev[1:], col[1:], op, fmt)
                    out[r, 0, :, j] = np.concatenate([col[:1], upd])
                else:
                    left = np.concatenate([[acc_for_lane0], prev[1:]])
                    out[r, 0, :, j] = combine(left, col, op, fmt)
            out[r, 0, :, spt - 1] = Lr
            T = Lr[-1]
        return out.reshape(n)
    # 3b. several warps
    acc = None
    for r in range(R):
        totals = L[r, :, lanes - 1]                                # warp totals of chunk r
        masked = [None] * warps                                    # prefix of the warps before w
        a = acc
        if a is None:
            a = totals[0]
            for w in range(warps):
                masked[w] = a if w >= 1 else totals[0]
            for i in range(1, warps):
                a = combine(a, totals[i], op, fmt)
                for w in range(warps):
                    if w >= i + 1:
                        masked[w] = a
        else:
            m_all = a
            for w in range(warps):
                masked[w] = m_all
            for i in range(warps):
                a = combine(a, totals[i], op, fmt)
                for w in range(warps):
                    if w >= i + 1:
                        masked[w] = a
        for w in range(warps):
            Lw = L[r, w]
            if r == 0 and w == 0:
                temp = Lw
            else:
                temp = combine(np.full(lanes, masked[w]), Lw, op, fmt)
            prev = np.concatenate([[masked[w]], temp[:-1]])        # lane 0 takes the warp prefix
            for j in range(spt - 1):
                col = v[r, w, :, j]
                upd = combine(prev, col, op, fmt)
                if r == 0 and w == 0:
                    upd[0] = col[0]
                out[r, w, :, j] = upd
            out[r, w, :, spt - 1] = temp
        acc = a
    return out.reshape(n)


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def add_(a, b):
        return a + b

    @triton.jit
    def mul_(a, b):
        return a * b

    @triton.jit
    def scan_k(x, o, N: tl.constexpr, OP: tl.constexpr):
        i = tl.arange(0, N)
        v = tl.load(x + i)
        if OP == 0:
            r = tl.associative_scan(v, 0, add_)
        else:
            r = tl.associative_scan(v, 0, mul_)
        tl.store(o + i, r)
    return scan_k


def layout_of(k):
    from kernel_analyzer.reference_eval.layouts import parse_layouts
    ttgir = k.asm["ttgir"]
    lays = parse_layouts(ttgir)
    m = re.search(r'"tt\.scan"\(%[\w#]+\).*?:\s*\(tensor<\d+x\w+,\s*(#\w+)>\)', ttgir, re.S)
    lay = lays[m.group(1)]
    return lay["sizePerThread"][0], lay["threadsPerWarp"][0], lay["warpsPerCTA"][0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "results/scan_rule.json"))
    a = ap.parse_args()
    import torch
    K = kernel()
    rng = np.random.default_rng(31)
    out = []
    for fmt, td in (("f32", torch.float32), ("f16", torch.float16), ("bf16", torch.bfloat16)):
        for op_name, op_code in (("add", 0), ("mul", 1)):
            for N in (128, 256, 1024, 4096):
                for nw in (1, 2, 4, 8):
                    if N < 32 * nw:
                        continue
                    if op_name == "mul":
                        x = 1 + rng.standard_normal(N) * 0.01
                    else:
                        x = rng.standard_normal(N) * 2.0 ** rng.integers(-6, 6, N)
                    xt = torch.from_numpy(x).to(td).cuda()
                    o = torch.empty_like(xt)
                    k = K[(1,)](xt, o, N=N, OP=op_code, num_warps=nw)
                    torch.cuda.synchronize()
                    spt, tpw, wpc = layout_of(k)
                    xv = xt.double().cpu().numpy()
                    dev = o.double().cpu().numpy()
                    rec = {"fmt": fmt, "op": op_name, "N": N, "num_warps": nw, "sizePerThread": spt,
                           "threadsPerWarp": tpw, "warpsPerCTA": wpc}
                    if N % (spt * tpw * wpc) or tpw != 32:
                        rec["status"] = "layout replicates data along the axis (outside the model)"
                    else:
                        model = scan_model(xv, spt, wpc, op_name, fmt)
                        seq = np.empty(N)
                        acc = xv[0]
                        seq[0] = acc
                        for i in range(1, N):
                            acc = combine(acc, xv[i], op_name, fmt)
                            seq[i] = acc
                        rec.update(bit_identical=int((model == dev).sum()), elements=N,
                                   sequential_order_identical=int((seq == dev).sum()))
                    out.append(rec)
                    print(rec, flush=True)
    Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
