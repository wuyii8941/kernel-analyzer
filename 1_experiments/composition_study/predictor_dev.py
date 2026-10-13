"""Development set of the bias predictor (kernel_analyzer.reference_eval.bias_predict).

These kernels are used while the predictor is written and debugged; they are NOT part of its validation.  The
validation (predictor_validate.py) uses held-out programs generated after the predictor and its contract table are
frozen.

For each kernel: capture the launch without running it, predict; then run it on the device and compare the measured
error against the exact program value (the predictor's own exact evaluation of the same inputs).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "2_tool/src"))


def kernels():
    import triton
    import triton.language as tl

    @triton.jit
    def dot_k(a_ptr, b_ptr, c_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, BK: tl.constexpr,
              PREC: tl.constexpr):
        rm = tl.arange(0, M)
        rn = tl.arange(0, N)
        acc = tl.zeros((M, N), dtype=tl.float32)
        for k0 in range(0, K, BK):
            rk = k0 + tl.arange(0, BK)
            a = tl.load(a_ptr + rm[:, None] * K + rk[None, :])
            b = tl.load(b_ptr + rk[:, None] * N + rn[None, :])
            acc = tl.dot(a, b, acc, input_precision=PREC)
        tl.store(c_ptr + rm[:, None] * N + rn[None, :], acc)

    @triton.jit
    def loop_acc_k(x_ptr, y_ptr, T: tl.constexpr, B: tl.constexpr):
        r = tl.arange(0, B)
        acc = tl.zeros((B,), dtype=tl.bfloat16)
        for t in range(T):
            acc += tl.load(x_ptr + t * B + r)
        tl.store(y_ptr + r, acc)

    @triton.jit
    def chain_k(x_ptr, y_ptr, z_ptr, B: tl.constexpr):
        r = tl.arange(0, B)
        x = tl.load(x_ptr + r)
        y = tl.load(y_ptr + r)
        z = (x * y + x) * 0.5 - y
        tl.store(z_ptr + r, z)

    return dot_k, loop_acc_k, chain_k


def run(case, kern, grid, make_args, **meta):
    import torch
    from kernel_analyzer.reference_eval import bias_predict as BP
    from kernel_analyzer.reference_eval.capture import TritonLaunchRecorder
    args = make_args()
    with TritonLaunchRecorder(execute=False) as R:
        kern[grid](*args, **meta)
    L = R.launches[0]
    assert not L.executed
    pred = BP.predict(L)
    summ = BP.summarize(pred)
    kern[grid](*args, **meta)                      # the device run (after the prediction)
    torch.cuda.synchronize()
    out = {}
    for name, o in pred["outputs"].items():
        idx = [a.name for a in L.ttir_params()].index(name)
        t = [a for a in args if isinstance(a, torch.Tensor)][idx]
        dev = t.double().cpu().numpy().reshape(-1)
        ok = o["ok"].reshape(-1)
        g = o["exact"].reshape(-1)[ok]
        e = dev[ok] - g
        gg = float((g * g).sum())
        u = BP.ulp_of(g, o["elem"])
        out[name] = {"predicted_alignment": summ["outputs"][name]["alignment"],
                     "predicted_noise_se": summ["outputs"][name]["alignment_noise_se"],
                     "measured_alignment": float((e * g).sum() / gg),
                     "measured_se_iid": float(np.std(e * g) * np.sqrt(g.size) / gg),
                     "predicted_rms_rel": summ["outputs"][name]["rms_relative_error"],
                     "measured_rms_rel": float(np.sqrt(np.mean((e[g != 0] / g[g != 0]) ** 2))),
                     "predicted_mean_ulp": summ["outputs"][name]["mean_ulp"],
                     "measured_mean_ulp": float(np.mean(e / u)),
                     "max_abs_error_over_radius": float(np.max(np.abs(e - o["delta"].reshape(-1)[ok])
                                                               / np.maximum(o["radius"].reshape(-1)[ok], 1e-300)))}
    return {"case": case, "status": summ["status"], "failures": summ["failures"], "sites": summ["sites"],
            "outputs": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE.parents[1] / ".cache/composition_study/predictor_dev.json"))
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    import torch
    dot_k, loop_acc_k, chain_k = kernels()
    g = torch.Generator(device="cuda").manual_seed(7)
    res = []
    cases = {
        "tf32 dot, normal fp32 operands": lambda: run(
            "tf32", dot_k, (1,), lambda: (torch.randn(64, 256, device="cuda", generator=g),
                                          torch.randn(256, 64, device="cuda", generator=g),
                                          torch.empty(64, 64, device="cuda")),
            M=64, N=64, K=256, BK=32, PREC="tf32"),
        "tf32 dot, bf16-valued weights": lambda: run(
            "tf32 bf16w", dot_k, (1,), lambda: (torch.randn(64, 256, device="cuda", generator=g),
                                                torch.randn(256, 64, device="cuda", generator=g).bfloat16().float(),
                                                torch.empty(64, 64, device="cuda")),
            M=64, N=64, K=256, BK=32, PREC="tf32"),
        "bf16 dot": lambda: run(
            "bf16", dot_k, (1,), lambda: (torch.randn(64, 256, device="cuda", generator=g).bfloat16(),
                                          torch.randn(256, 64, device="cuda", generator=g).bfloat16(),
                                          torch.empty(64, 64, device="cuda")),
            M=64, N=64, K=256, BK=32, PREC="tf32"),
        "ieee dot": lambda: run(
            "ieee", dot_k, (1,), lambda: (torch.randn(64, 256, device="cuda", generator=g),
                                          torch.randn(256, 64, device="cuda", generator=g),
                                          torch.empty(64, 64, device="cuda")),
            M=64, N=64, K=256, BK=32, PREC="ieee"),
        "bf16 loop accumulation": lambda: run(
            "loop", loop_acc_k, (1,), lambda: (torch.rand(64 * 1024, device="cuda", generator=g).bfloat16(),
                                               torch.empty(1024, device="cuda", dtype=torch.bfloat16)),
            T=64, B=1024),
        "f32 chain": lambda: run(
            "chain", chain_k, (1,), lambda: (torch.randn(4096, device="cuda", generator=g),
                                             torch.randn(4096, device="cuda", generator=g),
                                             torch.empty(4096, device="cuda")), B=4096),
    }
    for name, fn in cases.items():
        if a.only and a.only not in name:
            continue
        try:
            r = fn()
        except Exception as e:
            import traceback
            r = {"case": name, "error": f"{type(e).__name__}: {e}", "where": traceback.format_exc().splitlines()[-4:]}
        r["case"] = name
        res.append(r)
        print(json.dumps(r, ensure_ascii=False, default=str)[:900], flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
