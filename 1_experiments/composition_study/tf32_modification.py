"""A real modification of the round-2 TF32 kernel, predicted before it runs (predictor_protocol.md, part B).

Kernel: the Triton tutorial matmul of the round-2 study (fp32 inputs, tl.dot at the default tf32 precision), with
one change selected by MODE:

  V0  MODE 0, Qwen2.5-0.5B weights (bf16 checkpoint values)   -- the development case (round 2, tf32_attribution.py)
  V1  MODE 1, Qwen weights: the activation tile is rounded to tf32 to nearest (cvt.rna.tf32.f32) before tl.dot
  V2  MODE 0, random fp32 weights (normal, the layer's standard deviation): both operands are cut by the tensor core
  V3  MODE 2, random fp32 weights: both tiles rounded to nearest before tl.dot

phase predict: capture every launch without running it and predict the alignment <K - G, G> / <G, G> on a fixed
               subset of output tiles (bias_predict); write the predictions and their SHA-256.
phase measure: run the kernels on the device; the exact G is the float64 product of the same operands.
phase verdict: the declared rule per (variant, stratum), units as samples.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "2_tool/src"))
sys.path.insert(0, str(ROOT / "1_experiments/tolerance_study"))
CACHE = ROOT / ".cache/composition_study/predictor"
STRATA = [(layer, mat) for layer in (2, 12, 22) for mat in ("q_proj", "down_proj")]
UNITS = range(4)
VARIANTS = {"V0": (0, "qwen"), "V1": (1, "qwen"), "V2": (0, "random"), "V3": (2, "random")}
PROGRAMS = list(range(0, 28, 4))          # 7 of the 28 output tiles (M = 128, N = 896, 64 x 64 tiles)
Z, REL, SECOND_ORDER = 3.0, 0.1, 2.0


def kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def matmul_mod(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                   BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
                   GROUP_SIZE_M: tl.constexpr, MODE: tl.constexpr):
        pid = tl.program_id(axis=0)
        num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m
        offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
            if MODE >= 1:
                a = tl.inline_asm_elementwise("cvt.rna.tf32.f32 $0, $1;", "=r,r", [a], dtype=tl.float32,
                                              is_pure=True, pack=1)
            if MODE == 2:
                b = tl.inline_asm_elementwise("cvt.rna.tf32.f32 $0, $1;", "=r,r", [b], dtype=tl.float32,
                                              is_pure=True, pack=1)
            accumulator = tl.dot(a, b, accumulator)
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, accumulator, mask=c_mask)
    return matmul_mod


def operands(variant, layer, mat, unit):
    import torch
    import calls_tf32 as C
    x = C._load(unit, layer, mat).cuda().float().contiguous()
    w = C.weight(layer, mat).float()
    if VARIANTS[variant][1] == "random":
        g = torch.Generator(device="cuda").manual_seed(10_000 + 100 * layer + unit)
        w = torch.randn(w.shape, device="cuda", generator=g) * float(w.std())
    return x, w.t()                                         # the tutorial's b: (K, N) view of the weight


def launch(kern, x, wt, mode):
    import torch
    import triton
    M, K = x.shape
    N = wt.shape[1]
    c = torch.zeros((M, N), device="cuda", dtype=torch.float32)
    grid = (triton.cdiv(M, 64) * triton.cdiv(N, 64),)
    kern[grid](x, wt, c, M, N, K, x.stride(0), x.stride(1), wt.stride(0), wt.stride(1), c.stride(0), c.stride(1),
               BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=32, GROUP_SIZE_M=8, MODE=mode)
    return c


def tile_mask(M, N):
    """Output elements of the predicted programs (the tutorial's grouped program order, GROUP_SIZE_M = 8)."""
    m = np.zeros((M, N), dtype=bool)
    npm, npn = math.ceil(M / 64), math.ceil(N / 64)
    for pid in PROGRAMS:
        group = pid // (8 * npn)
        first = group * 8
        size = min(npm - first, 8)
        pm = first + ((pid % (8 * npn)) % size)
        pn = (pid % (8 * npn)) // size
        m[pm * 64:(pm + 1) * 64, pn * 64:(pn + 1) * 64] = True
    return m


def predict_all(out_path, only=None):
    from kernel_analyzer.reference_eval import bias_predict as BP
    from kernel_analyzer.reference_eval.capture import TritonLaunchRecorder
    kern = kernel()
    done = {(r["variant"], r["layer"], r["matrix"], r["unit"]) for r in
            (json.loads(l) for l in out_path.read_text().splitlines())} if out_path.exists() else set()
    with open(out_path, "a") as f:
        for variant, (mode, _) in VARIANTS.items():
            if only and variant != only:
                continue
            for layer, mat in STRATA:
                for unit in UNITS:
                    if (variant, layer, mat, unit) in done:
                        continue
                    x, wt = operands(variant, layer, mat, unit)
                    with TritonLaunchRecorder(execute=False) as R:
                        launch(kern, x, wt, mode)
                    L = R.launches[0]
                    assert not L.executed
                    t0 = time.time()
                    s = BP.summarize(BP.predict(L, programs=[(p, 0, 0) for p in PROGRAMS]))
                    rec = {"variant": variant, "layer": layer, "matrix": mat, "unit": unit, "status": s["status"],
                           "failures": s["failures"], "sites": s["sites"], "c_ptr": s["outputs"].get("c_ptr"),
                           "seconds": round(time.time() - t0, 1)}
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()
                    print(variant, layer, mat, unit, rec["status"], rec["c_ptr"] and rec["c_ptr"]["alignment"],
                          rec["seconds"], flush=True)


def measure_all(out_path):
    import torch
    kern = kernel()
    with open(out_path, "w") as f:
        for variant, (mode, _) in VARIANTS.items():
            for layer, mat in STRATA:
                for unit in UNITS:
                    x, wt = operands(variant, layer, mat, unit)
                    c = launch(kern, x, wt, mode)
                    torch.cuda.synchronize()
                    g = x.double().cpu().numpy() @ wt.double().cpu().numpy()
                    e = c.double().cpu().numpy() - g
                    m = tile_mask(*g.shape)
                    rec = {"variant": variant, "layer": layer, "matrix": mat, "unit": unit}
                    for name, sel in (("tiles", m), ("all", np.ones_like(m))):
                        gg, ee = g[sel], e[sel]
                        nz = gg != 0
                        rec[name] = {"alignment": float((ee * gg).sum() / (gg * gg).sum()),
                                     "se_iid": float(np.std(ee * gg) * math.sqrt(gg.size) / (gg * gg).sum()),
                                     "rms_relative_error": float(np.sqrt(np.mean((ee[nz] / gg[nz]) ** 2)))}
                    f.write(json.dumps(rec) + "\n")
                    print(variant, layer, mat, unit, rec["tiles"]["alignment"], rec["all"]["alignment"], flush=True)


def verdict(pred_path, meas_path):
    P = [json.loads(l) for l in pred_path.read_text().splitlines()]
    M = [json.loads(l) for l in meas_path.read_text().splitlines()]
    out = []
    for variant in VARIANTS:
        for layer, mat in STRATA:
            p = [r for r in P if (r["variant"], r["layer"], r["matrix"]) == (variant, layer, mat)]
            m = [r for r in M if (r["variant"], r["layer"], r["matrix"]) == (variant, layer, mat)]
            rec = {"variant": variant, "layer": layer, "matrix": mat}
            if any(r["status"] != "predicted" for r in p):
                rec.update(verdict="无法判断", reason="; ".join(sorted({f for r in p for f in r["failures"]}))[:300])
                out.append(rec)
                continue
            a_hat = np.array([r["c_ptr"]["alignment"] for r in p])
            a_meas = np.array([r["tiles"]["alignment"] for r in m])
            se_pred = float(np.std(a_hat, ddof=1) / math.sqrt(len(a_hat)))
            se_meas = max(float(np.std(a_meas, ddof=1) / math.sqrt(len(a_meas))),
                          math.sqrt(sum(r["tiles"]["se_iid"] ** 2 for r in m)) / len(m))
            rms = float(np.mean([r["c_ptr"]["rms_relative_error"] for r in p]))
            tol = Z * math.sqrt(se_pred ** 2 + se_meas ** 2) + REL * abs(a_hat.mean()) + SECOND_ORDER * rms ** 2
            diff = float(a_meas.mean() - a_hat.mean())
            rec.update(predicted=float(a_hat.mean()), measured=float(a_meas.mean()),
                       measured_all_tiles=float(np.mean([r["all"]["alignment"] for r in m])), tolerance=tol,
                       difference=diff, verdict="成立" if abs(diff) <= tol else "不成立")
            out.append(rec)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["predict", "merge", "measure", "verdict"], required=True)
    ap.add_argument("--variant", default=None, help="predict one variant (parallel runs; then --phase merge)")
    a = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    pred_path = CACHE / "tf32_modification_predictions.jsonl"
    meas_path = CACHE / "tf32_modification_measurements.jsonl"
    if a.phase == "predict":
        part = CACHE / f"tf32_modification_predictions_{a.variant}.jsonl" if a.variant else pred_path
        predict_all(part, a.variant)
        print("sha256", hashlib.sha256(part.read_bytes()).hexdigest())
    elif a.phase == "merge":
        rows = []
        for v in VARIANTS:
            rows += (CACHE / f"tf32_modification_predictions_{v}.jsonl").read_text().splitlines()
        pred_path.write_text("\n".join(rows) + "\n")
        print("sha256", hashlib.sha256(pred_path.read_bytes()).hexdigest())
    elif a.phase == "measure":
        measure_all(meas_path)
    else:
        V = verdict(pred_path, meas_path)
        for v in V:
            print(v)
        # the modification's effect: predicted and measured change of the alignment, per stratum
        effects = []
        for base, mod in (("V0", "V1"), ("V2", "V3")):
            for layer, mat in STRATA:
                b = next(v for v in V if (v["variant"], v["layer"], v["matrix"]) == (base, layer, mat))
                m = next(v for v in V if (v["variant"], v["layer"], v["matrix"]) == (mod, layer, mat))
                if "predicted" in b and "predicted" in m:
                    effects.append({"change": f"{base}->{mod}", "layer": layer, "matrix": mat,
                                    "predicted": m["predicted"] - b["predicted"],
                                    "measured": m["measured"] - b["measured"]})
        for e in effects:
            print(e)
        (CACHE / "tf32_modification_verdicts.json").write_text(json.dumps({"verdicts": V, "effects": effects},
                                                                           indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
