"""First tool case: the Triton 3.6.0 tutorial flash attention forward (tutorial_fwd.py), as is and with planted
defects (planted_fwd.py), run through the tool end to end:

  1. capture without running the kernel and predict the bias (bias_predict);
  2. run on the device; bitwise check of the lowering rules (emulate.verify) and per-node attribution
     (emulate.localize) on a sample of programs;
  3. the exact value of the declared program G (reference evaluator) and the specification S (float64 attention in
     torch): numerical deviation K - G, semantic deviation G - S.

Outputs per case: .cache/flash_attention_case/<case>.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "2_tool/src"))
sys.path.insert(0, str(HERE))
OUT = ROOT / ".cache/flash_attention_case"


def make_inputs(dtype, Z, H, N, D, seed, scale=1.0):
    import torch
    g = torch.Generator(device="cuda").manual_seed(seed)
    return [(torch.randn(Z, H, N, D, device="cuda", generator=g) * scale).to(dtype) for _ in range(3)]


def spec_attention(q, k, v, sm_scale, causal):
    """The specification: softmax(q k^T * sm_scale) v in float64 (torch)."""
    import torch
    q, k, v = (t.double() for t in (q, k, v))
    s = q @ k.transpose(-1, -2) * sm_scale
    if causal:
        N = s.shape[-1]
        s = s.masked_fill(torch.triu(torch.ones(N, N, dtype=torch.bool, device=s.device), 1), float("-inf"))
    return torch.softmax(s, -1) @ v


def alignment(e, g):
    gg = float((g * g).sum())
    return float((e * g).sum() / gg) if gg > 0 else float("nan")


def run_case(name, kernel, dtype_name, Z, H, N, D, causal, block_m, block_n, programs, seed=0, sm_scale=None,
             in_scale=1.0):
    import torch
    import triton
    from kernel_analyzer.reference_eval import bias_predict as BP
    from kernel_analyzer.reference_eval import emulate as E
    from kernel_analyzer.reference_eval.capture import TritonLaunchRecorder
    from kernel_analyzer.reference_eval.ttir_eval import KernelReferenceEvaluator, NumericMode, ST_OK
    from kernel_analyzer.reference_eval.ttir_parser import parse_ttir
    dtype = getattr(torch, dtype_name)
    sm_scale = sm_scale if sm_scale is not None else 1.0 / math.sqrt(D)
    q, k, v = make_inputs(dtype, Z, H, N, D, seed, in_scale)
    o = torch.zeros_like(q)
    M = torch.zeros(Z, H, N, device="cuda")
    triton.set_allocator(lambda size, align, stream: torch.empty(size, dtype=torch.int8, device="cuda"))
    grid = (triton.cdiv(N, block_m), Z * H, 1)

    def launch():
        kernel[grid](sm_scale, M, Z, H, q, k, v, o, N_CTX=N, HEAD_DIM=D, BLOCK_M=block_m, BLOCK_N=block_n,
                     FP8_OUTPUT=False, STAGE=3 if causal else 1, warp_specialize=False, IS_HOPPER=False,
                     num_stages=2, num_warps=4)
    progs = [(p % grid[0], p // grid[0], 0) for p in programs]
    rec = {"case": name, "dtype": dtype_name, "shape": [Z, H, N, D], "causal": causal, "blocks": [block_m, block_n],
           "programs": progs, "timing": {}}
    # 1. prediction, kernel not run
    with TritonLaunchRecorder(execute=False) as R:
        launch()
    L0 = R.launches[0]
    assert not L0.executed
    t0 = time.time()
    try:
        pred = BP.predict(L0, programs=progs)
        rec["prediction"] = BP.summarize(pred)
        rec["prediction"]["per_output_delta_rms"] = {k_: float(np.sqrt(np.mean(o_["delta"][o_["ok"]] ** 2)))
                                                     for k_, o_ in pred["outputs"].items()}
    except Exception as e:
        rec["prediction"] = {"error": f"{type(e).__name__}: {str(e)[:300]}",
                             "where": traceback.format_exc().splitlines()[-4:]}
    rec["timing"]["predict_s"] = round(time.time() - t0, 1)
    # 2. device run, bitwise check, attribution
    with TritonLaunchRecorder() as R:
        launch()
        torch.cuda.synchronize()
    L = R.launches[0]
    t0 = time.time()
    try:
        ver = E.verify(L, ungrouped=[], swapped=[], programs=progs)
        rec["verify"] = {k_: ver[k_] for k_ in ("status", "buffers", "not_emulable_nodes", "abort_reasons",
                                                 "lowering_choices")}
    except Exception as e:
        rec["verify"] = {"error": f"{type(e).__name__}: {str(e)[:300]}", "where": traceback.format_exc().splitlines()[-4:]}
    rec["timing"]["verify_s"] = round(time.time() - t0, 1)
    t0 = time.time()
    try:
        loc = E.localize(L, programs=progs[:2])
        rec["attribution"] = [{"node": r["node"], "lowering": r["lowering"], "text": r["text"][:110],
                               "o_ptr": {kk: r["buffers"].get("desc_o", {}).get(kk) for kk in ("mean", "mean_abs",
                                                                                                "mean_relative")}}
                              for r in loc.get("nodes", loc.get("rows", []))] if isinstance(loc, dict) else None
    except Exception as e:
        rec["attribution"] = {"error": f"{type(e).__name__}: {str(e)[:300]}",
                              "where": traceback.format_exc().splitlines()[-4:]}
    rec["timing"]["localize_s"] = round(time.time() - t0, 1)
    # 3. exact declared value G on the sampled programs; specification S
    module = parse_ttir(L.asm["ttir"])
    module.source_text = L.asm["ttir"]
    ev = KernelReferenceEvaluator(module, mode=NumericMode.NUMERICAL_DIFFERENCE, ttgir=L.asm["ttgir"])
    from kernel_analyzer.reference_eval import intervals as iv
    original = iv.elementary_bounds
    iv.elementary_bounds = BP._float64_elementary(original)
    t0 = time.time()
    try:
        res = ev.evaluate(L, programs=progs)
    finally:
        iv.elementary_bounds = original
    rec["timing"]["reference_s"] = round(time.time() - t0, 1)
    S = spec_attention(q, k, v, sm_scale, causal).cpu().numpy().reshape(-1)
    K = o.double().cpu().numpy().reshape(-1)
    for ptr, buf in res.buffers.items():
        if buf.name != "desc_o":
            continue
        ok = buf.written & (buf.st == ST_OK)
        idx = buf.global_indices()[ok]
        G = 0.5 * (buf.lo[ok] + buf.hi[ok])
        Kd, Sd = K[idx], S[idx]
        rec["measured"] = {"elements": int(ok.sum()), "written": int(buf.written.sum()),
                           "alignment_K_vs_G": alignment(Kd - G, G), "alignment_G_vs_S": alignment(G - Sd, Sd),
                           "alignment_K_vs_S": alignment(Kd - Sd, Sd),
                           "rms_rel_K_vs_G": float(np.sqrt(np.mean(((Kd - G) / np.where(G != 0, G, 1)) ** 2))),
                           "max_abs_K_vs_S": float(np.max(np.abs(Kd - Sd)))}
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--N", type=int, default=1024)
    ap.add_argument("--programs", default="0,1,2,3")
    ap.add_argument("--suffix", default="")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    import tutorial_fwd as T
    progs = [int(x) for x in a.programs.split(",")]
    cases = {
        "tutorial_fp16": lambda: run_case("tutorial_fp16", T._attn_fwd, "float16", 1, 2, a.N, 64, False, 128, 64, progs),
        "tutorial_fp16_causal": lambda: run_case("tutorial_fp16_causal", T._attn_fwd, "float16", 1, 2, a.N, 64, True,
                                                 128, 64, progs),
    }
    try:
        import planted_fwd as P
        cases.update(P.cases(run_case, a, progs))
    except ImportError:
        pass
    rec = cases[a.case]()
    (OUT / f"{a.case}{a.suffix}.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False, default=str))
    print(json.dumps(rec, ensure_ascii=False, default=str)[:3000])


if __name__ == "__main__":
    main()
