"""RQ5: an unseen kernel decomposed -- the Triton TF32 shrink of the round-2 study (group C).

The round-2 study measured that Triton's tutorial matmul with fp32 inputs (tl.dot at the default tf32 precision)
shrinks every Qwen2.5-0.5B layer output by about 3.5e-4 (alignment <e, r> / <r, r> from -3.1e-4 to -3.7e-4), while
cuBLAS TF32 shrinks by 1e-5 or less.  The study recorded the effect; this script explains it from the composition rule
of the tensor-core dot (kernel_analyzer.reference_eval.tensor_core):

1. the rule reproduces the device output of the same kernel on the same captured inputs, bit for bit;
2. switching one site of the rule from truncation to round-to-nearest at a time -- the activation operand's tf32
   conversion, the weight operand's conversion, the accumulation inside the mma -- gives each site's share of the
   shrink (the full switch is the all-nearest counterfactual);
3. the weights come from a bf16 checkpoint (8 significant bits), so their tf32 conversion is exact and the prediction
   is that the whole shrink is the activation truncation, about -0.5 * 2^-10 * E[2^e / |x|] (-3.5e-4 for
   log-uniform significands).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "2_tool/src"))
sys.path.insert(0, str(ROOT / "1_experiments/tolerance_study"))
from kernel_analyzer.reference_eval import tensor_core as TC  # noqa: E402


def alignment(y, r):
    e = y - r
    return float((e * r).sum() / (r * r).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "results/tf32_attribution.json"))
    ap.add_argument("--units", type=int, default=8)
    a = ap.parse_args()
    import torch
    import calls_tf32 as C
    out = []
    for layer, mat in C.STRATA:
        w = C.weight(layer, mat).float()
        wt = w.t().contiguous()
        wn = wt.double().cpu().numpy()
        w_tf32_exact = bool(np.array_equal(TC.round_sig(wn, 11, "rz"), wn))
        for seed in range(a.units):
            x = C._load(seed, layer, mat).cuda().float().contiguous()
            dev = C._triton(x, w, False).double().cpu().numpy()
            xn = x.double().cpu().numpy()
            r = xn @ wn                                                      # float64 reference (error ~1e-16)
            zeros = np.zeros_like(r)
            model, covered = TC.mma_chain(xn, wn, zeros, "tf32")
            variants = {}
            for name, (ma, mb, fin) in {"activation conversion nearest": ("rne", "rz", "rz"),
                                        "weight conversion nearest": ("rz", "rne", "rz"),
                                        "accumulation nearest": ("rz", "rz", "rne"),
                                        "all nearest": ("rne", "rne", "rne")}.items():
                xa = TC.round_sig(xn, 11, ma)
                wb = TC.round_sig(wn, 11, mb)
                y, _ = TC.mma_chain(xa, wb, zeros, "tf32", conv_mode="rz", final=fin)
                variants[name] = alignment(y, r)
            nz = xn[xn != 0]
            m, e = np.frexp(nz)
            tau = (np.ldexp(np.trunc(np.ldexp(m, 11)), e - 11) - nz) / nz
            rec = {"layer": layer, "matrix": mat, "unit": seed, "shape": [int(x.shape[0]), int(x.shape[1]),
                                                                         int(wt.shape[1])],
                   "elements": int(dev.size), "covered": int(covered.sum()), "bit_identical": int((dev == model).sum()),
                   "weights_tf32_exact": w_tf32_exact,
                   "alignment_device": alignment(dev, r), "alignment_model": alignment(model, r),
                   "alignment_if": variants,
                   "share_activation_truncation": None,
                   "predicted_from_activation_tau": float(np.mean(tau))}
            base = rec["alignment_model"]
            rec["site_contributions"] = {"activation truncation": base - variants["activation conversion nearest"],
                                         "weight truncation": base - variants["weight conversion nearest"],
                                         "accumulation truncation": base - variants["accumulation nearest"],
                                         "remaining with every site nearest": variants["all nearest"]}
            rec["share_activation_truncation"] = rec["site_contributions"]["activation truncation"] / base
            out.append(rec)
            print({k: rec[k] for k in ("layer", "matrix", "unit", "bit_identical", "elements", "alignment_device",
                                       "predicted_from_activation_tau")},
                  {k: round(v, 9) for k, v in rec["site_contributions"].items()}, flush=True)
    Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
