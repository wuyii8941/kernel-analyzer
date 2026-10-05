#!/usr/bin/env python3
"""Direct-differential baseline on the same cases, inputs and seeds as tool_spec_check (plan WP3).

For every case: Dynamo reset, the case's own setup (compile), then per seed
  K    = the case's compiled call (exactly what tool_spec_check measures),
  E32  = the same op eagerly in float32 (OpInfo cases; the PyTorch CI reference),
  F64  = the case's float64 specification point values,
and per output
  ci      torch.testing.assert_close(K, E32, rtol=1.3e-5, atol=1.5e-5, equal_nan=True) with the CUDA float32 overrides of
          test/inductor/test_torchinductor_opinfo.py (v2.10.0)  -- "PyTorch CI if it ran this sample",
  hp      relative RMS and max |K - F64| over the finite elements, special-value class mismatches,
  eager   the same for E32 against F64 (how far eager float32 itself is).
A compile or run exception counts as found (CI fails on it as well).

    python scripts/baseline_direct_diff.py --group opinfo --case oi_x_1,oi_y_2 --out DIR --seeds 9
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]

import tool_spec_check as tsc  # noqa: E402

OVERRIDES = json.load(open(ROOT / "scripts/data/inductor_override_cuda_f32.json"))
CI_RTOL, CI_ATOL = 1.3e-5, 1.5e-5


def _np(t):
    t = t.detach().cpu()
    if t.dtype == torch.bool:
        return t.numpy().astype(np.float64)
    if t.is_complex():
        raise TypeError("complex output")
    return t.to(torch.float64).numpy()


def _eager32(case, inp):
    import tool_spec_cases_opinfo as oi
    sample = inp["sample"]
    if isinstance(case, oi.OpInfoBwdCase):
        return case._grads(sample, case.op.op, inp["g"], None)
    with torch.no_grad():
        y = case._call(sample, case.op.op)
    return {f"out{i}": t for i, t in enumerate(oi._outputs(y))}


def _ci(k, e, op_name, backward=False):
    kw = OVERRIDES.get(op_name, {})
    if kw.get("assert_equal") is False:
        return {"verdict": "skipped (assert_equal False)"}
    if backward and kw.get("check_gradient") is False:
        return {"verdict": "skipped (check_gradient False)"}
    rtol, atol = kw.get("rtol", CI_RTOL), kw.get("atol", CI_ATOL)
    try:
        torch.testing.assert_close(k, e, rtol=rtol, atol=atol, equal_nan=True, check_dtype=False)
        return {"verdict": "pass", "rtol": rtol, "atol": atol}
    except AssertionError as err:
        return {"verdict": "fail", "rtol": rtol, "atol": atol, "why": str(err).strip().splitlines()[0][:160]}


def _hp(k, f):
    k, f = _np(k).reshape(-1), np.asarray(f, dtype=np.float64).reshape(-1)
    if k.shape != f.shape:
        return {"shape_mismatch": True}
    fin = np.isfinite(f) & np.isfinite(k)
    cls = lambda a: np.where(np.isnan(a), 1, np.where(a == np.inf, 2, np.where(a == -np.inf, 3, 0)))  # noqa: E731
    d = (k - f)[fin]
    scale = float(np.sqrt(np.mean(f[fin] ** 2))) if fin.any() else 0.0
    rel = float(np.sqrt(np.mean(d ** 2)) / scale) if fin.any() and scale > 0 else (0.0 if not d.size or not d.any() else float("inf"))
    return {"rel_rms": rel, "max_abs": float(np.abs(d).max()) if d.size else 0.0,
            "special_mismatch": int((cls(k) != cls(f)).sum())}


def run_case(case, seeds):
    torch._dynamo.reset()
    t0 = time.time()
    case.setup()
    op_name = getattr(getattr(case, "op", None), "name", "") + (
        "." + case.op.variant_test_name if getattr(case, "op", None) is not None and case.op.variant_test_name else "")
    per = {}
    for seed in seeds:
        inp = case.inputs(seed)
        k_out = case.launch(inp)
        torch.cuda.synchronize()
        e_out = _eager32(case, inp)
        specs = case.spec(inp)
        for name, k in k_out.items():
            rec = per.setdefault(name, {"ci": [], "hp": [], "eager": []})
            e = e_out.get(name)
            rec["ci"].append(_ci(k, e, op_name, backward=type(case).__name__ == "OpInfoBwdCase")
                             if e is not None else {"verdict": "no eager output"})
            f_lo, f_hi = specs[name]
            f_mid = np.where(np.isfinite(f_lo) & np.isfinite(f_hi), 0.5 * (np.asarray(f_lo) + np.asarray(f_hi)), f_lo)
            rec["hp"].append(_hp(k, f_mid))
            rec["eager"].append(_hp(e, f_mid) if e is not None else None)
    outputs = {}
    for name, rec in per.items():
        fails = [c for c in rec["ci"] if c["verdict"] == "fail"]
        hps = [h for h in rec["hp"] if not h.get("shape_mismatch")]
        rel = max((h["rel_rms"] for h in hps), default=None)
        outputs[name] = {
            "ci_found": bool(fails), "ci_fail_seeds": len(fails), "ci_example": fails[0] if fails else rec["ci"][0],
            "hp_rel_rms_max": rel, "hp_max_abs": max((h["max_abs"] for h in hps), default=None),
            "hp_special_mismatch": max((h["special_mismatch"] for h in hps), default=0),
            "hp_found": {str(t): bool(rel is not None and rel > t) for t in (1e-3, 1e-5, 1e-6)},
            "eager_rel_rms_max": max((h["rel_rms"] for h in rec["eager"] if h and not h.get("shape_mismatch")),
                                     default=None),
            "shape_mismatch": any(h.get("shape_mismatch") for h in rec["hp"]),
        }
    return {"case": case.name, "op": op_name, "seconds": time.time() - t0, "seeds": list(seeds), "outputs": outputs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True)
    ap.add_argument("--case", required=True, help="comma-separated case names")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, default=9)
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    import torch._inductor.config as ic
    ic.use_static_cuda_launcher = False  # same compiled configuration as the tool run
    cases = tsc.load_cases(a.group)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in a.case.split(","):
        try:
            res = run_case(cases[name], range(a.seeds))
        except Exception as e:  # noqa: BLE001
            res = {"case": name, "error": f"{type(e).__name__}: {e}"[:2000], "traceback": traceback.format_exc()[-4000:]}
        (out / f"{name}.json").write_text(json.dumps(res, indent=1))
        print(name, "error" if "error" in res else {k: v["ci_found"] for k, v in res["outputs"].items()}, flush=True)


if __name__ == "__main__":
    main()
