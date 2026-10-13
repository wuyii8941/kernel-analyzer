"""Validation of the bias predictor on generated programs (predictor_protocol.md).

phase predict: for every program and prediction sample, capture the launch without running the kernel
               (TritonLaunchRecorder(execute=False)) and predict (bias_predict.predict / summarize); write the
               predictions with their SHA-256.
phase measure: for every program and measurement sample (other seeds), run the kernel on the device, evaluate the
               exact program value with the reference evaluator (numerical-difference mode) and measure the error.
phase verdict: the declared rule, per program: 成立 / 不成立 / 无法判断.
"""
from __future__ import annotations

import argparse
import hashlib
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
import programs as PG  # noqa: E402

CACHE = ROOT / ".cache/composition_study/predictor"
PRED_SEEDS = range(6)
MEAS_SEEDS = range(1000, 1006)
Z, REL, SECOND_ORDER = 3.0, 0.1, 2.0


def _tensor_args(args):
    import torch
    return [a for a in args if isinstance(a, torch.Tensor)]


def predict_one(kern, spec, seed):
    from kernel_analyzer.reference_eval import bias_predict as BP
    from kernel_analyzer.reference_eval.capture import TritonLaunchRecorder
    grid, args, meta = PG.launch_args(spec, seed)
    with TritonLaunchRecorder(execute=False) as R:
        kern[grid](*args, **meta)
    L = R.launches[0]
    assert not L.executed
    t0 = time.time()
    pred = BP.predict(L)
    s = BP.summarize(pred)
    s["seconds"] = round(time.time() - t0, 2)
    return s


def measure_one(kern, spec, seed):
    import torch
    from kernel_analyzer.reference_eval.capture import TritonLaunchRecorder
    from kernel_analyzer.reference_eval.ttir_eval import KernelReferenceEvaluator, NumericMode, ST_OK
    from kernel_analyzer.reference_eval.ttir_parser import parse_ttir
    grid, args, meta = PG.launch_args(spec, seed)
    with TritonLaunchRecorder() as R:
        kern[grid](*args, **meta)
        torch.cuda.synchronize()
    L = R.launches[0]
    module = parse_ttir(L.asm["ttir"])
    module.source_text = L.asm["ttir"]
    ev = KernelReferenceEvaluator(module, mode=NumericMode.NUMERICAL_DIFFERENCE, ttgir=L.asm["ttgir"])
    # float64 elementary functions (relative error ~1e-16, far below the errors measured) instead of the rigorous
    # enclosures, for speed
    from kernel_analyzer.reference_eval import bias_predict as BP
    from kernel_analyzer.reference_eval import intervals as iv
    original = iv.elementary_bounds
    iv.elementary_bounds = BP._float64_elementary(original)
    try:
        res = ev.evaluate(L)
    finally:
        iv.elementary_bounds = original
    out = {}
    for ptr, buf in res.buffers.items():
        if buf.kind != "f" or not buf.written.any():
            continue
        ok = buf.written & (buf.st == ST_OK) & (buf.actual_after_st == 0) & np.isfinite(buf.actual_after)
        g = 0.5 * (buf.lo[ok] + buf.hi[ok])
        e = buf.actual_after[ok] - g
        gg = float((g * g).sum())
        nz = g != 0
        out[buf.name] = {"elements": int(ok.sum()), "written": int(buf.written.sum()),
                         "alignment": float((e * g).sum() / gg) if gg > 0 else 0.0,
                         "se_iid": float(np.std(e * g) * math.sqrt(g.size) / gg) if gg > 0 else 0.0,
                         "rms_relative_error": float(np.sqrt(np.mean((e[nz] / g[nz]) ** 2))) if nz.any() else 0.0,
                         "aborted_programs": len(res.aborted)}
    return out


def run_phase(phase, specs, kernels, seeds, out_path):
    done = {}
    if out_path.exists():
        for l in out_path.read_text().splitlines():
            r = json.loads(l)
            done[(r["program"], r["seed"])] = r
    with open(out_path, "a") as f:
        for spec in specs:
            kern = getattr(kernels, spec["name"])
            for seed in seeds:
                if (spec["name"], seed) in done:
                    continue
                try:
                    r = predict_one(kern, spec, seed) if phase == "predict" else {"outputs": measure_one(kern, spec,
                                                                                                         seed)}
                except Exception as e:
                    r = {"error": f"{type(e).__name__}: {str(e)[:300]}",
                         "where": traceback.format_exc().splitlines()[-3:]}
                r.update(program=spec["name"], seed=seed)
                f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
                f.flush()
                print(phase, spec["name"], seed, r.get("status") or ("error" if "error" in r else "ok"),
                      {k: round(v.get("alignment", 0), 9) for k, v in (r.get("outputs") or {}).items()},
                      (r.get("failures") or [])[:2], r.get("error", ""), flush=True)


def verdict(specs, pred_rows, meas_rows) -> list:
    """The declared rule (predictor_protocol.md)."""
    out = []
    for spec in specs:
        P = [r for r in pred_rows if r["program"] == spec["name"]]
        M = [r for r in meas_rows if r["program"] == spec["name"]]
        rec = {"program": spec["name"], "form": spec["form"], "spec": {k: v for k, v in spec.items() if k != "source"}}
        errs = [r for r in P + M if "error" in r]
        if errs:
            rec.update(verdict="无法判断", reason="harness or predictor error: " + errs[0]["error"][:200])
            out.append(rec)
            continue
        failures = sorted({f for r in P for f in r.get("failures", [])})
        names = sorted(set.intersection(*[set(r["outputs"]) for r in P + M]))
        per = {}
        for name in names:
            a_hat = np.array([r["outputs"][name]["alignment"] for r in P])
            a_meas = np.array([r["outputs"][name]["alignment"] for r in M])
            se_pred = float(np.std(a_hat, ddof=1) / math.sqrt(len(a_hat))) if len(a_hat) > 1 else 0.0
            se_iid = math.sqrt(sum(r["outputs"][name]["se_iid"] ** 2 for r in M)) / len(M)
            se_meas = max(float(np.std(a_meas, ddof=1) / math.sqrt(len(a_meas))) if len(a_meas) > 1 else 0.0,
                          se_iid)
            rms_pred = float(np.mean([r["outputs"][name]["rms_relative_error"] for r in P]))
            tol = (Z * math.sqrt(se_meas ** 2 + se_pred ** 2) + REL * abs(float(a_hat.mean()))
                   + SECOND_ORDER * rms_pred ** 2)
            diff = float(a_meas.mean() - a_hat.mean())
            per[name] = {"predicted": float(a_hat.mean()), "se_pred": se_pred, "measured": float(a_meas.mean()),
                         "se_meas": se_meas, "tolerance": tol, "difference": diff,
                         "z_like": diff / max(math.sqrt(se_meas ** 2 + se_pred ** 2), 1e-300),
                         "rms_rel_pred": rms_pred,
                         "rms_rel_meas": float(np.mean([r["outputs"][name]["rms_relative_error"] for r in M])),
                         "holds": abs(diff) <= tol,
                         # control: the null prediction (RN is unbiased), same rule without the contract terms
                         "null_holds": abs(float(a_meas.mean())) <= Z * se_meas + SECOND_ORDER * rms_pred ** 2}
        rec["outputs"] = per
        rec["null_verdict"] = ("成立" if per and all(v["null_holds"] for v in per.values()) else
                               ("不成立" if per else "无法判断"))
        if failures:
            rec.update(verdict="无法判断", reason="; ".join(failures)[:400])
        elif not per:
            rec.update(verdict="无法判断", reason="no comparable output")
        else:
            rec["verdict"] = "成立" if all(v["holds"] for v in per.values()) else "不成立"
        out.append(rec)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, help="dev-<k> or heldout")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", default=None, help="generator seed (held-out: from predictor_protocol.md)")
    ap.add_argument("--phase", choices=["predict", "measure", "verdict"], required=True)
    ap.add_argument("--only", default="")
    ap.add_argument("--shard", default="0/1", help="i/n: programs i, i+n, ... (parallel runs, one file each)")
    a = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    seed = a.seed or a.set
    specs = PG.generate(seed, a.n)
    if a.only:
        specs = [s for s in specs if s["name"] in a.only.split(",")]
    tag = a.set
    i, n = (int(x) for x in a.shard.split("/"))
    suffix = f"_s{i}of{n}" if n > 1 else ""
    pred_path = CACHE / f"{tag}_predictions{suffix}.jsonl"
    meas_path = CACHE / f"{tag}_measurements{suffix}.jsonl"
    if a.phase == "verdict":
        def rows(kind):
            return [json.loads(l) for f in sorted(CACHE.glob(f"{tag}_{kind}*.jsonl")) for l in f.read_text().splitlines()]
        P, M = rows("predictions"), rows("measurements")
        V = verdict(specs, P, M)
        from collections import Counter
        print("predictor", Counter(v["verdict"] for v in V))
        print("null prediction", Counter(v.get("null_verdict") for v in V))
        print("predictor x null", Counter((v["verdict"], v.get("null_verdict")) for v in V))
        for v in V:
            print(v["program"], v["form"], v["verdict"], v.get("reason", "")[:150],
                  {k: (f"{o['predicted']:.3g}", f"{o['measured']:.3g}", f"{o['tolerance']:.2g}")
                   for k, o in v.get("outputs", {}).items()})
        (CACHE / f"{tag}_verdicts.json").write_text(json.dumps(V, indent=1, ensure_ascii=False, default=str))
        return
    kernels = PG.materialize(specs, CACHE / f"programs_{tag.replace('-', '_')}{suffix}.py")
    specs = specs[i::n]
    if a.phase == "predict":
        run_phase("predict", specs, kernels, PRED_SEEDS, pred_path)
        print("sha256", hashlib.sha256(pred_path.read_bytes()).hexdigest())
    else:
        run_phase("measure", specs, kernels, MEAS_SEEDS, meas_path)


if __name__ == "__main__":
    main()
