#!/usr/bin/env python3
"""Phase 1 of blind_test_v1: the same pipeline for every program (kernel output level).

Per program and seed (0-31 development, 32-95 confirmation): the program runs under the launch recorder,
the automatic reference K_R is evaluated on all program instances, and the residual e = K - K_R enters
as the interval [K - hi, K - lo].  Reported per program (protocol section 5): coverage classes, residual
signs / mean / width, the mean effect under R1-R5 (endpoint-conservative t on the confirmation seeds; R5
learned on the development seeds), the tool's default detector (detect.py), output hashes for the
family-wise bitwise comparison, and the time.  Holm over program x rule and the localization run in the
aggregation step (--aggregate), after all programs.

    python scripts/run_blind_test_v1.py --package .cache/blind/blind_test_v1 --out results/reference_eval/blind_test_v1 \
        --programs prog_01,prog_02           # per program, any GPU
    python scripts/run_blind_test_v1.py --package ... --out ... --aggregate
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from kernel_analyzer.reference_eval import intervals as iv  # noqa: E402
from kernel_analyzer.reference_eval.analysis import _holm, _summarize  # noqa: E402
from kernel_analyzer.reference_eval.capture import TritonLaunchRecorder, save_launch  # noqa: E402
from kernel_analyzer.reference_eval.detect import detect  # noqa: E402
from kernel_analyzer.reference_eval.ttir_eval import ST_OK, KernelReferenceEvaluator, TORCH_TO_ELEM  # noqa: E402
from kernel_analyzer.reference_eval.ttir_mapping import kernel_coverage  # noqa: E402
from kernel_analyzer.reference_eval.ttir_parser import parse_ttir  # noqa: E402

DEV = range(0, 32)
CONF = range(32, 96)


def run_program(package: Path, pid: str, family: str, work: Path) -> dict:
    import torch

    sys.path.insert(0, str(package / "programs"))
    inputs = importlib.import_module("inputs")
    mod = importlib.import_module(pid)
    t0 = time.time()
    per_seed, hashes, coverage, launch_info = [], [], None, None
    for seed in list(DEV) + list(CONF):
        inp = inputs.make_inputs(mod.FAMILY, seed)
        rec = TritonLaunchRecorder()
        with rec:
            y = mod.launch(inp)
            torch.cuda.synchronize()
        if len(rec.launches) != 1:
            raise RuntimeError(f"{pid}: expected one launch, got {len(rec.launches)}")
        launch = rec.launches[0]
        module = parse_ttir(launch.asm["ttir"])
        if coverage is None:
            coverage = kernel_coverage(module)
            launch_info = {"kernel": launch.kernel_name, "grid": list(launch.grid),
                           "num_warps": launch.metadata.get("num_warps"),
                           "enable_fp_fusion": launch.metadata.get("enable_fp_fusion")}
        if seed in (0, 32):  # kept for localization
            save_launch(launch, work / pid / f"seed{seed:03d}")
        res = KernelReferenceEvaluator(module).evaluate(launch)
        out_ptr = y.untyped_storage().data_ptr() if torch.is_tensor(y) else None
        b = res.buffers.get(out_ptr) if out_ptr is not None else None
        if b is None:  # the output buffer is the one argument written in floating point
            cands = [bb for bb in res.buffers.values() if bb.kind == "f" and bb.written.any()]
            b = cands[-1]
        m = b.written
        k = b.actual_after[m]
        r_lo = iv.add_bounds(k, -b.hi[m])[0]
        r_hi = iv.add_bounds(k, -b.lo[m])[1]
        hashes.append(hashlib.sha256(np.asarray(b.after_raw).tobytes()).hexdigest()[:16])
        per_seed.append({"lo": r_lo, "hi": r_hi, "k": k, "kr": 0.5 * (b.lo[m] + b.hi[m]), "st": b.st[m],
                         "cond": b.cond[m], "index": b.global_indices()[m], "aborted": len(res.aborted)})
    seconds = time.time() - t0
    return {"coverage": coverage, "launch": launch_info, "per_seed": per_seed, "hashes": hashes, "seconds": seconds,
            "output_shape": tuple(y.shape) if torch.is_tensor(y) else None}


def rules_for(family: str, d: int, shape) -> list:
    rules = ["R1", "R2", "R3"]
    if family == "F4":
        rules.append("R4")
    return rules + ["R5"]


def direction(rule: str, kr: np.ndarray, family: str, shape) -> np.ndarray:
    n = kr.size
    if rule == "R1":
        return np.full(n, -1.0 / math.sqrt(n))
    if rule == "R2":
        return -np.sign(kr) / math.sqrt(n)
    if rule == "R3":
        norm = np.linalg.norm(kr)
        return -kr / norm if norm > 0 else np.zeros(n)
    if rule == "R4":  # F4: first 64 dims -1, dims 65-128 +1, others 0, per row; normalized
        cols = shape[-1]
        w = np.zeros(cols)
        w[:64], w[64:128] = -1.0, 1.0
        w = np.tile(w, n // cols)
        return w / np.linalg.norm(w)
    raise ValueError(rule)


def analyse_output(name, family, data, sel, shape):
    """Statistics for one output (all of it, or one column of F3) over the seeds."""

    lo = np.stack([s["lo"][sel] for s in data])
    hi = np.stack([s["hi"][sel] for s in data])
    kr = np.stack([s["kr"][sel] for s in data])
    k = np.stack([s["k"][sel] for s in data])
    st = np.stack([s["st"][sel] for s in data])
    cond = np.stack([s["cond"][sel] for s in data])
    total = st.size
    out = {"output": name, "elements_per_seed": int(lo.shape[1]),
           "coverage_classes": {"complete_composed": float(((st == ST_OK) & ~cond).sum() / total),
                                "conditional_local": float(((st == ST_OK) & cond).sum() / total),
                                "not_established": float((st != ST_OK).sum() / total)}}
    ok = (st == ST_OK) & ~cond
    mid = 0.5 * (lo + hi)
    out["residual"] = {"positive_frac": float(((lo > 0) & ok).sum() / ok.sum()),
                       "negative_frac": float(((hi < 0) & ok).sum() / ok.sum()),
                       "contains_zero_frac": float(((lo <= 0) & (hi >= 0) & ok).sum() / ok.sum()),
                       "mean": float(mid[ok].mean()), "max_width": float((hi - lo)[ok].max())}
    valid = ok[: len(DEV)].all(axis=0)  # coordinate set fixed on the development seeds
    conf_rows = slice(len(DEV), len(DEV) + len(CONF))
    if not valid.any() or not ok[conf_rows][:, valid].all():
        out["rules"] = {"verdict": "UNRESOLVED_REFERENCE"}
        return out
    lo, hi, kr, k, mid = lo[:, valid], hi[:, valid], kr[:, valid], k[:, valid], mid[:, valid]
    rules = {}
    for rule in rules_for(family, lo.shape[1], shape):
        if rule == "R5":
            w = mid[: len(DEV)].mean(axis=0)
            norm = np.linalg.norm(w)
            if norm == 0:
                rules[rule] = {"verdict": "UNRESOLVED_MEASUREMENT", "reason": "development direction is zero"}
                continue
            ws = np.broadcast_to(w / norm, lo[conf_rows].shape)
        else:
            ws = np.stack([direction(rule, kr[i], family, shape) for i in range(len(DEV), len(DEV) + len(CONF))])
        a, b = lo[conf_rows] * ws, hi[conf_rows] * ws
        r = _summarize(name, rule, np.minimum(a, b).sum(axis=1), np.maximum(a, b).sum(axis=1), 0.05)
        rules[rule] = {"mu_interval": [r["lower_bound_of_E_l"], r["upper_bound_of_E_h"]],
                       "mean": r["mean_projection"], "verdict": r["verdict"],
                       "p": r["p_value_two_sided_conservative"], "n": r["n"]}
    out["rules"] = rules
    mshape = None
    if shape is not None and len(shape) == 2 and valid.all() and lo.shape[1] == shape[0] * shape[1]:
        mshape = shape
    out["default_detector"] = detect(mid, k, len(DEV), shape=mshape, seed=0)
    return out


def per_program(args):
    package = args.package
    manifest = json.loads((package / "manifest.json").read_text())
    wanted = set(args.programs.split(",")) if args.programs else None
    work = ROOT / ".cache" / "blind_v1_work"
    for entry in manifest["programs"]:
        pid, family = entry["id"], entry["family"]
        if wanted and pid not in wanted:
            continue
        target = args.out / "programs" / f"{pid}.json"
        if target.exists():
            continue
        try:
            r = run_program(package, pid, family, work)
        except Exception as exc:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps({"program": pid, "family": family,
                                          "error": f"{type(exc).__name__}: {str(exc)[:400]}"}, indent=2) + "\n")
            print(pid, "ERROR", exc, flush=True)
            continue
        data = r["per_seed"]
        shape = r["output_shape"]
        outputs = []
        if family == "F3":  # two columns reported separately (protocol)
            idx = data[0]["index"]
            for col, name in ((0, "mean"), (1, "rstd")):
                sel = (idx % 2) == col
                outputs.append(analyse_output(name, family, data, sel, None))
        else:
            outputs.append(analyse_output("output", family, data, np.ones(data[0]["lo"].size, dtype=bool), shape))
        cov = r["coverage"]
        report = {"program": pid, "family": family, "launch": r["launch"], "coverage_complete": cov["complete"],
                  "unsupported": [x["rejected"] for x in cov["rejected"]], "operations": cov["operations"],
                  "aborted_programs": sum(s["aborted"] for s in data), "output_hashes": r["hashes"],
                  "outputs": outputs, "seconds": round(r["seconds"], 1)}
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2, default=float) + "\n")
        print(pid, family, round(r["seconds"], 1), [(o["output"], {k: v.get("verdict") for k, v in o["rules"].items()}
                                                     if isinstance(o["rules"], dict) and "verdict" not in o["rules"]
                                                     else o["rules"]) for o in outputs], flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--programs", default=None)
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    if args.aggregate:
        from scripts.blind_test_v1_aggregate import aggregate

        aggregate(args.package, args.out)
    else:
        per_program(args)


if __name__ == "__main__":
    main()
