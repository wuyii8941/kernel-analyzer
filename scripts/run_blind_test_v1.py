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
from kernel_analyzer.reference_eval.analysis import _holm, assess_units, residual_interval  # noqa: E402
from kernel_analyzer.reference_eval.capture import TritonLaunchRecorder, save_launch  # noqa: E402
from kernel_analyzer.reference_eval.ttir_eval import ST_OK, KernelReferenceEvaluator, TORCH_TO_ELEM  # noqa: E402
from kernel_analyzer.reference_eval.ttir_mapping import kernel_coverage  # noqa: E402
from kernel_analyzer.reference_eval.ttir_parser import parse_ttir  # noqa: E402

DEV = range(0, 32)
CONF = range(32, 96)


def set_seed_offset(offset: int):
    """Protocol v1.1 replication: the same rules on a disjoint seed set (offset 96: 96-127 development,
    128-191 confirmation).  Only the seed numbers change."""

    global DEV, CONF
    DEV = range(offset, offset + 32)
    CONF = range(offset + 32, offset + 96)


def run_program(package: Path, pid: str, family: str, work: Path, cache: Path = None) -> dict:
    """``cache``: when given, every seed's K (float32 values), K_R bounds, classes and indices are saved to
    cache/<pid>/seedNNN.npz, so later phases (specification, update layer) reuse the identical captures."""

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
        if seed in (DEV[0], CONF[0]):  # kept for localization and for the reviewer's independent check
            save_launch(launch, work / pid / f"seed{seed:03d}")
        res = KernelReferenceEvaluator(module).evaluate(launch)
        out_ptr = y.untyped_storage().data_ptr() if torch.is_tensor(y) else None
        b = res.buffers.get(out_ptr) if out_ptr is not None else None
        if b is None:  # the output buffer is the one argument written in floating point
            cands = [bb for bb in res.buffers.values() if bb.kind == "f" and bb.written.any()]
            b = cands[-1]
        m = b.written
        k = b.actual_after[m]
        r_lo, r_hi = residual_interval(k, b.lo[m], b.hi[m])  # directed: [K - hi, K - lo]
        if cache is not None:
            (cache / pid).mkdir(parents=True, exist_ok=True)
            np.savez(cache / pid / f"seed{seed:03d}.npz", k=k.astype(np.float32), kr_lo=b.lo[m], kr_hi=b.hi[m],
                     st=b.st[m], cond=b.cond[m], index=b.global_indices()[m].astype(np.int64))
        hashes.append(hashlib.sha256(np.asarray(b.after_raw).tobytes()).hexdigest()[:16])
        per_seed.append({"lo": r_lo, "hi": r_hi, "k": k, "kr": 0.5 * (b.lo[m] + b.hi[m]), "st": b.st[m],
                         "cond": b.cond[m], "index": b.global_indices()[m], "aborted": len(res.aborted)})
    seconds = time.time() - t0
    return {"coverage": coverage, "launch": launch_info, "per_seed": per_seed, "hashes": hashes, "seconds": seconds,
            "output_shape": tuple(y.shape) if torch.is_tensor(y) else None}


def rules_for(family: str, d: int = 0, shape=None) -> list:
    rules = ["R1", "R2", "R3"]
    if family == "F4":
        rules.append("R4")
    return rules + ["R5"]


def f4_pattern(shape) -> np.ndarray:
    """R4 of the protocol: per row, the first 64 dimensions -1, dimensions 65-128 +1, the rest 0."""

    cols = shape[-1]
    w = np.zeros(cols)
    w[:64], w[64:128] = -1.0, 1.0
    return np.tile(w, int(np.prod(shape)) // cols)


def direction(rule: str, kr: np.ndarray, family: str, shape) -> np.ndarray:
    """The protocol directions written out (used by the export scripts; the decisions go through
    analysis.assess_units, whose rule registry defines the same directions)."""

    n = kr.size
    if rule == "R1":
        return np.full(n, -1.0 / math.sqrt(n))
    if rule == "R2":
        return -np.sign(kr) / math.sqrt(n)
    if rule == "R3":
        norm = np.linalg.norm(kr)
        return -kr / norm if norm > 0 else np.zeros(n)
    if rule == "R4":
        w = f4_pattern(shape)
        return w / np.linalg.norm(w)
    raise ValueError(rule)


def coverage_classes(st, cond) -> dict:
    total = st.size
    return {"complete_composed": float(((st == ST_OK) & ~cond).sum() / total),
            "conditional_local": float(((st == ST_OK) & cond).sum() / total),
            "not_established": float((st != ST_OK).sum() / total)}


def assess_output(name, family, data, sel, shape, alignment="k", measurement=None):
    """Binding only: stack the seeds of one output and hand them to the unified decision layer
    (analysis.assess_units).  ``alignment`` picks the reference of the detector's alignment tests: the actual
    output K ("k", phase 1) or the K_R midpoint ("kr")."""

    lo = np.stack([s["lo"][sel] for s in data])
    hi = np.stack([s["hi"][sel] for s in data])
    kr = np.stack([s["kr"][sel] for s in data])
    k = np.stack([s["k"][sel] for s in data])
    st = np.stack([s["st"][sel] for s in data])
    cond = np.stack([s["cond"][sel] for s in data])
    ok = (st == ST_OK) & ~cond
    declared = {"R4": f4_pattern(shape)[sel]} if family == "F4" and shape is not None else None
    mshape = tuple(shape) if shape is not None and len(shape) == 2 and sel.all() else None
    record, arrays = assess_units(name, lo, hi, kr, ok, len(DEV), rules_for(family), declared_vectors=declared,
                                  alignment_reference=k if alignment == "k" else kr, detector_shape=mshape,
                                  measurement=measurement, unit_ids=list(DEV) + list(CONF))
    record["coverage_classes"] = coverage_classes(st, cond)
    # residual signs over every element with a complete reference (all seeds), as in the protocol tables
    okm = ok
    record["residual_all_elements"] = {"positive_frac": float(((lo > 0) & okm).sum() / okm.sum()),
                                       "negative_frac": float(((hi < 0) & okm).sum() / okm.sum()),
                                       "contains_zero_frac": float(((lo <= 0) & (hi >= 0) & okm).sum() / okm.sum()),
                                       "mean": float((0.5 * (lo + hi))[okm].mean()),
                                       "max_width": float((hi - lo)[okm].max())}
    return record, arrays


def analyse_output(name, family, data, sel, shape, alignment="k"):
    """Statistics for one output (all of it, or one column of F3) over the seeds, in the protocol's table
    format; the decisions come from analysis.assess_units."""

    record, _ = assess_output(name, family, data, sel, shape, alignment)
    out = {"output": name, "elements_per_seed": int(np.sum(sel)), "coverage_classes": record["coverage_classes"],
           "residual": record["residual_all_elements"], "record": record}
    if "rules" not in record:
        out["rules"] = {"verdict": record.get("verdict", "UNRESOLVED_REFERENCE")}
        return out
    out["rules"] = {}
    for r in record["rules"]:
        if "p_value_two_sided_conservative" not in r:
            out["rules"][r["rule"]] = {"verdict": r["verdict"], "reason": r.get("reason")}
            continue
        out["rules"][r["rule"]] = {"mu_interval": [r["lower_bound_of_E_l"], r["upper_bound_of_E_h"]],
                                   "mean": r["mean_projection"], "verdict": r["verdict"],
                                   "p": r["p_value_two_sided_conservative"], "n": r["n"],
                                   "interpretation": r.get("interpretation")}
    out["default_detector"] = record.get("default_detector")
    return out


def per_program(args):
    package = args.package
    manifest = json.loads((package / "manifest.json").read_text())
    wanted = set(args.programs.split(",")) if args.programs else None
    work = args.work
    for entry in manifest["programs"]:
        pid, family = entry["id"], entry["family"]
        if wanted and pid not in wanted:
            continue
        target = args.out / "programs" / f"{pid}.json"
        if target.exists():
            continue
        try:
            r = run_program(package, pid, family, work, args.cache_arrays)
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
        from kernel_analyzer.reference_eval.capture import load_launch
        from kernel_analyzer.reference_eval.interface import inventory

        launch = load_launch(work / pid / f"seed{DEV[0]:03d}")
        report = {"program": pid, "family": family, "launch": r["launch"], "coverage_complete": cov["complete"],
                  "interface": inventory(launch, parse_ttir(launch.asm["ttir"])),
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
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--cache-arrays", type=Path, default=None, help="save every seed's K and K_R bounds here")
    parser.add_argument("--work", type=Path, default=ROOT / ".cache" / "blind_v1_work",
                        help="saved launches (seeds DEV[0], CONF[0]) for localization and checks")
    args = parser.parse_args()
    set_seed_offset(args.seed_offset)
    if args.aggregate:
        from scripts.blind_test_v1_aggregate import aggregate

        aggregate(args.package, args.out, args.work)
    else:
        per_program(args)


if __name__ == "__main__":
    main()
