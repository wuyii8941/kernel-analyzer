#!/usr/bin/env python3
"""Kernel against its specification with the tool: e_num = K - K_R and e_sem = K_R - f.

For each case the kernel call runs under ``TritonLaunchRecorder``; the automatic reference K_R of the measured
buffer comes from the captured TTIR (``evaluate_sequence``, composed over all launches of the call), and the
specification f is the documented math of the function the kernel implements or replaces, evaluated on the same
inputs as a rigorous float64 enclosure (``reference_eval.intervals``).  Both residuals enter as directed intervals
and go through the unified decision layer (``analysis.assess_units``: rules R1, R2, R3, R5, endpoint-conservative
inference on the confirmation seeds, default detector 2.1).  Seeds 0-31 are development, 32-95 confirmation.

Reading: e_num detected / e_sem not -> numerical (class 1); e_sem detected with |e_sem| far above the K_R width
and the rounding scale -> the kernel's own semantics differ from f (class 4).  The per-coordinate profile of
|e_sem| (``sem_profile``) shows where in the output the deviation sits.

    python scripts/tool_spec_check.py --group flex --case all --out results/tool_spec/flex
"""

from __future__ import annotations

import argparse
import importlib.metadata as md
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kernel_analyzer.reference_eval import intervals as iv  # noqa: E402
from kernel_analyzer.reference_eval.analysis import assess_units, residual_interval  # noqa: E402
from kernel_analyzer.reference_eval.capture import TritonLaunchRecorder  # noqa: E402
from kernel_analyzer.reference_eval.ttir_eval import ST_OK, evaluate_sequence  # noqa: E402
from kernel_analyzer.reference_eval.ttir_mapping import kernel_coverage  # noqa: E402
from kernel_analyzer.reference_eval.ttir_parser import parse_ttir  # noqa: E402

DEV = list(range(0, 32))
CONF = list(range(32, 96))
RULES = ["R1", "R2", "R3", "R5"]


def f64(t):
    return t.detach().double().cpu().numpy()


def exact_mul(a, b):
    """Products of float32 values are exact in float64 (24 + 24 significant bits)."""
    p = a * b
    return p, p


def to_storage_order(out: torch.Tensor, lo: np.ndarray, hi: np.ndarray):
    """Spec arrays in the logical shape of ``out`` -> arrays over its storage (element offsets)."""
    n = out.untyped_storage().nbytes() // out.element_size()
    pos = torch.arange(n).as_strided(out.shape, out.stride(), out.storage_offset()).reshape(-1).numpy()
    s_lo, s_hi = np.full(n, np.nan), np.full(n, np.nan)
    s_lo[pos], s_hi[pos] = lo.reshape(-1), hi.reshape(-1)
    return s_lo, s_hi, pos


# ---------------------------------------------------------------------------------------------------------------
# Cases: Case subclasses live in tool_spec_cases_<group>.py; launch() returns {output name: tensor} and spec()
# returns {output name: (lo, hi)} in the tensor's logical shape.
# ---------------------------------------------------------------------------------------------------------------


class Case:
    name = ""
    implementation = ""   # what runs (the kernel)
    specification = ""    # what f is
    spec_bound = "rigorous float64 enclosure"

    def setup(self):
        """Once before the seeds (compilation, warm-up); runs outside the recorder."""

    def inputs(self, seed):
        raise NotImplementedError

    def launch(self, inp):
        raise NotImplementedError

    def spec(self, inp):
        raise NotImplementedError


def f64_point_spec(value, rel=2.0 ** -40):
    """A float64 evaluation of the specification taken as f with a declared bound rel * max|f| (screening use;
    a confirmed finding gets a rigorous enclosure)."""
    value = np.asarray(value, dtype=np.float64)
    b = rel * float(np.max(np.abs(value))) if value.size else 0.0
    return iv.down(value - b), iv.up(value + b)


GROUPS = {"liger": "tool_spec_cases_liger", "flex": "tool_spec_cases_flex", "inductor": "tool_spec_cases_inductor",
          "tridao": "tool_spec_cases_tridao", "fla": "tool_spec_cases_fla"}


def load_cases(group):
    import importlib

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    return {c.name: c for c in importlib.import_module(GROUPS[group]).CASES}


# ---------------------------------------------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------------------------------------------


def run(case, dev=DEV, conf=CONF):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    TritonLaunchRecorder.install_hook()
    case.setup()
    per, coverage, launch_info = {}, None, None
    not_triton, modified_after = set(), set()
    t0 = time.time()
    for seed in list(dev) + list(conf):
        inp = case.inputs(seed)
        rec = TritonLaunchRecorder()
        with rec:
            outs = case.launch(inp)
            torch.cuda.synchronize()
        if coverage is None:
            coverage = [kernel_coverage(parse_ttir(l.asm["ttir"]))["complete"] for l in rec.launches]
            launch_info = [{"kernel": l.kernel_name, "grid": list(l.grid), "triton": l.environment.get("triton")}
                           for l in rec.launches]
        seq = evaluate_sequence(rec.launches)
        specs = case.spec(inp)
        reasons, aborted = {}, {}
        for r in seq.launches:
            for key, n in getattr(r, "reasons", {}).items():
                reasons[key] = reasons.get(key, 0) + n
            for why in (r.aborted.values() if isinstance(r.aborted, dict) else r.aborted):
                aborted[str(why)[:200]] = aborted.get(str(why)[:200], 0) + 1
        for name, out in outs.items():
            buf = seq.memory.get(out.untyped_storage().data_ptr())
            if buf is None or not buf.written.any():
                not_triton.add(name)  # produced by a non-Triton op (cuBLAS, ATen reduction): nothing to evaluate
                continue
            m = buf.written
            idx = buf.global_indices()[m]
            final = torch.empty(0, dtype=out.dtype, device=out.device).set_(out.untyped_storage()).reshape(-1)
            final = final.detach().cpu().numpy()[idx] if idx.size else final.detach().cpu().numpy()[:0]
            if not np.array_equal(final.astype(np.float64), np.asarray(buf.actual_after[m], dtype=np.float64),
                                  equal_nan=True):
                modified_after.add(name)  # a non-Triton op changed it after the last recorded launch
                continue
            f_lo_s, f_hi_s, pos = to_storage_order(out, *specs[name])
            f_lo, f_hi = f_lo_s[idx], f_hi_s[idx]
            k, r_lo, r_hi = buf.actual_after[m], buf.lo[m], buf.hi[m]
            n_lo, n_hi = residual_interval(k, r_lo, r_hi)
            s_lo, s_hi = iv.isub(r_lo, r_hi, f_lo, f_hi)
            per.setdefault(name, []).append({
                "n": (n_lo, n_hi), "s": (s_lo, s_hi), "kr": 0.5 * (r_lo + r_hi), "k": k,
                "ok": (buf.st[m] == ST_OK) & ~buf.cond[m] & np.isfinite(f_lo), "idx": idx, "pos": pos,
                "shape": tuple(out.shape), "width": r_hi - r_lo, "reasons": reasons, "aborted": aborted})
    seconds = time.time() - t0
    n_dev = len(list(dev))
    report = {"case": case.name, "implementation": case.implementation, "specification": case.specification,
              "spec_bound": case.spec_bound, "launches": launch_info, "ttir_coverage_complete": coverage,
              "versions": {k: _version(k) for k in ("liger-kernel", "transformers", "torch", "triton")},
              "seeds": {"development": [list(dev)[0], list(dev)[-1]], "confirmation": [list(conf)[0], list(conf)[-1]]},
              "seconds": round(seconds, 1), "outputs": {},
              "outputs_not_written_by_triton": sorted(not_triton),
              "outputs_modified_after_last_triton_write": sorted(modified_after)}
    for name, rows in per.items():
        ok = np.stack([p["ok"] for p in rows])
        kr = np.stack([p["kr"] for p in rows])
        entry = {"elements_per_seed": int(rows[0]["idx"].size), "shape": list(rows[0]["shape"]),
                 "reference_classes": {"complete_fraction": float(ok.mean())},
                 "not_established_reasons_seed0": rows[0]["reasons"],
                 "aborted_programs_seed0": rows[0]["aborted"]}
        for key, label in (("n", "e_num = K - K_R"), ("s", "e_sem = K_R - f")):
            lo = np.stack([p[key][0] for p in rows])
            hi = np.stack([p[key][1] for p in rows])
            rec, _ = assess_units(f"{name}: {label}", lo, hi, kr, ok, n_dev, RULES, alignment_reference=kr,
                                  unit_ids=list(dev) + list(conf))
            mid = 0.5 * (lo + hi)
            if ok.any():
                rec["scale"] = {"mean_abs_residual": float(np.abs(mid[ok]).mean()),
                                "max_abs_residual": float(np.abs(mid[ok]).max()),
                                "rms_reference": float(np.sqrt((kr[ok] ** 2).mean())),
                                "relative_rms": float(np.sqrt((mid[ok] ** 2).mean() / max((kr[ok] ** 2).mean(), 1e-300))),
                                "max_K_R_width": float(np.stack([p["width"] for p in rows])[ok].max())}
            entry["numerical" if key == "n" else "semantic"] = rec
        # where e_sem sits in seed 0: mean |e_sem| per index of the last logical axis
        r0 = rows[0]
        inv = np.full(int(r0["pos"].max()) + 1, -1)
        inv[r0["pos"]] = np.arange(r0["pos"].size)
        full = np.zeros(r0["pos"].size)
        full[inv[r0["idx"]]] = np.abs(0.5 * (r0["s"][0] + r0["s"][1]))
        entry["sem_profile_last_axis"] = [round(float(x), 8) for x in full.reshape(-1, r0["shape"][-1]).mean(0)]
        report["outputs"][name] = entry
    return report


def _version(name):
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


def verdicts(rec):
    out = {r["rule"]: r.get("verdict") for r in rec.get("rules", [])}
    det = rec.get("default_detector") or {}
    out["detector_vector_mean"] = (det.get("vector_mean") or {}).get("verdict")
    out["detector_alignment"] = (det.get("alignment") or {}).get("verdict")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", required=True, choices=sorted(GROUPS))
    parser.add_argument("--case", required=True, help="case name, or 'all'")
    parser.add_argument("--out", type=Path, required=True, help="directory; one JSON per case")
    parser.add_argument("--seeds", type=int, default=96, help="development = first third")
    args = parser.parse_args()
    cases = load_cases(args.group)
    names = sorted(cases) if args.case == "all" else args.case.split(",")
    n_dev = args.seeds // 3
    args.out.mkdir(parents=True, exist_ok=True)
    for name in names:
        try:
            report = run(cases[name], dev=range(0, n_dev), conf=range(n_dev, args.seeds))
        except Exception as exc:  # noqa: BLE001
            import traceback

            report = {"case": name, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()[-1500:]}
            print(name, "ERROR", report["error"][:300], flush=True)
        (args.out / f"{name}.json").write_text(json.dumps(report, indent=1, default=float) + "\n")
        for oname, entry in report.get("outputs", {}).items():
            sem, num = entry["semantic"], entry["numerical"]
            print(f"{name:34s} {oname:6s} complete={entry['reference_classes']['complete_fraction']:.2f} "
                  f"num={verdicts(num)} sem={verdicts(sem)} "
                  f"sem_rel_rms={sem.get('scale', {}).get('relative_rms', float('nan')):.1e} "
                  f"num_rel_rms={num.get('scale', {}).get('relative_rms', float('nan')):.1e}", flush=True)


if __name__ == "__main__":
    main()
