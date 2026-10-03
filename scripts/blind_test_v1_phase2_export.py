#!/usr/bin/env python3
"""Captures for the reviewer's independent check of blind_test_v1 phase 2 (e_sem = K_R - f).

Programs: every program with a phase 2 finding (a rule detection in either scalar mode, a default-detector
alert, or e_sem intervals excluding zero on at least 99% of the coordinates).  For seeds 0 and 32, as in the
phase 1 package: the output K, the tool's K_R interval, the tool's e_sem interval for both scalar modes
(constructed with exactly the pipeline's arithmetic), and the projections under R1-R5.  Each projection is
given twice: as the pipeline computed it (ordinary products and sum, what the statistics used) and as a
strict outward bound (directed products, exact fsum, one ulp outward).  R5 directions are learned on the
e_sem midpoints of seeds 0-31 over the coordinates valid on all of them, exactly as in the pipeline.
Inputs and f depend only on the family and are stored once per family.

    python scripts/blind_test_v1_phase2_export.py --package .cache/blind/blind_test_v1 --programs prog_18,prog_33
    python scripts/blind_test_v1_phase2_export.py --package .cache/blind/blind_test_v1 --index
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from kernel_analyzer.reference_eval import intervals as iv  # noqa: E402
from kernel_analyzer.reference_eval.capture import TritonLaunchRecorder  # noqa: E402
from kernel_analyzer.reference_eval.ttir_eval import ST_OK, KernelReferenceEvaluator  # noqa: E402
from kernel_analyzer.reference_eval.ttir_parser import parse_ttir  # noqa: E402
from scripts.run_blind_test_v1 import direction  # noqa: E402

REPORTS = ROOT / "results" / "reference_eval" / "blind_test_v1"
FCACHE = ROOT / ".cache" / "blind_v1_phase2_f"
OUT = ROOT / ".cache" / "blind_v1_phase2_verification"
MODES = ("given", "fp32")
SEEDS = (0, 32)
DEV = range(32)


def hit(o):
    return isinstance(o["rules"], dict) and any(str(v.get("final_verdict", "")).startswith("DETECTED")
                                                for v in o["rules"].values() if isinstance(v, dict))


def detector_hit(o):
    return any(o.get("default_detector", {}).get(f, {}).get("final_verdict") == "DETECTED"
               for f in ("vector_mean", "alignment"))


def selection(report):
    sel = {}
    for pid, rep in report.items():
        why = set()
        for mode in MODES:
            for o in rep["modes"][mode]:
                if hit(o):
                    why.add(f"rule_detection_{mode}")
                if mode == "given" and detector_hit(o):
                    why.add("default_detector_alert")
                if mode == "given" and o["residual"]["contains_zero_frac"] <= 0.01:
                    why.add("e_sem_excludes_zero_99pct")
        if why:
            sel[pid] = sorted(why)
    return sel


def evaluate(mod, inputs, seed):
    import torch

    inp = inputs.make_inputs(mod.FAMILY, seed)
    rec = TritonLaunchRecorder()
    with rec:
        y = mod.launch(inp)
        torch.cuda.synchronize()
    launch = rec.launches[0]
    res = KernelReferenceEvaluator(parse_ttir(launch.asm["ttir"])).evaluate(launch)
    b = res.buffers[y.untyped_storage().data_ptr()]
    m = b.written
    k = b.actual_after[m]
    return {"inp": inp, "y": y, "k": k, "lo": b.lo[m], "hi": b.hi[m], "index": b.global_indices()[m],
            "ok": (b.st[m] == ST_OK) & ~b.cond[m],
            "hash": hashlib.sha256(np.asarray(b.after_raw).tobytes()).hexdigest()[:16]}


def e_sem(d, family, seed, mode):
    """The pipeline's construction (run_blind_test_v1.run_program + run_blind_test_v1_phase2.stage_programs)."""

    f = np.load(FCACHE / f"{family}_{seed:03d}.npz")
    f_lo, f_hi = f[f"{mode}_lo"][d["index"]], f[f"{mode}_hi"][d["index"]]
    s_lo = iv.add_bounds(d["k"], -d["hi"])[0]
    s_hi = iv.add_bounds(d["k"], -d["lo"])[1]
    r_lo = iv.add_bounds(d["k"], -s_hi)[0]
    r_hi = iv.add_bounds(d["k"], -s_lo)[1]
    return iv.add_bounds(r_lo, -f_hi)[0], iv.add_bounds(r_hi, -f_lo)[1]


def projections(e_lo, e_hi, w):
    a, b = e_lo * w, e_hi * w
    p_lo, p_hi = iv.imul(e_lo, e_hi, w, w)
    return {"pipeline": [float(np.minimum(a, b).sum()), float(np.maximum(a, b).sum())],
            "directed": [float(np.nextafter(math.fsum(p_lo), -np.inf)), float(np.nextafter(math.fsum(p_hi), np.inf))]}


def export_program(pid, inputs, report, phase1, why):
    import torch

    mod = importlib.import_module(pid)
    family = mod.FAMILY
    rep = report[pid]
    target = OUT / "programs" / pid
    target.mkdir(parents=True, exist_ok=True)
    dev = [evaluate(mod, inputs, s) for s in DEV]
    index = dev[0]["index"]
    cols = {"output": np.ones(index.size, dtype=bool)} if family != "F3" else \
        {"mean": (index % 2) == 0, "rstd": (index % 2) == 1}
    shape = tuple(dev[0]["y"].shape)
    valid = {name: np.stack([d["ok"][sel] for d in dev]).all(axis=0) for name, sel in cols.items()}
    r5 = {}
    for mode in MODES:
        es = [e_sem(d, family, s, mode) for s, d in zip(DEV, dev)]
        for name, sel in cols.items():
            mid = np.stack([0.5 * (lo[sel] + hi[sel]) for lo, hi in es])[:, valid[name]]
            w = mid.mean(axis=0)
            norm = np.linalg.norm(w)
            r5[(mode, name)] = w / norm if norm > 0 else None
            if r5[(mode, name)] is not None:
                np.save(target / f"R5_direction_{mode}_{name}.npy", r5[(mode, name)])
    for name, v in valid.items():
        np.save(target / f"valid_coordinates_{name}.npy", v)
    hash_check = {}
    for seed in SEEDS:
        d = dev[seed] if seed in DEV else evaluate(mod, inputs, seed)
        hash_check[f"seed{seed:03d}"] = {"exported": d["hash"], "phase1_run": phase1[pid]["output_hashes"][seed]}
        fam_dir = OUT / "family_inputs_and_f" / family / f"seed{seed:03d}"
        if not (fam_dir / "inputs.npz").exists():
            fam_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(fam_dir / "inputs.npz", **{key: v.detach().cpu().numpy()
                                                          for key, v in d["inp"].items() if torch.is_tensor(v)})
            (fam_dir / "scalars.json").write_text(json.dumps(
                {key: {"value": v, "repr": repr(v), "float32": float(np.float32(v))}
                 for key, v in d["inp"].items() if not torch.is_tensor(v)}, indent=2))
            f = np.load(FCACHE / f"{family}_{seed:03d}.npz")
            for mode in MODES:
                np.save(fam_dir / f"f_{mode}_lo.npy", f[f"{mode}_lo"])
                np.save(fam_dir / f"f_{mode}_hi.npy", f[f"{mode}_hi"])
        sd = target / f"seed{seed:03d}"
        sd.mkdir(exist_ok=True)
        np.save(sd / "K.npy", d["y"].detach().cpu().numpy())
        (sd / "K_raw.bin").write_bytes(d["y"].detach().cpu().contiguous().numpy().tobytes())
        np.save(sd / "output_flat_index.npy", d["index"])
        np.save(sd / "KR_lo.npy", d["lo"])
        np.save(sd / "KR_hi.npy", d["hi"])
        proj = {}
        kr_mid = 0.5 * (d["lo"] + d["hi"])
        for mode in MODES:
            e_lo, e_hi = e_sem(d, family, seed, mode)
            np.save(sd / f"e_sem_{mode}_lo.npy", e_lo)
            np.save(sd / f"e_sem_{mode}_hi.npy", e_hi)
            proj[mode] = {}
            for name, sel in cols.items():
                v = valid[name]
                lo, hi, kr = e_lo[sel][v], e_hi[sel][v], kr_mid[sel][v]
                rules = ["R1", "R2", "R3"] + (["R4"] if family == "F4" else [])
                proj[mode][name] = {}
                for rule in rules + ["R5"]:
                    w = r5[(mode, name)] if rule == "R5" else direction(rule, kr, family, shape)
                    if w is not None:
                        proj[mode][name][rule] = projections(lo, hi, w)
        (sd / "tool_values.json").write_text(json.dumps({
            "program": pid, "family": family, "seed": seed, "projections": proj,
            "e_sem": "[L_R - U_f, U_R - L_f]; e_sem_<mode>_lo/hi.npy over output_flat_index.npy (row-major)",
            "K_R": "KR_lo.npy / KR_hi.npy over output_flat_index.npy; the pipeline recovers K_R from the residual "
                   "interval [K - hi, K - lo] with directed additions, so e_sem can be one ulp wider than KR - f",
            "f": f"family_inputs_and_f/{family}/seed{seed:03d}/f_<mode>_lo/hi.npy, full output flattened row-major",
            "directions": "R1 -1/sqrt(n); R2 -sign(K_R mid)/sqrt(n); R3 -K_R mid/|K_R mid|; R4 F4 pattern; "
                          "R5 saved .npy (per mode and output); all over valid_coordinates_<output>.npy",
            "projection_kinds": {"pipeline": "ordinary products and numpy sum, as in the statistics",
                                 "directed": "directed products (intervals.imul), fsum, one ulp outward"}},
            indent=2))
    verdicts = {mode: {o["output"]: {"residual": o["residual"],
                                     "rules": {r: {"final_verdict": x.get("final_verdict"), "mu_interval": x.get("mu_interval")}
                                               for r, x in o["rules"].items() if isinstance(x, dict)},
                                     "default_detector": {f: o.get("default_detector", {}).get(f, {}).get("final_verdict")
                                                          for f in ("vector_mean", "alignment")}}
                       for o in rep["modes"][mode]} for mode in MODES}
    (target / "program.json").write_text(json.dumps({
        "program": pid, "family": family, "selected_because": why, "output_hash_check": hash_check,
        "phase2_report_values": verdicts}, indent=2, default=float))
    ok = all(v["exported"] == v["phase1_run"] for v in hash_check.values())
    print(pid, family, "exported; K hash matches phase 1 run:", ok, flush=True)


def write_index(sel, report):
    entries = []
    for pid in sorted(sel, key=lambda p: (report[p]["family"], p)):
        pj = json.loads((OUT / "programs" / pid / "program.json").read_text())
        entries.append({"program": pid, "family": report[pid]["family"], "selected_because": sel[pid],
                        "K_hash_matches_phase1_run": all(v["exported"] == v["phase1_run"]
                                                         for v in pj["output_hash_check"].values())})
    (OUT / "index.json").write_text(json.dumps({
        "selection": "programs with a phase 2 finding: a rule detection in either scalar mode, a default-detector "
                     "alert, or e_sem intervals excluding zero on >= 99% of coordinates",
        "seeds": list(SEEDS), "programs": entries}, indent=2) + "\n")
    print("index written:", len(entries), "programs")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--programs", default=None)
    parser.add_argument("--index", action="store_true")
    args = parser.parse_args()
    report = json.loads((REPORTS / "phase2" / "phase2_report.json").read_text())
    sel = selection(report)
    if args.index:
        write_index(sel, report)
        return
    phase1 = json.loads((REPORTS / "phase1_report.json").read_text())
    sys.path.insert(0, str(args.package / "programs"))
    inputs = importlib.import_module("inputs")
    wanted = args.programs.split(",") if args.programs else sorted(sel)
    for pid in wanted:
        export_program(pid, inputs, report, phase1, sel[pid])


if __name__ == "__main__":
    main()
