#!/usr/bin/env python3
"""Runner for the importance calibration (docs/importance_calibration_protocol_20261006.md).

    python scripts/importance/run.py work --phase b1 --worker 0 --workers 8     # rulers, dose-response, reference states
    python scripts/importance/run.py single-step                                 # B-2 measurements (needs b1 states)
    python scripts/importance/run.py work --phase b3 --worker 0 --workers 8     # replacement trainings (needs predictions)

Runs of R' and P (outcomes of A1 and X4) and every replacement run are written to results/importance/runs_sealed/;
``summary`` reads them only when results/importance/predictions.json exists.
"""
import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
OUT = ROOT / "results/importance"
CKPT = ROOT / ".cache/importance/ckpt"
STATE_STEPS = (250, 500, 1000, 1500)

B = {"rmsnorm": "torch", "rope": "torch", "swiglu": "torch", "ce": "torch"}
REFS = {"R": {}, "Rp": {"precision": "bf16"}, "P": {"precision": "bf16", "optimizer": "adamw_bf16"}}
ITEMS = {  # name: (configuration kwargs, reference)
    "A1": (REFS["Rp"], "R"),
    "A2": ({"precision": "tf32"}, "R"),
    "A3": ({"compile": True}, "R"),
    "X1": ({**REFS["Rp"], "impl": {**B, "rope": "unsloth"}}, "Rp"),
    "X2": ({**REFS["Rp"], "impl": {**B, "swiglu": "unsloth"}}, "Rp"),
    "X3": ({**REFS["Rp"], "impl": {**B, "ce": "unsloth"}}, "Rp"),
    "X4": (REFS["P"], "Rp"),
    "X5": ({**REFS["P"], "impl": {**B, "rmsnorm": "compiled_hf"}}, "P"),
    "X6": ({**REFS["P"], "impl": {**B, "rope": "unsloth"}}, "P"),
}
GAMMAS = (1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, -3e-2, -1e-1)
BS = (1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1)


def jobs(phase):
    out = []
    if phase == "b1":
        for s in range(8):
            out.append({"name": "R", "seed": s, "kw": {}, "sealed": False, "ckpt": s == 0})
        for ref in ("Rp", "P"):
            out.append({"name": ref, "seed": 0, "kw": REFS[ref], "sealed": True, "ckpt": True})
        for g in GAMMAS:
            for s in range(4):
                out.append({"name": f"inject_gamma_{g:g}", "seed": s, "kw": {"inject": "gamma", "dose": g}, "sealed": False})
        for b in BS:
            for s in range(4):
                out.append({"name": f"inject_direction_{b:g}", "seed": s, "kw": {"inject": "direction", "dose": b}, "sealed": False})
    elif phase == "b3":
        for ref in ("Rp", "P"):
            for s in range(1, 4):
                out.append({"name": ref, "seed": s, "kw": REFS[ref], "sealed": True})
        for name, (kw, ref) in ITEMS.items():
            if name in ("A1", "X4"):  # the same runs as R' and P
                continue
            for s in range(4):
                out.append({"name": name, "seed": s, "kw": kw, "sealed": True})
    return out


def run_job(job):
    import torch

    import small_lm as S

    sub = "runs_sealed" if job["sealed"] else "runs"
    path = OUT / sub / f"{job['name']}__s{job['seed']}.json"
    if path.exists():
        return
    S._COMPILED.clear()
    torch._dynamo.reset()
    c = S.Config(seed=job["seed"], **job["kw"], checkpoints=STATE_STEPS if job.get("ckpt") else ())
    res = S.train(c, out_dir=CKPT / f"{job['name']}_s{job['seed']}" if job.get("ckpt") else None)
    res["job"] = job
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res, indent=1, default=float) + "\n")
    print(f"done {job['name']} s{job['seed']} {res['seconds']}s", flush=True)  # the loss is not printed (sealing)


def single_step():
    import numpy as np

    import single_step as SS
    import small_lm as S

    (OUT / "single_step").mkdir(parents=True, exist_ok=True)
    for name, (kw, ref) in ITEMS.items():
        path = OUT / "single_step" / f"{name}.json"
        if path.exists():
            continue
        S._COMPILED.clear()
        ck = [CKPT / f"{ref}_s0" / f"ckpt_{t:05d}.pt" for t in STATE_STEPS]
        ref_cfg = S.Config(**REFS[ref])
        cand_cfg = S.Config(**kw)
        u, r, n_dev = SS.measure(ck, ref_cfg, cand_cfg)
        eff = SS.effects(u, r, n_dev)
        from kernel_analyzer.reference_eval.analysis import assess_units

        units = u.shape[0]
        rec, _ = assess_units(f"{name}: u at the update layer", u, u, r, np.ones_like(u, dtype=bool), n_dev,
                              ["R1", "R2", "R3", "R5"], alignment_reference=r, run_detector=False,
                              unit_ids=list(range(units)))
        eff["rules"] = [{k: x.get(k) for k in ("rule", "verdict", "mean_projection", "p_value_two_sided_conservative",
                                                 "unit_skewness", "t_approximation")} for x in rec["rules"]]
        eff.update(item=name, reference=ref, config=kw, states=list(STATE_STEPS), units=units, development=n_dev)
        path.write_text(json.dumps(eff, indent=1, default=float) + "\n")
        print(f"single-step {name}: gamma {eff['gamma']['mean']:.3e} [{eff['gamma']['lo']:.3e}, {eff['gamma']['hi']:.3e}] "
              f"b {eff['b']['mean']:.3e} [{eff['b']['lo']:.3e}, {eff['b']['hi']:.3e}] rel_rms {eff['relative_rms']:.2e}",
              flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["work", "single-step", "list"])
    ap.add_argument("--phase", default="b1")
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    if a.command == "list":
        for j in jobs(a.phase):
            print(j)
        return
    if a.command == "single-step":
        sys.path.insert(0, str(ROOT / "src"))
        single_step()
        return
    if a.phase == "b3" and not (OUT / "predictions.json").exists():
        raise SystemExit("predictions.json must be committed before the replacement trainings (protocol section 6)")
    for j in jobs(a.phase)[a.worker::a.workers]:
        run_job(j)


if __name__ == "__main__":
    main()
