#!/usr/bin/env python3
"""Layer table, stage A item 5 (docs/next_phase_plan_20261006.md): the actual-write layer (L4) of the RMSNorm layer
table replayed with the same gradients and states on the CUDA optimizer paths, next to the original CPU
single-tensor path.

    L4 path            device  implementation
    cpu_single         CPU     torch.optim.{SGD,AdamW}(foreach=False)        (the original table)
    cuda_foreach       CUDA    foreach=True
    cuda_fused         CUDA    fused=True

Gradients: K and the enclosure K_R of the compiled RMSNorm's weight-gradient kernel per seed (seeds 0-95, the same
seeds and deterministic kernels as the original run; the arrays are saved to .cache/layer_table/grad_rows.npz and
reused for every path).  States: zero (t = 1) and the history state of the protocol (9 steps of specification
gradients, generator seeds 2000-2008).  Inputs and history are synthetic (N(0, 1) bf16 activations), as declared.

    python scripts/layer_table_cuda_replay.py     # -> results/reference_eval/layer_table/cuda_replay.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import layer_table_m4 as m4  # noqa: E402
from kernel_analyzer import check  # noqa: E402
from kernel_analyzer.reference_eval import intervals as iv  # noqa: E402
from kernel_analyzer.reference_eval.analysis import assess_units, holm_adjusted  # noqa: E402

CACHE = ROOT / ".cache/layer_table/grad_rows.npz"
OUT = ROOT / "results/reference_eval/layer_table/cuda_replay.json"
PATHS = {"cpu_single": ("cpu", {"foreach": False}), "cuda_foreach": ("cuda", {"foreach": True}),
         "cuda_fused": ("cuda", {"fused": True})}


def grad_rows():
    if CACHE.exists():
        z = np.load(CACHE)
        return [{"k": z["k"][i], "r_lo": z["lo"][i], "r_hi": z["hi"][i], "ok": z["ok"][i]} for i in range(z["k"].shape[0])]
    keep = {}
    check.run(m4.RMSNormCase(), dev=m4.DEV, conf=m4.CONF, keep=keep)
    rows = keep["grad_w"]
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, k=np.stack([r["k"] for r in rows]), lo=np.stack([r["r_lo"] for r in rows]),
                        hi=np.stack([r["r_hi"] for r in rows]), ok=np.stack([r["ok"] for r in rows]))
    return rows


def step(opt_name, path, th, grad, state):
    device, flags = PATHS[path]
    p = torch.nn.Parameter(torch.from_numpy(th.astype(np.float32).copy()).to(device))
    if opt_name == "sgd":
        opt = torch.optim.SGD([p], lr=float(m4.LR), **flags)
    else:
        opt = torch.optim.AdamW([p], lr=float(m4.LR), betas=tuple(float(b) for b in m4.BETAS), eps=float(m4.EPS),
                                weight_decay=0.0, **flags)
        if state is not None:
            m, v, steps = state
            step_t = torch.tensor(float(steps), device=device if flags.get("fused") else "cpu")
            opt.state[p] = {"step": step_t, "exp_avg": torch.from_numpy(m.astype(np.float32)).to(device),
                            "exp_avg_sq": torch.from_numpy(v.astype(np.float32)).to(device)}
    p.grad = torch.from_numpy(np.asarray(grad, dtype=np.float32).copy()).to(device)
    opt.step()
    if device == "cuda":
        torch.cuda.synchronize()
    return p.detach().cpu().numpy().astype(np.float64)


def main():
    rows = grad_rows()
    th = m4.theta().numpy().astype(np.float64)
    m_h, v_h = m4.history_state()
    states = {"sgd": None, "adamw_zero": None, "adamw_history": (m_h, v_h, 9)}
    table = []
    for opt_name, state in states.items():
        for path in PATHS:
            name = "sgd" if opt_name == "sgd" else "adamw"
            lo_all, hi_all, ref_all, ok_all = [], [], [], []
            replay = {"unique": 0, "enumerated": 0, "not_established": 0}
            for r in rows:
                k, lo, hi, ok = r["k"], r["r_lo"], r["r_hi"], r["ok"].astype(bool)
                t_k = step(name, path, th, k, state)
                cands, count = m4.candidates(lo, hi)
                outs = np.stack([step(name, path, th, c, state) for c in cands])
                a_lo, a_hi = iv.isub(t_k, t_k, outs.min(axis=0), outs.max(axis=0))
                ref = step(name, path, th, np.float32(0.5 * (lo + hi)), state) - th
                ok_a = ok & (count <= m4.MAX_CANDIDATES)
                replay["unique"] += int((ok & (count == 1)).sum())
                replay["enumerated"] += int((ok & (count > 1) & (count <= m4.MAX_CANDIDATES)).sum())
                replay["not_established"] += int((~ok_a).sum())
                lo_all.append(a_lo), hi_all.append(a_hi), ref_all.append(ref), ok_all.append(ok_a)
            lo, hi, ref, ok = (np.stack(x) for x in (lo_all, hi_all, ref_all, ok_all))
            rec, _ = assess_units(f"L4 {opt_name} {path}", np.where(ok, lo, 0), np.where(ok, hi, 0), np.where(ok, ref, 0),
                                  ok, len(m4.DEV), check.RULES, alignment_reference=np.where(ok, ref, 0),
                                  unit_ids=list(m4.DEV) + list(m4.CONF))
            mid = 0.5 * (lo + hi)
            ref_norm = float(np.sqrt((ref[ok] ** 2).mean()) * np.sqrt(m4.HIDDEN))
            row = m4.summarize(f"L4 actual write ({opt_name}, {path})", rec, ref_norm)
            row.update(optimizer=opt_name, path=path, device=PATHS[path][0], flags=PATHS[path][1], replay=replay,
                       state_source="zero" if opt_name != "adamw_history" else "9 steps of specification gradients (seeds 2000-2008)",
                       relative_rms=float(np.sqrt((mid[ok] ** 2).mean() / max((ref[ok] ** 2).mean(), 1e-300))))
            table.append(row)
            print(f"{opt_name:14s} {path:12s} " + " ".join(
                f"{x['rule']}:{x.get('verdict', '')[:12]}({x.get('effect_relative', float('nan')):.1e})" for x in row["rules"]),
                f"rel_rms {row['relative_rms']:.2e}", flush=True)
    tests = [(i, j) for i, row in enumerate(table) for j, r in enumerate(row["rules"]) if r.get("p") is not None]
    for (i, j), adj in zip(tests, holm_adjusted([table[i]["rules"][j]["p"] for i, j in tests])):
        r = table[i]["rules"][j]
        r["holm_p"] = adj
        r["final"] = r["verdict"] if adj <= 0.05 and str(r["verdict"]).startswith("DETECTED") else "NOT_CONFIRMED"
    OUT.write_text(json.dumps({"inputs": "synthetic N(0,1) bf16 activations and upstream gradients (seeds 0-95)",
                               "holm_family": len(tests), "rows": table}, indent=1, default=float) + "\n")


if __name__ == "__main__":
    main()
