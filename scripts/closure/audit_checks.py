#!/usr/bin/env python3
"""Evidence for two numbers the closure reports quoted from interactive checks (audit 2026-10-08):
(1) EMB-A1: the float64 eager max_norm renormalisation equals w * max_norm / (norm + 1e-7) (high-precision recomputation);
(2) SPEC-ISSUE-1: the torch SGD maximize momentum buffer equals minus the spec's b, parameters agree.
Writes results/closure/audit_checks.json."""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from mpmath import mp

ROOT = Path(__file__).resolve().parents[2]
for p in ("scripts/essential", "scripts/closure", "specs/phase2"):
    sys.path.insert(0, str(ROOT / p))
mp.prec = 200
out = {}
import p2b_embedding as EM  # noqa: E402
c = {x["id"]: x for x in EM.conditions()}["emb_embedding_padNone_mn1.0_nofreq"]
d = pickle.loads((EM.CACHE / "eager_cpu_float64.pkl").read_bytes())
for cc in (0.0, 1e-7):
    worst = 0.0
    for seed in EM.SEEDS:
        inp = EM.make_inputs(c, seed)
        W = np.asarray(inp["w"], float)
        idx = np.asarray(inp["idx"]).ravel()
        K = np.asarray(d["res"][(c["id"], seed)]["base"]["outputs"]["out"], float).reshape(len(idx), -1)
        for r, j in enumerate(idx):
            row = [mp.mpf(float(v)) for v in W[j]]
            n = mp.sqrt(sum(v * v for v in row))
            ref = [v / (n + cc) if n > 1 else v for v in row]
            worst = max(worst, max(abs(float(K[r, k]) - float(ref[k])) / (1 + abs(float(ref[k]))) for k in range(len(row))))
    out[f"emb_max_norm_float64_vs_w_over_norm_plus_{cc:g}"] = worst
import f_eval_2b as FE  # noqa: E402
import p2b_optimizers as OP  # noqa: E402
cid = "opt_sgd_nesterov_cold_max_wd0.0"
for dt, name in (("float64", "torch_for_loop_cpu_float64"), ("float32", "torch_foreach_cuda")):
    _, _, _, fa, _ = FE.optimizer_job((cid, 0, "base", dt))
    r = FE.load(OP.CACHE / f"{name}.pkl")["res"][(cid, 0)]["base"]
    g = FE.opt_groups(r, fa)
    k, lo, hi = g["state.momentum_buffer"]
    mid = (lo + hi) / 2
    pk, plo, phi = g["param"]
    pmid = (plo + phi) / 2
    out[f"sgd_maximize_{name}"] = {"buffer_vs_spec_b_max_rel": float(np.max(np.abs(k - mid) / (1 + np.abs(mid)))),
                                   "buffer_vs_minus_spec_b_max_rel": float(np.max(np.abs(k + mid) / (1 + np.abs(mid)))),
                                   "param_vs_spec_max_rel": float(np.max(np.abs(pk - pmid) / (1 + np.abs(pmid)))),
                                   "buffer_elements": int(k.size), "param_elements": int(pk.size)}
(ROOT / "results/closure/audit_checks.json").write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps(out, indent=1))
