#!/usr/bin/env python3
"""Numerical-effect stream of the closure (protocol v3 appendix A, declared and committed in 84fe39e before this run).

20 low-precision Inductor programs, mode A (e_num = K - K_R, tool 2.3, frozen detector thresholds), 32 development units
(seed 2000-2031) and 64 confirmation units (seed 2032-2095), output `out`; the frozen direction rules and default detector,
then contract_v3.statistical_judgment.  Output: results/closure/numerical_stream.json.

    python scripts/closure/numerical_stream.py
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
for p in ("scripts/essential", "scripts/closure", "src"):
    sys.path.insert(0, str(ROOT / p))
import contract_v3 as CV  # noqa: E402
import p2b_g6_g7 as G  # noqa: E402

OUT = ROOT / "results/closure/numerical_stream.json"
DEV = list(range(2000, 2032))
CONF = list(range(2032, 2096))
gen = G.gen


def rot(x):
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], -1)


def programs():
    keep = [p for p in G.g7_programs() if p[0] not in ("var_bf16", "scatter_add_bf16", "embedding_bag_mean_bf16")]
    assert len(keep) == 17
    bf, hf = torch.bfloat16, torch.float16
    inv = 1.0 / (10000 ** (torch.arange(0, 16, 2, dtype=torch.float64) / 16))
    ang = torch.arange(11, dtype=torch.float64)[:, None] * inv[None, :]
    emb = torch.cat([ang, ang], -1)
    keep.append(("rope_rotate_half_bf16", bf, lambda s: [gen(s, 2, 4, 11, 16), torch.cos(emb), torch.sin(emb)],
                 lambda x, c, sn: x * c + rot(x) * sn))
    keep.append(("clip_scale_bf16", bf, lambda s: [gen(s, 64, 16)],
                 lambda g: g * torch.clamp(1.0 / (torch.sqrt((g.float() ** 2).sum()) + 1e-6), max=1.0).to(g.dtype)))
    keep.append(("layer_norm_fp16", hf, lambda s: [gen(s, 8, 64)], lambda x: F.layer_norm(x, (64,))))
    assert len(keep) == 20
    return keep


def run_program(name, dt, make, fn):
    from kernel_analyzer import check
    st = {}

    class Case(check.Case):
        def setup(self):
            torch._dynamo.reset()
            st["fn"] = torch.compile(fn, dynamic=False)
            self.launch(self.inputs(DEV[0]))
            torch.cuda.synchronize()

        def inputs(self, seed):
            return {"args": [a.to("cuda", dt) for a in make(seed)], "_seed": seed}

        def launch(self, t):
            return {"out": st["fn"](*t["args"])}

    c = Case()
    c.name = c.implementation = f"stream/{name}"
    c.specification = "none (mode A)"
    t0 = time.time()
    rep = check.run(c, dev=DEV, conf=CONF)
    o = rep.get("outputs", {}).get("out", {})
    num = o.get("numerical") or {}
    rules = num.get("rules") or []
    judg = {r.get("rule"): CV.statistical_judgment(r) for r in rules if isinstance(r, dict)}
    return {"status": "ok", "dtype": str(dt), "tool_version": rep.get("tool_version"),
            "launches": len(rep.get("launches") or []),
            "complete_fraction": (o.get("reference_classes") or {}).get("finite_complete_fraction"),
            "not_written_by_triton": rep.get("outputs_not_written_by_triton"),
            "binding_not_established": rep.get("outputs_binding_not_established"),
            "frozen_rules": [{"rule": r.get("rule"), "verdict": r.get("verdict"), "n": r.get("n"),
                              "unit_skewness": r.get("unit_skewness"), "mean_projection": r.get("mean_projection"),
                              "t_interval": r.get("t_interval"), "robust": (r.get("robust") or {}).get("verdict")} for r in rules],
            "judgment_v3": judg,
            "default_detector": {k: (num.get("default_detector") or {}).get(k) for k in ("verdict", "decision", "p_value")},
            "scale": num.get("scale"), "seconds": round(time.time() - t0, 1)}


def main():
    res, t0 = {}, time.time()
    for name, dt, make, fn in programs():
        try:
            res[name] = run_program(name, dt, make, fn)
        except Exception as exc:  # noqa: BLE001
            res[name] = {"status": "error", "reason": f"{type(exc).__name__}: {exc}"[:400], "trace": traceback.format_exc()[-1500:]}
        r = res[name]
        if r["status"] == "ok":
            print(name, "complete", r["complete_fraction"], {k: v["judgment"] for k, v in r["judgment_v3"].items()},
                  "rel_rms", (r["scale"] or {}).get("relative_rms"), f"{r['seconds']} s", flush=True)
        else:
            print(name, r["status"], r["reason"][:300], flush=True)
    OUT.write_text(json.dumps({"declared_in": "docs/protocol_closure_v3_20261008.md appendix A (commit 84fe39e)",
                               "units": {"development": DEV[:1] + ["..."] + DEV[-1:], "confirmation": CONF[:1] + ["..."] + CONF[-1:]},
                               "seconds": round(time.time() - t0, 1), "programs": res}, indent=1, default=str) + "\n")


if __name__ == "__main__":
    main()
