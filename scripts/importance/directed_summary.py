#!/usr/bin/env python3
"""Stage C summary (docs/directed_search_protocol_20261007.md section 3): classification of every candidate against
the frozen delta (0.05, relative to the per-step update norm), per seed and combined, and rho against the accepted
changes of stage B.

    python scripts/importance/directed_summary.py     # -> results/directed_search/summary.json
"""
import json
import math
from pathlib import Path

from scipy.stats import t as tdist

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/directed_search"
DELTA = 0.05
NAMES = {"C1": "torch AdamW bf16, foreach", "C2": "torch AdamW bf16, fused", "C3": "torchao AdamW8bit",
         "C4": "torchao AdamW4bit", "C6": "torchao _AdamW bf16 stochastic rounding", "C7": "bitsandbytes AdamW8bit",
         "C8": "torch Muon, Newton-Schulz in bf16 (vs fp32)", "C9": "weight EMA stored in bf16 (vs fp32)",
         "C10": "bf16 gradient accumulation in pure bf16 training (vs fp32 buffer)"}


def classify(e):
    def outside(c):  # the whole interval beyond delta in magnitude
        return c["lo"] > DELTA or c["hi"] < -DELTA

    def inside(c):
        return max(abs(c["lo"]), abs(c["hi"])) < DELTA

    if outside(e["gamma"]) or outside(e["b"]):
        return "exceeds"
    if inside(e["gamma"]) and inside(e["b"]):
        return "within"
    return "undecided"


def ci975(e):
    q975, q95 = tdist.ppf(1 - 0.0125, e["n"] - 1), tdist.ppf(1 - 0.025, e["n"] - 1)
    h = (e["hi"] - e["mean"]) * q975 / q95
    return e["mean"] - h, e["mean"] + h


def rho(x, a):
    x0, x1 = ci975(x)
    a0, a1 = ci975(a)
    if a0 <= 0 <= a1:
        return "undefined"
    r = [p / q for p in (x0, x1) for q in (a0, a1)]
    return {"point": x["mean"] / a["mean"], "lo": min(r), "hi": max(r)}


def main():
    per = {}
    for p in sorted(OUT.glob("shadow_*_s*.json")):
        res = json.loads(p.read_text())
        for name, e in res.get("candidates_effects", {}).items():
            per.setdefault(name, {})[f"s{res['seed']}"] = {"effects": e, "class": classify(e),
                                                          "ema_val": {k: res.get(f"ema_{k}_val_loss") for k in ("fp32", "bf16")}
                                                          if name == "C9" else None}
    c10 = OUT / "single_step_C10.json"
    if c10.exists():
        e = json.loads(c10.read_text())
        per["C10"] = {"states_P_s0": {"effects": e, "class": classify(e)}}
    stage_b = {a: json.loads((ROOT / "results/importance/single_step" / f"{a}.json").read_text()) for a in ("A1", "A2", "A3")}
    table = {}
    for name, seeds in sorted(per.items(), key=lambda kv: int(kv[0][1:])):
        classes = {s: v["class"] for s, v in seeds.items()}
        combined = list(classes.values())[0] if len(set(classes.values())) == 1 else "undecided (seeds disagree)"
        first = list(seeds.values())[0]["effects"]
        table[name] = {"candidate": NAMES.get(name, name), "classes": classes, "combined": combined,
                       "gamma": {s: v["effects"]["gamma"] for s, v in seeds.items()},
                       "b": {s: v["effects"]["b"] for s, v in seeds.items()},
                       "relative_rms": {s: v["effects"]["relative_rms"] for s, v in seeds.items()},
                       "T_star": {s: v["effects"].get("T_star_tau1") for s, v in seeds.items()},
                       "rules": {s: [(r["rule"], r.get("verdict")) for r in v["effects"]["rules"]] for s, v in seeds.items()},
                       "rho_vs_A": {f"{k}/{a}": rho(first[k], stage_b[a][k]) for a in ("A1", "A2", "A3") for k in ("gamma", "b")},
                       "upper_bound_over_delta": max(max(abs(first[k]["lo"]), abs(first[k]["hi"])) for k in ("gamma", "b")) / DELTA}
        if name == "C9":
            table[name]["ema_val_loss"] = {s: v["ema_val"] for s, v in seeds.items()}
    (OUT / "summary.json").write_text(json.dumps({"delta": DELTA, "candidates": table}, indent=1, default=float) + "\n")
    for name, t in table.items():
        g = list(t["gamma"].values())[0]
        b = list(t["b"].values())[0]
        print(f"{name:4s} {t['candidate'][:44]:44s} {t['combined']:28s} gamma {g['mean']: .2e} [{g['lo']: .2e},{g['hi']: .2e}] "
              f"b {b['mean']: .2e} [{b['lo']: .2e},{b['hi']: .2e}] ub/delta {t['upper_bound_over_delta']:.3g}")


if __name__ == "__main__":
    main()
