#!/usr/bin/env python3
"""Shadow-state single-step measurement for stage C (docs/directed_search_protocol_20261007.md section 2).

The reference configuration trains normally.  Every candidate optimizer keeps its own state over a shadow copy of
the parameters: at every step the shadow parameters are reset to the reference theta_t (cast to the candidate's
parameter dtype), given the reference's clipped gradient (cast likewise) and stepped, so the candidate's state
accumulates its own format's drift while the trajectory does not diverge.  At the measured steps
    u_t = (theta_X+ - theta_X) - (theta_R+ - theta_R)
on the same theta_t and gradient.  The weight EMA candidate (bf16 against fp32 EMA buffers) is measured the same
way on the EMA update.

Two passes over the identical deterministic run: pass 1 accumulates the development-unit sums (nu_hat) and the
per-unit scalars; pass 2 projects the confirmation units on nu_hat (and checks that its per-unit gamma equals pass 1).

    python scripts/importance/shadow.py --group ka --seed 0       # R' + shadows C1, C2, and the EMA pair (C9)
    python scripts/importance/shadow.py --group liger --seed 0    # R' + shadows C3-C7   (liger environment)
    python scripts/importance/shadow.py --group muon --seed 0     # Muon with fp32 Newton-Schulz + shadow C8
"""
import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import small_lm as S  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/directed_search"
CACHE = ROOT / ".cache/directed_search"
MEASURE = list(range(100, 2000, 20))  # 95 units; development units: index % 3 == 0
RP = {"precision": "bf16"}
GROUPS = {
    "ka": (dict(RP), {"C1": {"optimizer": "adamw_bf16_foreach"}, "C2": {"optimizer": "adamw_bf16_fused"}}, True),
    "liger": (dict(RP), {"C3": {"optimizer": "ao_adamw8bit"}, "C4": {"optimizer": "ao_adamw4bit"},
                         "C6": {"optimizer": "ao_adamw_bf16sr"},
                         "C7": {"optimizer": "bnb_adamw8bit"}}, False),
    "muon": ({**RP, "optimizer": "muon_fp32ns"}, {"C8": {"optimizer": "muon"}}, False),
    # deep case C4 (protocol section 9): attribution and the block-size intervention, next to C4 in the same run
    "c4deep": (dict(RP), {"C4": {"optimizer": "ao_adamw4bit"}, "C4a": {"optimizer": "ao_adamw4bit_vonly"},
                          "C4b": {"optimizer": "ao_adamw4bit_monly"}, "C4c": {"optimizer": "ao_adamw4bit_b32"}}, False),
}
BF16_PARAMS = {"adamw_bf16", "adamw_bf16_foreach", "adamw_bf16_fused", "ao_adamw_bf16sr"}
STEPS = None  # smoke runs only


def flat(params):
    return torch.cat([p.detach().double().reshape(-1) for p in params])


def run(group, seed, nu=None):
    ref_kw, cands, with_ema = GROUPS[group]
    c = S.Config(seed=seed, **ref_kw, **({"steps": STEPS} if STEPS else {}))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    S.setup_precision(c)
    model, opt = S.build(c)
    data = S.Data(c, "cuda")
    shadows = {}
    for name, kw in cands.items():
        cc = replace(c, **kw)
        torch.manual_seed(seed)
        sm = S.SmallLM(cc).cuda()
        if cc.optimizer in BF16_PARAMS:
            sm = sm.to(torch.bfloat16)
        shadows[name] = (sm, S.make_optimizer(sm, cc), cc)
    ema = {"fp32": [p.detach().clone().float() for p in model.parameters()],
           "bf16": [p.detach().clone().to(torch.bfloat16) for p in model.parameters()]} if with_ema else None
    decay = 0.999
    rows = {name: [] for name in list(shadows) + (["C9"] if with_ema else [])}
    sums = {name: {"dev": None, "h0": None, "h1": None, "n0": 0, "n1": 0, "sq": 0.0, "sum": None, "n": 0} for name in rows}
    measure = {s: i for i, s in enumerate(MEASURE)}
    t0 = time.time()
    for step in range(c.steps):
        lr = S.lr_at(step, c)
        S.set_lr(opt, lr)
        x, y = data.batch(step)
        with S.autocast(c):
            loss = S.forward(model, x, y, c)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), c.clip)
        params = list(model.parameters())
        theta = [p.detach().clone() for p in params]
        deltas = {}
        for name, (sm, so, cc) in shadows.items():
            S.set_lr(so, lr)
            with torch.no_grad():
                for sp, t, p in zip(sm.parameters(), theta, params):
                    sp.copy_(t.to(sp.dtype))
                    sp.grad = p.grad.detach().to(sp.dtype)
            before = flat(sm.parameters()) if step in measure else None
            so.step()
            if step in measure:
                deltas[name] = flat(sm.parameters()) - before
            for sp in sm.parameters():
                sp.grad = None
        before_r = flat(params) if step in measure else None
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step in measure:
            r = flat(params) - before_r
        if ema is not None:
            with torch.no_grad():
                e_before = (torch.cat([e.double().reshape(-1) for e in ema["fp32"]]),
                            torch.cat([e.double().reshape(-1) for e in ema["bf16"]])) if step in measure else None
                for key in ("fp32", "bf16"):
                    for e, p in zip(ema[key], params):
                        e.mul_(decay).add_(p.detach().to(e.dtype), alpha=1 - decay)
                if step in measure:
                    d32 = torch.cat([e.double().reshape(-1) for e in ema["fp32"]]) - e_before[0]
                    d16 = torch.cat([e.double().reshape(-1) for e in ema["bf16"]]) - e_before[1]
                    deltas["C9"] = (d16 - d32, d32)
        if step in measure:
            i = measure[step]
            for name in rows:
                if name == "C9":
                    u, rr = deltas["C9"]
                else:
                    u, rr = deltas[name] - r, r
                n = rr.numel()
                rn = float(rr.norm())
                row = {"step": step, "dev": i % 3 == 0, "r_norm": rn, "u_norm": float(u.norm()),
                       "gamma": float(u @ rr) / rn ** 2, "p_R1": -float(u.sum()) / math.sqrt(n),
                       "p_R2": -float(u @ torch.sign(rr)) / math.sqrt(n), "p_R3": -float(u @ rr) / rn}
                sm_ = sums[name]
                if row["dev"]:
                    sm_["dev"] = u.clone() if sm_["dev"] is None else sm_["dev"] + u
                else:
                    h = sm_["n"] % 2
                    sm_[f"h{h}"] = u.clone() if sm_[f"h{h}"] is None else sm_[f"h{h}"] + u
                    sm_[f"n{h}"] += 1
                    sm_["sq"] += float(u @ u)
                    sm_["sum"] = u.clone() if sm_["sum"] is None else sm_["sum"] + u
                    sm_["n"] += 1
                    if nu is not None:
                        row["b"] = float(u @ nu[name]) / rn
                        row["p_R5"] = float(u @ nu[name])
                rows[name].append(row)
    out = {"group": group, "seed": seed, "reference": asdict(c), "candidates": {k: v for k, v in cands.items()},
           "seconds": round(time.time() - t0, 1), "val_loss_reference": S.evaluate(model, data, c), "rows": rows}
    if ema is not None:
        saved = [p.detach().clone() for p in model.parameters()]
        for key in ("fp32", "bf16"):
            with torch.no_grad():
                for p, e in zip(model.parameters(), ema[key]):
                    p.copy_(e.to(p.dtype))
            out[f"ema_{key}_val_loss"] = S.evaluate(model, data, c)
        with torch.no_grad():
            for p, s in zip(model.parameters(), saved):
                p.copy_(s)
    extra = {}
    for name, sm_ in sums.items():
        mu2 = float((sm_["h0"] / sm_["n0"]) @ (sm_["h1"] / sm_["n1"]))
        mean_u = sm_["sum"] / sm_["n"]
        extra[name] = {"mu_norm2_split_half": mu2, "C0": sm_["sq"] / sm_["n"] - float(mean_u @ mean_u)}
    nu_new = {name: (sm_["dev"] / max(float(sm_["dev"].norm()), 1e-300)) for name, sm_ in sums.items()}
    return out, extra, nu_new


def effects(rows, extra):
    sys.path.insert(0, str(ROOT / "src"))
    from scipy.stats import t as tdist

    from kernel_analyzer.reference_eval.analysis import _summarize

    def ci(v):
        v = np.asarray(v, dtype=np.float64)
        n = v.size
        m, sd = float(v.mean()), float(v.std(ddof=1))
        h = float(tdist.ppf(0.975, n - 1)) * sd / math.sqrt(n)
        return {"mean": m, "lo": m - h, "hi": m + h, "sd": sd, "n": n}

    conf = [r for r in rows if not r["dev"]]
    rules = []
    for key, rule in (("p_R1", "R1"), ("p_R2", "R2"), ("p_R3", "R3"), ("p_R5", "R5")):
        p = np.array([r[key] for r in conf])
        rec = _summarize(f"u: {rule}", rule, p, p, 0.05)
        rules.append({k: rec.get(k) for k in ("rule", "verdict", "mean_projection", "p_value_two_sided_conservative",
                                              "unit_skewness", "t_approximation", "reason") if k in rec})
    rn2 = np.array([r["r_norm"] ** 2 for r in rows])
    un2 = np.array([r["u_norm"] ** 2 for r in rows])
    mu2 = extra["mu_norm2_split_half"]
    return {"gamma": ci([r["gamma"] for r in conf]), "b": ci([r["b"] for r in conf]),
            "relative_rms": float(np.sqrt(un2.mean() / rn2.mean())), "rules": rules, **extra,
            "T_star_tau1": (extra["C0"] / mu2) if mu2 > 0 else float("inf")}


def c10(seed):
    """C10: bf16 against fp32 gradient accumulation (4 micro-batches) in pure bf16 training; ordinary single step
    from the P states of stage B (protocol section 2)."""
    import single_step as SS

    P = {"precision": "bf16", "optimizer": "adamw_bf16", "grad_accum": 4}
    ck = [ROOT / ".cache/importance/ckpt/P_s0" / f"ckpt_{t:05d}.pt" for t in (250, 500, 1000, 1500)]
    rows, n_dev, extra = SS.measure(ck, S.Config(**P, accum_fp32=True), S.Config(**P))
    eff = SS.effects(rows, n_dev, extra)
    eff.update(item="C10", reference="P with fp32 gradient accumulation", candidate="P with bf16 accumulation")
    path = OUT / "single_step_C10.json"
    path.write_text(json.dumps(eff, indent=1, default=float) + "\n")
    print(f"C10: gamma {eff['gamma']['mean']: .3e} [{eff['gamma']['lo']: .3e}, {eff['gamma']['hi']: .3e}] "
          f"b {eff['b']['mean']: .3e} [{eff['b']['lo']: .3e}, {eff['b']['hi']: .3e}] rel {eff['relative_rms']:.2e}")


def main():
    global MEASURE, OUT, STEPS
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", choices=list(GROUPS) + ["c10"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="60 steps into .cache (code check only, not a result)")
    ap.add_argument("--effects", type=Path, help="add the decision-layer summary to a saved shadow run")
    a = ap.parse_args()
    if a.effects:
        add_effects(a.effects)
        return
    if a.group == "c10":
        OUT.mkdir(parents=True, exist_ok=True)
        c10(a.seed)
        return
    if a.smoke:
        MEASURE = [10, 20, 30, 40, 50, 55]
        OUT = CACHE / "smoke"
        STEPS = 60
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"shadow_{a.group}_s{a.seed}.json"
    if path.exists():
        return
    out1, extra, nu = run(a.group, a.seed)
    out2, _, _ = run(a.group, a.seed, nu=nu)
    for name in out1["rows"]:
        g1 = [r["gamma"] for r in out1["rows"][name]]
        g2 = [r["gamma"] for r in out2["rows"][name]]
        if g1 != g2:
            raise SystemExit(f"pass 2 does not reproduce pass 1 for {name}: the two-pass design needs determinism")
    res = {k: v for k, v in out2.items() if k != "rows"}
    res["extra"] = extra
    res["rows"] = out2["rows"]
    res["pass1_seconds"] = out1["seconds"]
    path.write_text(json.dumps(res, indent=1, default=float) + "\n")
    add_effects(path)


def add_effects(path):
    """The decision layer needs gmpy2 (ka_main); runs from the liger environment are summarised afterwards with
    ``--effects``."""
    res = json.loads(path.read_text())
    try:
        res["candidates_effects"] = {name: effects(res["rows"][name], res["extra"][name]) for name in res["rows"]}
    except ModuleNotFoundError as exc:
        print(f"effects deferred ({exc}); run: python scripts/importance/shadow.py --effects {path}")
        return
    path.write_text(json.dumps(res, indent=1, default=float) + "\n")
    for name, e in res["candidates_effects"].items():
        print(f"{res['group']} s{res['seed']} {name}: gamma {e['gamma']['mean']: .3e} [{e['gamma']['lo']: .3e}, {e['gamma']['hi']: .3e}] "
              f"b {e['b']['mean']: .3e} [{e['b']['lo']: .3e}, {e['b']['hi']: .3e}] rel {e['relative_rms']:.2e}", flush=True)


if __name__ == "__main__":
    main()
