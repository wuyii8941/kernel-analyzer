#!/usr/bin/env python3
"""Non-precision directed search (docs/nonprecision_search_protocol_20261007.md).

Shadow-state single step as in stage C (scripts/importance/shadow.py): the reference trains; every candidate keeps
its own optimizer state over a shadow copy of the parameters that is reset to the reference theta_t each step, and
gets the gradient the candidate's training program would form from the same batch:

    D  (reference: one paragraph per row, k = 4 micro-batches, loss = sum over valid tokens / all valid tokens)
       N1: each micro-batch's mean / k (HF Trainer before 4.46).  From the reference's per-micro-batch gradients
           G_i (of sum_i / N) the candidate's is sum_i G_i * N / (k N_i); each program clips its own gradient.
    F  (reference: fp16 autocast, GradScaler, unscale_ then clip)
       N3: clip the scaled gradient, then unscale; the scale and the skipped steps are the reference scaler's.
    R  (stage B reference)
       N4: weight decay on the 1-D parameters too;  N5: learning rate of step + 1.
       Both get the reference's clipped gradient.

At the measured steps u_t = (theta_X+ - theta_X) - (theta_R+ - theta_R).  Two passes over the identical
deterministic run: pass 1 gives nu_hat from the development units, pass 2 projects the confirmation units on it.

    python scripts/importance/nonprecision.py shadow --group D --seed 0
    python scripts/importance/nonprecision.py train --name D --seed 0       # training-level runs
    python scripts/importance/nonprecision.py summary
    python scripts/importance/nonprecision.py weights     # exploratory: N1's token weights by position and length
    python scripts/importance/nonprecision.py n4-attribution   # exploratory: N4 through the RMSNorm gains
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
from shadow import effects, flat  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/nonprecision"
MEASURE = list(range(100, 2000, 20))  # 95 units; development units: index % 3 == 0
STEPS = None  # smoke runs only

D = {"data": "docs", "grad_accum": 4, "loss_norm": "token"}
F = {"precision": "fp16"}
GROUPS = {
    "D": (D, {"N1": {"loss_norm": "microbatch"}}),
    "F": (F, {"N3": {"clip_order": "clip_first"}}),
    # attribution of N3 (protocol section 7 item 5): the same with eps / 2^16, the scale during every measured step
    "F_attr": (F, {"N3": {"clip_order": "clip_first"}, "N3e": {"clip_order": "clip_first", "eps": 1e-8 / 65536}}),
    "R": ({}, {"N4": {"wd_all": True}, "N5": {"lr_shift": 1}}),
}
TRAIN = {"R": {}, "D": D, "N1": {**D, "loss_norm": "microbatch"}, "F": F, "N3": {**F, "clip_order": "clip_first"},
         "N4": {"wd_all": True}, "N5": {"lr_shift": 1}}
NAMES = {"N1": "per-micro-batch mean / k under gradient accumulation (vs all-token normalisation)",
         "N3": "fp16 AMP: clip before unscale_ (vs unscale_ then clip)",
         "N4": "weight decay on the 1-D (RMSNorm) parameters too (vs 2-D only)",
         "N5": "learning-rate schedule one step ahead",
         "N3e": "attribution: N3 with Adam eps / 2^16"}


class Tally:
    """Per-unit scalars and the vector sums of shadow.run, for one candidate."""

    def __init__(self):
        self.rows = []
        self.s = {"dev": None, "h0": None, "h1": None, "n0": 0, "n1": 0, "sq": 0.0, "sum": None, "n": 0}

    def add(self, step, i, u, rr, nu):
        n, rn = rr.numel(), float(rr.norm())
        row = {"step": step, "dev": i % 3 == 0, "r_norm": rn, "u_norm": float(u.norm()),
               "gamma": float(u @ rr) / rn ** 2, "p_R1": -float(u.sum()) / math.sqrt(n),
               "p_R2": -float(u @ torch.sign(rr)) / math.sqrt(n), "p_R3": -float(u @ rr) / rn}
        s = self.s
        if row["dev"]:
            s["dev"] = u.clone() if s["dev"] is None else s["dev"] + u
        else:
            h = s["n"] % 2
            s[f"h{h}"] = u.clone() if s[f"h{h}"] is None else s[f"h{h}"] + u
            s[f"n{h}"] += 1
            s["sq"] += float(u @ u)
            s["sum"] = u.clone() if s["sum"] is None else s["sum"] + u
            s["n"] += 1
            if nu is not None:
                row["b"] = float(u @ nu) / rn
                row["p_R5"] = float(u @ nu)
        self.rows.append(row)

    def finish(self):
        s = self.s
        mu2 = float((s["h0"] / s["n0"]) @ (s["h1"] / s["n1"]))
        mean_u = s["sum"] / s["n"]
        extra = {"mu_norm2_split_half": mu2, "C0": s["sq"] / s["n"] - float(mean_u @ mean_u)}
        return extra, s["dev"] / max(float(s["dev"].norm()), 1e-300)


def run(group, seed, nu=None):
    ref_kw, cands = GROUPS[group]
    kind = group.split("_")[0]  # the reference program: "D", "F" or "R"
    c = S.Config(seed=seed, **ref_kw, **({"steps": STEPS} if STEPS else {}))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    S.setup_precision(c)
    model, opt = S.build(c)
    data = S.make_data(c, "cuda")
    scaler = S.make_scaler(c)
    shadows = {}
    for name, kw in cands.items():
        cc = replace(c, **kw)
        torch.manual_seed(seed)
        sm = S.SmallLM(cc).cuda()
        shadows[name] = (sm, S.make_optimizer(sm, cc), cc)
    tallies = {name: Tally() for name in shadows}
    measure = {s: i for i, s in enumerate(MEASURE)}
    skipped, skipped_units, t0 = [], [], time.time()
    params = list(model.parameters())
    for step in range(c.steps):
        lr = S.lr_at(step, c)
        S.set_lr(opt, lr)
        x, y = data.batch(step)
        cand_grads = {}
        if kind == "D":
            n_all = int((y != -100).sum())
            micro = []
            for xs, ys in zip(x.chunk(c.grad_accum), y.chunk(c.grad_accum)):
                with S.autocast(c):
                    loss = S.forward(model, xs, ys, c, "sum") / n_all
                loss.backward()
                micro.append((int((ys != -100).sum()), [p.grad for p in params]))
                for p in params:
                    p.grad = None
            for j, p in enumerate(params):  # the reference's accumulation, in the order autograd would add
                g = micro[0][1][j].clone()
                for _, gs in micro[1:]:
                    g.add_(gs[j])
                p.grad = g
            k = c.grad_accum
            cand = [torch.zeros_like(p) for p in params]
            for n_i, gs in micro:
                for acc, g in zip(cand, gs):
                    acc.add_(g, alpha=n_all / (k * n_i))
            cand_grads["N1"] = cand
            del micro
            torch.nn.utils.clip_grad_norm_(params, c.clip)
        elif kind == "F":
            with S.autocast(c):
                loss = S.forward(model, x, y, c)
            scaler.scale(loss).backward()
            scale = scaler.get_scale()
            scaled = [p.grad.detach().clone() for p in params]
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, c.clip)
        else:
            with S.autocast(c):
                loss = S.forward(model, x, y, c)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, c.clip)
        theta = [p.detach().clone() for p in params]
        # an overflowed scaled gradient makes GradScaler skip the step (checked against the scale below)
        skip = kind == "F" and not all(bool(torch.isfinite(g).all()) for g in scaled)
        deltas = {}
        for name, (sm, so, cc) in shadows.items():
            S.set_lr(so, S.lr_at(step + cc.lr_shift, c))
            sps = list(sm.parameters())
            with torch.no_grad():
                for sp, t in zip(sps, theta):
                    sp.copy_(t)
            if name == "N1":
                for sp, g in zip(sps, cand_grads.pop("N1")):
                    sp.grad = g
                torch.nn.utils.clip_grad_norm_(sps, c.clip)
            elif name.startswith("N3"):
                for sp, g in zip(sps, scaled):
                    sp.grad = g.clone()  # clipping and unscaling act in place; every shadow needs its own copy
                torch.nn.utils.clip_grad_norm_(sps, c.clip)  # on the scaled gradient
                with torch.no_grad():
                    for sp in sps:
                        sp.grad.mul_(1.0 / scale)  # a power of two: exact, as GradScaler's unscale
            else:  # the reference's clipped gradient
                for sp, p in zip(sps, params):
                    sp.grad = p.grad.detach().clone()
            before = flat(sps) if step in measure else None
            if not skip:
                so.step()
            if step in measure:
                deltas[name] = flat(sps) - before
            for sp in sps:
                sp.grad = None
        if kind == "F":
            del scaled
        before_r = flat(params) if step in measure else None
        if scaler is not None:
            scaler.step(opt)
            scaler.update()
            if (scaler.get_scale() < scale) != skip:
                raise SystemExit(f"step {step}: GradScaler's skip decision differs from the overflow check")
            if skip:
                skipped.append(step)
        else:
            opt.step()
        opt.zero_grad(set_to_none=True)
        r = flat(params) - before_r if step in measure else None
        if step in measure:
            i = measure[step]
            if skip:
                skipped_units.append(step)
                continue
            for name, tal in tallies.items():
                tal.add(step, i, deltas[name] - r, r, None if nu is None else nu[name])
    out = {"group": group, "seed": seed, "reference": asdict(c), "candidates": cands,
           "seconds": round(time.time() - t0, 1), "val_loss_reference": S.evaluate(model, data, c),
           "rows": {name: t.rows for name, t in tallies.items()}}
    if scaler is not None:
        out.update(skipped_steps=skipped, skipped_units=skipped_units, final_scale=scaler.get_scale())
    extra, nu_new = {}, {}
    for name, t in tallies.items():
        extra[name], nu_new[name] = t.finish()
    return out, extra, nu_new



def shadow(group, seed, smoke=False):
    global MEASURE, STEPS
    out_dir = OUT
    if smoke:
        MEASURE, STEPS, out_dir = [10, 20, 30, 40, 50, 55], 60, ROOT / ".cache/nonprecision/smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"shadow_{group}_s{seed}.json"
    if path.exists():
        return
    out1, extra, nu = run(group, seed)
    out2, _, _ = run(group, seed, nu=nu)
    for name in out1["rows"]:
        if [r["gamma"] for r in out1["rows"][name]] != [r["gamma"] for r in out2["rows"][name]]:
            raise SystemExit(f"pass 2 does not reproduce pass 1 for {name}: the two-pass design needs determinism")
    res = {k: v for k, v in out2.items() if k != "rows"}
    res.update(extra=extra, rows=out2["rows"], pass1_seconds=out1["seconds"])
    res["candidates_effects"] = {name: effects(res["rows"][name], extra[name]) for name in res["rows"]}
    path.write_text(json.dumps(res, indent=1, default=float) + "\n")
    for name, e in res["candidates_effects"].items():
        print(f"{group} s{seed} {name}: gamma {e['gamma']['mean']: .3e} [{e['gamma']['lo']: .3e}, {e['gamma']['hi']: .3e}] "
              f"b {e['b']['mean']: .3e} [{e['b']['lo']: .3e}, {e['b']['hi']: .3e}] rel {e['relative_rms']:.2e} "
              + " ".join(f"{r['rule']}:{r.get('verdict')}" for r in e["rules"]), flush=True)


def train(name, seed):
    path = OUT / "runs" / f"{name}__s{seed}.json"
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    res = S.train(S.Config(seed=seed, **TRAIN[name]))
    res["name"] = name
    path.write_text(json.dumps(res, indent=1, default=float) + "\n")
    print(f"{name} s{seed}: val {res['val_loss']:.4f} ({res['seconds']} s)", flush=True)


def holm(pvals, alpha=0.05):
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    reject = [False] * len(pvals)
    for rank, i in enumerate(order):
        if pvals[i] > alpha / (len(pvals) - rank):
            break
        reject[i] = True
    return reject


def summary():
    from scipy.stats import t as tdist

    from directed_summary import DELTA, classify

    shadows = {}
    for g in ("D", "F", "R"):  # the pre-registered family of section 3 (F_attr is attribution only)
        for seed in (0, 1):
            p = OUT / f"shadow_{g}_s{seed}.json"
            if p.exists():
                res = json.loads(p.read_text())
                for name, e in res["candidates_effects"].items():
                    shadows.setdefault(name, {})[seed] = (e, res)
    tests = [(name, seed, r) for name in sorted(shadows) for seed in sorted(shadows[name])
             for r in shadows[name][seed][0]["rules"]]
    pv = [r.get("p_value_two_sided_conservative", 1.0) if r.get("p_value_two_sided_conservative") is not None else 1.0
          for _, _, r in tests]
    rej = holm(pv)
    table = {}
    for name in sorted(shadows):
        per = shadows[name]
        by_rule = {}
        for (n, seed, r), p, h in zip(tests, pv, rej):
            if n == name:
                by_rule.setdefault(r["rule"], {})[seed] = {"verdict": r.get("verdict"), "p": p, "holm_reject": h,
                                                           "mean_projection": r.get("mean_projection")}
        strict = [rule for rule, v in by_rule.items() if len(v) == 2 and all(x["holm_reject"] for x in v.values())
                  and len({math.copysign(1, x["mean_projection"]) for x in v.values()}) == 1]
        # loose reading (reported, not the criterion): some rule rejected on seed 0 and some rule on seed 1, same sign
        signs = {seed: {math.copysign(1, x["mean_projection"]) for rule, v in by_rule.items() for s_, x in v.items()
                        if s_ == seed and x["holm_reject"]} for seed in (0, 1)}
        loose = bool(signs[0] & signs[1])
        classes = {f"s{seed}": classify(e) for seed, (e, _) in per.items()}
        cl = set(classes.values())
        table[name] = {
            "candidate": NAMES[name], "setting": next(g for g, (_, c) in GROUPS.items() if name in c and g != "F_attr"),
            "detected_strict": bool(strict) and len(per) == 2, "rules_detected_both_seeds": strict,
            "detected_loose": loose and len(per) == 2, "rules": by_rule,
            "gamma": {f"s{seed}": e["gamma"] for seed, (e, _) in per.items()},
            "b": {f"s{seed}": e["b"] for seed, (e, _) in per.items()},
            "relative_rms": {f"s{seed}": e["relative_rms"] for seed, (e, _) in per.items()},
            "T_star": {f"s{seed}": e.get("T_star_tau1") for seed, (e, _) in per.items()},
            "class_vs_delta": classes, "combined": cl.pop() if len(cl) == 1 else "undecided (seeds disagree)",
            "upper_bound_over_delta": {f"s{seed}": max(max(abs(e[k]["lo"]), abs(e[k]["hi"])) for k in ("gamma", "b")) / DELTA
                                       for seed, (e, _) in per.items()},
            "skipped_units": {f"s{seed}": res.get("skipped_units") for seed, (_, res) in per.items()},
        }

    def val(path):
        return json.loads(path.read_text())["val_loss"] if path.exists() else None

    def ref_path(ref, seed):
        if ref == "R":
            return ROOT / "results/importance/runs" / f"R__s{seed}.json"
        return OUT / "runs" / f"{ref}__s{seed}.json"

    def interval(v):
        v = np.asarray(v, dtype=np.float64)
        m = float(v.mean())
        if v.size < 2:
            return {"mean": m, "n": int(v.size)}
        h = float(tdist.ppf(0.975, v.size - 1)) * float(v.std(ddof=1)) / math.sqrt(v.size)
        return {"mean": m, "lo": m - h, "hi": m + h, "n": int(v.size)}

    sigma = {}
    for ref, seeds in (("D", range(8)), ("F", range(4))):
        v = [val(ref_path(ref, s_)) for s_ in seeds]
        v = [x for x in v if x is not None]
        sigma[ref] = {"sd": float(np.std(v, ddof=1)) if len(v) > 1 else None, "mean": float(np.mean(v)) if v else None,
                      "n": len(v)}
    sigma["R"] = {"sd": 0.0145, "source": "stage B sigma_seed (8 seeds)"}
    training = {}
    for cand, ref in (("N1", "D"), ("N3", "F"), ("N4", "R"), ("N5", "R")):
        pairs = [(val(OUT / "runs" / f"{cand}__s{s_}.json"), val(ref_path(ref, s_))) for s_ in range(4)]
        d = [a - b for a, b in pairs if a is not None and b is not None]
        if not d:
            continue
        iv = interval(d)
        sd = sigma[ref]["sd"]
        outside = sd is not None and "lo" in iv and abs(iv["mean"]) >= sd and (iv["lo"] > 0 or iv["hi"] < 0)
        training[cand] = {"reference": ref, "delta_L_per_seed": d, "delta_L": iv, "sigma_ref": sd,
                          "outside_seed_fluctuation": outside,
                          "val_loss": {"candidate": [a for a, _ in pairs], "reference": [b for _, b in pairs]}}
    r0 = OUT / "runs" / "R__s0.json"
    check = None
    if r0.exists():
        check = {"stage_B_R_s0": val(ref_path("R", 0)), "rerun_R_s0": val(r0)}
        check["bit_identical"] = check["stage_B_R_s0"] == check["rerun_R_s0"]
    attr = None
    pa = OUT / "shadow_F_attr_s0.json"
    if pa.exists():
        ra = json.loads(pa.read_text())
        attr = {name: {k: e[k] for k in ("gamma", "b", "relative_rms")} | {"rules": [(r["rule"], r.get("verdict")) for r in e["rules"]]}
                for name, e in ra["candidates_effects"].items()}
        attr["N3_identical_to_family_run"] = (ra["rows"]["N3"] == json.loads((OUT / "shadow_F_s0.json").read_text())["rows"]["N3"])
        attr["skipped_units"] = ra.get("skipped_units")
    out = {"delta": DELTA, "holm_tests": len(pv), "candidates": table, "sigma": sigma, "training": training,
           "harness_regression_R_s0": check, "N3_attribution_s0": attr}
    (OUT / "summary.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    for name, t in table.items():
        g, b = t["gamma"].get("s0"), t["b"].get("s0")
        print(f"{name} [{t['setting']}] strict {t['detected_strict']} {t['rules_detected_both_seeds']} loose {t['detected_loose']} "
              f"{t['combined']}: gamma {g['mean']: .2e} [{g['lo']: .2e},{g['hi']: .2e}] b {b['mean']: .2e} "
              f"[{b['lo']: .2e},{b['hi']: .2e}]")
    for cand, t in training.items():
        print(f"{cand} vs {t['reference']}: dL {t['delta_L']} sigma {t['sigma_ref']} outside {t['outside_seed_fluctuation']}")
    print("sigma", sigma, "regression", check)


def weights(seed=0):
    """Exploratory (not pre-registered): N1 weights every token of micro-batch i by N / (k N_i) relative to the
    reference.  Averaged over the 2,000 training batches of seed 0, by target position and by row length."""
    c = S.Config(seed=seed, **D)
    off = np.load(S.DATA / "wikitext103_gpt2_docs_train_offsets.npy")
    n_valid = np.minimum(np.diff(off), c.seq + 1) - 1
    order = np.random.default_rng(1_000_000 + seed).permutation(off.size - 1)  # as DocData
    k, m = c.grad_accum, c.batch // c.grad_accum
    w_pos, n_pos = np.zeros(c.seq), np.zeros(c.seq)
    by_len = {b: [0.0, 0] for b in range(4)}
    mb = []
    for step in range(c.steps):
        n = n_valid[order[step * c.batch:(step + 1) * c.batch]]
        for i in range(k):
            ni = n[i * m:(i + 1) * m]
            w = n.sum() / (k * ni.sum())
            mb.append(w)
            for length in ni:
                w_pos[:length] += w
                n_pos[:length] += 1
                b = min(length // 64, 3)
                by_len[b][0] += w * length
                by_len[b][1] += length
    mb = np.array(mb)
    pos = w_pos / n_pos
    out = {"seed": seed, "microbatch_weight": {"mean": mb.mean(), "sd": mb.std(), "p5": np.percentile(mb, 5),
                                               "p95": np.percentile(mb, 95)},
           "token_weight_by_position": {int(q): pos[q] for q in (0, 16, 32, 64, 128, 192, 255)},
           "token_weight_by_row_targets": {f"{64 * b}-{64 * b + 63 if b < 3 else c.seq}": v[0] / v[1]
                                           for b, v in by_len.items()}}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "n1_token_weights.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    print(json.dumps(out, indent=1, default=float))


def n4_attribution(seed=0):
    """Exploratory (protocol section 7 item 6): final RMSNorm gains of R and N4, and R's final model with N4's
    final gain / all nine gains substituted."""
    ck = ROOT / ".cache/nonprecision/ckpt"
    paths = {}
    for name, kw in (("R", {}), ("N4", TRAIN["N4"])):
        d = ck / f"{name}_s{seed}"
        if not (d / "ckpt_02000.pt").exists():
            res = S.train(S.Config(seed=seed, checkpoints=(2000,), **kw), out_dir=d)
            ref = json.loads((ROOT / "results/importance/runs" / f"R__s{seed}.json").read_text())["val_loss"] if name == "R" \
                else json.loads((OUT / "runs" / f"N4__s{seed}.json").read_text())["val_loss"]
            if res["val_loss"] != ref:
                raise SystemExit(f"{name} rerun with a checkpoint is not bit-identical ({res['val_loss']} vs {ref})")
        paths[name] = d / "ckpt_02000.pt"
    sd = {n: torch.load(p, map_location="cuda")["model"] for n, p in paths.items()}
    gains = [k for k, v in sd["R"].items() if v.dim() == 1]
    c = S.Config(seed=seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    S.setup_precision(c)
    model = S.SmallLM(c).cuda()
    data = S.Data(c, "cuda")

    def val(state):
        model.load_state_dict(state, strict=False)
        return S.evaluate(model, data, c)

    base = {"R": val(sd["R"]), "N4": val(sd["N4"])}
    swap_final = dict(sd["R"], nf=sd["N4"]["nf"])
    swap_all = dict(sd["R"], **{k: sd["N4"][k] for k in gains})
    out = {"seed": seed, "gain_mean": {k: {"R": float(sd["R"][k].mean()), "N4": float(sd["N4"][k].mean())} for k in gains},
           "val_loss": {**base, "R_with_N4_final_gain": val(swap_final), "R_with_N4_all_gains": val(swap_all)}}
    dl = base["N4"] - base["R"]
    out["share_of_delta_L"] = {k: (out["val_loss"][k] - base["R"]) / dl for k in ("R_with_N4_final_gain", "R_with_N4_all_gains")}
    (OUT / "n4_attribution.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    print(json.dumps(out, indent=1, default=float))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a1 = sub.add_parser("shadow")
    a1.add_argument("--group", choices=list(GROUPS), required=True)
    a1.add_argument("--seed", type=int, default=0)
    a1.add_argument("--smoke", action="store_true", help="60 steps into .cache (code check only, not a result)")
    a2 = sub.add_parser("train")
    a2.add_argument("--name", choices=list(TRAIN), required=True)
    a2.add_argument("--seed", type=int, nargs="+", default=[0])
    sub.add_parser("summary")
    sub.add_parser("weights")
    sub.add_parser("n4-attribution")
    a = ap.parse_args()
    if a.cmd == "shadow":
        shadow(a.group, a.seed, a.smoke)
    elif a.cmd == "train":
        for seed in a.seed:
            train(a.name, seed)
    elif a.cmd == "weights":
        weights()
    elif a.cmd == "n4-attribution":
        n4_attribution()
    else:
        summary()


if __name__ == "__main__":
    main()
