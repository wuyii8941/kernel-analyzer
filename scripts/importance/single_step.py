"""Single-step replay of a replacement at the update layer (stage B, docs/next_phase_plan_20261006.md 2.1).

For a training state S = (theta, optimizer state, step) saved by ``small_lm.train`` and a batch b, the reference
configuration R and a candidate configuration X each take one optimizer step from the same S on the same b:
    r = theta_R+ - theta_R,   u = (theta_X+ - theta_X) - r
(theta_X is theta cast to X's parameter dtype; the cast itself is not part of u).  Per unit (state, batch):
    gamma_i = <u, r> / |r|^2                                   (scale-type effect, the dose of u = gamma r)
    b_i     = <u, nu_hat> / |r|, nu_hat = mean(u on the development units) / |.|   (fixed-direction effect, cross-fit)
and the projections of the tool's rules (R1: -1/sqrt(n); R2: -sign(r)/sqrt(n); R3: -r/|r|; R5: nu_hat) per unit, fed
to the decision layer's endpoint-conservative t inference (point residuals, so both endpoints coincide).

The replay streams over units: the parameter vectors (13M coordinates) are never stacked; development units are
processed first (they fix nu_hat), then the confirmation units.
"""
from __future__ import annotations

import copy
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

import small_lm as S


def load_state(path, device="cuda"):
    ck = torch.load(path, map_location=device, weights_only=False)
    c = S.Config(**{k: (tuple(v) if k in ("betas", "checkpoints") else v) for k, v in ck["config"].items()})
    return ck, c


class Stepper:
    """One model per configuration, reloaded from a saved state for every unit (so a compiled model is compiled
    once); a fresh optimizer per unit, whose load_state_dict casts the moments to the parameter dtype."""

    def __init__(self, c: S.Config, device="cuda"):
        self.c, self.device = c, device
        torch.manual_seed(0)
        self.model = S.SmallLM(c).to(device)
        if c.optimizer == "adamw_bf16":
            self.model = self.model.to(torch.bfloat16)

    def step(self, ck, batch):
        c, model = self.c, self.model
        S.setup_precision(c)
        model.load_state_dict({k: v for k, v in ck["model"].items()})  # copies into the existing parameters
        opt = S.make_optimizer(model, c)
        opt.load_state_dict(copy.deepcopy(ck["opt"]))
        S.set_lr(opt, S.lr_at(ck["step"], c))
        before = torch.cat([p.detach().double().reshape(-1) for p in model.parameters()])
        x, y = batch
        if c.grad_accum == 1:
            with S.autocast(c):
                loss = S.forward(model, x, y, c)
            loss.backward()
        else:  # as small_lm.train_step: micro-batches, accumulated in the parameter dtype or an fp32 buffer
            buf = {n: torch.zeros_like(p, dtype=torch.float32) for n, p in model.named_parameters()} if c.accum_fp32 else None
            for xs, ys in zip(x.chunk(c.grad_accum), y.chunk(c.grad_accum)):
                with S.autocast(c):
                    l = S.forward(model, xs, ys, c) / c.grad_accum
                l.backward()
                if buf is not None:
                    for n, p in model.named_parameters():
                        buf[n].add_(p.grad.float())
                        p.grad = None
            if buf is not None:
                for n, p in model.named_parameters():
                    p.grad = buf[n].to(p.dtype)
        if c.clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), c.clip)
        opt.step()
        after = torch.cat([p.detach().double().reshape(-1) for p in model.parameters()])
        for p in model.parameters():
            p.grad = None
        return after - before  # float64 on the device


def measure(ck_paths, ref_cfg: S.Config, cand_cfg: S.Config, units_per_ckpt=24, dev_per_ckpt=8, batch_seed=777,
            device="cuda"):
    """Streams the paired replay; returns per-unit scalars (development units first) and running sums."""
    cks, batches = [], []
    for ci, path in enumerate(ck_paths):
        ck, saved = load_state(path, device)
        data = S.Data(saved, device)
        rng = np.random.default_rng([batch_seed, ci])
        cks.append((ck, saved))
        batches.append([data.random_batch(rng) for _ in range(units_per_ckpt)])
    steps = cks[0][1].steps
    steppers = {"ref": Stepper(replace(ref_cfg, steps=steps), device), "cand": Stepper(replace(cand_cfg, steps=steps), device)}
    order = [(ci, j) for j in range(dev_per_ckpt) for ci in range(len(ck_paths))] + \
            [(ci, j) for j in range(dev_per_ckpt, units_per_ckpt) for ci in range(len(ck_paths))]
    n_dev = dev_per_ckpt * len(ck_paths)
    dev_sum = nu = sum_u = None
    halves = [None, None]
    half_n = [0, 0]
    conf_sq, conf_n = 0.0, 0
    rows = []
    for k, (ci, j) in enumerate(order):
        ck, _ = cks[ci]
        r = steppers["ref"].step(ck, batches[ci][j])
        u = steppers["cand"].step(ck, batches[ci][j]) - r
        n = r.numel()
        rn = float(r.norm())
        row = {"checkpoint": ci, "batch": j, "dev": k < n_dev, "r_norm": rn, "u_norm": float(u.norm()),
               "gamma": float(u @ r) / rn ** 2, "p_R1": -float(u.sum()) / math.sqrt(n),
               "p_R2": -float(u @ torch.sign(r)) / math.sqrt(n), "p_R3": -float(u @ r) / rn}
        if k < n_dev:
            dev_sum = u.clone() if dev_sum is None else dev_sum + u
            if k == n_dev - 1:
                nu = dev_sum / max(float(dev_sum.norm()), 1e-300)
        else:
            row["b"] = float(u @ nu) / rn
            row["p_R5"] = float(u @ nu)
            h = conf_n % 2
            halves[h] = u.clone() if halves[h] is None else halves[h] + u
            half_n[h] += 1
            conf_sq += float(u @ u)
            sum_u = u.clone() if sum_u is None else sum_u + u
            conf_n += 1
        rows.append(row)
        del r, u
    mu2 = float((halves[0] / half_n[0]) @ (halves[1] / half_n[1]))
    mean_u = sum_u / conf_n
    c0 = conf_sq / conf_n - float(mean_u @ mean_u)
    return rows, n_dev, {"mu_norm2_split_half": mu2, "C0": c0}


def effects(rows, n_dev, extra, alpha=0.05):
    from scipy.stats import t as tdist

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from kernel_analyzer.reference_eval.analysis import _summarize

    def ci(v):
        v = np.asarray(v, dtype=np.float64)
        n = v.size
        m, sd = float(v.mean()), float(v.std(ddof=1))
        q = float(tdist.ppf(1 - alpha / 2, n - 1))
        return {"mean": m, "lo": m - q * sd / math.sqrt(n), "hi": m + q * sd / math.sqrt(n), "sd": sd, "n": n}

    conf = [r for r in rows if not r["dev"]]
    rn2 = np.array([r["r_norm"] ** 2 for r in rows])
    un2 = np.array([r["u_norm"] ** 2 for r in rows])
    rules = []
    for key, rule in (("p_R1", "R1"), ("p_R2", "R2"), ("p_R3", "R3"), ("p_R5", "R5")):
        p = np.array([r[key] for r in conf])
        rec = _summarize(f"u: {rule}", rule, p, p, alpha)
        rules.append({k: rec.get(k) for k in ("rule", "verdict", "mean_projection", "p_value_two_sided_conservative",
                                              "unit_skewness", "t_approximation", "reason") if k in rec})
    mu2 = extra["mu_norm2_split_half"]
    return {"gamma": ci([r["gamma"] for r in conf]), "b": ci([r["b"] for r in conf]),
            "relative_rms": float(np.sqrt(un2.mean() / rn2.mean())), "update_norm_rms": float(np.sqrt(rn2.mean())),
            "mu_norm2_split_half": mu2, "C0": extra["C0"],
            "T_star_tau1": (extra["C0"] / mu2) if mu2 > 0 else float("inf"), "rules": rules, "units": rows}
