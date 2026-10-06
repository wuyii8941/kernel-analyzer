"""Single-step replay of a replacement at the update layer (stage B, docs/next_phase_plan_20261006.md 2.1).

For a training state S = (theta, optimizer state, step) saved by ``small_lm.train`` and a batch b, the reference
configuration R and a candidate configuration X each take one optimizer step from the same S on the same b:
    r = theta_R+ - theta_R,   u = (theta_X+ - theta_X) - r
(theta_X is theta cast to X's parameter dtype; the cast itself is not part of u).  Per unit (state, batch):
    gamma_i = <u, r> / |r|^2                                   (scale-type effect, the dose of u = gamma r)
    b_i     = <u, nu_hat> / |r|, nu_hat = mean(u on the development units) / |.|   (fixed-direction effect, cross-fit)
and the decision layer of the tool (``assess_units``: R1, R2, R3, R5) on u against the reference update r.
"""
from __future__ import annotations

import copy
import math
from dataclasses import replace

import numpy as np
import torch

import small_lm as S


def load_state(path, device="cuda"):
    ck = torch.load(path, map_location=device, weights_only=False)
    c = S.Config(**{k: (tuple(v) if k in ("betas", "checkpoints") else v) for k, v in ck["config"].items()})
    return ck, c


def one_step(ck, c: S.Config, batch, device="cuda"):
    """theta+ - theta (flattened float64 on CPU) for configuration c from saved state ck on the given batch."""
    S.setup_precision(c)
    torch.manual_seed(0)
    model = S.SmallLM(c).to(device)
    model.load_state_dict(ck["model"])
    if c.optimizer == "adamw_bf16":
        model = model.to(torch.bfloat16)  # optimizer.load_state_dict casts the moment estimates to the parameter dtype
    opt = S.make_optimizer(model, c)
    opt.load_state_dict(copy.deepcopy(ck["opt"]))
    for g in opt.param_groups:
        g["lr"] = S.lr_at(ck["step"], c)
    before = torch.cat([p.detach().double().reshape(-1) for p in model.parameters()]).cpu()
    x, y = batch
    with S.autocast(c):
        loss = S.forward(model, x, y, c)
    loss.backward()
    if c.clip:
        torch.nn.utils.clip_grad_norm_(model.parameters(), c.clip)
    opt.step()
    after = torch.cat([p.detach().double().reshape(-1) for p in model.parameters()]).cpu()
    return (after - before).numpy()


def measure(ck_paths, ref_cfg: S.Config, cand_cfg: S.Config, units_per_ckpt=24, dev_per_ckpt=8, batch_seed=777,
            device="cuda"):
    """Paired single-step replay over checkpoints x batches. Returns per-unit arrays (u, r) ordered development
    units first, plus the effect estimates."""
    us, rs, is_dev = [], [], []
    for ci, path in enumerate(ck_paths):
        ck, saved = load_state(path, device)
        data = S.Data(replace(saved, seed=saved.seed), device)
        rng = np.random.default_rng([batch_seed, ci])
        for j in range(units_per_ckpt):
            batch = data.random_batch(rng)
            r = one_step(ck, replace(ref_cfg, steps=saved.steps), batch, device)
            xs = one_step(ck, replace(cand_cfg, steps=saved.steps), batch, device)
            us.append(xs - r)
            rs.append(r)
            is_dev.append(j < dev_per_ckpt)
    order = np.argsort(~np.asarray(is_dev), kind="stable")  # development units first
    u, r = np.stack(us)[order], np.stack(rs)[order]
    n_dev = int(np.sum(is_dev))
    return u, r, n_dev


def effects(u, r, n_dev, alpha=0.05):
    from scipy.stats import t as tdist

    def ci(v):
        n = v.size
        m, sd = float(v.mean()), float(v.std(ddof=1))
        q = float(tdist.ppf(1 - alpha / 2, n - 1))
        return {"mean": m, "lo": m - q * sd / math.sqrt(n), "hi": m + q * sd / math.sqrt(n), "sd": sd, "n": n}

    rn = np.linalg.norm(r, axis=1)
    gamma = np.einsum("ij,ij->i", u, r) / rn ** 2
    nu = u[:n_dev].mean(0)
    nu = nu / max(np.linalg.norm(nu), 1e-300)
    b = (u[n_dev:] @ nu) / rn[n_dev:]
    # step-scale auxiliary (plan 2.3): C(0) = E|w|^2 and |mu|^2 from two halves (unbiased), tau = 1 (fresh batches)
    conf = u[n_dev:]
    half = conf.shape[0] // 2
    mu2 = float(conf[:half].mean(0) @ conf[half:2 * half].mean(0))
    c0 = float(((conf - conf.mean(0)) ** 2).sum(1).mean())
    return {"gamma": ci(gamma[n_dev:]), "b": ci(b), "relative_rms": float(np.sqrt((np.linalg.norm(u, axis=1) ** 2).mean() /
                                                                                (rn ** 2).mean())),
            "mu_norm2_split_half": mu2, "C0": c0, "T_star_tau1": (c0 / mu2) if mu2 > 0 else float("inf"),
            "update_norm_rms": float(np.sqrt((rn ** 2).mean()))}
