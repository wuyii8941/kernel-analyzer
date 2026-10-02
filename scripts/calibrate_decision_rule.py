#!/usr/bin/env python3
"""Calibrate the bias decision rule on data with a known mean (CPU only).

The rule is the one used by the unified entry: learn a fixed direction from
the calibration units (normalized mean), project the confirmation units on
it, two-sided t interval at alpha, endpoint-conservative verdict on [l, h],
Holm over a declared family.  Units are u_i = mu * v + noise_i in R^d with v
unknown to the rule.  Noise per coordinate has unit variance and is Gaussian,
Student-t(3), or sparse with rare large values.

Reported: false-positive rate at mu = 0 with a Clopper-Pearson interval,
family-wise error under Holm, and detection probability as a function of
effect size, dimension and confirmation sample size.  The interval width w
models reference uncertainty entering the endpoints.

Sensitivity in high dimension: the learned direction m = mu v + e_bar has
|m|^2 ~ mu^2 + d sigma^2 / n_dev, so the mean projection is about
mu^2 / sqrt(mu^2 + d sigma^2 / n_dev) and detection about
Phi(sqrt(n_conf) * projection / sigma - t_crit).  This prediction (integrated
over the direction noise) is stored next to the simulated rates.  Three
remedies are calibrated in d = 1024 (16 groups of 64): a direction given by
the mechanism (the aligned rule), cross-fitting (naive pooled t, fold-wise
Bonferroni, sign-flip randomization of the pooled statistic) and structural
dimension reduction (sums over declared groups).

    python scripts/calibrate_decision_rule.py --out results/reference_eval/decision_rule_calibration.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import beta, norm, t

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kernel_analyzer.reference_eval.analysis import _holm, _summarize  # noqa: E402

ALPHA = 0.05


def noise(rng, kind, shape):
    if kind == "gaussian":
        return rng.standard_normal(shape)
    if kind == "student_t3":
        return rng.standard_t(3, shape) / math.sqrt(3.0)
    if kind == "sparse_spike":
        # 98% small Gaussian, 2% large; unit variance overall.
        small, big, p = 0.3, 1.0, 0.02
        big_scale = math.sqrt((1 - (1 - p) * small ** 2) / p)
        mask = rng.random(shape) < p
        return np.where(mask, rng.standard_normal(shape) * big_scale, rng.standard_normal(shape) * small)
    raise ValueError(kind)


def simulate(rng, kind, d, mu, n_cal, n_conf, reps, chunk=200):
    """Projections of the confirmation units on the learned direction: (reps, n_conf)."""

    out = []
    for start in range(0, reps, chunk):
        r = min(chunk, reps - start)
        cal = noise(rng, kind, (r, n_cal, d))
        cal[:, :, 0] += mu
        direction = cal.mean(axis=1)
        direction /= np.linalg.norm(direction, axis=1, keepdims=True)
        conf = noise(rng, kind, (r, n_conf, d))
        conf[:, :, 0] += mu
        out.append(np.einsum("rnd,rd->rn", conf, direction))
    return np.concatenate(out)


def verdicts(proj, width):
    """Vectorized endpoint-conservative verdicts: +1 positive, -1 negative, 0 not confirmed."""

    n = proj.shape[1]
    mean = proj.mean(axis=1)
    sd = proj.std(axis=1, ddof=1)
    half = t.ppf(1 - ALPHA / 2, n - 1) * sd / math.sqrt(n)
    lower = mean - width - half
    upper = mean + width + half
    return np.where(lower > 0, 1, np.where(upper < 0, -1, 0)), mean, sd


def p_values(proj, width):
    n = proj.shape[1]
    sd = proj.std(axis=1, ddof=1)
    se = sd / math.sqrt(n)
    mean = proj.mean(axis=1)
    p_l = 2 * t.sf(np.abs(mean - width) / se, n - 1)
    p_h = 2 * t.sf(np.abs(mean + width) / se, n - 1)
    return np.maximum(p_l, p_h)


def clopper_pearson(k, n, level=0.95):
    a = (1 - level) / 2
    lo = 0.0 if k == 0 else beta.ppf(a, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - a, k + 1, n - k)
    return [float(lo), float(hi)]


def check_equivalence(rng):
    """The vectorized verdict equals analysis._summarize on sampled trials."""

    mismatches = 0
    for trial in range(200):
        proj = rng.standard_normal((1, 64)) * 1.0 + rng.choice([0.0, 0.2, 0.5])
        w = rng.choice([0.0, 0.05])
        v, _, _ = verdicts(proj, w)
        s = _summarize("x", "fixed_direction", proj[0] - w, proj[0] + w, ALPHA)
        expect = {"DETECTED_POSITIVE": 1, "DETECTED_NEGATIVE": -1, "NOT_CONFIRMED": 0}[s["verdict"]]
        mismatches += int(v[0] != expect)
    return mismatches


def predicted_detection(d, mu, n_dev, n_conf, draws=200_000, seed=0):
    """Detection probabilities from the projection formula, integrated over the learned direction's noise."""

    rng = np.random.default_rng(seed)
    m1 = mu + rng.standard_normal(draws) / math.sqrt(n_dev)
    rest2 = rng.chisquare(d - 1, draws) / n_dev if d > 1 else np.zeros(draws)
    proj = mu * m1 / np.sqrt(m1 ** 2 + rest2)
    crit = t.ppf(1 - ALPHA / 2, n_conf - 1)
    z = math.sqrt(n_conf) * proj
    return float(norm.cdf(z - crit).mean()), float(norm.cdf(-z - crit).mean())


def closed_form(d, mu, n_dev=32, n_conf=64, sigma=1.0):
    proj = mu ** 2 / math.sqrt(mu ** 2 + d * sigma ** 2 / n_dev)
    return {"d": d, "mu": mu, "projection": proj,
            "detection": float(norm.cdf(proj / (sigma / math.sqrt(n_conf)) - 1.96))}


D_REM, GROUP, N_DEV, N_CONF = 1024, 64, 32, 64
N_ALL = N_DEV + N_CONF


def remedy_vectors(rng):
    signs = np.repeat(rng.choice([-1.0, 1.0], D_REM // GROUP), GROUP)
    vectors = {"group_constant": signs / math.sqrt(D_REM),
               "within_group_mixed": rng.choice([-1.0, 1.0], D_REM) / math.sqrt(D_REM)}
    halves = {}
    for name, v in vectors.items():
        other = rng.standard_normal(D_REM)
        other -= (other @ v) * v
        other /= np.linalg.norm(other)
        halves[name] = 0.5 * v + math.sqrt(0.75) * other  # a declared direction with cosine 0.5 to the truth
    return vectors, halves


def t_verdict(proj, crit):
    n = proj.shape[-1]
    mean = proj.mean(axis=-1)
    se = proj.std(axis=-1, ddof=1) / math.sqrt(n)
    tstat = mean / se
    return np.where(tstat > crit, 1, np.where(tstat < -crit, -1, 0)), tstat


def cross_fit_stats(gram, folds, flips):
    """Pooled cross-fit projections for every sign pattern; gram (r, n, n), flips (B, n) -> (r, B, n)."""

    n = gram.shape[1]
    edges = np.linspace(0, n, folds + 1).astype(int)
    proj = np.empty((gram.shape[0], flips.shape[0], n))
    for k in range(folds):
        a, b = edges[k], edges[k + 1]
        keep = np.ones(n)
        keep[a:b] = 0.0
        e = flips * keep  # (B, n): signs of the units the direction is learned from
        acc = np.einsum("rij,bj->rbi", gram, e)  # (r, B, n): u_i . sum_j e_j u_j
        norm2 = np.einsum("rbi,bi->rb", acc, e)
        proj[:, :, a:b] = flips[None, :, a:b] * acc[:, :, a:b] / np.sqrt(np.maximum(norm2, 1e-300))[:, :, None]
    return proj


def remedies(rng, null_reps, power_reps, flips=199):
    """Mechanism direction, cross-fitting and grouping in d = 1024, against the learned fixed direction."""

    vectors, halves = remedy_vectors(rng)
    crit_conf = t.ppf(1 - ALPHA / 2, N_CONF - 1)
    crit_all = t.ppf(1 - ALPHA / 2, N_ALL - 1)
    sign_patterns = np.vstack([np.ones(N_ALL), rng.choice([-1.0, 1.0], (flips, N_ALL))])

    def run(kind, mu, vname, reps, with_flip):
        v, half = vectors[vname], halves[vname]
        hits = {}
        for start in range(0, reps, 50):
            r = min(50, reps - start)
            u = noise(rng, kind, (r, N_ALL, D_REM)) + mu * v
            res = {}
            # learned fixed direction (the current default)
            w = u[:, :N_DEV].mean(axis=1)
            w /= np.linalg.norm(w, axis=1, keepdims=True)
            res["fixed_direction"] = t_verdict(np.einsum("rnd,rd->rn", u[:, N_DEV:], w), crit_conf)[0]
            # direction given by the mechanism: exact, and with cosine 0.5 to the truth
            res["mechanism_exact"] = t_verdict(u[:, N_DEV:] @ v, crit_conf)[0]
            res["mechanism_cos0.5"] = t_verdict(u[:, N_DEV:] @ half, crit_conf)[0]
            # cross-fitting
            gram = np.einsum("rid,rjd->rij", u, u)
            for folds in (2, 4):
                pat = sign_patterns if (with_flip and folds == 2) else sign_patterns[:1]
                proj = cross_fit_stats(gram, folds, pat)
                v_pool, t_obs = t_verdict(proj[:, 0], crit_all)
                res[f"cross_fit{folds}_pooled_t"] = v_pool
                edges = np.linspace(0, N_ALL, folds + 1).astype(int)
                fold_v = np.stack([t_verdict(proj[:, 0, a:b], t.ppf(1 - ALPHA / (2 * folds), b - a - 1))[0]
                                   for a, b in zip(edges[:-1], edges[1:])], axis=1)
                any_pos = (fold_v == 1).any(axis=1)
                any_neg = (fold_v == -1).any(axis=1)
                res[f"cross_fit{folds}_foldwise_bonferroni"] = np.where(any_pos & ~any_neg, 1,
                                                                        np.where(any_neg & ~any_pos, -1,
                                                                                 np.where(any_pos, 2, 0)))
                if pat.shape[0] > 1:
                    _, t_all = t_verdict(proj, crit_all)  # (r, B+1)
                    pval = (np.abs(t_all[:, 1:]) >= np.abs(t_all[:, :1])).sum(axis=1) + 1
                    pval = pval / pat.shape[0]
                    res["cross_fit2_signflip"] = np.where(pval <= ALPHA, np.sign(t_all[:, 0]).astype(int), 0)
            # structural reduction: sums over groups of 64 coordinates (16 groups)
            g = u.reshape(r, N_ALL, D_REM // GROUP, GROUP).sum(axis=3)
            wg = g[:, :N_DEV].mean(axis=1)
            wg /= np.linalg.norm(wg, axis=1, keepdims=True)
            res["grouped_fixed_direction"] = t_verdict(np.einsum("rnd,rd->rn", g[:, N_DEV:], wg), crit_conf)[0]
            for key, val in res.items():
                h = hits.setdefault(key, [0, 0, 0])
                h[0] += int((val == 1).sum())
                h[1] += int((val == -1).sum())
                h[2] += int((val == 2).sum())
        return hits

    null_rows, power_rows = [], []
    for kind in ("gaussian", "student_t3", "sparse_spike"):
        hits = run(kind, 0.0, "group_constant", null_reps, True)
        for rule, (pos, neg, both) in hits.items():
            k = pos + neg + both
            null_rows.append({"noise": kind, "d": D_REM, "rule": rule, "reps": null_reps, "false_positives": k,
                              "rate": k / null_reps, "interval_95": clopper_pearson(k, null_reps)})
        print("remedy null", kind, {r["rule"]: r["rate"] for r in null_rows if r["noise"] == kind}, flush=True)
    for vname in ("group_constant", "within_group_mixed"):
        for mu in (0.3, 0.5, 1.0, 2.0):
            hits = run("gaussian", mu, vname, power_reps, True)
            for rule, (pos, neg, both) in hits.items():
                power_rows.append({"noise": "gaussian", "d": D_REM, "effect_shape": vname, "mu": mu, "rule": rule,
                                   "reps": power_reps, "detect_positive": pos / power_reps,
                                   "detect_negative": neg / power_reps, "detect_mixed_folds": both / power_reps})
            print("remedy power", vname, mu, {r["rule"]: r["detect_positive"] for r in power_rows
                                              if r["effect_shape"] == vname and r["mu"] == mu}, flush=True)
    return {"d": D_REM, "group_size": GROUP, "groups": D_REM // GROUP, "split": [N_DEV, N_CONF],
            "sign_flips": flips, "false_positive_rate": null_rows, "detection_probability": power_rows,
            "notes": [
                "fixed_direction / mechanism / grouped rules test the 64 confirmation units; cross-fitting uses all 96.",
                "mechanism_exact uses the true direction; mechanism_cos0.5 a declared direction with cosine 0.5 to it.",
                "cross_fit*_pooled_t pools the fold projections into one t test; the projections are dependent "
                "through the shared units, so its false-positive rate is measured, not assumed.",
                "cross_fit*_foldwise_bonferroni tests every fold at alpha/K (valid under any dependence); a "
                "rejection with opposite signs in different folds counts as a false positive (detect_mixed_folds).",
                "cross_fit2_signflip: randomization test of the pooled statistic over random unit sign flips "
                "(exact when the unit distribution is symmetric about zero under the null).",
                "grouped_fixed_direction sums the 1024 coordinates over 16 contiguous groups of 64 first; "
                "effect_shape group_constant matches that structure, within_group_mixed does not.",
            ]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--figure", type=Path, default=ROOT / "results/reference_eval/figures/decision_rule_power.png")
    parser.add_argument("--null-reps", type=int, default=4000)
    parser.add_argument("--power-reps", type=int, default=1000)
    parser.add_argument("--remedy-null-reps", type=int, default=4000)
    parser.add_argument("--remedy-power-reps", type=int, default=1000)
    args = parser.parse_args()
    rng = np.random.default_rng(20261003)
    report = {"schema": "kernel-analyzer-decision-rule-calibration-v1", "alpha": ALPHA,
              "rule": "fixed direction from 32 calibration units; t interval on confirmation projections; "
                      "endpoint-conservative on [a - w, a + w]",
              "equivalence_mismatches_vs_analysis_summarize": check_equivalence(rng)}
    noises = ("gaussian", "student_t3", "sparse_spike")
    dims = (1, 64, 1024)

    null_rows = []
    for kind in noises:
        for d in dims:
            for w in (0.0, 0.1):
                proj = simulate(rng, kind, d, 0.0, 32, 64, args.null_reps)
                v, _, _ = verdicts(proj, w)
                k = int((v != 0).sum())
                null_rows.append({"noise": kind, "d": d, "width": w, "reps": args.null_reps, "false_positives": k,
                                  "rate": k / args.null_reps, "interval_95": clopper_pearson(k, args.null_reps)})
                print("null", kind, d, w, k / args.null_reps)
    report["false_positive_rate"] = null_rows

    # Family-wise error with Holm over six independent null comparisons.
    fam = []
    for kind in noises:
        reps = 2000
        pvals = np.stack([p_values(simulate(rng, kind, 64, 0.0, 32, 64, reps), 0.0) for _ in range(6)], axis=1)
        rejected = sum(any(_holm(list(row), ALPHA)) for row in pvals)
        fam.append({"noise": kind, "family_size": 6, "reps": reps, "familywise_errors": int(rejected),
                    "rate": rejected / reps, "interval_95": clopper_pearson(int(rejected), reps)})
        print("holm", kind, rejected / reps)
    report["holm_familywise_error"] = fam

    effects = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0)
    power_rows = []
    for kind in noises:
        for d in dims:
            for mu in effects[1:]:
                proj = simulate(rng, kind, d, mu, 32, 64, args.power_reps)
                v, _, _ = verdicts(proj, 0.0)
                power_rows.append({"noise": kind, "d": d, "n_conf": 64, "mu": mu, "width": 0.0,
                                   "detect_positive": float((v == 1).mean()), "detect_negative": float((v == -1).mean())})
    for n_conf in (16, 32, 128):
        for mu in effects[1:]:
            proj = simulate(rng, "gaussian", 64, mu, 32, n_conf, args.power_reps)
            v, _, _ = verdicts(proj, 0.0)
            power_rows.append({"noise": "gaussian", "d": 64, "n_conf": n_conf, "mu": mu, "width": 0.0,
                               "detect_positive": float((v == 1).mean()), "detect_negative": float((v == -1).mean())})
    for w in (0.05, 0.1, 0.2):
        for mu in effects[1:]:
            proj = simulate(rng, "gaussian", 64, mu, 32, 64, args.power_reps)
            v, _, _ = verdicts(proj, w)
            power_rows.append({"noise": "gaussian", "d": 64, "n_conf": 64, "mu": mu, "width": w,
                               "detect_positive": float((v == 1).mean()), "detect_negative": float((v == -1).mean())})
    for row in power_rows:
        if row["noise"] == "gaussian" and row["width"] == 0.0:
            row["predicted_positive"], row["predicted_negative"] = predicted_detection(row["d"], row["mu"], 32,
                                                                                       row["n_conf"])
    report["detection_probability"] = power_rows
    report["closed_form_examples"] = [closed_form(1024, 1.0), closed_form(1, 0.5)]
    report["remedies"] = remedies(np.random.default_rng(20261005), args.remedy_null_reps, args.remedy_power_reps)
    report["notes"] = [
        "mu is the mean of each unit along the true direction, in units of the per-coordinate noise sd.",
        "In d dimensions the learned direction is noisy, so the same mu is harder to detect as d grows; "
        "the curves are the tool's sensitivity, not an a-priori magnitude bound (no bound M is used).",
        "Wrong-sign detections are reported separately (detect_negative at mu > 0).",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    plot(report, args.figure)


def plot(report, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = report["detection_probability"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, kind in zip(axes, ("gaussian", "student_t3", "sparse_spike")):
        for d, color in zip((1, 64, 1024), ("#1f5f8b", "#d1495b", "#3d8f5f")):
            pts = sorted((r["mu"], r["detect_positive"]) for r in rows
                         if r["noise"] == kind and r["d"] == d and r["n_conf"] == 64 and r["width"] == 0.0)
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", color=color, label=f"d = {d}")
        ax.axhline(0.05, color="#888", lw=0.8, ls="--")
        ax.set_title(kind.replace("_", " "))
        ax.set_xscale("log")
        ax.set_xlabel("mean along the true direction (noise sd units)")
        ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("detection probability (64 confirmation units)")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)


if __name__ == "__main__":
    main()
