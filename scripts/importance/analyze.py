#!/usr/bin/env python3
"""Analysis of the importance calibration (docs/importance_calibration_protocol_20261006.md).

    python scripts/importance/analyze.py curves     # seed band, dose-response, thresholds  -> results/importance/curves.json
    python scripts/importance/analyze.py predict    # predictions from single-step + curves -> results/importance/predictions.json
    python scripts/importance/analyze.py summary    # unseals the replacement runs (needs predictions.json) -> summary.json
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, t as tdist

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run import BS, GAMMAS, ITEMS, OUT  # noqa: E402


def load(sub, name, seeds):
    vals = []
    for s in seeds:
        p = OUT / sub / f"{name}__s{s}.json"
        if not p.exists():
            return None
        vals.append(json.loads(p.read_text())["val_loss"])
    return np.array(vals)


def ci(v, alpha=0.05):
    v = np.asarray(v, dtype=np.float64)
    m, sd = float(v.mean()), float(v.std(ddof=1)) if v.size > 1 else float("nan")
    q = float(tdist.ppf(1 - alpha / 2, v.size - 1)) if v.size > 1 else float("nan")
    h = q * sd / math.sqrt(v.size)
    return {"mean": m, "lo": m - h, "hi": m + h, "sd": sd, "n": int(v.size), "values": [float(x) for x in v]}


def curves():
    R = load("runs", "R", range(8))
    band = {"R_seeds": [float(x) for x in R], "mean": float(R.mean()), "sigma_seed": float(R.std(ddof=1)),
            "deepseek_relative_0p25pct": 0.0025 * float(R.mean())}
    ref4 = R[:4]
    dose = {"gamma": {}, "direction": {}}
    for kind, doses in (("gamma", GAMMAS), ("direction", BS)):
        for d in doses:
            v = load("runs", f"inject_{kind}_{d:g}", range(4))
            if v is None:
                continue
            c = ci(v - ref4)
            c["exceeds_seed_band"] = bool(abs(c["mean"]) >= band["sigma_seed"] and (c["lo"] > 0 or c["hi"] < 0))
            dose[kind][f"{d:g}"] = c
    thresholds = {}
    for kind in dose:
        pos = sorted((float(k), v) for k, v in dose[kind].items() if float(k) > 0)
        hit = [d for d, v in pos if v["exceeds_seed_band"]]
        thresholds[kind] = {"d_star": min(hit) if hit else None, "largest_tested": max(d for d, _ in pos) if pos else None}
    res = {"seed_band": band, "dose_response": dose, "thresholds": thresholds}
    (OUT / "curves.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({"seed_band": band, "thresholds": thresholds}, indent=1))
    for kind in dose:
        for k, v in dose[kind].items():
            print(f"{kind:9s} {k:>7s}  dL {v['mean']: .5f} [{v['lo']: .5f}, {v['hi']: .5f}]  exceeds {v['exceeds_seed_band']}")
    return res


def interp(curve, x):
    """|mean dL| as a function of |dose| (log-linear between tested positive doses)."""
    pts = sorted((float(k), abs(v["mean"])) for k, v in curve.items() if float(k) > 0)
    xs, ys = np.log10([p[0] for p in pts]), np.array([p[1] for p in pts])
    if x <= 0:
        return float(ys[0]), "below the smallest dose"
    lx = math.log10(x)
    if lx < xs[0]:
        return float(ys[0]), "below the smallest dose"
    if lx > xs[-1]:
        return float(ys[-1]), "beyond the calibrated range"
    return float(np.interp(lx, xs, ys)), "interpolated"


def predict():
    cv = json.loads((OUT / "curves.json").read_text())
    sigma = cv["seed_band"]["sigma_seed"]
    preds = {}
    for name in ITEMS:
        ss = json.loads((OUT / "single_step" / f"{name}.json").read_text())
        g, b = ss["gamma"], ss["b"]
        pg, ng = interp(cv["dose_response"]["gamma"], abs(g["mean"]))
        pb, nb = interp(cv["dose_response"]["direction"], abs(b["mean"]))
        ug, _ = interp(cv["dose_response"]["gamma"], max(abs(g["lo"]), abs(g["hi"])))
        ub, _ = interp(cv["dose_response"]["direction"], max(abs(b["lo"]), abs(b["hi"])))
        preds[name] = {"gamma_hat": g, "b_hat": b, "predicted_abs_dL": pg + pb, "upper_prediction": ug + ub,
                       "notes": [ng, nb], "predicted_category": "exceeds" if pg + pb >= sigma else "within seed band",
                       "relative_rms": ss["relative_rms"], "T_star_tau1": ss["T_star_tau1"]}
    # rho against the accepted changes (rectangle of 97.5% intervals, protocol section 4)
    from scipy.stats import t as tt

    def ci975(e):
        q975, q95 = tt.ppf(1 - 0.0125, e["n"] - 1), tt.ppf(1 - 0.025, e["n"] - 1)
        h = (e["hi"] - e["mean"]) * q975 / q95
        return e["mean"] - h, e["mean"] + h

    for name in ITEMS:
        preds[name]["rho"] = {}
        for a in ("A1", "A2", "A3"):
            for key in ("gamma_hat", "b_hat"):
                x0, x1 = ci975(preds[name][key])
                a0, a1 = ci975(preds[a][key])
                if a0 <= 0 <= a1:
                    preds[name]["rho"][f"{key}/{a}"] = "undefined: the accepted change's effect is not established"
                else:
                    r = [x / y for x in (x0, x1) for y in (a0, a1)]
                    preds[name]["rho"][f"{key}/{a}"] = {"point": preds[name][key]["mean"] / preds[a][key]["mean"],
                                                       "lo": min(r), "hi": max(r)}
    res = {"sigma_seed": sigma, "rule": "protocol section 6", "predictions": preds}
    (OUT / "predictions.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    for name, p in preds.items():
        print(f"{name}: gamma {p['gamma_hat']['mean']: .2e} b {p['b_hat']['mean']: .2e} -> |dL| {p['predicted_abs_dL']:.5f} "
              f"(upper {p['upper_prediction']:.5f}) {p['predicted_category']}")


def summary():
    if not (OUT / "predictions.json").exists():
        raise SystemExit("sealed: predictions.json is required first")
    cv = json.loads((OUT / "curves.json").read_text())
    pr = json.loads((OUT / "predictions.json").read_text())
    sigma = cv["seed_band"]["sigma_seed"]
    refs = {"R": load("runs", "R", range(4)), "Rp": load("runs_sealed", "Rp", range(4)), "P": load("runs_sealed", "P", range(4))}
    obs = {}
    for name, (kw, ref) in ITEMS.items():
        sub_name = {"A1": ("runs_sealed", "Rp"), "X4": ("runs_sealed", "P")}.get(name, ("runs_sealed", name))
        v = load(sub_name[0], sub_name[1], range(4))
        if v is None or refs[ref] is None:
            obs[name] = {"missing": True}
            continue
        c = ci(v - refs[ref])
        c["observed_category"] = "exceeds" if (abs(c["mean"]) >= sigma and (c["lo"] > 0 or c["hi"] < 0)) else "within seed band"
        c["predicted_category"] = pr["predictions"][name]["predicted_category"]
        c["predicted_abs_dL"] = pr["predictions"][name]["predicted_abs_dL"]
        obs[name] = c
    done = [n for n in obs if not obs[n].get("missing")]
    agree = sum(obs[n]["observed_category"] == obs[n]["predicted_category"] for n in done)
    rho = spearmanr([obs[n]["predicted_abs_dL"] for n in done], [abs(obs[n]["mean"]) for n in done]).correlation if len(done) > 2 else None
    verdict = "good" if (agree >= 7 and rho is not None and rho >= 0.6) else "poor"
    th = cv["thresholds"]
    delta = {k: (v["d_star"] / 2 if v["d_star"] else v["largest_tested"]) for k, v in th.items()}
    res = {"sigma_seed": sigma, "observed": obs, "category_agreement": f"{agree}/{len(done)}", "spearman": rho,
           "prediction_validity": verdict, "delta": delta,
           "delta_rule": "d*/2 per injection type; the largest tested dose when no dose exceeds the seed band (upper bound only)",
           "accepted_changes_within_seed_band": {a: obs[a].get("observed_category") for a in ("A1", "A2", "A3") if a in obs}}
    (OUT / "summary.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    for n in done:
        o = obs[n]
        print(f"{n}: observed dL {o['mean']: .5f} [{o['lo']: .5f}, {o['hi']: .5f}] {o['observed_category']:17s} predicted "
              f"{o['predicted_abs_dL']:.5f} {o['predicted_category']}")
    print("agreement", res["category_agreement"], "spearman", rho, "->", verdict, "delta", delta)


def figure():
    """Calibration figure: single-step effect (x, log) against |paired dL| (y), dose-response with 95% intervals,
    the seed band, the 0.25% line, and each replacement at its single-step effect with its observed |dL|."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cv = json.loads((OUT / "curves.json").read_text())
    sm = json.loads((OUT / "summary.json").read_text()) if (OUT / "summary.json").exists() else None
    pr = json.loads((OUT / "predictions.json").read_text())
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for kind, color, label in (("gamma", "tab:blue", "injected u = γ·r"), ("direction", "tab:orange", "injected u = b·|r|·ν")):
        pts = sorted((float(k), v) for k, v in cv["dose_response"][kind].items() if float(k) > 0)
        x = [d for d, _ in pts]
        y = [abs(v["mean"]) for _, v in pts]
        lo = [max(abs(v["mean"]) - (v["hi"] - v["mean"]), 1e-6) for _, v in pts]
        hi = [abs(v["mean"]) + (v["hi"] - v["mean"]) for _, v in pts]
        ax.fill_between(x, lo, hi, color=color, alpha=0.15)
        ax.plot(x, y, "o-", color=color, label=label)
    sig = cv["seed_band"]["sigma_seed"]
    ax.axhline(sig, color="k", ls="--", lw=1, label=f"seed σ = {sig:.4f}")
    ax.axhline(cv["seed_band"]["deepseek_relative_0p25pct"], color="gray", ls=":", lw=1, label="0.25 % of loss")
    if sm:
        for name, o in sm["observed"].items():
            if o.get("missing"):
                continue
            ss = pr["predictions"][name]
            x = max(abs(ss["gamma_hat"]["mean"]), abs(ss["b_hat"]["mean"]), 1e-9)
            ax.plot([x], [max(abs(o["mean"]), 1e-6)], "s", color="tab:red" if name.startswith("X") else "tab:green")
            ax.annotate(name, (x, max(abs(o["mean"]), 1e-6)), textcoords="offset points", xytext=(4, 3), fontsize=8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("single-step effect at the update layer (|γ̂| or |b̂|, relative to the update norm)")
    ax.set_ylabel("|paired Δ validation loss| after 2,000 steps")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "calibration.png", dpi=150)
    print("wrote", OUT / "calibration.png")


if __name__ == "__main__":
    {"curves": curves, "predict": predict, "summary": summary, "figure": figure}[sys.argv[1]]()
