#!/usr/bin/env python3
"""Layer table (RQ4 / M4, docs/layer_table_protocol_20261006.md): one implementation difference measured at four
points -- the kernel output, the parameter gradient, the ideal optimizer response and the actual FP32 write.

Scenario: HF ``LlamaRMSNorm`` compiled by Inductor, fp32 weight, bf16 activations.  L1 / L2 come from the unified
engine (``check.run``, mode B); L3 from ``update_layer`` (interval arithmetic in the optimizer's real semantics);
L4 from the FP32 torch optimizer with the gradient replaced by RN32(K_R) (enumerated when the enclosure rounds to
several FP32 values, at most 4).

    python scripts/layer_table_m4.py            # -> results/reference_eval/layer_table/{engine.json, table.json}
"""
import json
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from kernel_analyzer import check  # noqa: E402
from kernel_analyzer.check import Case, f64_point_spec  # noqa: E402
from kernel_analyzer.reference_eval import intervals as iv  # noqa: E402
from kernel_analyzer.reference_eval import update_layer as ul  # noqa: E402
from kernel_analyzer.reference_eval.analysis import assess_units, holm_adjusted  # noqa: E402

OUT = ROOT / "results/reference_eval/layer_table"
ROWS, HIDDEN, EPS_NORM = 64, 2048, 1e-6
LR, BETAS, EPS = Fraction(1, 1024), ("0.9", "0.999"), Fraction(1, 2 ** 27)
DEV, CONF = range(0, 32), range(32, 96)
MAX_CANDIDATES = 4


class RMSNorm(torch.nn.Module):
    """transformers' LlamaRMSNorm (forward copied from modeling_llama)."""

    def __init__(self, weight):
        super().__init__()
        self.weight = torch.nn.Parameter(weight)
        self.variance_epsilon = EPS_NORM

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


def theta():
    g = torch.Generator().manual_seed(7777)
    return (1 + 0.1 * torch.randn(HIDDEN, generator=g)).float()


def draw(seed):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(ROWS, HIDDEN, generator=g).to(torch.bfloat16),
            torch.randn(ROWS, HIDDEN, generator=g).to(torch.bfloat16))


def spec64(x, dy, w):
    x, dy = x.double(), dy.double()
    w = w.double().requires_grad_(True)
    y = w * (x / torch.sqrt((x * x).mean(-1, keepdim=True) + EPS_NORM))
    (gw,) = torch.autograd.grad(y, [w], dy)
    return y.detach(), gw


class RMSNormCase(Case):
    name = "hf_llama_rmsnorm_compiled"
    implementation = "transformers LlamaRMSNorm, torch.compile (Inductor), fp32 weight, bf16 activations"
    specification = "y = w * x / sqrt(mean(x^2) + eps); dL/dw by autograd, float64"

    def setup(self):
        self.mod = RMSNorm(theta().cuda())
        self.fn = torch.compile(self.mod)
        x, dy = draw(10_000)
        y = self.fn(x.cuda())
        torch.autograd.grad(y, [self.mod.weight], dy.cuda())  # compile forward and backward outside the recorder

    def inputs(self, seed):
        x, dy = draw(seed)
        return {"x": x.cuda(), "dy": dy.cuda()}

    def launch(self, inp):
        y = self.fn(inp["x"])
        (gw,) = torch.autograd.grad(y, [self.mod.weight], inp["dy"])
        return {"y": y, "grad_w": gw}

    def spec(self, inp):
        y, gw = spec64(inp["x"].cpu(), inp["dy"].cpu(), self.mod.weight.detach().cpu())
        return {"y": f64_point_spec(y.numpy()), "grad_w": f64_point_spec(gw.numpy())}


def history_state():
    """(m, v) after 9 AdamW steps on specification gradients (generator seeds 2000-2008), float64."""
    w = theta()
    grads = [spec64(*draw(s), w)[1].numpy() for s in range(2000, 2009)]
    return ul.ema_state(grads, *BETAS)


def fp32_step(opt_name, th, grad, state):
    p = torch.nn.Parameter(torch.from_numpy(th.astype(np.float32).copy()))
    if opt_name == "sgd":
        opt = torch.optim.SGD([p], lr=float(LR), foreach=False)
    else:
        opt = torch.optim.AdamW([p], lr=float(LR), betas=tuple(float(b) for b in BETAS), eps=float(EPS),
                                weight_decay=0.0, foreach=False)
        if state is not None:
            m, v, steps = state
            opt.state[p] = {"step": torch.tensor(float(steps)), "exp_avg": torch.from_numpy(m.astype(np.float32)),
                            "exp_avg_sq": torch.from_numpy(v.astype(np.float32))}
    p.grad = torch.from_numpy(np.asarray(grad, dtype=np.float32).copy())
    opt.step()
    return p.detach().numpy().astype(np.float64)


def candidates(lo, hi):
    c_lo, c_hi = np.float32(lo), np.float32(hi)
    cands, count, cur = [c_lo], np.ones(c_lo.shape, dtype=np.int64), c_lo
    for _ in range(MAX_CANDIDATES - 1):
        nxt = np.nextafter(cur, np.float32(np.inf))
        more = nxt <= c_hi
        count += more
        cur = np.where(more, nxt, cur)
        cands.append(cur)
    beyond = np.nextafter(cur, np.float32(np.inf)) <= c_hi
    return cands, np.where(beyond, MAX_CANDIDATES + 1, count)


def update_layers(rows):
    """L3 / L4 arrays per optimizer from the kept gradient rows (K, K_R enclosure per seed)."""
    th = theta().numpy().astype(np.float64)
    m_h, v_h = history_state()
    opts = {"sgd": None, "adamw_zero": None, "adamw_history": (m_h, v_h, 9)}
    res = {}
    for name, state in opts.items():
        if name == "sgd":
            delta = lambda a, b: ul.sgd_delta(a, b, LR)  # noqa: E731
        else:
            m = state[0] if state else np.zeros(HIDDEN)
            v = state[1] if state else np.zeros(HIDDEN)
            t = 10 if state else 1
            delta = lambda a, b, m=m, v=v, t=t: ul.adamw_delta(a, b, m, v, t, LR, BETAS[0], BETAS[1], EPS)  # noqa: E731
        ideal = {"lo": [], "hi": [], "ref": [], "ok": []}
        actual = {"lo": [], "hi": [], "ref": [], "ok": []}
        replay = {"unique": 0, "enumerated": 0, "not_established": 0}
        for r in rows:
            k, lo, hi, ok = r["k"], r["r_lo"], r["r_hi"], r["ok"]
            i_lo, i_hi, i_ref = ul.update_difference(k, lo, hi, delta)
            for key, val in (("lo", i_lo), ("hi", i_hi), ("ref", i_ref), ("ok", ok)):
                ideal[key].append(val)
            t_k = fp32_step(name, th, k, state)
            cands, count = candidates(lo, hi)
            outs = np.stack([fp32_step(name, th, c, state) for c in cands])
            a_lo, a_hi = iv.isub(t_k, t_k, outs.min(axis=0), outs.max(axis=0))
            ref = fp32_step(name, th, np.float32(0.5 * (lo + hi)), state) - th
            ok_a = ok & (count <= MAX_CANDIDATES)
            replay["unique"] += int((ok & (count == 1)).sum())
            replay["enumerated"] += int((ok & (count > 1) & (count <= MAX_CANDIDATES)).sum())
            replay["not_established"] += int((~ok_a).sum())
            for key, val in (("lo", a_lo), ("hi", a_hi), ("ref", ref), ("ok", ok_a)):
                actual[key].append(val)
        res[name] = {"ideal": {k: np.stack(v) for k, v in ideal.items()},
                     "actual": {k: np.stack(v) for k, v in actual.items()}, "replay": replay}
    return res


def summarize(label, rec, ref_norm):
    out = {"layer": label, "rules": []}
    for r in rec["rules"]:
        row = {"rule": r["rule"], "verdict": r.get("verdict"), "p": r.get("p_value_two_sided_conservative"),
               "t_approximation": r.get("t_approximation"), "unit_skewness": r.get("unit_skewness")}
        if "mean_projection" in r and ref_norm:
            row["effect_relative"] = r["mean_projection"] / ref_norm
            row["interval_relative"] = [r["lower_bound_of_E_l"] / ref_norm, r["upper_bound_of_E_h"] / ref_norm]
        out["rules"].append(row)
    out["scale"] = rec.get("scale")
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    keep = {}
    report = check.run(RMSNormCase(), dev=DEV, conf=CONF, keep=keep)
    (OUT / "engine.json").write_text(json.dumps(report, indent=1, default=float) + "\n")
    table = []
    for out_name, label in (("y", "L1 output"), ("grad_w", "L2 gradient")):
        e = report["outputs"].get(out_name)
        if e is None:
            table.append({"layer": label, "not_evaluated": True,
                          "why": report["outputs_not_written_by_triton"] + report["outputs_modified_after_last_triton_write"]})
            continue
        rms = e["numerical"]["scale"]["rms_reference"] * np.sqrt(e["elements_per_seed"])
        for q in ("numerical", "semantic"):
            row = summarize(f"{label} ({'e_num' if q == 'numerical' else 'e_sem'})", e[q], rms)
            row["reference_classes"] = e["reference_classes"]
            table.append(row)
    rows = keep["grad_w"]
    layers = update_layers(rows)
    n_dev = len(DEV)
    for name, d in layers.items():
        for kind, label in (("ideal", "L3 ideal response"), ("actual", "L4 actual write")):
            a = d[kind]
            lo, hi, ref, ok = a["lo"], a["hi"], a["ref"], a["ok"].astype(bool)
            rec, _ = assess_units(f"{label} {name}", np.where(ok, lo, 0), np.where(ok, hi, 0), np.where(ok, ref, 0), ok,
                                  n_dev, check.RULES, alignment_reference=np.where(ok, ref, 0),
                                  unit_ids=list(DEV) + list(CONF))
            mid = 0.5 * (lo + hi)
            rec["scale"] = {"relative_rms": float(np.sqrt((mid[ok] ** 2).mean() / max((ref[ok] ** 2).mean(), 1e-300))),
                            "max_interval_width_over_rms_ref": float((hi - lo)[ok].max() / np.sqrt((ref[ok] ** 2).mean()))}
            ref_norm = float(np.sqrt((ref[ok] ** 2).mean()) * np.sqrt(HIDDEN))
            row = summarize(f"{label} ({name})", rec, ref_norm)
            if kind == "actual":
                row["replay"] = d["replay"]
            table.append(row)
    tests = [(i, j) for i, row in enumerate(table) for j, r in enumerate(row.get("rules", [])) if r.get("p") is not None]
    for (i, j), adj in zip(tests, holm_adjusted([table[i]["rules"][j]["p"] for i, j in tests])):
        r = table[i]["rules"][j]
        r["holm_p"] = adj
        r["final"] = r["verdict"] if adj <= 0.05 and str(r["verdict"]).startswith("DETECTED") else "NOT_CONFIRMED"
    (OUT / "table.json").write_text(json.dumps({"protocol": "docs/layer_table_protocol_20261006.md",
                                                "holm_family": len(tests), "rows": table}, indent=1, default=float) + "\n")
    for row in table:
        rs = " ".join(f"{r['rule']}:{str(r.get('final', r.get('verdict')))[:12]}({r.get('effect_relative', float('nan')):.1e})"
                      for r in row.get("rules", []))
        print(f"{row['layer']:40s} {rs}", flush=True)


if __name__ == "__main__":
    main()
