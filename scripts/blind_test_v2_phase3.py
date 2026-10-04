#!/usr/bin/env python3
"""blind_test_v2 phase 3: the parameter-update layer, with the rules frozen in docs/blind_test_v2_phase3_protocol.md.

Main table: the actual write difference u = theta'(K) - theta'(g_R) of the same FP32 torch optimizer on the same
parameters and state, only the gradient replaced (g_R = RN32(K_R); replayed when unique, enumerated over at most 4
FP32 candidates otherwise, not established beyond).  Second table: the ideal response of the real optimizer
(update_layer, interval arithmetic).  Optimizers: SGD, AdamW with a history state built from the specification f
on the history seeds, AdamW from zero state.  Rules and seeds as in phase 1 (unified decision layer, detector-v2.1).

    python scripts/blind_test_v2_phase3.py --package .cache/blind/blind_test_v2 --stage history
    python scripts/blind_test_v2_phase3.py --package .cache/blind/blind_test_v2 --stage programs
    python scripts/blind_test_v2_phase3.py --package .cache/blind/blind_test_v2 --stage aggregate
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import sys
import time
from fractions import Fraction
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from kernel_analyzer.reference_eval import intervals as iv  # noqa: E402
from kernel_analyzer.reference_eval.analysis import holm_adjusted, interpret  # noqa: E402
from kernel_analyzer.reference_eval.ttir_eval import ST_NE  # noqa: E402

OUT = ROOT / "results" / "reference_eval" / "blind_test_v2" / "phase3"
ARRAYS = ROOT / ".cache" / "blind_v2_arrays"
STATE = ROOT / ".cache" / "blind_v2_phase3_state"
LR = 2.0 ** -10
BETAS = ("0.9", "0.999")
EPS = 2.0 ** -27
OPTIMIZERS = ("sgd", "adamw_history", "adamw_zero")
MEASURES = ("actual_write", "ideal_response")
RUNS = {"seeds_0_95": 0, "seeds_96_191": 96}
MAX_CANDIDATES = 4
UNSCORED = {"G7", "G8"}
SIGN = {"DETECTED_POSITIVE": "+", "DETECTED_NEGATIVE": "−", "NOT_CONFIRMED": "·"}
SHORT = {"DETECTED": "检出", "NOT_CONFIRMED": "未确认", "EXPLORATORY_ONLY": "探索", "CANNOT_JUDGE": "无法判断"}


def _optimizer(name, p, state=None):
    import torch

    if name == "sgd":
        return torch.optim.SGD([p], lr=LR, foreach=False)
    opt = torch.optim.AdamW([p], lr=LR, betas=tuple(float(b) for b in BETAS), eps=EPS, weight_decay=0,
                            foreach=False, fused=False)
    if state is not None:
        opt.state[p] = {"step": torch.tensor(float(state["step"]), dtype=torch.float32),
                        "exp_avg": torch.from_numpy(state["exp_avg"].copy()),
                        "exp_avg_sq": torch.from_numpy(state["exp_avg_sq"].copy())}
    return opt


def fp32_step(name, theta, grad, state=None):
    """theta' of one step of the FP32 torch optimizer (CPU), as float64 values of the FP32 result."""

    import torch

    p = torch.nn.Parameter(torch.from_numpy(theta.astype(np.float32).copy()))
    opt = _optimizer(name, p, state)
    p.grad = torch.from_numpy(np.asarray(grad, dtype=np.float32).copy())
    opt.step()
    return p.detach().numpy().astype(np.float64)


def theta_for(seed, n):
    import torch

    g = torch.Generator("cpu").manual_seed(5000 + seed)
    return torch.randn(n, generator=g, dtype=torch.float32).numpy()


# --------------------------------------------------------------------------- history state


def _history_task(args):
    package, fam, s = args
    target = STATE / f"{fam}_{s:03d}.npz"
    if target.exists():
        return fam, s
    import torch

    sys.path.insert(0, str(Path(package) / "programs"))
    inputs = importlib.import_module("inputs")
    from blind_test_v2_spec import SPEC

    grads, ambiguous = [], 0
    for i in range(16):
        lo, hi = SPEC[fam](inputs.make_inputs(fam, 20000 + 16 * s + i, device="cpu"))
        grads.append(np.float32(0.5 * (lo + hi)))  # RN32 of the enclosure midpoint (frozen rule)
        ambiguous += int((np.float32(lo) != np.float32(hi)).sum())
    p = torch.nn.Parameter(torch.zeros(grads[0].size, dtype=torch.float32))
    opt = _optimizer("adamw", p)
    for g in grads:
        p.grad = torch.from_numpy(g.astype(np.float32))
        opt.step()
    st = opt.state[p]
    np.savez(target, exp_avg=st["exp_avg"].numpy(), exp_avg_sq=st["exp_avg_sq"].numpy(),
             step=float(st["step"]), ambiguous_rn32=ambiguous)
    return fam, s


def stage_history(package, workers):
    STATE.mkdir(parents=True, exist_ok=True)
    families = json.loads((package / "manifest.json").read_text())["families"]
    jobs = [(str(package), fam, s) for fam in families for s in range(192)]
    with Pool(workers) as pool:
        for i, _ in enumerate(pool.imap_unordered(_history_task, jobs, chunksize=2)):
            if (i + 1) % 128 == 0:
                print(f"history: {i + 1}/{len(jobs)}", flush=True)


# --------------------------------------------------------------------------- per program


def _candidates(kr_lo, kr_hi):
    """FP32 candidates of RN32(K_R) per coordinate: up to MAX_CANDIDATES consecutive values from RN32(lo) to
    RN32(hi), and the number of candidates (MAX_CANDIDATES + 1 means more than allowed)."""

    c_lo, c_hi = np.float32(kr_lo), np.float32(kr_hi)
    cands, count = [c_lo], np.ones(c_lo.shape, dtype=np.int64)
    cur = c_lo
    for _ in range(MAX_CANDIDATES):
        nxt = np.nextafter(cur, np.float32(np.inf))
        more = nxt <= c_hi
        count += more
        cur = np.where(more, nxt, cur)
        cands.append(cur)
    return cands[:MAX_CANDIDATES], count


def measure_seed(opt, fam, seed, c, theta, ideal_delta):
    """(actual-write residual interval, reference update, ok mask, replay stats), and the ideal-response ones."""

    from kernel_analyzer.reference_eval import update_layer as ul

    state = np.load(STATE / f"{fam}_{seed:03d}.npz") if opt == "adamw_history" else None
    name = "sgd" if opt == "sgd" else "adamw"
    order = np.argsort(c["index"])
    k = c["k"].astype(np.float64)[order]
    kr_lo, kr_hi = c["kr_lo"][order], c["kr_hi"][order]
    ok = ((c["st"] == 0) & ~c["cond"])[order]
    t_k = fp32_step(name, theta, k, state)
    cands, count = _candidates(kr_lo, kr_hi)
    outs = np.stack([fp32_step(name, theta, g, state) for g in cands])
    u_lo, u_hi = iv.isub(t_k, t_k, outs.min(axis=0), outs.max(axis=0))
    r = fp32_step(name, theta, np.float32(0.5 * (kr_lo + kr_hi)), state) - theta.astype(np.float64)
    ok_actual = ok & (count <= MAX_CANDIDATES)
    stats = {"unique": int((ok & (count == 1)).sum()), "enumerated": int((ok & (count > 1) & (count <= MAX_CANDIDATES)).sum()),
             "not_established": int((~ok_actual).sum())}
    if opt == "sgd":
        delta = lambda a, b: ul.sgd_delta(a, b, Fraction(1, 1024))  # noqa: E731
    else:
        m = state["exp_avg"].astype(np.float64) if state is not None else np.zeros_like(k)
        v = state["exp_avg_sq"].astype(np.float64) if state is not None else np.zeros_like(k)
        t = 17 if state is not None else 1
        delta = lambda a, b: ul.adamw_delta(a, b, m, v, t, Fraction(1, 1024), BETAS[0], BETAS[1],  # noqa: E731
                                            Fraction(1, 2 ** 27))
    i_lo, i_hi, i_r = ul.update_difference(k, kr_lo, kr_hi, delta)
    return (u_lo, u_hi, r, ok_actual, stats), (i_lo, i_hi, i_r, ok)


def _program_task(args):
    package, pid, family = args
    target = OUT / "programs" / f"{pid}.json"
    if target.exists():
        return pid
    import run_blind_test_v1 as p1
    from blind_test_v2_phase2 import _shape

    t0 = time.time()
    shape = _shape(package, family)
    report = {"program": pid, "family": family, "output_shape": list(shape)}
    for opt in OPTIMIZERS:
        report[opt] = {m: {} for m in MEASURES}
        for run, offset in RUNS.items():
            p1.set_seed_offset(offset)
            data = {m: [] for m in MEASURES}
            replay = {"unique": 0, "enumerated": 0, "not_established": 0}
            u_width = {m: 0.0 for m in MEASURES}
            for seed in list(p1.DEV) + list(p1.CONF):
                c = np.load(ARRAYS / pid / f"seed{seed:03d}.npz")
                theta = theta_for(seed, c["k"].size)
                (u_lo, u_hi, r, ok_a, stats), (i_lo, i_hi, i_r, ok_i) = measure_seed(opt, family, seed, c, theta, None)
                for key in replay:
                    replay[key] += stats[key]
                n = u_lo.size
                for m, lo, hi, ref, ok in (("actual_write", u_lo, u_hi, r, ok_a), ("ideal_response", i_lo, i_hi, i_r, ok_i)):
                    st = np.where(ok, 0, ST_NE).astype(np.int8)
                    data[m].append({"lo": lo, "hi": hi, "kr": ref, "k": ref, "st": st, "cond": np.zeros(n, dtype=bool),
                                    "index": np.arange(n)})
                    if ok.any():
                        u_width[m] = max(u_width[m], float((hi - lo)[ok].max()))
            for m in MEASURES:
                out = p1.analyse_output("output", family, data[m], np.ones(data[m][0]["lo"].size, dtype=bool), shape,
                                        alignment="kr")
                out["record"]["measurement"] = {"quantity": f"{opt}/{m}", "point": "parameter update"}
                out["u_max_width"] = u_width[m]
                if m == "actual_write":
                    out["replay"] = replay
                report[opt][m][run] = out
    report["seconds"] = round(time.time() - t0, 1)
    target.write_text(json.dumps(report, indent=2, default=float) + "\n")
    return pid


def stage_programs(package, workers):
    (OUT / "programs").mkdir(parents=True, exist_ok=True)
    manifest = json.loads((package / "manifest.json").read_text())
    jobs = [(str(package), e["id"], e["family"]) for e in manifest["programs"]]
    with Pool(workers) as pool:
        for pid in pool.imap_unordered(_program_task, jobs):
            print(pid, "done", flush=True)


# --------------------------------------------------------------------------- aggregate


def stage_aggregate(package):
    manifest = json.loads((package / "manifest.json").read_text())
    recs = {e["id"]: json.loads((OUT / "programs" / f"{e['id']}.json").read_text()) for e in manifest["programs"]}
    order = sorted(recs, key=lambda p: (recs[p]["family"], p))
    for opt in OPTIMIZERS:
        for m in MEASURES:
            for run in RUNS:
                outs = [recs[p][opt][m][run] for p in order]
                tests = [x for o in outs if isinstance(o["rules"], dict) for x in o["rules"].values()
                         if isinstance(x, dict) and "p" in x]
                for x, adj in zip(tests, holm_adjusted([x["p"] for x in tests])):
                    x["holm_adjusted_p"] = adj
                    x["final_verdict"] = x["verdict"] if adj <= 0.05 else "NOT_CONFIRMED"
                for fam in ("vector_mean", "alignment"):
                    dt = [t for o in outs if o.get("default_detector") for t in o["default_detector"][fam]["tests"]
                          if t.get("p") is not None]
                    for t, adj in zip(dt, holm_adjusted([t["p"] for t in dt])):
                        sup = t.get("diagnostics", {}).get("tail_assumption_supported", True)
                        t["final_verdict"] = ("DETECTED" if sup else "EXPLORATORY_ONLY") if adj <= 0.05 else "NOT_CONFIRMED"
                    for o in outs:
                        if o.get("default_detector"):
                            d = o["default_detector"][fam]
                            vs = [t.get("final_verdict", t["verdict"]) for t in d["tests"]]
                            d["final_verdict"] = ("DETECTED" if "DETECTED" in vs else "EXPLORATORY_ONLY"
                                                  if "EXPLORATORY_ONLY" in vs else "CANNOT_JUDGE"
                                                  if vs and all(v == "CANNOT_JUDGE" for v in vs) else "NOT_CONFIRMED")
    (OUT / "phase3_report.json").write_text(json.dumps(recs, indent=2, default=float) + "\n")
    template = (package / "report_template_phase3.csv").read_text().splitlines()[0].split(",")

    def rule_cell(x):
        return (f"[{x['mu_interval'][0]:.3e}, {x['mu_interval'][1]:.3e}]", x["final_verdict"]) \
            if isinstance(x, dict) and "mu_interval" in x else ("", (x or {}).get("verdict", "") if isinstance(x, dict) else "")

    def fmt(x):
        if not isinstance(x, dict) or "mu_interval" not in x:
            return "—" if not isinstance(x, dict) else x.get("verdict", "—")
        return f"{SIGN.get(x['final_verdict'], '?')} [{x['mu_interval'][0]:.2e}, {x['mu_interval'][1]:.2e}]"

    sections, summary = [], []
    for m in MEASURES:
        for opt in OPTIMIZERS:
            rows, mat_rows, det_rows, n_det, n_rep = [], [], [], [0, 0], 0
            for p in order:
                r = recs[p]
                a, b = r[opt][m]["seeds_0_95"], r[opt][m]["seeds_96_191"]
                reproduced = [rule for rule in ("R1", "R2", "R3", "R5")
                              if isinstance(a["rules"].get(rule), dict) and isinstance(b["rules"].get(rule), dict)
                              and str(a["rules"][rule].get("final_verdict", "")).startswith("DETECTED")
                              and a["rules"][rule].get("final_verdict") == b["rules"][rule].get("final_verdict")]
                n_rep += len(reproduced)
                for i, o in enumerate((a, b)):
                    n_det[i] += sum(1 for x in o["rules"].values() if isinstance(x, dict)
                                    and str(x.get("final_verdict", "")).startswith("DETECTED"))
                rep = a.get("replay")
                method = ("FP32 replay: unique {unique}, enumerated {enumerated}, not established {not_established} "
                          "(coordinates x seeds, seeds 0-95)".format(**rep)) if rep else "interval propagation (real optimizer)"
                row = {"optimizer": opt, "program": p, "family": r["family"], "u_max_width": f"{a['u_max_width']:.3e}",
                       "replay_method": method, "reproduced_seeds_96_191": ";".join(reproduced) or "none",
                       "seconds": r["seconds"]}
                for rule in ("R1", "R2", "R3", "R5"):
                    row[f"mu_{rule}_interval"], row[f"mu_{rule}_verdict"] = rule_cell(a["rules"].get(rule))
                rows.append(row)
                fam = r["family"] + ("（不计分）" if r["family"] in UNSCORED else "")
                cells = []
                for rule in ("R1", "R2", "R3", "R5"):
                    mark = " ✔" if rule in reproduced else ""
                    cells.append(f"{fmt(a['rules'].get(rule))} / {fmt(b['rules'].get(rule))}{mark}")
                mat_rows.append(f"| {p} | {fam} | {a['u_max_width']:.1e} | " + " | ".join(cells) + " |")
                det_rows.append(f"| {p} | " + " | ".join(SHORT.get(o["default_detector"][f]["final_verdict"], "")
                                                       for o in (a, b) for f in ("vector_mean", "alignment")) + " |")
            name = f"phase3_{opt}{'' if m == 'actual_write' else '_ideal_response'}.csv"
            with open(OUT / name, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=template, extrasaction="ignore")
                w.writeheader()
                w.writerows(rows)
            summary.append(f"| {opt} | {'实际写入差（主表）' if m == 'actual_write' else '理想响应（副表）'} | {n_det[0]} | {n_det[1]} | {n_rep} |")
            sections.append(f"### {opt}，{'实际写入差（主表）' if m == 'actual_write' else '理想响应（副表）'}\n\n"
                            "每格「0–95 / 96–191」：Holm 后判定与 μ 的端点保守区间；✔ 两轮同号检出。\n\n"
                            "| 程序 | 家族 | u 最大宽度 | R1 | R2 | R3 | R5 |\n|---|---|---:|---|---|---|---|\n"
                            + "\n".join(mat_rows) + "\n\n默认检测器（第二指标）：\n\n"
                            "| 程序 | 向量均值，0–95 | 对齐，0–95 | 向量均值，96–191 | 对齐，96–191 |\n|---|---|---|---|---|\n"
                            + "\n".join(det_rows) + "\n")
    amb = sum(int(np.load(f)["ambiguous_rn32"]) for f in sorted(STATE.glob("*.npz")))
    md = f"""# blind_test_v2 阶段 3 报告：参数更新层

口径在运行前冻结于 `docs/blind_test_v2_phase3_protocol.md`：把每个程序的输出当作同形状参数块的梯度；共同状态（θ_s、由规格 f
在历史 seed 上生成的 AdamW 状态）对 K 与 K_R 相同。**主表是实际写入差**：同一个 FP32 torch optimizer、同一参数与状态，只替换梯度，
u = θ′(K) − θ′(RN32(K_R))；K_R 区间两端的 RN32 相同时直接重放，不同时逐坐标枚举 FP32 候选（不超过 4 个）取可靠包围，更多则为参照
未建立。**副表是理想响应**：实数 optimizer 在区间算术中对 K 与 K_R 各求一步，不是实际写入差。规则 R1、R2（−sign(r)/√n）、
R3（−r/‖r‖，正号表示沿参照更新方向推得少）、R5，seed 0–95 与 96–191，每种 optimizer、每个口径、每轮各自对全部「程序 × 规则」
做 Holm；判定层 `detector-v2.1`。历史梯度中 f 的包围两端舍入到不同 FP32 值的坐标：{amb} 个（取中点的 RN32，冻结规则）。

| optimizer | 口径 | 检出格（0–95） | 检出格（96–191） | 两轮同号 |
|---|---|---:|---:|---:|
{chr(10).join(summary)}

{chr(10).join(sections)}
"""
    (OUT / "phase3_report.md").write_text(md)
    print("written", OUT / "phase3_report.md")
    print("\n".join(summary))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--stage", choices=("history", "programs", "aggregate"), required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    {"history": lambda: stage_history(args.package, args.workers),
     "programs": lambda: stage_programs(args.package, args.workers),
     "aggregate": lambda: stage_aggregate(args.package)}[args.stage]()


if __name__ == "__main__":
    main()
