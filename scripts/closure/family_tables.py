#!/usr/bin/env python3
"""Per-family tables of the closure (task book section 10): units and G3 coverage, contract classes, the four groups
(E from the 2b runs; F, P, FR from the closure runs), the four columns, the lists of not established / unsupported /
timeout, and the cost by group.  Every count carries its denominator.

Inputs: results/essential/phase2b/{coverage_plan.json,*/analysis.json}, results/closure/{contract_classification.json,
f_eval/*.json, p_extra/*.json, fr_modeB/*.json, candidate_conventions.json}.  Output: results/closure/family_tables.json and
docs/closure_family_tables_20261008.md.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
C = ROOT / "results/closure"
B = ROOT / "results/essential/phase2b"
FAMS = ["matmul_linear", "reductions", "activations", "gather_layout", "checkpoint", "embedding", "attention", "packing", "rope",
        "normalization", "optimizers", "schedulers", "clip_amp", "training_program", "moe"]


def load(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


# ------------------------------------------------------------------------------------------------ groups

def g3(fam, plan):
    t = plan["tier1"].get(fam) or {}
    fac = t.get("factors") or {}
    full = math.prod(len(v) for v in fac.values()) if fac else None
    contract = load(C / "contract_classification.json")["families"].get(fam, {})
    executed = contract.get("conditions")
    return {"factor_levels": {k: len(v) for k, v in fac.items()}, "full_factorial": full,
            "planned": t.get("planned_conditions"), "executed": executed,
            "uncovered_combinations": (full - executed) if (full and executed is not None) else None,
            "risk_points": t.get("points"), "state_sequences": t.get("state_sequences"), "methods_planned": t.get("methods")}


def e_group(fam):
    d = load(B / fam / "analysis.json")
    if d is None:
        return {"status": "no 2b analysis file"}
    out = {}
    top = d.get("E") if isinstance(d.get("E"), dict) else None
    for name, v in d.get("candidates", {}).items():
        if not isinstance(v, dict):
            continue
        e = v.get("E") if v.get("E") is not None else v.get("E_vs_2.10")
        if e is None and top and name in top:
            e = top[name]
        if e is None:
            out[name] = "reference or recorded only (no E)"
        elif isinstance(e, dict):
            keys = ("conditions_compared", "violating_conditions", "compared", "violating", "status", "reference", "by_output")
            out[name] = {k: e[k] for k in keys if k in e}
        else:
            out[name] = e
    return out


def f_group(fam):
    d = load(C / "f_eval" / f"{fam}.json")
    if d is None:
        return {"status": "no spec: F not open" if fam not in ("training_program", "checkpoint") else None}
    s = d.get("summary") or {}
    per = {}
    for name, v in (s.get("per_candidate") or {}).items():
        per[name] = {o: f"{x['beyond']}/{x['elements']}" + (f" (non-finite {x['nonfinite']})" if x.get("nonfinite") else "")
                     for o, x in v["by_output"].items()}
    status = s.get("spec_status") or {k: v for k, v in (s.get("spec_not_ok") or {}).items()}
    return {"spec": d.get("spec"), "beyond_over_elements": per, "spec_status": status,
            "precision_invariance": s.get("precision_invariance_conditions"), "no_f_outputs": s.get("no_f"),
            "seconds": d.get("seconds")}


def p_group(fam):
    d = load(B / fam / "analysis.json") or {}
    pre = {}
    for name, v in (d.get("candidates") or {}).items():
        if isinstance(v, dict):
            p = v.get("P") or ({"P_no_leakage": v["P_no_leakage"]} if "P_no_leakage" in v else None)
            if p:
                pre[name] = p
    if isinstance(d.get("P"), dict):
        pre["(family)"] = d["P"]
    ex = load(C / "p_extra" / f"{fam}.json")
    return {"preregistered_2b": pre, "closure_gradcheck_jvp": ex.get("tally") if ex else "not run (no differentiable "
            "candidate output in this family or covered by the 2b P)", "seconds": ex.get("seconds") if ex else None}


def fr_group(fam):
    d = load(C / "fr_modeB" / f"{fam}.json")
    if d is None:
        return {"status": "mode B not run (no Triton candidate with spec f in this family)"}
    tot = defaultdict(int)
    worst_pure = 0.0
    for key, r in d["cases"].items():
        if r["status"] != "ok":
            continue
        for o, v in r["outputs"].items():
            mixed = bool(v.get("mixed_non_triton_sources"))
            sv = v.get("special_values") or {}
            # the tool counts class mismatches in both directions; elements where f is undefined (NaN, e.g. ATT-A1 rows)
            # are not "K special against a finite f"
            knf = max(0, int(sv.get("k_vs_f_class_mismatch") or 0) - int(v.get("f_undefined_elements") or 0))
            tot["ok_elements"] += v["ok_elements"]
            tot["e_num_beyond_tau32_finite_K"] += max(0, v["e_num_beyond_tau32"] - knf)
            tot["K_special_vs_f_finite_column4"] += knf
            tot["f_undefined_excluded"] += v.get("f_undefined_elements", 0)
            if mixed:
                tot["ok_elements_mixed"] += v["ok_elements"]
                tot["e_sem_certified_mixed (not a semantic verdict)"] += v["e_sem_certified_elements"]
            else:
                tot["ok_elements_pure"] += v["ok_elements"]
                tot["e_sem_certified_pure"] += v["e_sem_certified_elements"]
                worst_pure = max(worst_pure, v.get("max_e_sem_gap_relative") or 0.0)
            if "e_sem_certified_with_received_hyperparameters" in v:
                tot["e_sem_certified_after_attribution"] += v["e_sem_certified_with_received_hyperparameters"]
                tot["e_sem_certified_where_attribution_ran"] += v["e_sem_certified_elements"]
    nr = {k: (r.get("reason") or "")[:120] for k, r in d["cases"].items() if r["status"] != "ok"}
    return {"case_status": d["status_counts"], "elements": dict(tot), "max_pure_e_sem_gap_relative": worst_pure,
            "cases_not_ok": nr, "seconds": d.get("seconds")}


# ------------------------------------------------------------------------------------------------ four columns (curated
# from the runs above; every entry names its record)

COLUMNS = {
    "activations": {
        "4": ["B023 gelu(approximate='tanh') backward NaN for |x| >= 1.8447e19 (float32 / bf16; eager CPU, CUDA, Inductor): "
              "bugs/B023_*.md; seen again by the closure P (jvp vs VJP non-finite on acti_gelu_tanh_huge, results/closure/p_extra/activations.json)",
              "swiglu / geglu huge: the real product exceeds FLT_MAX, K = ±inf (IEEE overflow) against a finite f "
              "(results/closure/fr_modeB/activations.json special_values; F non-finite counts)"]},
    "embedding": {
        "2": ["EMB-A1: max_norm renormalisation divides by (norm + 1e-7); visible against f at float64 (891/3552 out, 340/960 "
              "weight_after beyond τ64), precision invariance 'semantic' (results/closure/f_eval/embedding.json); "
              "residual after dividing by (norm + 1e-7): <= 7.3e-17"],
        "pending": ["E-D2 (max_norm scope): weight_after not adjudicated (docs/doc_check_seven_items_20261008.md)"]},
    "attention": {
        "3": ["ATT-A1 rows without an allowed key: zero (SDPA math / efficient, flex, xformers), NaN with non-finite dK/dV "
              "(manual eager, Inductor), finite non-zero (HF eager, cuDNN bf16) -- results/essential/phase2b/attention/analysis.json"]},
    "moe": {"2": ["MOE-C4: the candidates' aux loss counts top-1 assignments, the spec counts dispatched experts (declared difference)"]},
    "reductions": {
        "3": ["BASE-A1 empty amax / amin: the candidates raise (Inductor and eager); empty mean: NaN "
              "(results/essential/phase2b/reductions/analysis.json conventions)"],
        "pending": ["BASE-A1 var / std with N - correction <= 0 (NaN observed): clause stopped by the doc check"]},
    "optimizers": {"pending": ["O-D2 SGD maximize placement: momentum buffer = -(spec b) to 1.5e-16, parameters agree "
                               "(docs/spec_issues_phase2.md SPEC-ISSUE-1)"]},
    "schedulers": {"pending": ["SCH-A1 for T_cur >= T_max (0/24 beyond against the closed form, recorded only)",
                               "contract label of HF total > warmup (2 conditions marked D)"]},
    "matmul_linear": {"pending": ["contract label of batch broadcast (3 conditions marked D; doc says legal, spec scope)"]},
    "clip_amp": {"4": ["clip_grad_norm_ with non-finite gradients (error_if_nonfinite=False): scaled by the non-finite "
                       "coefficient (documented) -- 15 condition x seed"]},
    "normalization": {"difference": ["float32 huge_offset: F beyond τ32 in 247/5400 y elements of eager CPU float32; "
                                     "precision invariance: no semantic element (numerical); contract states no accuracy "
                                     "requirement -> 契约外，待审阅 (no defect verdict)"]},
}
COMMON_PENDING = ("K_R ≠ f_r caused by compile-time float32 constants (Inductor GELU sqrt(2/pi), 0.044715, 1/sqrt(2); flex "
                  "RCP_LN2 = 1.44269504; BatchNorm 1 - momentum = 0.9 and n/(n-1) = 1.1111111111111112 folded at compile time; "
                  "hyperparameters held as float32): difference established, contract silent -> 契约外，待审阅")


def main():
    plan = load(B / "coverage_plan.json")
    contract = load(C / "contract_classification.json")["families"]
    out = {}
    for fam in FAMS:
        out[fam] = {"g3": g3(fam, plan), "contract_classes": contract.get(fam, {}).get("class_counts"),
                    "E": e_group(fam), "F": f_group(fam), "P": p_group(fam), "FR": fr_group(fam),
                    "columns": COLUMNS.get(fam, {})}
    (C / "family_tables.json").write_text(json.dumps(out, indent=1, ensure_ascii=False, default=str) + "\n")
    lines = ["# 第一档 15 个家族：四组与四栏汇总（2026-10-08，收束包 v2.0）", "",
             "由 `scripts/closure/family_tables.py` 生成（机器可读版 `results/closure/family_tables.json`）。E 来自 2b 运行"
             "（`results/essential/phase2b/*/analysis.json`）；F、P 补充项、FR 模式 B 来自本轮（`results/closure/`）。所有计数附分母；"
             "bf16 候选只记录不判定（τ 不适用）。工具版本 2.3（混合来源检测修正，见协议 v3 第 10 节）。", "",
             f"共同的待审阅项：{COMMON_PENDING}。", "",
             "## 成本（本轮各组的工具时间，秒；E 与 2b 的 P 见 2b 记录的预算栏）", "",
             "| 家族 | F | P 补充（gradcheck / jvp） | FR 模式 B |", "|---|---|---|---|"]
    for fam in FAMS:
        r = out[fam]
        lines.append(f"| {fam} | {(r['F'] or {}).get('seconds', '—')} | {r['P'].get('seconds') or '—'} | {r['FR'].get('seconds', '—')} |")
    lines.append("")
    for fam in FAMS:
        r = out[fam]
        g = r["g3"]
        lines += [f"## {fam}", "",
                  f"- 单位（G3）：因子水平 {g['factor_levels']}，全组合 {g['full_factorial']}，计划 {g['planned']}，已执行 {g['executed']}，"
                  f"未覆盖组合 {g['uncovered_combinations']}；风险点 {g['risk_points']}",
                  f"- 契约 v2 分类：{r['contract_classes']}"]
        lines.append(f"- E（2b）：{json.dumps(r['E'], ensure_ascii=False)[:1500]}")
        f = r["F"]
        if f and f.get("beyond_over_elements") is not None:
            lines.append(f"- F（规格 {f['spec']}）：超出/元素 {json.dumps(f['beyond_over_elements'], ensure_ascii=False)[:1800]}；"
                         f"规格状态 {json.dumps(f['spec_status'], ensure_ascii=False)[:600]}；精度不变性（按条件计）{f['precision_invariance']}；"
                         f"无 f 的输出 {f['no_f_outputs']}")
        else:
            lines.append(f"- F：{f}")
        p = r["P"]
        lines.append(f"- P（2b 预注册）：{json.dumps(p['preregistered_2b'], ensure_ascii=False)[:1500]}")
        lines.append(f"- P（本轮 gradcheck / jvp 对 VJP）：{json.dumps(p['closure_gradcheck_jvp'], ensure_ascii=False)[:1500]}")
        lines.append(f"- FR（模式 B）：{json.dumps(r['FR'], ensure_ascii=False)[:1800]}")
        cols = r["columns"]
        for k, name in (("1", "第 1 栏 文档明确条款的违反"), ("2", "第 2 栏 声明解释下的差异"), ("3", "第 3 栏 未定义情形的实现约定"),
                        ("4", "第 4 栏 数值失败"), ("pending", "条款待审阅 / 契约外"), ("difference", "差异（非缺陷判定）")):
            if k in "1234":
                lines.append(f"- {name}：" + ("；".join(cols.get(k, [])) or "无"))
            elif cols.get(k):
                lines.append(f"- {name}：" + "；".join(cols[k]))
        lines.append("")
    (ROOT / "docs/closure_family_tables_20261008.md").write_text("\n".join(lines) + "\n")
    print("written", len(FAMS), "families")


if __name__ == "__main__":
    main()
