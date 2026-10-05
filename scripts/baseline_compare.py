#!/usr/bin/env python3
"""Join the direct-differential baseline with the tool's results on the same cases (plan WP3).

    python scripts/baseline_compare.py --pair opinfo_onesample:opinfo_onesample_rerun --pair opinfo_loose:opinfo_loose_rerun \
        --pair opinfo_screen:opinfo_screen_rerun --out results/baseline/compare

Each --pair TOOLDIR[:OVERRIDE] names results/tool_spec/TOOLDIR (with the case JSONs of results/tool_spec/OVERRIDE taking
precedence: clean reruns) and results/baseline/TOOLDIR.  Per output:

  tool    the e_sem bin of tool_spec_summary.sem_bin, the special-value class mismatches (K_R / K vs f), or the error
  ci      CI-style direct differential failed on at least one seed (or the case raised); check_gradient=False of
          test_torchinductor_opinfo.py counts as not checked
  hp      relative RMS of K - f above 1e-5 (high-precision direct differential), or special-value mismatch
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from tool_spec_summary import sem_bin  # noqa: E402

OVERRIDES = json.load(open(ROOT / "scripts/data/inductor_override_cuda_f32.json"))
CI_LISTS = json.load(open(ROOT / "scripts/data/inductor_skips_xfails_cuda_f32.json"))
# limits of the baseline harness itself (not CI results): complex outputs, case names not built in this selection
BASELINE_NA = ("TypeError: complex output", "KeyError:")


def tool_rows(tool_dir, override_dir):
    files = {Path(f).stem: f for f in glob.glob(str(ROOT / "results/tool_spec" / tool_dir / "*.json"))}
    if override_dir:
        files.update({Path(f).stem: f for f in glob.glob(str(ROOT / "results/tool_spec" / override_dir / "*.json"))})
    out = {}
    for case, f in files.items():
        r = json.load(open(f))
        if "error" in r:
            out[(case, "*")] = {"tool": "error", "error": r["error"].splitlines()[0][:120]}
            continue
        for o in r.get("outputs_not_written_by_triton", []):
            out[(case, o)] = {"tool": "not Triton"}
        for o in r.get("outputs_modified_after_last_triton_write", []):
            out[(case, o)] = {"tool": "modified after Triton write"}
        for o in r.get("outputs_whose_writing_programs_aborted", {}):
            out[(case, o)] = {"tool": "programs aborted"}
        for o, e in r.get("outputs", {}).items():
            sv = e.get("special_values") or {}
            ext = bool(e.get("depends_on_non_triton_intermediates"))
            b = sem_bin(e.get("semantic"), ext, (e.get("total") or {}).get("relative_rms"), e.get("semantic_elementwise"))
            out[(case, o)] = {"tool": b, "special": (sv.get("kr_vs_f_class_mismatch", 0), sv.get("k_vs_f_class_mismatch", 0)),
                              "sem_rel": ((e.get("semantic") or {}).get("scale") or {}).get("relative_rms"),
                              "num_rel": ((e.get("numerical") or {}).get("scale") or {}).get("relative_rms"),
                              "seconds": r.get("seconds")}
    return out


def base_rows(base_dir):
    out = {}
    for f in glob.glob(str(ROOT / "results/baseline" / base_dir / "*.json")):
        r = json.load(open(f))
        case = r["case"]
        if "error" in r:
            if r["error"].startswith(BASELINE_NA):
                out[(case, "*")] = {"na": r["error"].splitlines()[0][:80]}
            else:
                out[(case, "*")] = {"ci": True, "hp": True, "error": r["error"].splitlines()[0][:120]}
            continue
        bwd = case.startswith("oib_")
        op = r.get("op", "")
        if op in CI_LISTS["skips"] or op in CI_LISTS["expected_failures"] or (
                bwd and op in CI_LISTS["gradient_expected_failures"]):
            for o in r["outputs"]:
                out[(case, o)] = {"na": "CI skip / expected failure"}
            continue
        skip_grad = bwd and OVERRIDES.get(r.get("op", ""), {}).get("check_gradient") is False
        for o, e in r["outputs"].items():
            ci = e["ci_found"] and not skip_grad and not str(e["ci_example"].get("verdict", "")).startswith("skipped")
            rel = e["hp_rel_rms_max"]
            out[(case, o)] = {"ci": bool(ci or e.get("shape_mismatch")), "ci_skipped_grad": skip_grad,
                              "hp": bool((rel is not None and rel > 1e-5) or e["hp_special_mismatch"] or e.get("shape_mismatch")),
                              "hp_rel": rel, "eager_rel": e["eager_rel_rms_max"], "seconds": r["seconds"]}
    return out


def tool_alarm(t):
    if t["tool"] in ("candidate", "error"):
        return True
    return t["tool"] not in ("not Triton", "modified after Triton write", "programs aborted") and any(t.get("special", (0, 0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", action="append", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    rows = []
    for pair in a.pair:
        tool_dir, _, override = pair.partition(":")
        t, b = tool_rows(tool_dir, override or None), base_rows(tool_dir)
        na_cases = {c for (c, o), v in b.items() if "na" in v}
        b = {k: v for k, v in b.items() if k[0] not in na_cases}
        cases_err_b = {c for (c, o) in b if o == "*"}
        cases_err_t = {c for (c, o) in t if o == "*"}
        for key in sorted(set(t) | set(b)):
            case, o = key
            tr = t.get(key) or ({"tool": "error"} if case in cases_err_t else {"tool": "absent"})
            br = b.get(key) or ({"ci": True, "hp": True, "error": "case error"} if case in cases_err_b else None)
            if br is None:
                continue
            rows.append({"set": tool_dir, "case": case, "output": o, **{f"tool_{k}": v for k, v in tr.items()},
                         **{f"base_{k}": v for k, v in br.items()}})
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "rows.json").write_text(json.dumps(rows, indent=0, default=str))

    lines = ["# 直接差分基线与工具的逐输出对照", ""]
    for s in sorted({r["set"] for r in rows}) + ["all"]:
        rs = [r for r in rows if s == "all" or r["set"] == s]
        tal = [r for r in rs if tool_alarm({"tool": r["tool_tool"], "special": r.get("tool_special", (0, 0))})]
        lines += [f"## {s}（{len({r['case'] for r in rs})} 个用例，{len(rs)} 个输出）", "",
                  "| 工具分档 | 输出数 | CI 式差分失败 | 高精度差分 > 1e-5 |", "|---|---:|---:|---:|"]
        by = Counter(r["tool_tool"] for r in rs)
        for bin_, n in by.most_common():
            sub = [r for r in rs if r["tool_tool"] == bin_]
            lines.append(f"| {bin_} | {n} | {sum(r['base_ci'] for r in sub)} | {sum(r['base_hp'] for r in sub)} |")
        both = sum(1 for r in tal if r["base_ci"])
        lines += ["", f"工具报警（候选、错误、特殊值类别不一致）{len(tal)} 个输出，其中 CI 式差分也失败 {both} 个；"
                  f"CI 式差分失败而工具未报警 {sum(1 for r in rs if r['base_ci'] and r not in tal)} 个。", ""]
    lines += ["## CI 式差分失败而工具未报警的输出", "", "| 集合 | 用例 | 输出 | 工具分档 | 工具 e_num 相对 RMS | 高精度相对 RMS | eager fp32 相对 RMS |", "|---|---|---|---|---:|---:|---:|"]
    for r in rows:
        if r["base_ci"] and not tool_alarm({"tool": r["tool_tool"], "special": r.get("tool_special", (0, 0))}):
            lines.append(f"| {r['set']} | {r['case']} | {r['output']} | {r['tool_tool']} | {r.get('tool_num_rel')} | {r.get('base_hp_rel')} | {r.get('base_eager_rel')} |")
    lines += ["", "## 工具报警而 CI 式差分通过的输出", "", "| 集合 | 用例 | 输出 | 工具分档 | e_sem 相对 RMS | 特殊值 | 高精度相对 RMS | 梯度检查被关 |", "|---|---|---|---|---:|---|---:|---|"]
    for r in rows:
        if tool_alarm({"tool": r["tool_tool"], "special": r.get("tool_special", (0, 0))}) and not r["base_ci"]:
            lines.append(f"| {r['set']} | {r['case']} | {r['output']} | {r['tool_tool']} | {r.get('tool_sem_rel')} | "
                         f"{r.get('tool_special')} | {r.get('base_hp_rel')} | {r.get('base_ci_skipped_grad')} |")
    tt = [r["tool_seconds"] for r in rows if r.get("tool_seconds")]
    bt = [r["base_seconds"] for r in rows if r.get("base_seconds")]
    if tt and bt:
        import statistics as st
        lines += ["", f"耗时（每用例，不含进程启动）：工具中位数 {st.median(tt):.1f} s，直接差分中位数 {st.median(bt):.1f} s。"]
    (a.out / "compare.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:60]))


if __name__ == "__main__":
    main()
