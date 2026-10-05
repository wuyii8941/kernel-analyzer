#!/usr/bin/env python3
"""Census table over results/tool_spec/<group>/*.json (tool_spec_check output).

Per output: reference completeness, e_sem / e_num verdicts (any rule or default-detector family detected) and
relative RMS (midpoints over the complete coordinates).  e_sem bins: none (not detected), constant-level
(detected, rel < 1e-7: constant rounding such as fp32 log2(e)), small (1e-7 .. 1e-5), candidate (>= 1e-5).

e_sem is "detected" when a rule or the default detector confirms it, or when it is certified: some coordinate's
residual interval K_R - f excludes zero (the spec's declared bound is inside f's interval, so this is a proof that
the kernel's real-arithmetic semantics differ from f there, with no statistics).  Certification covers deviations
whose sign follows the inputs (no systematic direction for the rules to find), e.g. dropped attention keys.

    python scripts/tool_spec_summary.py --groups flex inductor tridao fla inductor2 --out results/tool_spec/summary.md
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def detected(rec):
    if not rec or "rules" not in rec:
        return None
    hit = any(str(r.get("verdict", "")).startswith("DETECTED") for r in rec["rules"])
    det = rec.get("default_detector") or {}
    hit = hit or any((det.get(k) or {}).get("verdict") == "DETECTED" for k in ("vector_mean", "alignment"))
    return hit


def certified(rec):
    """Some coordinate's residual interval excludes zero (fractions over all units and used coordinates)."""
    res = (rec or {}).get("residual") or {}
    return (res.get("positive_frac") or 0) + (res.get("negative_frac") or 0) > 0


def sem_bin(rec, external=False, total=None):
    d = detected(rec)
    if d is None:
        return "unresolved"
    d = d or certified(rec)
    rel = (rec.get("scale") or {}).get("relative_rms", float("nan"))
    if d and total is not None and rel >= 1e-5 and total < 0.1 * rel and total < 1e-5:
        # e_sem undone by the rounded execution (K close to f): a rounding-dependent decision, not an error in K
        return "compensated (rounding-dependent decision)"
    rel = (rec.get("scale") or {}).get("relative_rms", float("nan"))
    if not d:
        return "none"
    if external:
        return "mixed (external re-entry)"
    if rel < 1e-7:
        return "constant-level"
    if rel < 1e-5:
        return "small"
    return "candidate"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--groups", nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    lines = ["# 对规格检查普查表（工具：e_num = K − K_R，e_sem = K_R − f）", ""]
    allrows = []
    for grp in a.groups:
        files = sorted(glob.glob(str(ROOT / "results" / "tool_spec" / grp / "*.json")))
        rows, errors, skipped = [], [], []
        for f in files:
            r = json.load(open(f))
            if "error" in r:
                errors.append((r["case"], r["error"].splitlines()[0][:120]))
                continue
            for o in r.get("outputs_not_written_by_triton", []) + r.get("outputs_modified_after_last_triton_write", []):
                skipped.append(f"{r['case']}:{o}")
            for o, e in r["outputs"].items():
                s, n = e["semantic"], e["numerical"]
                ext = bool(e.get("depends_on_non_triton_intermediates"))
                sv = e.get("special_values") or {}
                rows.append({"group": grp, "case": r["case"], "output": o,
                             "special_mismatch": (sv.get("kr_vs_f_class_mismatch", 0), sv.get("k_vs_f_class_mismatch", 0)),
                             "complete": e["reference_classes"]["complete_fraction"],
                             "sem": sem_bin(s, ext, (e.get("total") or {}).get("relative_rms")),
                             "sem_rel": (s.get("scale") or {}).get("relative_rms"),
                             "total_rel": (e.get("total") or {}).get("relative_rms"),
                             "num_detected": detected(n), "num_rel": (n.get("scale") or {}).get("relative_rms")})
        allrows += rows
        bins = {}
        for x in rows:
            bins[x["sem"]] = bins.get(x["sem"], 0) + 1
        special_rows = [x for x in rows if x["special_mismatch"][0] or x["special_mismatch"][1]]
        lines += [f"## {grp}", "",
                  f"用例 {len(files)}，输出 {len(rows)}；e_sem 分档：" +
                  "，".join(f"{k} {v}" for k, v in sorted(bins.items())) +
                  f"；报错 {len(errors)}；不评（非 Triton 写出或之后被改）{len(skipped)}", "",
                  "| 用例 | 输出 | 参照完整 | e_sem | e_sem 相对 RMS | e_num 检出 | e_num 相对 RMS |", "|---|---|---|---|---|---|---|"]
        for x in rows:
            fmt = lambda v: "—" if v is None or v != v else f"{v:.1e}"  # noqa: E731
            lines.append(f"| {x['case']} | {x['output']} | {x['complete']:.2f} | {x['sem']} | {fmt(x['sem_rel'])} | "
                         f"{'是' if x['num_detected'] else ('否' if x['num_detected'] is False else '—')} | {fmt(x['num_rel'])} |")
        if special_rows:
            lines += ["", "特殊值类别不一致（NaN / ±inf，K_R 对 f、K 对 f 的元素数）："] + [
                f"- {x['case']}:{x['output']} K_R≠f {x['special_mismatch'][0]}，K≠f {x['special_mismatch'][1]}"
                for x in special_rows]
        if errors:
            lines += ["", "报错："] + [f"- {c}: {m}" for c, m in errors]
        if skipped:
            lines += ["", "不评：" + "，".join(skipped)]
        lines.append("")
    a.out.write_text("\n".join(lines) + "\n")
    (a.out.with_suffix(".json")).write_text(json.dumps(allrows, indent=1, default=float) + "\n")
    print("\n".join(l for l in lines if l.startswith("## ") or l.startswith("用例")))


if __name__ == "__main__":
    main()
