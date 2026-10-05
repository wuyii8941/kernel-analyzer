#!/usr/bin/env python3
"""Census table over results/tool_spec/<group>/*.json (tool_spec_check output).

Per output: reference completeness, e_sem / e_num verdicts (any rule or default-detector family detected) and
relative RMS (midpoints over the complete coordinates).  e_sem bins: none (not detected), constant-level
(detected, rel < 1e-7: constant rounding such as fp32 log2(e)), small (1e-7 .. 1e-5), candidate (>= 1e-5).

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


def sem_bin(rec, external=False):
    d = detected(rec)
    if d is None:
        return "unresolved"
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
                rows.append({"group": grp, "case": r["case"], "output": o,
                             "complete": e["reference_classes"]["complete_fraction"],
                             "sem": sem_bin(s, ext), "sem_rel": (s.get("scale") or {}).get("relative_rms"),
                             "num_detected": detected(n), "num_rel": (n.get("scale") or {}).get("relative_rms")})
        allrows += rows
        bins = {}
        for x in rows:
            bins[x["sem"]] = bins.get(x["sem"], 0) + 1
        lines += [f"## {grp}", "",
                  f"用例 {len(files)}，输出 {len(rows)}；e_sem 分档：" +
                  "，".join(f"{k} {v}" for k, v in sorted(bins.items())) +
                  f"；报错 {len(errors)}；不评（非 Triton 写出或之后被改）{len(skipped)}", "",
                  "| 用例 | 输出 | 参照完整 | e_sem | e_sem 相对 RMS | e_num 检出 | e_num 相对 RMS |", "|---|---|---|---|---|---|---|"]
        for x in rows:
            fmt = lambda v: "—" if v is None or v != v else f"{v:.1e}"  # noqa: E731
            lines.append(f"| {x['case']} | {x['output']} | {x['complete']:.2f} | {x['sem']} | {fmt(x['sem_rel'])} | "
                         f"{'是' if x['num_detected'] else ('否' if x['num_detected'] is False else '—')} | {fmt(x['num_rel'])} |")
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
