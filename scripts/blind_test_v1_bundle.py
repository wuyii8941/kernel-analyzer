#!/usr/bin/env python3
"""Submission bundle for blind_test_v1: verdict matrices, reports and the verification archives.

Writes the phase 1 verdict matrix (protocol v1.1: 37 programs, F3 as two columns, R1-R5, both seed sets with
the final Holm verdicts and the mu intervals) and the phase 2 verdict matrix (e_sem = K_R - f, both scalar
modes) as CSV, copies the reports, and assembles

    .cache/blind_test_v1_reports/      matrices + reports (small)
    .cache/blind_test_v1_submission/   the same plus the four verification archives

each with README.md and SHA256SUMS.

    python scripts/blind_test_v1_bundle.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "reference_eval" / "blind_test_v1"
CACHE = ROOT / ".cache"
RULES = ("R1", "R2", "R3", "R4", "R5")
ARCHIVES = {
    "blind_v1_verification.tar.gz": "阶段 1 协议 v1.1 第 5 节独立核实材料（15 个程序，seed 0 与 32）",
    "blind_v1_subset_priority.tar.gz": "阶段 2 发布文件第 6 节子集，优先部分（F1、F3、F5）",
    "blind_v1_subset_others.tar.gz": "阶段 2 第 6 节子集，其余",
    "blind_v1_phase2_verification.tar.gz": "阶段 2 独立核实材料（19 个程序，seed 0 与 32）",
}


def interval(x):
    return "" if x is None else f"[{x[0]:.6e}, {x[1]:.6e}]"


def phase1_matrix(path: Path):
    matrix = json.loads((RES / "phase1_report_v1.1.json").read_text())
    header = ["program", "output", "family"]
    for r in RULES:
        header += [f"{r}_verdict_seeds_0_95", f"{r}_mu_interval_seeds_0_95", f"{r}_verdict_seeds_96_191",
                   f"{r}_mu_interval_seeds_96_191", f"{r}_detected_and_reproduced"]
    rows = []
    for name, m in matrix.items():
        pid, _, out = name.partition("（")
        row = {"program": pid, "output": out.rstrip("）") or "output", "family": m["family"]}
        for r in RULES:
            c = m["cells"].get(r)
            if c is None:
                for suffix in ("verdict_seeds_0_95", "mu_interval_seeds_0_95", "verdict_seeds_96_191",
                               "mu_interval_seeds_96_191", "detected_and_reproduced"):
                    row[f"{r}_{suffix}"] = "not_applicable"
                continue
            a, b = c["seeds_0_95"], c["seeds_96_191"] or {}
            row.update({f"{r}_verdict_seeds_0_95": a["verdict"], f"{r}_mu_interval_seeds_0_95": interval(a["interval"]),
                        f"{r}_verdict_seeds_96_191": b.get("verdict", ""),
                        f"{r}_mu_interval_seeds_96_191": interval(b.get("interval")),
                        f"{r}_detected_and_reproduced": str(c["detected_and_reproduced"]).lower()})
        rows.append(row)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def phase2_matrix(path: Path):
    report = json.loads((RES / "phase2" / "phase2_report.json").read_text())
    header = ["program", "output", "family", "e_sem_mean", "e_sem_max_width", "e_sem_positive_frac",
              "e_sem_negative_frac", "e_sem_contains_zero_frac"]
    for mode in ("given", "fp32"):
        for r in RULES:
            header += [f"{r}_verdict_{mode}", f"{r}_mu_interval_{mode}"]
    header += ["default_detector_vector_mean_given", "default_detector_alignment_given"]
    rows = []
    for pid in sorted(report, key=lambda p: (report[p]["family"], p)):
        rep = report[pid]
        for o, o32 in zip(rep["modes"]["given"], rep["modes"]["fp32"]):
            res = o["residual"]
            row = {"program": pid, "output": o["output"], "family": rep["family"], "e_sem_mean": f"{res['mean']:.6e}",
                   "e_sem_max_width": f"{res['max_width']:.3e}", "e_sem_positive_frac": f"{res['positive_frac']:.4f}",
                   "e_sem_negative_frac": f"{res['negative_frac']:.4f}",
                   "e_sem_contains_zero_frac": f"{res['contains_zero_frac']:.4f}"}
            for mode, oo in (("given", o), ("fp32", o32)):
                for r in RULES:
                    x = oo["rules"].get(r) if isinstance(oo["rules"], dict) else None
                    if not isinstance(x, dict) or "mu_interval" not in x:
                        row[f"{r}_verdict_{mode}"] = "not_applicable"
                        row[f"{r}_mu_interval_{mode}"] = ""
                    else:
                        row[f"{r}_verdict_{mode}"] = x["final_verdict"]
                        row[f"{r}_mu_interval_{mode}"] = interval(x["mu_interval"])
            dd = o.get("default_detector", {})
            row["default_detector_vector_mean_given"] = dd.get("vector_mean", {}).get("final_verdict", "")
            row["default_detector_alignment_given"] = dd.get("alignment", {}).get("final_verdict", "")
            rows.append(row)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows)
    return len(rows)


README = """# blind_test_v1 被测方提交材料

代码仓库 https://github.com/wuyii8941/kernel-analyzer （main）。检测器冻结于 `e43616c`，评测前的覆盖补充为 `cf7631c`，
之后没有改动检测器、方向规则或阈值；答案文件 `answer_key_SEALED.json` 从未打开。环境：NVIDIA RTX A6000（sm_86），
Triton 3.6.0，torch 2.10.0。没有修改任何程序源码。

## 判定矩阵

- `phase1/phase1_verdict_matrix_v1.1.csv`：阶段 1（e = K − K_R），37 个程序（F3 的 mean、rstd 两列分开，共 {n1} 行）×
  R1–R5。每格给出 seed 0–95 与 seed 96–191 两轮的 Holm 后最终判定（DETECTED_POSITIVE / DETECTED_NEGATIVE /
  NOT_CONFIRMED，规则不适用为 not_applicable）、μ 的端点保守区间（未做多重校正的单格 95% 区间），以及两轮都检出且同号
  的标记。每轮各自对全部 170 个「程序 × 规则」检验做 Holm。R5 的方向在每轮的开发 seed 上分别学习，所以 R5 的复现是
  程序级复现。
- `phase2/phase2_verdict_matrix.csv`：阶段 2（e_sem = K_R − f），{n2} 行 × R1–R5，两种运行时标量取法（given = 发布值，
  fp32 = 先舍入到 float32），以及 e_sem 的符号比例、均值与宽度。默认检测器一栏只作第二指标；它对 e_sem 取区间中点，
  在 e_sem 区间全部含零的行上也会报警（阶段 2 报告解读第 5 条），不应计入判定。

## 报告

- `phase1/phase1_report_v1.1.md`（及 `.json`）：按协议修订 v1.1 与阶段 2 发布文件第 4 节改写的阶段 1 报告。
- `phase1/phase1_report_seeds_0_95.csv`、`phase1/phase1_report_seeds_96_191.csv`：按 `report_template_phase1.csv` 填写的
  两轮完整报告（覆盖、残差、μ、定位、耗时）；`.json` 为对应的完整机器可读结果。
- `phase1/phase1_report_original.md`：最初提交的阶段 1 报告（v1.1 之前），留作对照。
- `phase2/phase2_report.md`（及 `.json`）：阶段 2 报告；`phase2/check_f3_merge.json` 是解读第 3 条的事后核对结果。
{archives}
## 校验

`SHA256SUMS` 覆盖本目录下的全部文件。
"""


def assemble(target: Path, with_archives: bool):
    if target.exists():
        shutil.rmtree(target)
    (target / "phase1").mkdir(parents=True)
    (target / "phase2").mkdir()
    n1 = phase1_matrix(target / "phase1" / "phase1_verdict_matrix_v1.1.csv")
    n2 = phase2_matrix(target / "phase2" / "phase2_verdict_matrix.csv")
    copies = {
        RES / "phase1_report_v1.1.md": "phase1/phase1_report_v1.1.md",
        RES / "phase1_report_v1.1.json": "phase1/phase1_report_v1.1.json",
        RES / "phase1_report.csv": "phase1/phase1_report_seeds_0_95.csv",
        RES / "phase1_report.json": "phase1/phase1_report_seeds_0_95.json",
        RES / "replication_96_191" / "phase1_report.csv": "phase1/phase1_report_seeds_96_191.csv",
        RES / "replication_96_191" / "phase1_report.json": "phase1/phase1_report_seeds_96_191.json",
        RES / "phase1_report.md": "phase1/phase1_report_original.md",
        RES / "phase2" / "phase2_report.md": "phase2/phase2_report.md",
        RES / "phase2" / "phase2_report.json": "phase2/phase2_report.json",
        RES / "phase2" / "check_f3_merge.json": "phase2/check_f3_merge.json",
    }
    for src, dst in copies.items():
        shutil.copy2(src, target / dst)
    archives = ""
    if with_archives:
        (target / "data").mkdir()
        lines = ["", "## 核实数据（`data/`）", ""]
        for name, desc in ARCHIVES.items():
            shutil.copy2(CACHE / name, target / "data" / name)
            lines.append(f"- `data/{name}`：{desc}。")
        lines.append("")
        lines.append("每个压缩包内都有 README 或 index.json，说明文件格式与构造方式。")
        archives = "\n".join(lines) + "\n"
    (target / "README.md").write_text(README.format(n1=n1, n2=n2, archives=archives))
    sums = []
    for p in sorted(target.rglob("*")):
        if p.is_file() and p.name != "SHA256SUMS":
            sums.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(target)}")
    (target / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    print(target, n1, "phase 1 rows,", n2, "phase 2 rows,", len(sums), "files")


def main():
    assemble(CACHE / "blind_test_v1_reports", with_archives=False)
    assemble(CACHE / "blind_test_v1_submission", with_archives=True)


if __name__ == "__main__":
    main()
