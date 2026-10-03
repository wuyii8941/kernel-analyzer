#!/usr/bin/env python3
"""Audit package for blind_test_v1 after the reviewer group 1 fixes.

    .cache/blind_test_v1_audit_<date>/                 frozen submission (scoring basis), post-submission
                                                       findings, tool fixes, regression, code snapshot
    .cache/blind_test_v1_regression_vectors_<date>/    the direction vectors of the regression (large)

each with README.md and SHA256SUMS, packed as .tar.gz next to the directories.

    python scripts/blind_test_v1_audit_package.py --date 20261004
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
RES = ROOT / "results" / "reference_eval"
BT = RES / "blind_test_v1"
CACHE = ROOT / ".cache"

README = """# blind_test_v1 审阅材料（{date}）

被测方：kernel-analyzer。代码仓库 https://github.com/wuyii8941/kernel-analyzer ，本包对应提交 `{head}`（`05_code/` 有该提交
的源码快照与提交列表）。答案文件 `answer_key_SEALED.json` 从未打开。环境：NVIDIA RTX A6000（sm_86），Triton 3.6.0，torch 2.10.0。

## 目录

| 目录 | 内容 | 用途 |
|---|---|---|
| `01_frozen_submission/` | 冻结的判定矩阵与报告：阶段 1（协议 v1.1，seed 0–95 与 96–191）、阶段 2（两种标量口径） | **计分依据**，与此前提交的内容相同，未作任何改动 |
| `02_post_submission_findings/` | 提交之后发现的一处统计实现问题及其影响复核 | 计分时参考，见下文第 2 节 |
| `03_tool_fixes/` | 按两位审阅方第一组意见所作的修改说明、默认检测器第 1 版与第 2 版的校准、更新后的主张与范围 | 审阅修复 |
| `04_regression_v2/` | 修复版把 blind_test_v1 当回归集重跑的报告、与冻结结果的逐格对比、离线复算记录 | 审阅修复；**不替代**冻结结果 |
| `05_code/` | 提交 `{head}` 的源码快照（src、scripts、tests、相关文档）与提交列表 | 复核实现 |

方向向量（坐标集、R4、R5、默认检测器的学习方向与分组方向，约 175 MB）在另一个压缩包
`blind_test_v1_regression_vectors_{date}.tar.gz`，只在需要从原始数据重算投影时用到；用投影上下界复算统计不需要它。
此前交付的核实数据（阶段 1 第 5 节材料、阶段 2 第 6 节子集、阶段 2 核实材料）仍在仓库的 `blind-v1-deliverables`
分支，本包不重复。

## 1 请审阅方做的事

1. **冻结结果计分、揭盲**：以 `01_frozen_submission/` 的两个判定矩阵为准。默认检测器（第二指标）请与协议规则分开计分；
   第 1 版默认检测器在阶段 2 的 6 个输出上的报警是已确认的误报（见第 3 节），按冻结结果原样计入。
2. **参数更新层的盲测**：测量点从 kernel 输出换成经 optimizer 之后的更新差 u（研究主张依赖的那一层）。请给出这一层的
   阳性设计与协议（optimizer 种类、moment 状态的来源）；被测方先按 AdamW 与真实训练保存的状态准备测量点。
3. **新组合上的成绩**：默认检测器第 2 版冻结于提交 `{head}`，请提供一批未参与本轮修改的新程序，评价检出、误报、覆盖与成本。
4. 对 `03_tool_fixes/` 与 `04_regression_v2/` 的修复审阅。

## 2 提交之后发现的问题：端点保守 p 值

冻结运行用的保守 p 值是两个端点 t 检验双侧 p 值的较大者。对跨零的盒（E[l] < 0 < E[h] 都显著）它会很小，而区间判定是
未确认；Holm 于是把这类检验算作拒绝，放宽了其余检验的阈值。正确的形式是 2·min(p(E[l] > 0), p(E[h] < 0))，与区间判定
一致，已在代码中修正。对冻结结果的复核（`holm_pvalue_recheck.json`）：

| 运行 | 检验数 | 被误算为 Holm 拒绝的跨零检验 | 最终检出（提交 → 复算） |
|---|---:|---:|---|
{recheck_rows}

所有最终检出都不变：检出格的 p 值远小于最严的 Holm 阈值，阶段 1 没有受影响的检验。

## 3 修复与回归的结论

- 默认检测器第 2 版：每个出口都按区间做端点保守检验；校准中点残差的误报率 3.0–5.4%（向量均值族）、3.0–4.2%（对齐族），
  真值为零而区间中点带系统偏移时 0/1000。
- 修复版重跑（`04_regression_v2/regression_report.md`）：协议规则判定在实现、语义两种口径三个量上 0 处变化；默认检测器只在
  6 个语义残差区间全部含零的输出上由检出变为未确认（prog_07、11、12、21、26 的对齐族，prog_25 rstd 的两族），没有其他变化。
- 新拆出的量：接口项 f_interface − f_published（启动时的标量舍入，F4 的 t、F1/F3 的 eps）与总差异 K − f_published；
  编译期常数清单（prog_05、08、27 的 8 个分块合并系数）。
- 离线复算记录 `regression_records.json`：每个程序、输出、量、规则的每个确认 seed 的投影上下界，实际方向（常数、按单位的
  归一化标量或向量名），开发/确认 seed，原始 p 与 Holm 校正后 p，排除原因，默认检测器各检验的 p 与校正后 p。

这批程序在修复过程中已被看过，回归结果不能当作泛化成绩。

## 校验

`SHA256SUMS` 覆盖本目录全部文件。
"""


def sha_tree(target: Path):
    lines = []
    for p in sorted(target.rglob("*")):
        if p.is_file() and p.name != "SHA256SUMS":
            lines.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(target)}")
    (target / "SHA256SUMS").write_text("\n".join(lines) + "\n")
    return len(lines)


def pack(target: Path):
    out = target.with_suffix(".tar.gz") if target.suffix == "" else Path(str(target) + ".tar.gz")
    with tarfile.open(out, "w:gz") as tar:
        tar.add(target, arcname=target.name)
    return out


def main():
    import json

    from blind_test_v1_bundle import phase1_matrix, phase2_matrix

    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    args = parser.parse_args()
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                          check=True).stdout.strip()
    target = CACHE / f"blind_test_v1_audit_{args.date}"
    if target.exists():
        shutil.rmtree(target)
    dirs = {name: target / name for name in ("01_frozen_submission", "02_post_submission_findings", "03_tool_fixes",
                                             "04_regression_v2", "05_code")}
    for d in dirs.values():
        d.mkdir(parents=True)
    f = dirs["01_frozen_submission"]
    phase1_matrix(f / "phase1_verdict_matrix_v1.1.csv")
    phase2_matrix(f / "phase2_verdict_matrix.csv")
    copies = {
        BT / "phase1_report_v1.1.md": f / "phase1_report_v1.1.md",
        BT / "phase1_report_v1.1.json": f / "phase1_report_v1.1.json",
        BT / "phase1_report.csv": f / "phase1_report_seeds_0_95.csv",
        BT / "phase1_report.json": f / "phase1_report_seeds_0_95.json",
        BT / "replication_96_191" / "phase1_report.csv": f / "phase1_report_seeds_96_191.csv",
        BT / "replication_96_191" / "phase1_report.json": f / "phase1_report_seeds_96_191.json",
        BT / "phase2" / "phase2_report.md": f / "phase2_report.md",
        BT / "phase2" / "phase2_report.json": f / "phase2_report.json",
        BT / "phase2" / "check_f3_merge.json": f / "check_f3_merge.json",
        BT / "holm_pvalue_recheck.json": dirs["02_post_submission_findings"] / "holm_pvalue_recheck.json",
        ROOT / "docs" / "tool_changes_20261003.md": dirs["03_tool_fixes"] / "tool_changes_20261003.md",
        ROOT / "docs" / "claims.md": dirs["03_tool_fixes"] / "claims.md",
        RES / "detector_calibration.json": dirs["03_tool_fixes"] / "detector_calibration_v1.json",
        RES / "detector_calibration_v2.json": dirs["03_tool_fixes"] / "detector_calibration_v2.json",
        BT / "regression_v2" / "regression_report.md": dirs["04_regression_v2"] / "regression_report.md",
        BT / "regression_v2" / "regression_summary.json": dirs["04_regression_v2"] / "regression_summary.json",
        BT / "regression_v2" / "regression_records.json": dirs["04_regression_v2"] / "regression_records.json",
    }
    for src, dst in copies.items():
        shutil.copy2(src, dst)
    code = dirs["05_code"]
    subprocess.run(["git", "archive", "--format=tar.gz", "-o", str(code / f"source_{head}.tar.gz"), "HEAD", "src",
                    "scripts", "tests", "README.md", "docs/tool_changes_20261003.md", "docs/claims.md"],
                   cwd=ROOT, check=True)
    log = subprocess.run(["git", "log", "--format=%h %ad %s", "--date=short", "e43616c^..HEAD"], cwd=ROOT,
                         capture_output=True, text=True, check=True).stdout
    (code / "COMMITS.txt").write_text("检测器冻结于 e43616c（盲测前），覆盖补充 cf7631c；之后的提交：\n\n" + log)
    recheck = json.loads((BT / "holm_pvalue_recheck.json").read_text())
    names = {"phase1_seeds_0_95": "阶段 1，seed 0–95", "phase1_seeds_96_191": "阶段 1，seed 96–191",
             "phase2_published_scalars": "阶段 2，发布标量", "phase2_float32_scalars": "阶段 2，float32 标量"}
    rows = "\n".join(f"| {names[k]} | {v['tests']} | {v['straddling_tests_counted_as_holm_rejections']} | "
                     f"{v['final_detections_submitted']} → {v['final_detections_recomputed']} |"
                     for k, v in recheck.items() if k in names)
    (target / "README.md").write_text(README.format(date=args.date, head=head, recheck_rows=rows))
    n = sha_tree(target)
    print(pack(target), n, "files")

    vec = CACHE / f"blind_test_v1_regression_vectors_{args.date}"
    if vec.exists():
        shutil.rmtree(vec)
    vec.mkdir()
    for p in sorted((CACHE / "blind_v1_regression_v2").glob("prog_*.npz")):
        shutil.copy2(p, vec / p.name)
    (vec / "README.md").write_text(
        f"# blind_test_v1 回归的方向向量（{args.date}，提交 `{head}`）\n\n"
        "每个程序一个 .npz，键名为 `<程序>__<输出>__<量>__<规则或 detector__检验>`，坐标集为 "
        "`<程序>__<输出>__<量>__coordinate_set`（按输出位置的布尔数组；本轮 205 条记录全部使用完整坐标集）。R4 为声明向量，R5 为在开发 seed 上学习的方向，detector__learned / rows / cols 为默认检测器"
        "的学习方向与分组方向（已拉回到原坐标）。R1、R2、R3 与对齐检验的方向不存向量：R1 为常数，R2、R3 由 K_R 中点与记录中"
        "的归一化标量逐单位重建（`regression_records.json` 中每条规则的 `direction` 字段）。\n")
    n = sha_tree(vec)
    print(pack(vec), n, "files")


if __name__ == "__main__":
    main()
