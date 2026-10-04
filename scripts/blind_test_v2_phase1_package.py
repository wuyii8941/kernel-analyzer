#!/usr/bin/env python3
"""Phase 1 submission package for blind_test_v2.

    python scripts/blind_test_v2_phase1_package.py --date 20261004
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "reference_eval" / "blind_test_v2"
CACHE = ROOT / ".cache"

README = """# blind_test_v2 阶段 1 提交（{date}）

被测方：kernel-analyzer（https://github.com/wuyii8941/kernel-analyzer，本包对应提交 `{head}`）。盲测包 sha256 `{sha}`。
两个封存答案文件在解包时被排除，从未写到磁盘上；没有修改任何程序源码。

**被测版本**：默认检测器与判定层冻结于标签 `detector-v2.1`（提交 `408ab3a`），在打开盲测包之前。包内 README 写的是第 2 版
（`3490425`）；2.1 只多了上一轮回归审阅要求在开测前完成的两项修改（所有入口用有向减法构造残差区间；样本不足或数值失效时返回
无法判断）。打开盲测包之后，检测器、方向规则与阈值都没有改动。

## 文件

| 文件 | 内容 |
|---|---|
| `phase1_report_v2.md` | 阶段 1 报告：两组 seed 的判定与 μ 区间、默认检测器（第二指标）、逐位相同配对、定位、接口标量与编译期常数 |
| `phase1_verdict_matrix_v2.csv` | **判定矩阵**：38 个程序 × R1、R2、R3、R5，两组 seed 的 Holm 后判定、μ 区间、两组都检出且同号的标记 |
| `phase1_report_v2.json` | 同一矩阵的 JSON |
| `seeds_0_95/phase1_report.csv`、`seeds_96_191/phase1_report.csv` | 按 `report_template_phase1.csv` 填写的两组完整报告（含编译期常数与接口标量两列） |
| `seeds_0_95/phase1_report.json`、`seeds_96_191/phase1_report.json` | 每个程序的完整机器记录，含判定层的离线复算材料（每条规则每个确认 seed 的投影上下界、实际方向、原始 p） |
| `coverage_check.json` | 评测前覆盖检查（38 个程序覆盖完整，无被拒绝的操作） |

## 摘要

{summary}

## 阶段 3 开始前请出题方确认的两点

1. **AdamW（带历史）的动量来源**：协议用同家族干净程序在历史 seed 上的输出构造 m、v，但哪个程序是干净版本属于盲测信息。
   被测方建议用阶段 2 公布的规格 f 在历史输入上的值（舍入到 float32）作为干净输出；也可以由出题方直接给出 m、v。
2. **u 的计算口径**：被测方计划在 optimizer 的实数语义下，用区间算术分别对 K（点）与 K_R（区间）求一步，u = step(K) − step(K_R)，
   不含 optimizer 自身的 float32 舍入（那是另一个被测实现）。weight_decay = 0，θ 在 u 中抵消。

## 校验

`SHA256SUMS` 覆盖本目录全部文件。
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    args = parser.parse_args()
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                          check=True).stdout.strip()
    sha = hashlib.sha256((ROOT / "blind_test_v2.tar.gz").read_bytes()).hexdigest()
    target = CACHE / f"blind_test_v2_phase1_submission_{args.date}"
    if target.exists():
        shutil.rmtree(target)
    (target / "seeds_0_95").mkdir(parents=True)
    (target / "seeds_96_191").mkdir()
    for name in ("phase1_report_v2.md", "phase1_report_v2.json", "phase1_verdict_matrix_v2.csv", "coverage_check.json"):
        shutil.copy2(RES / name, target / name)
    for src, dst in (("phase1", "seeds_0_95"), ("replication_96_191", "seeds_96_191")):
        for name in ("phase1_report.csv", "phase1_report.json"):
            shutil.copy2(RES / src / name, target / dst / name)
    md = (RES / "phase1_report_v2.md").read_text()
    cov = md.split("## 覆盖与参照", 1)[1].split("##", 1)[0].strip()
    (target / "README.md").write_text(README.format(date=args.date, head=head, sha=sha, summary=cov))
    lines = []
    for p in sorted(target.rglob("*")):
        if p.is_file() and p.name != "SHA256SUMS":
            lines.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(target)}")
    (target / "SHA256SUMS").write_text("\n".join(lines) + "\n")
    out = Path(str(target) + ".tar.gz")
    with tarfile.open(out, "w:gz") as tar:
        tar.add(target, arcname=target.name)
    print(out, len(lines), "files", hashlib.sha256(out.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
