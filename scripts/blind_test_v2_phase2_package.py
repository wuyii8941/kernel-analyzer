#!/usr/bin/env python3
"""Phase 2 submission package for blind_test_v2.

    python scripts/blind_test_v2_phase2_package.py --date 20261004
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "reference_eval" / "blind_test_v2" / "phase2"
CACHE = ROOT / ".cache"

README = """# blind_test_v2 阶段 2 提交（{date}）

被测方：kernel-analyzer（提交 `{head}`）。阶段 1 冻结包 sha256 `51a7ce55…c86724b24` 不变。

- 规格 f 按阶段 2 发布文件第 1 节实现，独立于程序的 TTIR（`blind_test_v2_spec.py`，附在包内）：参照求值器的区间运算，求和为
  精确和外放一个 ulp，√ 与 (·)^(−1/2) 定向，G8 前缀和用精确整数（宽度 0）。与普通 float64 公式交叉核对，相对差在舍入水平。
- e_sem = K_R − f 以区间 [L_R − U_f, U_R − L_f] 进入（有向减法），K_R 是阶段 1 同一批捕获的参照（阶段 1 运行时逐 seed 缓存）。
- 规则 R1、R2、R3、R5，seed 0–95 与 96–191，端点保守推断与 Holm：与阶段 1 相同，判定层 `detector-v2.1`（`408ab3a`）。本阶段
  没有任何检验触发 2.1 相对第 2 版（`3490425`）新增的保护，所以判定与第 2 版相同；阶段 1 同样如此（796 个检验，0 个）。
- 只有一种标量口径（所有运行时标量 float32 精确，接口项为零）。

## 文件

| 文件 | 内容 |
|---|---|
| `phase2_report.md` | 报告：每个程序一行（e_sem 的符号比例、均值、最大宽度，R1–R5 两轮的 Holm 后判定与 μ 区间），检出含义，默认检测器（第二指标） |
| `phase2_verdict_matrix_v2.csv` | **判定矩阵**：38 个程序 × R1、R2、R3、R5，两轮判定、μ 区间、复现标记，e_sem 统计，默认检测器 |
| `phase2_report.json` | 每个程序两轮的完整记录（判定层的离线复算材料：每条规则每个确认 seed 的投影上下界、实际方向、原始与校正后 p） |
| `blind_test_v2_spec.py` | f 的求值代码 |

## 摘要

{summary}
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    args = parser.parse_args()
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                          check=True).stdout.strip()
    target = CACHE / f"blind_test_v2_phase2_submission_{args.date}"
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    for name in ("phase2_report.md", "phase2_verdict_matrix_v2.csv", "phase2_report.json"):
        shutil.copy2(RES / name, target / name)
    shutil.copy2(ROOT / "scripts" / "blind_test_v2_spec.py", target / "blind_test_v2_spec.py")
    md = (RES / "phase2_report.md").read_text()
    summary = [line for line in md.splitlines() if line.startswith("检出：")]
    (target / "README.md").write_text(README.format(date=args.date, head=head, summary="\n".join(summary)))
    lines = [f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(target)}"
             for p in sorted(target.rglob("*")) if p.is_file() and p.name != "SHA256SUMS"]
    (target / "SHA256SUMS").write_text("\n".join(lines) + "\n")
    out = Path(str(target) + ".tar.gz")
    with tarfile.open(out, "w:gz") as tar:
        tar.add(target, arcname=target.name)
    print(out, len(lines), "files", hashlib.sha256(out.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
