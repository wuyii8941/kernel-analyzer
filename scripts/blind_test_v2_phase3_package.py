#!/usr/bin/env python3
"""Phase 3 submission package for blind_test_v2.

    python scripts/blind_test_v2_phase3_package.py --date 20261004
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "reference_eval" / "blind_test_v2" / "phase3"
CACHE = ROOT / ".cache"

README = """# blind_test_v2 阶段 3 提交：参数更新层（{date}）

被测方：kernel-analyzer（提交 `{head}`）。阶段 1、2 的冻结包不变。判定层 `detector-v2.1`（`408ab3a`）。

口径：`blind_test_v2_phase3_protocol.md`（运行前冻结于 `70a5cee`；第 5 节记录出题方的确认）。

- **计分用的三张 38 × 4 判定矩阵**（按 `report_template_phase3.csv`）：`phase3_sgd.csv`、`phase3_adamw_history.csv`、
  `phase3_adamw_zero.csv`，口径为出题方确认的理想响应：optimizer 取实数语义，用区间算术对 K（点）与 K_R（区间）各走一步，
  u = step(K) − step(K_R)，不含 optimizer 自身的 FP32 舍入。模板的 μ 与判定列为 seed 0–95 一轮，`reproduced_seeds_96_191`
  列出两轮同号检出的规则；两轮完整判定见 `phase3_report.md`。
- **附加的三张**：`phase3_*_actual_write.csv`，实际 FP32 写入差（阶段 1 审阅方建议的口径）：同一个 FP32 torch optimizer、同一参数与
  状态、只替换梯度，u = θ′(K) − θ′(RN32(K_R))，RN32 不唯一时枚举 FP32 候选取可靠包围。不用于计分。
- 共同状态：θ_s 由 seed 5000 + s 生成；AdamW（带历史）的 m、v 由规格 f 在历史 seed 的输入上 RN32 后经同一 FP32 AdamW 16 步得到
  （出题方确认的替代办法），t = 17；AdamW（零动量）t = 1。

## 文件

| 文件 | 内容 |
|---|---|
| `phase3_report.md` | 报告：每种 optimizer、每个口径的两轮判定与 μ 区间，默认检测器（第二指标） |
| `phase3_sgd.csv`、`phase3_adamw_history.csv`、`phase3_adamw_zero.csv` | **计分矩阵**（理想响应） |
| `phase3_*_actual_write.csv` | 附加矩阵（实际写入差） |
| `phase3_report.json` | 每个程序、每种 optimizer、每个口径、每轮的完整记录（离线复算材料：每条规则每个确认 seed 的投影上下界、实际方向、原始与校正后 p） |
| `blind_test_v2_phase3_protocol.md` | 冻结口径 |

核实材料（seed 0 与 32 的输入、f、θ、AdamW 历史状态、K、K_R 区间与三种 optimizer 下计分口径的 u 区间）另见
`blind_test_v2_verification_{date}.tar.gz`。

## 摘要

{summary}
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    args = parser.parse_args()
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                          check=True).stdout.strip()
    target = CACHE / f"blind_test_v2_phase3_submission_{args.date}"
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    names = ["phase3_report.md", "phase3_report.json"] + [f"phase3_{o}{s}.csv" for o in ("sgd", "adamw_history", "adamw_zero")
                                                         for s in ("", "_actual_write")]
    for name in names:
        shutil.copy2(RES / name, target / name)
    shutil.copy2(ROOT / "docs" / "blind_test_v2_phase3_protocol.md", target / "blind_test_v2_phase3_protocol.md")
    md = (RES / "phase3_report.md").read_text()
    summary = "\n".join(line for line in md.splitlines() if line.startswith("| optimizer") or line.startswith("|---|---|---:")
                        or (line.startswith("| ") and ("理想响应" in line or "实际写入" in line) and line.count("|") == 6))
    (target / "README.md").write_text(README.format(date=args.date, head=head, summary=summary))
    lines = [f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(target)}"
             for p in sorted(target.rglob("*")) if p.is_file() and p.name != "SHA256SUMS"]
    (target / "SHA256SUMS").write_text("\n".join(lines) + "\n")
    out = Path(str(target) + ".tar.gz")
    with tarfile.open(out, "w:gz") as tar:
        tar.add(target, arcname=target.name)
    print(out, len(lines), "files", hashlib.sha256(out.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
