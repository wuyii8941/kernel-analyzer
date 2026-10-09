# Kernel Analyzer

测量 Triton kernel 的数值实现差异有没有系统性（平均）作用：从 kernel 的 IR（TTIR / TTGIR）自动生成实数语义 G 的严格包围，
再用固定的方向规则与端点保守统计判断 e_num = K − G 的平均作用是否非零。

当前版本、状态与待决事项只在 [`CURRENT.json`](CURRENT.json) 维护。仓库分四部分，每部分只保留最新、最全的版本：

| 部分 | 内容 | 入口 |
| --- | --- | --- |
| [1_experiments](1_experiments/) | 完整实验：DSL v2 开发评价（最新）、结构验收 v1.1（冻结运行，待计分）、通用能力轮与收束轮（general-v3.1）、本质错误一轮、盲测 v1 / v2 记录、独立规格 | [1_experiments/README.md](1_experiments/README.md) |
| [2_tool](2_tool/) | 现有工具：源码、脚本、测试、示例、规则注册表与工具文档 | [2_tool/README.md](2_tool/README.md) |
| [3_audits](3_audits/) | 完整审计：最新外部审计（基线 1aee15e）的发现与处理状态、此前审计的待决事项、收到的审计材料 | [3_audits/README.md](3_audits/README.md) |
| [4_bugs_cases](4_bugs_cases/) | 已找 bug 与重要案例：问题登记 B001–B023、复现、上游草稿、案例结论与主张 | [4_bugs_cases/README.md](4_bugs_cases/README.md) |

2026-10-09 整理前的全部内容（旧研究阶段的结果、脚本、测试与过程文档）都可从标签 `pre-reorg-20261009` 取回：
`git show pre-reorg-20261009:<旧路径>`。保留文件里出现的 `pre-reorg-20261009:<路径>` 即指该标签下的位置。

## 运行

```bash
export HOME=$PWD/.cache XDG_CACHE_HOME=$PWD/.cache TRITON_CACHE_DIR=$PWD/.cache/triton TMPDIR=$PWD/.cache/tmp
/data1/tzh/envs/ka_main/bin/python -m pytest -q 2_tool/tests          # 工具测试（不得改写已跟踪文件）
PYTHONPATH=2_tool/src /data1/tzh/envs/ka_main/bin/python 2_tool/scripts/measure.py --declaration 2_tool/examples/general/bsr_softmax.json --out report.json
```

环境与用法见 [2_tool/README.md](2_tool/README.md)。
