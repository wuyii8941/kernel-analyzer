# Kernel Analyzer

测量 Triton kernel 的数值实现差异有没有系统性（平均）作用：从 kernel 的 IR（TTIR / TTGIR）自动生成实数语义 G 的严格包围，
再用固定的方向规则与端点保守统计判断 e_num = K − G 的平均作用是否非零。

当前版本、状态与待决事项只在 [`CURRENT.json`](CURRENT.json) 维护。仓库分五部分，每部分只保留最新、最全的版本：

| 部分 | 内容 | 入口 |
| --- | --- | --- |
| [1_experiments](1_experiments/) | 完整实验：第二轮常见配置下的容差盲点与 bias（最新）、第一批验收、DSL v2 开发评价、结构验收 v1.1（冻结运行，待计分）、通用能力轮与收束轮（general-v3.1）、本质错误一轮、盲测 v1 / v2 记录 | [1_experiments/README.md](1_experiments/README.md) |
| [2_tool](2_tool/) | 现有工具：源码、脚本、测试、示例、规则注册表与工具文档 | [2_tool/README.md](2_tool/README.md) |
| [3_audits](3_audits/) | 完整审计：最新外部审计（基线 1aee15e）的发现与处理状态、第一批与第二轮的测量修复、此前审计的待决事项、收到的审计材料 | [3_audits/README.md](3_audits/README.md) |
| [4_bugs_cases](4_bugs_cases/) | 已找 bug 与重要案例：问题登记 B001–B024、复现、上游草稿、案例结论与主张 | [4_bugs_cases/README.md](4_bugs_cases/README.md) |
| [5_specification](5_specification/) | 规格：任务规格 f 的方法、来源登记、62 份合同、127 个 profile（65 个具备声明范围内的参照准入条件）、62 份真实调用计划（58 份已审阅）及证据记录 | [5_specification/README.md](5_specification/README.md) |

## 当前主线（2026-10-22）

Specification 的方法、正确性证据链和运行时准入机制已足以支撑当前研究。它仍然有明确的不支持范围，但已不是
主要瓶颈。后续只在真实案例暴露具体缺口时补公式、参数、状态、输出对应或参照计算能力，不再以合同数量作为主线进展。

当前有两项工作，顺序如下：

1. **先闭合一个有说服力的真实案例。** 至少完成两条路线之一：证明一个自然产生的 bias 会沿真实训练或推理路径传到后续量；
   或者在主流算子的常用配置上证明真实差异，定位后完成实际修复。首选开发案例是 PyTorch Inductor `avg_pool2d`
   `ceil_mode` 反向：其公开任务、完整前向/VJP 绑定和真实差异已经建立，现在缺的是面向修复的定位、源码改动和同输入的修复前后核对。
2. **把 bias 定位做到能直接指导修复。** 现有的逐个 IR 位点置为精确、再完整重算的方式成本高，对多位点交互、多 kernel 调用和公式错误也不够实用。
   新目标是从受影响的输出或状态沿真实数据流回溯，先确定 `K−G`、`G−f` 还是状态/来源问题，再给出源码区域、机制和可执行的修复候选。最终证据必须来自重新编译后的真实改动，
   而不只是把某个中间节点在模型里设为精确。

旗舰案例的验收要求是：差异来自未植入的发布实现；任务、输入、输出和状态完整对应；定位结果包含真实原因；实际修复消除目标差异且不损害其他承诺结果；
若走传递路线，还要在固定输入下证明局部修正会按预期改变下游量。仅有微基准上的局部差值不算闭环。

2026-10-09 整理前的全部内容（旧研究阶段的结果、脚本、测试与过程文档）都可从标签 `pre-reorg-20261009` 取回：
`git show pre-reorg-20261009:<旧路径>`。保留文件里出现的 `pre-reorg-20261009:<路径>` 即指该标签下的位置。

## 运行

```bash
export HOME=$PWD/.cache XDG_CACHE_HOME=$PWD/.cache TRITON_CACHE_DIR=$PWD/.cache/triton TMPDIR=$PWD/.cache/tmp
/data1/tzh/envs/ka_main/bin/python -m pytest -q 2_tool/tests          # 工具测试（不得改写已跟踪文件）
PYTHONPATH=2_tool/src /data1/tzh/envs/ka_main/bin/python 2_tool/scripts/measure.py --declaration 2_tool/examples/general/bsr_softmax.json --out report.json
```

环境与用法见 [2_tool/README.md](2_tool/README.md)。
