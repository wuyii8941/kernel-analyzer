# 1 完整实验

每个实验只保留最新、最全的版本：协议、最终结果文件与状态文档。成绩只按三种状态写（成立 / 不成立 / 无法判断；
冻结验收沿用其协议的「建立 / 未建立 / 无法判断」）。被取代的中间版本与早期研究阶段的原始数据可从标签 `pre-reorg-20261009` 取回；
保留文档里的 `pre-reorg-20261009:<路径>` 指该标签下的位置。

| 实验 | 工具 | 状态 | 主要结果 | 目录 |
| --- | --- | --- | --- | --- |
| 组合研究：Triton 操作组合中 bias 的成因与预测 | 4.0（仿真器组合规则扩展；预测层 `reference_eval/bias_predict.py`，协议 v1 冻结于 bd84c02、v2 冻结于 0f1e465） | 原始协议写于设备扫描之前；预测协议在留出程序生成之前冻结，预测在测量之前入库 | 预测原型已有证据、理论化标准尚未建立（外部审阅 c415b62）：两个留出集（各 60 个随机程序）成立 51 / 44、不成立 2 / 2（f32 近似函数链的亚 ulp 偏差）、其余按失效条件放弃；有可测 bias 的 45 个程序中说中 33 个（零预测全部失败）；真实修改（TF32 激活就近）的效果在 6 个 Qwen 层上事先说中、设备证实。原语表 986 个实例；原语 × 23 种形式 17,687 个单元：严格判定下逐位相同 12,074、部分 1,591（未定义值通道）、不一致元素 0；张量核 dot 实验模型（全截断）；改正后的库语料 231 个 kernel，213 个逐位相同，暴露并修复三处两原语相互作用的仿真缺口；B024（Triton 浮点 `%`，上游已知） | [composition_study/](composition_study/README.md) |
| 工具案例：Triton 教程 flash attention | 4.0（`reference_eval/audit.py` 审查入口） | 用一次调用审查真实算子；缺陷就地修复、先写失败测试 | 教程前向逐位复现、未发现显著偏差；植入的 tf32 截断、语义常数、低精度累加三类缺陷都被找到并定位到源码行，事前预测与实测一致，附改法与模型内反事实；fp16 累加器与强停滞如实报「无法判断」。修复工具缺陷六处：对 dot 输出的归约（MMA 布局）、近似函数下溢、归因候选漏掉张量核 dot、dot 提示、fma 链状态传播、审查入口与报告（显著性、格式分辨率、自检）。第二轮按开发者视角的审阅重写报告：人话摘要、常用容差检查分三类（三种常用检查都分辨不出 0.1%～0.3% 的偏差）、改法代价、全部位点都改的效果、含边界块的抽样与各 program 的偏差范围；修复「归因与测量不是同一组 program」；l_bf16 全部 32 个 program 的审查为 +0.29%（抽 6 个为 +0.23%）。阶段状态与下一步见 [2_tool/docs/audit_status.md](../2_tool/docs/audit_status.md) | [flash_attention_case/](flash_attention_case/README.md) |
| 第二轮：常见配置下的数值误差、容差盲点与 bias | 4.0（测量修复 5fd8754，增量观察层 aeecd33） | 协议冻结于确认数据之前（偏离 D-1–D-11 入协议）；开发证据 | A 纯 bf16 AdamW（torchtune 默认）：容差通过，参数增量 91–94% 未写入；600 步验证损失差 −0.113（非零，成立），bf16 EMA +0.095；B 编译 RAdam：单步偏差非零（成立），200 步训练作用不成立（等价）；C Triton TF32：单层收缩约 3.5e-4（成立），单步下游与训练作用无法判断；机器可读台账 `results/ledger.json` | [tolerance_study/](tolerance_study/README.md) |
| DSL v2 开发评价（增量 1–15，外部审计修复版） | 4.0（未冻结） | 开发证据，不是盲测成绩 | 官方测试广泛捕获 12,766 次启动：无条件完整（含集合目标）96.82%，另 1.23% 只在未证明前提（z3 不能证明结合的 scan）下完整，求值器错误 0；TTIR 与 NVIDIA / AMD TTGIR 两层对照不相交 0；W1 430/430；设备验证 0 | [dsl_v2/](dsl_v2/README.md) |
| 第一批验收（来源 / 精度 / 安装 / 查询） | 4.0（实现提交 d26d5f6） | 协议冻结于运行之前；执行方编写的保留集，不替代外部保留集 | 24 项：成立 24、不成立 0、无法判断 0；来源可靠性违反 0，保守 3（事先登记）；回归 28/28 相同；记有一处执行偏差（竞争程序误上 GPU，结果不入记录） | [batch1_acceptance/](batch1_acceptance/README.md) |
| 结构验收集 v1.1-rc1，运行 r20261008T2030 | 3.1（general-v3.1，冻结） | **已测量，未计分**（包为 ISSUER_RESEAL_REQUIRED，等原出题方直接交付原始答案） | 程序级「全部必要输出已建立」27/33；5 个 T1 程序因 Welford 写法未建立 | [structure_acceptance_v1_1/](structure_acceptance_v1_1/README.md) |
| 通用能力轮 | 3.1（冻结） | 计划项状态表（已执行可裁决 / 部分 / 未运行） | 统一入口、Welford、SumK / DotK、五类分布校准、失败分类、P05 证据；第 5 项未见结构验收由上一行承接 | [general_round_v3_1/](general_round_v3_1/README.md) |
| 收束包 v2.0 轮 | 2.3 | 计划项状态表 | 保证链（校验器 0 问题、12/12 降级路径有触发）、33 行复审、15 个家族四组四栏；组合覆盖 9/12 | [closure_round_v2/](closure_round_v2/README.md) |
| 本质错误一轮（第一阶段、2a、2b） | 2.x | 最终记录（2b 第二版） | 15 个第一档家族的 F / E / P / 模式 B；新发现 B020–B023（见 4_bugs_cases） | [essential_bugs_round/](essential_bugs_round/README.md) |
| 盲测 v1、v2 | 2.x | 已揭盲，只作回归集 | 75 个程序全部建立完整参照；独立复算 16,908,160 个元素 0 违反；已知阳性 16/17、阴性对照 13/13 | [blind_tests/](blind_tests/blind_records/README.md) |
| 评价计划轮（2026-10-06）的其他实验 | 2.x | 成绩核账（三种状态） | 外部受控集合、陌生组合子集、逐层表、重要性标定、定向搜索、直接差分基线、benchmark、普查等，逐项结果与当时的依据路径 | [ledger_20261006.md](ledger_20261006.md) |

## 实验输入

- 独立任务规格已移到 [5_specification](../5_specification/README.md)（2026-10-15，换成 v2.4）。模式 B 与本质错误一轮、收束轮用的是当时的
  第一阶段 v0.4 与第二阶段包，可用 `git show ff5e449:1_experiments/specs/<路径>` 取回。
- DSL v2 的设计依据：[dsl_v2/design_rc3/](dsl_v2/design_rc3/)（rc3 正式版副本，SHA-256 8050903f…6faf）。
- 冻结验收的运行记录压缩为一个文件
  [structure_acceptance_v1_1/structure_v1_1_rc1_r20261008T2030.tar.gz](structure_acceptance_v1_1/structure_v1_1_rc1_r20261008T2030.tar.gz)
  （内容与入库时逐字节相同）；计分或回归前解压到 `.cache/acceptance/`：
  `tar -xzf 1_experiments/structure_acceptance_v1_1/structure_v1_1_rc1_r20261008T2030.tar.gz -C .cache/acceptance`。

## 冻结与待交

- general-v3.1（标签，提交 d6e177e）冻结工具与它在结构验收 v1.1 上的测量不再重算；计分只等原出题方交付两份原始答案、各自的
  SHA-256、包版本、计分协议与承诺记录（审计材料账 M10）。
- 盲测 v1、v2 已揭盲，不再作盲测；DSL v2 的新盲测、独立语料 B、独立生成器、δ 与设备报告都属于外部待交材料（见 3_audits）。
