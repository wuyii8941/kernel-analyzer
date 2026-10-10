# 1 完整实验

每个实验只保留最新、最全的版本：协议、最终结果文件与状态文档。成绩只按三种状态写（成立 / 不成立 / 无法判断；
冻结验收沿用其协议的「建立 / 未建立 / 无法判断」）。被取代的中间版本与早期研究阶段的原始数据可从标签 `pre-reorg-20261009` 取回；
保留文档里的 `pre-reorg-20261009:<路径>` 指该标签下的位置。

| 实验 | 工具 | 状态 | 主要结果 | 目录 |
| --- | --- | --- | --- | --- |
| DSL v2 开发评价（增量 1–15，外部审计修复版） | 4.0（未冻结） | 开发证据，不是盲测成绩 | 官方测试广泛捕获 12,766 次启动：无条件完整（含集合目标）96.82%，另 1.23% 只在未证明前提（z3 不能证明结合的 scan）下完整，求值器错误 0；TTIR 与 NVIDIA / AMD TTGIR 两层对照不相交 0；W1 430/430；设备验证 0 | [dsl_v2/](dsl_v2/README.md) |
| 第一批验收（来源 / 精度 / 安装 / 查询） | 4.0（实现提交 d26d5f6） | 协议冻结于运行之前；执行方编写的保留集，不替代外部保留集 | 24 项：成立 24、不成立 0、无法判断 0；来源可靠性违反 0，保守 3（事先登记）；回归 28/28 相同；记有一处执行偏差（竞争程序误上 GPU，结果不入记录） | [batch1_acceptance/](batch1_acceptance/README.md) |
| 结构验收集 v1.1-rc1，运行 r20261008T2030 | 3.1（general-v3.1，冻结） | **已测量，未计分**（包为 ISSUER_RESEAL_REQUIRED，等原出题方直接交付原始答案） | 程序级「全部必要输出已建立」27/33；5 个 T1 程序因 Welford 写法未建立 | [structure_acceptance_v1_1/](structure_acceptance_v1_1/README.md) |
| 通用能力轮 | 3.1（冻结） | 计划项状态表（已执行可裁决 / 部分 / 未运行） | 统一入口、Welford、SumK / DotK、五类分布校准、失败分类、P05 证据；第 5 项未见结构验收由上一行承接 | [general_round_v3_1/](general_round_v3_1/README.md) |
| 收束包 v2.0 轮 | 2.3 | 计划项状态表 | 保证链（校验器 0 问题、12/12 降级路径有触发）、33 行复审、15 个家族四组四栏；组合覆盖 9/12 | [closure_round_v2/](closure_round_v2/README.md) |
| 本质错误一轮（第一阶段、2a、2b） | 2.x | 最终记录（2b 第二版） | 15 个第一档家族的 F / E / P / 模式 B；新发现 B020–B023（见 4_bugs_cases） | [essential_bugs_round/](essential_bugs_round/README.md) |
| 盲测 v1、v2 | 2.x | 已揭盲，只作回归集 | 75 个程序全部建立完整参照；独立复算 16,908,160 个元素 0 违反；已知阳性 16/17、阴性对照 13/13 | [blind_tests/](blind_tests/blind_records/README.md) |
| 评价计划轮（2026-10-06）的其他实验 | 2.x | 成绩核账（三种状态） | 外部受控集合、陌生组合子集、逐层表、重要性标定、定向搜索、直接差分基线、benchmark、普查等，逐项结果与当时的依据路径 | [ledger_20261006.md](ledger_20261006.md) |

## 实验输入

- [specs/](specs/)：审阅方交付的独立任务规格（第一阶段 v0.4、第二阶段包 v0.2），模式 B 与本质错误一轮、收束轮使用。
- DSL v2 的设计依据：[dsl_v2/design_rc3/](dsl_v2/design_rc3/)（rc3 正式版副本，SHA-256 8050903f…6faf）。
- 冻结验收的运行记录压缩为一个文件
  [structure_acceptance_v1_1/structure_v1_1_rc1_r20261008T2030.tar.gz](structure_acceptance_v1_1/structure_v1_1_rc1_r20261008T2030.tar.gz)
  （内容与入库时逐字节相同）；计分或回归前解压到 `.cache/acceptance/`：
  `tar -xzf 1_experiments/structure_acceptance_v1_1/structure_v1_1_rc1_r20261008T2030.tar.gz -C .cache/acceptance`。

## 冻结与待交

- general-v3.1（标签，提交 d6e177e）冻结工具与它在结构验收 v1.1 上的测量不再重算；计分只等原出题方交付两份原始答案、各自的
  SHA-256、包版本、计分协议与承诺记录（审计材料账 M10）。
- 盲测 v1、v2 已揭盲，不再作盲测；DSL v2 的新盲测、独立语料 B、独立生成器、δ 与设备报告都属于外部待交材料（见 3_audits）。
