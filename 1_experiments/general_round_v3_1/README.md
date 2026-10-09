# 通用能力轮：计划项状态表（2026-10-08；冻结版本 general-v3.1，替代 general-v3.0）

general-v3.0 的 SumK 界经独立审计证明不严格，3.1 修正后重新冻结（第 5 项的验收集尚未交付）。3.0 下的工具结果已全部用 3.1 重跑。
冻结哈希见 `2_tool/docs/changelog.md` 的 3.1 条目与标签 `general-v3.1`。审计见 `3_audits/audit_two_rounds_20261008.md`。

| 项 | 状态 | 结果 / 缺口 | 依据 |
|---|---|---|---|
| 1 统一入口与默认规则 | 已执行可裁决 | 四项声明、缺项逐条报出；默认规则自动展开（固定均值 R1 / R5、对齐 R2 / R3，类内 Holm）；未接入过的 torch.sparse Triton kernel 演示（无参照、无可疑节点）；调用级与 kernel 级完整率分开（上游非 Triton 值必须是声明输入的复制才算调用级完整） | `2_tool/src/kernel_analyzer/measure.py`；`1_experiments/general_round_v3_1/results/demo` |
| 2 多值归约（Welford） | 已执行可裁决 | 以前被拒绝的 bf16 方差现在完整建立；未加保护的比值、零权重行、舍入核对模式按前提处理（3.1 修正） | `2_tool/tests/test_welford_reduction.py`（8 项） |
| 2 嵌套控制 / 别名 / 跨启动 / 原子 / 位级 / 外部指令 | 部分方法可执行 | 注册表 502 条（支持 396、声明前提 2、未建立 2、拒绝 102），枚举测试通过，通用路径无 kernel 名特判；但 396 条支持条目只引用 15 条不同的测试（按类别共享），不是逐条专属测试 | `1_experiments/general_round_v3_1/results/rule_registry.json`；`2_tool/tests/test_rule_registry.py` |
| 2 scan 通用折叠 | 部分方法可执行 | 依赖未检查的结合性前提，交审阅方 | `1_experiments/general_round_v3_1/tables/rule_registry_report.md` |
| 3 精确累加 / 同变量 / 解析缓存 | 已执行可裁决 | SumK / DotK（3.1：p_n 最后加入，审计探针 0 违反）；宽度对 attention、packing、G6、新算子、数值作用流缩小 11–42 倍，normalization 约 3 倍，其余 6 个家族不变；达到 1/8 ulp 的比例在 3.1 与 γ 下相同 | `1_experiments/general_round_v3_1/tables/quality_table.md`；`2_tool/tests/test_sumk_rigorous.py` |
| 3 自适应工作精度 | 未运行 | 只报告触发比例：activations 6.1% 元素（1 + erf / tanh 相消）未达 1/8 ulp | 同上 |
| 3 按 program 批处理 | 未运行 | 逐 program 求值未改 | — |
| 4 两类分开、五类分布校准 | 已执行可裁决 | 零效应误报 0.026–0.055；效应 0.02 时 R1 约 0.59–0.62、对齐类对状态相关约 0.63；稀有尾部 R5 约 60% 无法判断（种子固定后重跑） | `1_experiments/general_round_v3_1/tables/calibration_table.md` |
| 4 有界路线 | 部分方法可执行 | 只在校准中使用（Hoeffding，协议写的是经验 Bernstein）；统一入口只记录声明的误差预算、不应用；M 的自动推出未实现；Hoeffding 在稀有尾部零功效 | `measure.bounded_mean_test`；协议偏离 5 |
| 5 未见结构验收 | 未运行 | 工具已冻结（3.1）；运行框架用自制冒烟包验证；等待审阅方交付封存的验收集 | `2_tool/scripts/general/acceptance_run.py` |
| 6 失败分类 | 已执行可裁决 | 绑定 101、仅语义结论受阻 85、特殊值不计 11、规格外不计 17、包围太宽 6、统计不足 4（3.1 修正了 5 条误分类与未分类原因的处理） | `1_experiments/general_round_v3_1/tables/failure_table.md` |
| 7 P05 证据入库 | 已执行可裁决 | 18 个文件，盲测校验和通过 | `1_experiments/general_round_v3_1/tables/evidence_P05.md` |
| 回归（任务书第 8 节） | 已执行可裁决 | 全部测试 1850 项通过（日志 `1_experiments/general_round_v3_1/results/full_tests_v3_1.log`；15 个模块在 ka_main 缺 transformers 无法收集）；回归集判定不变（3 例 e_sem 相对 RMS 在 1e-16 变化） | `1_experiments/general_round_v3_1/results/regression_v3_1` |
| 符号对照 | 已执行可裁决 | 待理论包原文核对 | `1_experiments/general_round_v3_1/tables/symbol_table.md` |

发现：check.run 中未定谓词（select 与 scf.if）取两分支并集，κ = conditional 只来自 pin_loads；收束轮组合覆盖缺的「κ 有条件」一条因此在
check.run 中结构上不会出现（交审阅方决定链表如何改写）。
