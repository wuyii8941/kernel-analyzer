# 证据账本

本页按论文主张组织，不按实验轮次组织。术语和计数以
[`current_mainline.md`](current_mainline.md) 为准。

状态含义：

- `SUPPORTED`：当前证据足以支持带范围的表述；
- `BOUNDED`：有案例级证据，但不能推广；
- `PLANNED`：方法已定义，统一实验尚未完成；
- `UNRESOLVED`：现有 artifact 或统计能力不足；
- `NOT_SUPPORTED`：当前证据不支持该主张。

## 方法与覆盖

| 主张 | 判定门槛 | 证据 | 状态 |
|---|---|---|---|
| 测试单元是 concrete forward 加 actual backward。 | saved tensors、cotangent、真实 VJP、参数可达和 repair boundary 全部绑定。 | `results/coverage/cases/directional_candidate_math_registry.json.gz`; [`method.md`](method.md) | `SUPPORTED`（冻结四模型范围） |
| 所有实际调用保留在分母。 | 无法判断、多处合并和重复实现模式不从分母静默删除。 | `results/coverage/cases/full_coordinate_audit.json.gz`; [`denominator.md`](denominator.md) | `SUPPORTED`：1,562/1,562 个具体输出位置有首轮处置 |
| 全量首轮检查不等于全量长程实验。 | coverage census 与 repair/trajectory funnel 分列。 | [`coverage_table_v1.md`](coverage_table_v1.md); [`current_mainline.md`](current_mainline.md) | `SUPPORTED` |
| Candidate/repair 使用同一完整训练状态。 | weights、inputs、RNG、saved states、optimizer moments、scheduler 等逐项相同。 | per-case certificates; [`bias_protocol.md`](bias_protocol.md) | `SUPPORTED_CASE_LEVEL` |

## 形成与三阶段测量

| 主张 | 判定门槛 | 证据 | 状态 |
|---|---|---|---|
| 所选 residual boundary 上的平均 update 可精确拆成 source asymmetry 与 response rectification 两项。 | 预声明 `e -> -e`，对 event distribution 和 response 作奇偶分解。 | [`effective_antithetic_symmetry.md`](effective_antithetic_symmetry.md); [`method.md`](method.md) | `SUPPORTED_AS_IDENTITY`；不是底层机制穷尽分类 |
| 方向或稳定缩放可能在 local、gradient 或 update 阶段形成、增强或消失。 | 同一 matched inputs 和 contrast 保存三层完整向量或 Gram，并用同一三分支规则。 | `results/property/training_bias_profile_v2/five_case_summary.json`; prospective batch summaries; [`five_case_training_bias_profile_v2.md`](five_case_training_bias_profile_v2.md); [`prospective_training_bias_profiles.md`](prospective_training_bias_profiles.md) | `SUPPORTED_FOR_FIVE_DEVELOPMENT_CASES_AND_BOUNDED_NEW_BATCHES`；不是 prevalence 或通用泛化结果 |
| 新 DeepSeek normalization 与 attention-projection backward 在 cold-start AdamW 下反复缩小正常 update。 | 16 个 calibration + 16 个 confirmation states；三分支统一测量；各自冻结组内 Holm 校正；完整 repair/sham 绑定。 | prospective batch 1/2 raw and summary JSON | `SUPPORTED_FOR_TWO_BOUNDED_CASES`：分别为 `−13.68%` 与 `−10.69%`；不能外推到所有 DeepSeek 算子或 warm moments |
| 上述两个新 update effect 会造成配对 loss 分叉。 | 四臂 live replay；形成证据独立存在；loss split 是事前停止条件。 | prospective batch 1/2 consequence JSON | `SUPPORTED_AS_TRAJECTORY_NON_IDENTITY`：两项均在 step 1 停止；不是 4096-step persistence 或最终质量结论 |
| 统一方法能接入 Gemma 4 的不同模型与实现位置。 | 使用历史冻结的两个位置、相同 16+16 测量和原记录一致的运行环境；不因结果替换位置。 | `results/property/generalization_benchmark_v1/gemma4_method_bridge_result.json` | `SUPPORTED_AS_METHOD_BRIDGE_WITH_NEGATIVE_RESULTS`：一个位置 candidate/repair 完全相同，另一个所有区间跨零；不是新选择的前瞻发现集合 |
| Phi 显示 backward 可放大剩余共同方向，而 AdamW 可把它变成相对正常 update 的缩小。 | 三阶段使用同一 32-window protocol；gradient residual 与 update aligned branch 分别通过解释/主要 Holm 组。 | same v2 five-case summary | `BOUNDED`：一个 checkpoint、cold-start AdamW |
| 严格正负 residual 仍可能得到不相反的 update。 | 同一 state、weights 和 moments 下 exact `+delta/-delta` replay，response remainder 非零。 | saved-P / SiLU artifacts; [`effective_antithetic_symmetry.md`](effective_antithetic_symmetry.md) | `SUPPORTED_CASE_LEVEL` |

## 统一统计与训练等价性

| 主张 | 判定门槛 | 证据 | 状态 |
|---|---|---|---|
| 误差能量与可复现平均方向必须分开报告。 | 每层同时报告 `E||u||²`、mean effect、normal-update scale。 | [`method.md`](method.md); existing RMS/direction comparisons | `SUPPORTED_AS_METHOD`; v1 empirical 数字保持探索性 |
| Aligned scaling 与剩余 residual direction 应分开。 | 保存 `G_uu`、`G_rr`、`G_ur`，对所有案例同时运行固定方向、aligned scaling 和 residual direction。 | [`training_bias_profile_v2.md`](training_bias_profile_v2.md); five-case raw/summary JSON | `SUPPORTED_FOR_FIVE_DEVELOPMENT_CASES`：Liger 为 residual direction；Phi、v-proj、Mamba 为 update scaling；不得预写普遍发生率 |
| 论文级判断要报告效应量与区间。 | calibration/confirmation 分离；区间范围必须与输入抽样方式一致。 | [`training_bias_profile_v2.md`](training_bias_profile_v2.md); `single_state_unit_validation.json`; five-case summary | `SUPPORTED_FOR_FROZEN_WINDOW_SUITE`；不是独立-run 或随机总体区间 |
| 多案例、多阶段判断需要控制总体误报。 | 结果揭示前冻结检验组；update 与解释阶段分开作 Holm；无法判断项不从分母移除。 | empirical protocol/amendments; five-case summary; prospective batch 1/2 protocols and summaries | `COMPLETED`：开发组 15/30 项；新 batch 1 为 12/24 项且两项 abstain 以 p=1 保留；batch 2 为 3/6 项 |
| 固定输入集合上的 update 等价是相对声明设置的操作性结论。 | 覆盖全部坐标的 update RMS 比例低于范围，三个方向区间也全部在各自范围内；只有保存完整向量时才允许签逐位零差异。 | `results/property/training_equivalence_v2/`; `results/property/generalization_benchmark_v1/equivalence_v2.json`; [`current_mainline.md`](current_mainline.md) | `METHOD_COMPLETE_FOR_FIXED_SUITE_UPDATE`：完整向量反例验证通过；大向量使用三个冻结随机摘要，不等于随机训练状态总体或完整训练质量等价。`1%` 范围是结果揭示后的方法修正，不冒充事前冻结 |

## Orbit mean

| 主张 | 判定门槛 | 证据 | 状态 |
|---|---|---|---|
| 多种等价求和顺序的平均差异可作为 reduction 类来源预测。 | 先在前 16 个输入确定方向，再在后 16 个输入检查；两个实现都使用 FP32，只改变 Liger `dW` 的分块加法顺序。 | `results/property/liger_fp32_chunk_order_v1/`; [`persistence_property_protocol.md`](persistence_property_protocol.md) | `SUPPORTED_FOR_ONE_BOUNDED_CASE`：后 16 个输入中 15 个沿预测方向；gradient 与 AdamW update 的方向结果通过 Holm 校正；效应很小且没有长程训练后果结论 |
| Orbit mean 是通用静态 Oracle。 | 必须覆盖非 reduction mechanisms 且无需 downstream measurement。 | 无 | `NOT_SUPPORTED` |
| Liger 的 BF16 求和顺序平均差异与 local mean 对齐，换 FP32 accumulator 后二者同步下降。 | 事前冻结求和顺序清单、方向和干预预测。 | 当前新增的是“FP32 对 FP32、只换顺序”的实验，不是这项 BF16/FP32 联合干预 | `NOT_TESTED_AS_STATED`；不能用新的纯 FP32 结果替代这项更强主张 |

## 短程筛查

| 主张 | 判定门槛 | 证据 | 状态 |
|---|---|---|---|
| 16/32 步方向性可用于安排后续实验优先级。 | 同一 optimizer、同一测量长度、每行自己的随机抵消范围；未升级不输出 SAFE。 | 校正后的 15 行 AdamW 案例集合；[`direct_persistence_screen.md`](direct_persistence_screen.md) | `SUPPORTED_AS_RETROSPECTIVE_TRIAGE` |
| 短程方向性比同层 update RMS 更有排序信息。 | 同一 15 行案例集合比较。 | direction AUROC `0.944`; RMS AUROC `0.528` | `SUPPORTED_FOR_RETROSPECTIVE_15_ROWS` |
| 16/32 步足以定义长期 persistent bias。 | 同协议长程确认不能出现反转。 | Qwen cold-start vs warm-state 结果构成反例 | `NOT_SUPPORTED` |

## 长程与训练后果

| 主张 | 判定门槛 | 证据 | 状态 |
|---|---|---|---|
| Direct long-run bias 已在多条记录中出现。 | 4096 same-state direct updates、case-specific null；late windows 若未导出必须单列。 | `results/property/declared_persistent_4096/all_bias_case_audit.json` | `SUPPORTED_CASE_LEVEL`：3 条 direct rows 有 late windows，另 8 条只有 aggregate long-run evidence |
| Feedback-sustained long-run bias 已在多条记录中出现。 | direct/feedback 分开，feedback direction 与 paired loss split 同时存在。 | same machine audit and per-row artifacts | `SUPPORTED_CASE_LEVEL`：32 条 machine labels；只有具备 late windows 的行可进一步称 window-confirmed |
| Loss split 本身证明目标算子 direct bias 持续。 | 必须同时有 direct bias-bearing component。 | v_proj、Mamba、saved-P consequence controls 反驳该推断 | `NOT_SUPPORTED` |
| 4096 步证明 loss 收敛到不同终点。 | 独立 full-training runs 和预声明稳定窗口。 | 当前无 | `NOT_SUPPORTED` |
| 最终参数变化必须拆成 direct 与 feedback。 | 四臂 recurrence 关闭，实际变化等于 direct + feedback；interaction 只作依赖性诊断。 | [`direct_persistence_evidence.md`](direct_persistence_evidence.md) | `SUPPORTED_CASE_LEVEL` |
| 已测参数偏移不是任意随机的 loss 方向。 | 在保存的 repair 参数附近，将测得方向与同长度随机方向作相同尺度的 loss 检查。 | `results/property/loss_direction_stress_v1/` | `SUPPORTED_FOR_TWO_LOCAL_SENSITIVITY_TESTS`：Liger 原尺度 loss 变化为随机方向中位数的约 `1039` 倍；Phi 为约 `9.55` 倍；不是未来训练直线外推 |
| 三个训练位置的配对轨迹在不同输入流中稳定分开。 | 每项四组互不重叠的 32 步输入流；分别检查 direct、feedback、actual 和 loss。 | `results/property/independent_consequence_v1/summary.json` | `SUPPORTED_AS_TRAJECTORY_NON_IDENTITY`：三项 actual/feedback 均为 `4/4`，direct 均为 `0/4`；稳定窗口平均 loss gap 均跨零，不支持稳定质量方向 |

## 干预与机制

| 主张 | 判定门槛 | 证据 | 状态 |
|---|---|---|---|
| Phi 的方向不只是误差能量下降造成。 | 同一 cold-start AdamW；natural/sham；多个 SR seeds；至少能量近似匹配的 repeats 方向消失。 | [`phi_adamw_source_intervention.md`](phi_adamw_source_intervention.md) | `SUPPORTED_CASE_LEVEL` |
| AdamW 会改变 direction verdict，但不是统一误差来源。 | 同一 gradient contrast 比较 SGD、captured AdamW、moment ablation。 | [`direct_persistence_optimizer.md`](direct_persistence_optimizer.md) | `SUPPORTED_CASE_LEVEL` |
| DeepSeek 的大缩小效应依赖 AdamW 历史状态。 | 两个正例与一个 Phi 负例统一比较 cold、warm 8、warm 32、warm 后重置 moments 和 SGD；45 项整体 Holm 校正。 | `results/property/optimizer_condition_benchmark_v1/summary.json` | `SUPPORTED_FOR_TWO_DEEPSEEK_CASES`：cold 为 `−10.69%/−13.68%`，warm 后不足 `1%`，重置 moments 后恢复为 `−9.03%/−15.13%`；Phi 五种设置均未确认 |
| 冻结方法可以在未参与方法开发的训练位置发现 update 缩小；修正后的总体 update 检查可以阻止未知方向的大差异被误签为等价。 | 16 项预先冻结；15 项有效测量；48 项统一 Holm 校正；结果揭示后增加覆盖全部坐标的 update RMS 兜底并保留修正记录。 | `results/property/generalization_benchmark_v1/summary.json`; `results/property/generalization_benchmark_v1/equivalence_v2.json` | `SUPPORTED_FOR_THIS_FROZEN_BENCHMARK`：9 项确认方向或缩放效应；3 项通过固定输入 update 等价，其中 2 项的三个冻结摘要均为零；3 项总体 update 比例超范围；1 项无法判断。不是跨所有模型与 checkpoint 的准确率 |
| Liger 的统一 orbit predictor 已闭合。 | 必须完成本页前述 BF16/FP32 confirmation prediction。 | 当前只有 chunk/accumulator 机制证据 | `UNRESOLVED` |

## 当前计数

- 23 个唯一主矩阵 case IDs；
- 301 条逐行长程审计记录；
- 43 条 `long-run bias + paired loss split` machine labels：3 条 late-window direct、
  8 条 aggregate direct without exported late windows、32 条 feedback-sustained；
- 4 条目前具有显式 late rolling-window confirmation；
- 5 条只有 paired loss split、direct bias 未通过 long-run gate；
- 105 条更宽的 outcome-relevant records 中包含 persistence 尚未测量的历史候选，
  不能全部叫 final persistent cases；
- 45 条 unresolved/abstain 继续保留在分母。

## 当前最安全的论文表述

> 在具体 LLM training implementation 上，tensor error magnitude 与真实 optimizer
> update 中的可复现平均方向是不同信息。通过 matched candidate/repair replay，
> 可以先用 source/response 分解解释方向怎样形成，再用 local、gradient、update
> 三阶段测量定位它在哪里出现，最后用 paired long-run training 区分 direct effect、
> feedback 和 loss consequence。

不能升级为“已建立通用全算子安全 Oracle”或“所有正交 residual 都有方向”。
