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
| 所有实际调用保留在分母。 | unresolved、many-to-one fusion 和 repeated pattern 不从分母静默删除。 | `results/coverage/cases/full_coordinate_audit.json.gz`; [`denominator.md`](denominator.md) | `SUPPORTED`：1,562/1,562 endpoints 有首轮处置 |
| 全量首轮检查不等于全量长程实验。 | coverage census 与 repair/trajectory funnel 分列。 | [`coverage_table_v1.md`](coverage_table_v1.md); [`current_mainline.md`](current_mainline.md) | `SUPPORTED` |
| Candidate/repair 使用同一完整训练状态。 | weights、inputs、RNG、saved states、optimizer moments、scheduler 等逐项相同。 | per-case certificates; [`bias_protocol.md`](bias_protocol.md) | `SUPPORTED_CASE_LEVEL` |

## 形成与三阶段测量

| 主张 | 判定门槛 | 证据 | 状态 |
|---|---|---|---|
| 所选 residual boundary 上的平均 update 可精确拆成 source asymmetry 与 response rectification 两项。 | 预声明 `e -> -e`，对 event distribution 和 response 作奇偶分解。 | [`effective_antithetic_symmetry.md`](effective_antithetic_symmetry.md); [`method.md`](method.md) | `SUPPORTED_AS_IDENTITY`；不是底层机制穷尽分类 |
| 方向可能在 local、gradient 或 update 阶段形成、增强或消失。 | 同一 matched states 和同一 contrast 保存三层完整向量或 Gram。 | `results/property/bias_formation_v2/`; Phi formation artifacts | `SUPPORTED_CASE_LEVEL`；统一多案例 profile 仍 `PLANNED` |
| Phi 显示 backward 可把 local-centered difference 变成 gradient/update direction。 | confirmation states 上 local centered，gradient/update 有方向。 | Phi v2.1 formation artifact; [`current_mainline.md`](current_mainline.md) | `BOUNDED` |
| 严格正负 residual 仍可能得到不相反的 update。 | 同一 state、weights 和 moments 下 exact `+delta/-delta` replay，response remainder 非零。 | saved-P / SiLU artifacts; [`effective_antithetic_symmetry.md`](effective_antithetic_symmetry.md) | `SUPPORTED_CASE_LEVEL` |

## 统一统计与训练等价性

| 主张 | 判定门槛 | 证据 | 状态 |
|---|---|---|---|
| 误差能量与可复现平均方向必须分开报告。 | 每层同时报告 `E||u||²`、mean effect、normal-update scale。 | [`method.md`](method.md); existing RMS/direction comparisons | `SUPPORTED_AS_METHOD`; v1 empirical 数字保持探索性 |
| Aligned scaling 与剩余 residual direction 应分开。 | 保存 `G_uu`、`G_rr`、`G_ur`，报告加权 aligned effect 和 perpendicular mean。 | [`training_bias_profile_v2.md`](training_bias_profile_v2.md) | `METHOD_VALIDATED_SYNTHETICALLY`；真实案例待 v2 重采，不得预写 residual 普遍有方向 |
| 论文级总体判断要报告效应量与置信区间。 | calibration/confirmation training units 分离；连续 states 不拆开；CI coverage 通过相关-cluster 合成验证。 | [`training_bias_profile_v2.md`](training_bias_profile_v2.md); `results/property/training_bias_profile_v2/synthetic_validation.json` | `METHOD_VALIDATED_SYNTHETICALLY`；empirical population inference 仍 `PLANNED` |
| 多案例、多阶段判断需要控制总体误报。 | 预先提交 confirmatory/discovery families；Holm 为主。 | v2 protocol; [`method.md`](method.md) | `SUPPORTED_AS_METHOD`；新 empirical family 尚未运行 |
| Training equivalence 是相对 protocol 的操作性结论。 | update effect CI 落入预声明工程范围，且声明 consequence endpoint 不失败。 | [`current_mainline.md`](current_mainline.md) | `PLANNED_FRAMEWORK`，不是现成通用 verifier |

## Orbit mean

| 主张 | 判定门槛 | 证据 | 状态 |
|---|---|---|---|
| Orbit mean 可以作为 reduction-family source predictor。 | 冻结合法 schedules；default held out；cross-half estimate；预测方向在 confirmation states 对齐。 | [`persistence_property_protocol.md`](persistence_property_protocol.md); reduction-orbit code/tests | `PLANNED_CANDIDATE` |
| Orbit mean 是通用静态 Oracle。 | 必须覆盖非 reduction mechanisms 且无需 downstream measurement。 | 无 | `NOT_SUPPORTED` |
| Liger 的 BF16 orbit mean 与 local mean 对齐，FP32 accumulator 同时降低二者。 | 事前冻结 schedule roster、方向和干预预测。 | Liger orbit experiment 尚待统一 confirmation | `PLANNED`；已有 chunk evidence 不能替代该预测实验 |

## 短程筛查

| 主张 | 判定门槛 | 证据 | 状态 |
|---|---|---|---|
| 16/32 步 directionality 可用于优先级排序。 | 同一 optimizer、同一 horizon、每行自己的 null；未升级不输出 SAFE。 | corrected 15-row AdamW cohort; [`direct_persistence_screen.md`](direct_persistence_screen.md) | `SUPPORTED_AS_RETROSPECTIVE_TRIAGE` |
| 短程 directionality 比同层 update RMS 更有排序信息。 | 同一 15-row cohort 比较。 | direction AUROC `0.944`; RMS AUROC `0.528` | `SUPPORTED_FOR_RETROSPECTIVE_15_ROWS` |
| 16/32 步足以定义长期 persistent bias。 | 同协议长程确认不能出现反转。 | Qwen cold-start vs warm-state 结果构成反例 | `NOT_SUPPORTED` |

## 长程与训练后果

| 主张 | 判定门槛 | 证据 | 状态 |
|---|---|---|---|
| Direct long-run bias 已在多条记录中出现。 | 4096 same-state direct updates、case-specific null；late windows 若未导出必须单列。 | `results/property/declared_persistent_4096/all_bias_case_audit.json` | `SUPPORTED_CASE_LEVEL`：3 条 direct rows 有 late windows，另 8 条只有 aggregate long-run evidence |
| Feedback-sustained long-run bias 已在多条记录中出现。 | direct/feedback 分开，feedback direction 与 paired loss split 同时存在。 | same machine audit and per-row artifacts | `SUPPORTED_CASE_LEVEL`：32 条 machine labels；只有具备 late windows 的行可进一步称 window-confirmed |
| Loss split 本身证明目标算子 direct bias 持续。 | 必须同时有 direct bias-bearing component。 | v_proj、Mamba、saved-P consequence controls 反驳该推断 | `NOT_SUPPORTED` |
| 4096 步证明 loss 收敛到不同终点。 | 独立 full-training runs 和预声明稳定窗口。 | 当前无 | `NOT_SUPPORTED` |
| 最终参数变化必须拆成 direct 与 feedback。 | 四臂 recurrence 关闭，实际变化等于 direct + feedback；interaction 只作依赖性诊断。 | [`direct_persistence_evidence.md`](direct_persistence_evidence.md) | `SUPPORTED_CASE_LEVEL` |

## 干预与机制

| 主张 | 判定门槛 | 证据 | 状态 |
|---|---|---|---|
| Phi 的方向不只是误差能量下降造成。 | 同一 cold-start AdamW；natural/sham；多个 SR seeds；至少能量近似匹配的 repeats 方向消失。 | [`phi_adamw_source_intervention.md`](phi_adamw_source_intervention.md) | `SUPPORTED_CASE_LEVEL` |
| AdamW 会改变 direction verdict，但不是统一误差来源。 | 同一 gradient contrast 比较 SGD、captured AdamW、moment ablation。 | [`direct_persistence_optimizer.md`](direct_persistence_optimizer.md) | `SUPPORTED_CASE_LEVEL` |
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
