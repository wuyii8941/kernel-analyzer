# 文档与版本入口

## 当前研究口径

只在以下文档维护当前研究定义，避免多份“唯一主线”互相覆盖：

1. [当前主线](current_mainline.md)：**框架与机制、真实 Triton、统计理论、强训练后果**；
   实现与参考 → 成因条件 → 三阶段测量与统计判断 → 修改预测与训练验证。
2. [实际写入分析 v2](training_numerical_analysis_v2.md)：真实 AdamW 重放、原案例恢复、新家族与训练确认；[v1](training_numerical_analysis_v1.md)保留历史写入模拟的限制。
3. [实验方法](method.md)：比较对象、推导条件、三阶段测量和训练后果。
4. [案例与证据地图](case_evidence_map.md)：每类证据的来源、配置和缺口。
5. [主张账本](claims.md)：什么已经支持，什么不能推出。
6. [研究出发点与贡献边界](novelty_positioning.md)：2026-09-07 公开文献核查及具体创新主张。
7. [完整新计划验收范围](new_mainline_execution_status.md)与[自动采集入口](numerical_coverage_execution.md)：区分已列出、可执行、已测量和仍未完成。
8. [随机状态超界比例检验](population_exceedance_inference.md)：精确有限样本端点，及其与总体平均 Q 的严格区分。
9. [随机状态总体推断合同](population_inference_contract.md)：有限样本平均 Q 为什么必须依赖
   事前尾部条件，以及条件不足时的自动停止规则。
10. [算子族与重点问题组证据深度](../results/property/numerical_coverage_v1/operator_problem_group_depth_v1.md)：
    把目录覆盖、真实 Triton、update、数学来源、修改和训练结果分开。

[讲稿](talk_beyond_tolerance.md)由用户单独维护，本轮未改。它不是实验状态数据库；
其中旧数字或标签以对应实验协议与上述来源说明理解，不因本轮整理静默改写。

## 数学推导：保留，不因为版本旧而删除

- [误差与正负响应的精确分解](effective_antithetic_symmetry.md)
- [矩阵乘与输出舍入分别修改](source_aligned_repair.md)
- [attention 保存状态与 backward](l23_qproj_tile.md)
- [Phi 同 AdamW 的随机舍入干预](phi_adamw_source_intervention.md)
- [求和顺序的来源预测](persistence_property_protocol.md)

推导给出成立条件；公式存在不自动证明某个案例非零。各页保留原实验数字和范围，
后续现状通过案例地图连接。

## 测量与实验记录：按版本读取

| 记录 | 用途与当前解释 |
|---|---|
| [三类实现补测](three_mechanism_profiles.md)、[统一轮次](unified_measurement_round.md)、[扩展轮次](extended_unified_profiles.md) | 各轮测量与干预，不覆盖后续同名案例 |
| [Profile v2 方法](training_bias_profile_v2.md)、[五例结果](five_case_training_bias_profile_v2.md) | 每个固定输入从零 moments 单独测量；不是 32 步自然连续训练 |
| [新案例批次](prospective_training_bias_profiles.md) | 依各批冻结规则解释，早期未决不代表以后一直未运行 |
| [16 项原始汇总](../results/property/generalization_benchmark_v1/summary.json) | 冻结验证集；不能自动计为成因与 loss 已闭合的案例数 |
| [等价 v2](../results/property/generalization_benchmark_v1/equivalence_v2.json) | 原数据上的方法修订；完整 Gram 与随机摘要保证不同 |
| [optimizer 状态对照](../results/property/optimizer_condition_benchmark_v1/summary.json) | warm/reset 区分参数与 moments 条件 |
| [optimizer-update 新家族](optimizer_update_family_audit.md) | TorchAO 8-bit AdamW 的固定 gradient 机制、实际参数写入与八条数据流训练确认 |
| [fused causal attention 新家族](fused_attention_family_audit.md) | Qwen 第一层 Flash-SDPA 与 math attention 的固定集合三阶段测量 |
| [fused RoPE / position scaling](fused_rotary_position_scaling_audit.md) | Ministral 真实 Triton 相同输入比较与 optimizer-state 条件核验 |
| [四输入流训练后果](../results/property/independent_consequence_v1/summary.json) | 同一 checkpoint、指定参数，非独立全参数预训练 |
| [最新 Liger 全参数训练](liger_single_boundary_collapse_experiment.md) | 2048、4096、10000 步设置及 loss 符号反转 |

## 历史记录：保留证据，不继续定义成功标准

- [旧案例登记](../case.md)、[旧 Flash-style 登记与推导](../cases_flash_style.md)、
  [Qwen3-VL 原始 SiLU 分析](../round2.md)
- [旧分类协议](bias_protocol.md)、[旧系统入口](system.md)
- [32 步归因](direct_persistence_evidence.md)、[optimizer 对照](direct_persistence_optimizer.md)、
  [短筛](direct_persistence_screen.md)、[held-out](direct_persistence_heldout.md)、
  [当轮限制](direct_persistence_limitations.md)
- [历史 4096 逐行机器审计](../results/property/declared_persistent_4096/all_bias_case_audit.json)、
  [可读表](all_bias_long_horizon_audit.md)、[Liger/SiLU 历史复核](liger_silu_long_horizon_recheck.md)
- [无法重放项](unresolved_long_replays.md)、[计数变更](gate_history.md)

这些路径保留是为了引用和复现；后续修订不回写成事前决定。
历史审计中的 301 行不是整个仓库至今所有实验的全集，旧“持续”标签也不替代数学成因。

## 覆盖与维护

- [首轮覆盖](coverage_table_v1.md)、[分母](denominator.md)、[模型覆盖](model_coverage_audit.md)
- [本轮整理记录](mainline_cleanup_20260905.md)
- [结果目录说明](../results/README.md)、[项目地图](../PROJECT.md)

原始结果、冻结协议、阴性、未决及推导全部保留。按时间追加的实验可能使用不同模型、
参考、optimizer 或参数范围；不能选一个最新数字覆盖所有旧协议。
当前结论引用具体 JSON 和源码，而不是从旧 Markdown 反推实验事实。
