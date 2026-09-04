# 文档入口

## 当前论文口径

按以下顺序阅读：

1. [当前科研主线](current_mainline.md)：唯一的研究问题、定义、当前数字和下一步。
2. [统一实验方法](method.md)：candidate/repair、成因分解、三阶段测量、统计输出与
   orbit predictor 的限制。
3. [Training Bias Profile v2](training_bias_profile_v2.md)：统一三分支、整体误报控制、
   大向量摘要和五例当前结果。
4. [正式讲稿](talk_beyond_tolerance.md)：面向非本项目听众的报告版本。
5. [证据账本](claims.md)：每条可写主张、证据门槛和当前状态。
6. [长程机器审计](../results/property/declared_persistent_4096/all_bias_case_audit.json)：
   当前 4096-step 结果、标签和 unresolved 记录。`all_bias_long_horizon_audit.md`
   是人类可读表，整理期间不能覆盖机器 JSON。

上述五份文件使用同一顺序：

```text
方向怎样形成
→ local / gradient / update 在哪里出现
→ 效应量和不确定性
→ 短程筛查
→ 长程参数与 loss 后果
```

## 详细证据

- [五案例 Training Bias Profile v2 统一结果](five_case_training_bias_profile_v2.md)
- [方法冻结后的新案例结果](prospective_training_bias_profiles.md)
- [修正后的固定输入 update 等价结果](../results/property/generalization_benchmark_v1/equivalence_v2.json)
- [正负成因分解](effective_antithetic_symmetry.md)
- [Normalization、softmax backward、attention BMM 的统一三阶段补测](three_mechanism_profiles.md)
- [Liger/Phi 统一重测、干预、统计自检与 DeepSeek 未见确认](unified_measurement_round.md)
- [Qwen lm-head/v-proj、Mamba in-proj、saved-P 与 SiLU 的统一补测](extended_unified_profiles.md)
- [Reduction orbit predictor protocol](persistence_property_protocol.md)
- [Phi 同协议随机舍入干预](phi_adamw_source_intervention.md)
- [直接作用、反馈和实际变化](direct_persistence_evidence.md)
- [Optimizer 对照](direct_persistence_optimizer.md)
- [短程筛查](direct_persistence_screen.md)
- [未见实现检查](direct_persistence_heldout.md)
- [当前限制](direct_persistence_limitations.md)
- [无法安全重放的记录](unresolved_long_replays.md)
- [覆盖总表](coverage_table_v1.md)
- [模型覆盖审计](model_coverage_audit.md)
- [分母定义](denominator.md)
- [历史计数变更](gate_history.md)

## 证据保留规则

- `results/` 中的 JSON、CSV、压缩审计表和原始测量不得因文档收口而删除。
- 旧文档中的短程 `positive` 只表示该旧协议下的观察，不能覆盖当前长程标签。
- 旧文档中的 `property` 或 `Oracle` 名称不能被解释为通用安全判断。
- 数字冲突时，以 [当前科研主线](current_mainline.md)、长程机器审计和具体结果 JSON
  为准。
