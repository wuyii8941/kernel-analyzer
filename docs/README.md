# 文档入口

当前科研口径只由下面五份文档维护：

1. [当前主线](current_mainline.md)：研究问题、贡献与完成边界。
2. [实验方法](method.md)：比较对象、三阶段测量和统计定义。
3. [主张账本](claims.md)：证据允许说什么、不能说什么。
4. [逐案例根因审计](case_causal_audit.md)：各问题组的代码来源、干预和未闭合部分。
5. [案例与证据地图](case_evidence_map.md)：原始结果和具体协议的位置。
6. [当前根因闭环账本](root_cause_closure_current.md)：按问题组去重后的最新根因、干预和训练后果边界。

这些入口发生冲突时，以方法、根因审计和主张账本中的较窄表述为准。覆盖数量、
旧阶段完成标签和单次 loss 差异不单独改变当前结论。

## 当前专题文档

- [算子接入与自动化边界](system.md)：给定算子后还需提供什么、哪些环节已自动化，以及实际端到端检查。
- [Bias 证明计划结果](bias_proof_plan_result.md)：区分误差能量与系统性 aligned effect。
- [统计与实验设计对齐](statistics_experiment_alignment.md)：固定集合、随机 history、
  配对训练和失败端点。
- [随机总体推断合同](population_inference_contract.md)与
  [超界比例检验](population_exceedance_inference.md)。
- [自动覆盖入口](numerical_coverage_execution.md)与
  [当前 kernel 清单](observed_kernel_catalog_v2.md)。
- [同行与创新边界](novelty_positioning.md)。
- [根因推导边界](root_cause_exhaustion.md)：现有记录能推出什么，以及哪些升级必须新增观测。

## 根因推导和主要案例

- [AdamW8bit optimizer 状态](optimizer_update_family_audit.md)与
  [残差结构干预](adamw8bit_residual_structure_20260913.md)。
- [Liger 分块梯度累加](liger_language_mechanism_followup.md)与
  [训练后果](liger_single_boundary_collapse_experiment.md)。
- [MM 算术与输出舍入](source_aligned_repair.md)。
- [Attention 状态到 q-projection](l23_qproj_tile.md)。
- [SiLU/saved-state 的奇偶响应分解](effective_antithetic_symmetry.md)。
- [Fused RoPE 与 optimizer state](fused_rotary_position_scaling_audit.md)。

这些专题页各自只负责一个推导或实验边界，不维护全仓库案例总数。根因是否闭合统一查
`case_causal_audit.md`，不要从专题标题推断。

## 历史材料

旧协议、阴性结果、失败记录和机器输出保留在 `results/` 及 Git 历史中。已经被当前
主线替换的阶段状态、运行恢复日志、旧清单和旧生成报告不再保留为顶层 Markdown，
避免它们继续被误读为当前结论。用户维护的 [讲稿](talk_beyond_tolerance.md)不是实验
状态数据库。

首轮覆盖和模型状态快照已移除，直接查[原覆盖表](../results/coverage/coverage_table_v1.json)
和[原模型审计](../results/coverage/model_coverage_audit_v1.json)。历史 API 留在代码中，
不再用旧 T1–T4 的阶段通过来定义当前自动误差测试能力。
