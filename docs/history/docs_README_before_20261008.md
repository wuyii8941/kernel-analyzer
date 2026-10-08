# 文档入口

**最高优先级入口：[阶段性总结与系统 prompt（2026-10-02）](stage_summary_20261002.md)。**
**当前任务：本质错误一轮——[第一阶段执行协议](protocol_essential_bugs_20261007.md)（结果冻结，[中期报告](essential_bugs_phase1_interim_20261007.md)）、[第二阶段执行协议](protocol_essential_bugs_phase2_20261007.md)（[2a 中期记录](essential_bugs_phase2a_record_20261007.md)、[2b 运行记录](essential_bugs_phase2b_record_20261007.md)）；独立规格以 `specs/` 中最新的审阅版本为准（第一阶段为 `specs/phase1`，v0.4）。** 已完成的历史计划：[下一阶段计划 A–D（2026-10-06）](next_phase_plan_20261006.md)；M1–M4 的记录见[评价设计与工作计划](evaluation_plan_20261006.md)；成绩核账见 [status_ledger_20261006.md](status_ledger_20261006.md)。
评价：[Benchmark](benchmark_20261006.md)；外部受控集合（[协议](external_eval_protocol_20261006.md)、[结果](external_eval_results_20261006.md)）；陌生组合子集（[协议](unfamiliar_subset_protocol_20261006.md)、[结果](unfamiliar_subset_results_20261006.md)）；[统计校准](statistics_calibration_20261006.md)；逐层表（[协议](layer_table_protocol_20261006.md)、[结果](layer_table_results_20261006.md)）；[诊断对照](diagnosis_comparison_20261006.md)；[成本](cost_equal_budget_20261006.md)；[普查统一报告](census_unified_20261006.md)。接入：[接入指南](binding_guide.md)、[外部接入试验协议](reuse_trial_protocol.md)。本轮进展：[执行进展](progress_20261006.md)、[工具改动 2026-10-06](tool_changes_20261006.md)（一个引擎、两种模式、命令行、回归比较）、[直接差分基线](baseline_direct_diff_20261006.md)（运行前的[预测](baseline_predictions_20261006.md)）、[问题登记](../bugs/README.md)。
凡与它冲突的旧结论以它为准；下列文档是其证据与细节来源。运行环境见
[environments.md](environments.md)；自动参照求值器（第 11 节步 1）在
`src/kernel_analyzer/reference_eval/`，验收测试为 `tests/test_reference_eval_*.py`；第 1–6 步的
实施结果见 [自动参照结果](auto_reference_results_20261002.md)，第二轮检验（统计校准、真实 kernel 改动、
与误差大小方法的对比、交叉熵除法追到更新、换 Triton 版本）见 [工具检验](tool_validation.md)。五步证据链的推导见
[讲稿](bias_chain_final.md)。

当前科研口径从下面四份文档进入：

1. [当前主线](current_mainline.md)：研究问题、贡献与完成边界。
2. [实验方法](method.md)：比较对象、三阶段测量和统计定义。
3. [主张账本](claims.md)：证据允许说什么、不能说什么。
4. [全部案例结论](root_cause_closure_current.md)：去重后的来源、平均 bias、实际写入、loss、未定位根因和原始记录。

案例状态以对应协议、机器结果和案例总表为准；统计解释以方法中的对象与假设为准。
发现矛盾时记录并修正，不按更新时间自动采用更强结论。覆盖数量、旧阶段完成标签和
单次 loss 差异不单独改变当前结论。

想使用或继续开发小工具，直接看[系统入口与实施计划](system.md#轻量诊断工具实施计划)；
想查最近两天的探针和多步训练，直接看[结果索引](../results/README.md)。两者不再各建
一份案例总结。

## 文档职责边界

- `current_mainline.md` 只写研究问题、贡献、当前边界和停止条件，不维护案例数字。
- `method.md` 与 `statistics_experiment_alignment.md` 只写测量对象、统计量、抽样单位和
  推断条件，不维护逐案例 verdict。
- `claims.md` 只写可以对外主张什么；`root_cause_closure_current.md` 是唯一的逐问题组
  案例汇总，统一维护根因等级、bias 证据、训练后果和未闭合原因。
- `system.md` 只维护实际接口、人工输入责任及小工具的后续计划；`results/README.md`
  只索引机器证据与数据用途，不复制判定表。
- 专题页只保留一个独立推导或实验边界，并链接回案例汇总；不复制全仓库案例总数。
- 原始协议、机器结果和历史实验保存在 `results/`。早期的 profile、persistence 和
  统一测量汇总页已删除，不再创建平行的人类可读总表。

## 当前专题文档

- [算子接入与自动化边界](system.md)：给定算子后还需提供什么、哪些环节已自动化，以及实际端到端检查。
- [统计与实验设计对齐](statistics_experiment_alignment.md)：固定集合、随机 history、
  配对训练和失败端点。
- [随机总体推断合同](population_inference_contract.md)：统一说明平均能量、超界比例和
  方向比例三类总体问题及其保证边界。
- [自动覆盖入口](numerical_coverage_execution.md)与
  [当前 kernel 清单](observed_kernel_catalog_v2.md)。
- [同行与创新边界](novelty_positioning.md)。
- [陌生 Triton 的轻量 bias 检查](bias_checker_triton_examples_20260914.md)：展示最小
  `check_bias` 接入，不扩展为训练级结论。

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
[全部案例结论](root_cause_closure_current.md)，不要从专题标题推断。若某页只描述旧
协议或单次复核，正文必须保留其历史范围，不得把它升级为当前总体结论。

## 历史材料

2026-10-02 整理移除了根目录的旧案例登记 `case.md`、`cases_flash_style.md`、`round2.md`，
历史协议 `bias_protocol.md`、`persistence_property_protocol.md`、4096 步历史审计
`all_bias_long_horizon_audit.md`、`cold_update_alignment_interpretation.md`，以及只记录
接入准备、没有独立结论的家族接入页（八个 `*_common_input.md`、depthwise 卷积、grouped
causal softmax 前向、fused attention）。它们都可从 Git 历史恢复；原始 JSON 未删除。

2026-09-21 整理移除了重复的项目地图、旧 Gemma/Llama 扫描状态、旧未测家族清单、
training-numerical-v2 阶段总结，以及 Liger 顺序训练、saved-P 轨迹、Liger/SiLU
长程复核、Gemma RMS 顺序的独立快照页。其来源限制和复现入口已并入系统、案例总表
及结果索引；原始 JSON、训练输出、阴性/失败记录与讲稿未删除。
后续扫描生成报告写入所属结果目录，不重新生成已撤下的当前文档。

旧协议、阴性结果、失败记录和机器输出保留在 `results/` 及 Git 历史中。已经被当前
主线替换的阶段状态、运行恢复日志、旧清单和旧生成报告不再作为顶层入口；因生成脚本
仍需保留的少数报告只作为机器审计的可读副本，不能被误读为当前结论。用户维护的
[讲稿](talk_beyond_tolerance.md)不是实验状态数据库。

首轮覆盖和模型状态快照已移除，直接查[原覆盖表](../results/coverage/coverage_table_v1.json)
和[原模型审计](../results/coverage/model_coverage_audit_v1.json)。历史 API 留在代码中，
不再用旧 T1–T4 的阶段通过来定义当前自动误差测试能力。

目录中未列入“当前专题文档”的 Markdown 只在仍有独立推导、单次机制实验或生成脚本
依赖时保留；它们不是当前主线入口，也不能产生新的案例计数。新增文档前应先判断
是否应补充现有页面，而不是再建一个平行汇总。
