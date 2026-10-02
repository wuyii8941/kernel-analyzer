# 实验结果索引

这里保存协议、机器输出与复算报告；逐问题组的当前结论只在
[案例总表](../docs/root_cause_closure_current.md)维护。文件名的日期、目录数量或历史
COMPLETE 标签不代表独立 bias 数，也不决定哪个统计结论有效。

## 近期工作怎么读

| 要查的问题 | 保存位置 | 解释范围 |
|---|---|---|
| 全部去重案例、来源等级和未解项 | [当前机器账本](property/case_causal_audit_v1/root_cause_closure_current.json)与[人工总表](../docs/root_cause_closure_current.md) | 来源闭合、mean bias、训练后果分栏，不混用计数 |
| 最近文本、音频、视觉和 MoE 的单变量探针 | [new_problem_group_search_v1](property/new_problem_group_search_v1/) | 多数是固定真实输入的一步 local/gradient/write，不因使用训练模型而成为独立长训练 |
| 路径隔离、source factorial 和归约复核 | [root_cause_closure_v1](property/root_cause_closure_v1/) | 每份记录只覆盖声明边界，未定位与阴性保留 |
| Liger 同精度顺序：真实写入和经验库方向 | [liger_fp32_chunk_order_v1](property/liger_fp32_chunk_order_v1/) | 旧公式摘要与后来的实际写入采集分开；经验库总体不等于自然训练总体 |
| Granite 同精度 expert 顺序的经验库抽样 | [granite_expert_order_population_v1](property/granite_expert_order_population_v1/) | 按该协议的输入库、载体与 optimizer 条件解释 |

## 多步训练与数据用途

| 实验 | 入口 | 不能混淆的事项 |
|---|---|---|
| AdamW8bit 同路径 OFF/ON 归因 | [same_path_training](property/result_analysis_v1/same_path_training/) | 已选数据上的重放不是新独立确认 |
| 残差坐标、时间、关键/其余参数组干预 | [结构干预协议](property/result_analysis_v3/structured_training/protocol.json)、[失败感知汇总](property/result_analysis_v3/structured_training/analysis/failure_aware_final.json) | 数值失败、执行失败、有限完成分别保留；不只报告幸存者 loss |
| 冻结补偿后的独立训练流确认 | [确认结果](property/result_analysis_v4/iid_training_confirmation/verification.json) | 样本单位是配对 run；范围限于协议中的 checkpoint、数据与评估 |
| Liger 纯 FP32 顺序的 1024 步对照 | [训练记录](property/liger_fp32_chunk_order_v1/fp32_order_training_1024.json) | 小模型机制对照，末端 loss 差回到零；不替代历史精度对照 |
| saved-P 声明轨迹 | [1024 步记录](property/case_causal_audit_v1/qwen_saved_p_declared_trajectory_20260915.json) | 非零写入、有限参数/loss 差异；loss 变号，不是 iid mean-bias 证明 |
| Liger 历史压力轨迹 | [single_point_collapse_v1](property/single_point_collapse_v1/)、[v2 延长汇总](property/single_point_collapse_v2/full_10000_summary.json) | 延长后 loss 差异反转，参数距离不代表质量恶化 |
| Liger/SiLU 历史持续性与 loss | [declared_persistent_4096](property/declared_persistent_4096/)、[paired_loss_4096](property/paired_loss_4096/) | direct 与 feedback、单轨迹与独立重复分别解释 |

## 轻量工具的验证，不是新训练结论

- [主流模型配置扫描](property/mainstream_model_bias_scan_v1/)保留各批次原始扫描及来源探针。
  配置形状的生成输入、checkpoint norm 向量与真实模型训练不是同一个协议。
  汇总入口为该目录中的 `mainstream_bias_root_cause_summary_20260918_v5.json`、
  `additional_mainstream_root_cause_summary_v3.json` 和
  `classic_encoder_decoder_root_cause_summary_v1.json`；各版本按对应批次读，不能拼成
  一次独立确认。statewise aligned 的确认也不等于总体向量均值非零。
- TileLang 的 [GEMM](property/tilelang_gemm_bias_smoke_v1.json)、
  [elementwise](property/tilelang_elementwise_bias_smoke_v1.json)、
  [reduction](property/tilelang_reduction_bias_smoke_v1.json) 是真实编译运行的 smoke；
  [已知偏移](property/tilelang_known_positive_bias_smoke_v1.json)是人工阳性控制。
  新的 reduction-order 家族调用模板见 `scripts/tilelang_reduction_family_smoke.py` 和
  `scripts/run_tilelang_family_check.py`；若在安装 TileLang 的 CUDA 环境运行，输出应另存为
  新报告，不覆盖历史 smoke。当前已完成一份真实 GPU 运行：
  [reduction-order family v9](property/tilelang_reduction_family_smoke_v9.json) 为
  `VALID / EVIDENCE_ONLY / NOT_ASSESSED`，4/4 个同语义变体样本观察到非零同号差异，
  取负输入奇对称检查通过；这仍是声明输入库的受控证据，不是总体 bias 证书。
  同一批量入口的真实 GPU 回归记录在
  [reduction-order CLI](property/tilelang_reduction_family_cli_v1.json) 和
  [elementwise CLI](property/tilelang_elementwise_cli_v1.json)；它们确认命令行加载、编译
  和统一报告路径与直接脚本调用一致。
  单算子、无 reference 的奇对称筛查记录在
  [真实 reduction](property/tilelang_single_operator_reduction_v2.json) 和
  [固定偏移阳性控制](property/tilelang_single_operator_positive_v2.json)；前者未观察到
  性质违反，后者 4/4 样本观察到违反，但两者都不越级签发完整 bias verdict。
  真实 elementwise 单算子筛查见
  [elementwise](property/tilelang_single_operator_elementwise_v2.json)，同样未观察到奇对称
  性质违反。
- 真实 Triton 的 reference-free 家族 smoke 见
  [2026-09-21 记录](property/bias_checker_examples_v1/reference_free_family_smoke_20260921.json)：
  同精度归约模板报告受控 variant 差异，softmax 模板通过 probability-mass 性质；两者
  都不会把性质证据越级写成总体 bias。
- `scripts/run_reference_free_diagnostic_examples.py` 使用人工构造输入，不是真实案例
  无 reference 复现。后续规格模板与真实案例验收见[系统计划](../docs/system.md#轻量诊断工具实施计划)。

## 历史采集与复算

[coverage](coverage/)保存旧位置清单、输入 bank、runtime release 和 T1–T4 证据；
[final](final/)保存早期推导与测量。旧案例登记 `cases_flash_style.md` 已于 2026-10-02 移除，见 Git 历史。
不存在一个历史 JSON 能覆盖所有后续实验。

[training_numerical_analysis_v2](property/training_numerical_analysis_v2/)保存修正后的
实际写入采集；v1 的额外舍入写入模拟仍按历史限制保留。旧随机总体渐近路线的边界
失败也不删除；它不否定后来带独立性/先验界条件的其他合同。当前保证查
[总体合同](../docs/population_inference_contract.md)，不能按 v2 旧阶段总结推断当前全库状态。

保留的复算脚本包括 `run_training_numerical_v2.py` 的 freeze/capture/report、
`verify_training_numerical_reports_v2.py`、`verify_liger_language_confirmation.py`、
`summarize_liger_temporal_extension.py` 和 `build_training_numerical_final_report.py`。
先读其参数和冻结协议；复算不产生新的独立数据，不覆盖既有原始记录。
Gemma RMS 的逐端点归约阴性控制使用 `run_gemma_rms_order_intervention.py`；
同时替换多个端点产生的上游混杂已在案例总表保留，不作为第二个端点的阳性来源。

## 保存与清理规则

原始协议、样本记录、失败日志和 checkpoint 不因文档整理删除。生成汇总与原始证据
分开；运行时报告放在对应结果目录，不另造一份 docs 当前总表。旧报告中的已删除文档
路径是历史记录，必要时从 Git 历史查看，不批量改写冻结文件。

新输出、缓存与临时数据应在本仓库内。部分历史原始张量与 compiler/cache 产品按当时
记录位于仓库外（例如 `/data1/tzh/cache/kernel-analyzer/`），尚未因此次整理迁移；
能否读取原始量须逐实验核对，不能用 JSON 摘要冒充完整张量。文档删除可从 Git 恢复。
