# 现有成绩核账（三种状态，2026-10-06）

评价计划（[evaluation_plan_20261006.md](evaluation_plan_20261006.md)）引用的每一项成绩，只用三种状态之一：
**已完成且有报告**（报告与原始数据在仓库）；**已运行，待核验或待入库**；**尚未运行**。新增或变化时改本表。

| # | 成绩 | 问题 | 状态 | 依据（仓库路径） |
|---|---|---|---|---|
| 1 | 盲测两轮的独立复算：16,908,160 个输出元素、974 个投影值，0 处违反 | RQ1 | 已完成且有报告 | `results/reference_eval/blind_test_records/blind_records/v*/..._verification_report.json`；复算 `recount.json`（`scripts/blind_records_recount.py`） |
| 2 | 盲测全部 75 个程序建立完整组合参照，无不支持指令 | RQ1 | 已完成且有报告 | `blind_test_v1/coverage_check.json`、`blind_test_v2/coverage_check.json`；两轮最终计分 |
| 3 | 锁定版本的全部注册指令均映射或显式拒绝（Triton 3.6.0，229 条） | RQ1 | 已完成且有报告 | `results/reference_eval/ttir_op_registry.json`；`docs/auto_reference_results_20261002.md` |
| 4 | 真实训练捕获的 142 个 launch 全部建立参照；逐位模拟全部通过 | RQ1 | 已完成且有报告 | README「核内定位」；`docs/tool_validation.md` |
| 5 | 真实修改的反事实（11 处）与设备输出逐位一致 | RQ1 | 已完成且有报告 | `docs/tool_validation.md` |
| 6 | 语义变体 14 个全部给出 K_R ≠ f 的区间证据；预登记符号 6/6 | RQ2 | 已完成且有报告 | 两轮最终计分；`v1/blind_test_v1_phase2_verification_and_scoring.md` |
| 7 | OpInfo 直接差分基线（2149 个用例），预测 6 条成立 | RQ2、RQ3 | 已完成且有报告 | `docs/baseline_direct_diff_20261006.md`；`results/baseline/compare/` |
| 8 | 已知阳性 16/17 检出且符号正确（去掉事先声明的探针 16/16） | RQ3 | 已完成且有报告 | 两轮最终计分；由冻结判定矩阵复算一致（`recount.json`） |
| 9 | 阴性对照 13/13 与伙伴逐位相同且判定一致（v1 7 个含勘误 5，v2 6 个） | RQ3 | 已完成且有报告 | 两轮最终计分 |
| 10 | 合成数据上误报率与族错误率符合名义水平；默认检测器第 2 版的校准 | RQ3 | 已完成且有报告 | `results/reference_eval/detector_calibration_v2.json`；`docs/tool_validation.md` |
| 11 | 盲测 v2 更新层：理想响应口径 SGD 8/8、AdamW 带历史 8/8 符号与封存推导一致；实际写入口径 11–14 个程序参照未建立 | RQ4 | 已完成且有报告 | `blind_test_v2/phase3/phase3_report.md`；v2 最终计分；更新层答案 |
| 12 | AdamW8bit：状态误差到 8 对独立 run 的训练改善闭合 | RQ4、RQ5 | 已完成且有报告 | `docs/claims.md`；`results/property/result_analysis_v4/iid_training_confirmation/verification.json` |
| 13 | bf16 参数下 AdamW 的问题被无提示地定位到二阶矩写出（B005） | RQ5 | 已完成且有报告 | `bugs/README.md` B005（提交 9d44d70）；`docs/bf16_adam_beta2_prediction.md` |
| 14 | B016、B012 深案例：修复验证、预测对实测、严格包围 | RQ5 | 已完成且有报告 | `bugs/B016_*`、`bugs/B012_*`、`docs/radam_prediction_20261006.md`、`results/tool_spec/final/{scatter1_patched,radam_deep*,strict}` |
| 15 | 真实问题登记 B012–B018（B017、B018 的 B 部分在 main 上仍在；B018 非工具检出） | RQ5 | 已完成且有报告 | `bugs/README.md` 与各报告；nightly 实测输出 |
| 16 | benchmark 修复前后 38 个单位，评分 38/38 | RQ5 | 已完成且有报告 | `docs/benchmark_20261006.md`；`results/benchmark/` |
| 17 | 接口整合回归：15 个用例 0 处判定变化；干净克隆安装 38 项测试通过 | RQ1、RQ5 | 已完成且有报告 | `docs/tool_changes_20261006.md`；`results/regression/` |
| 18 | 外部受控集合（The Correctness Illusion，699 个 Triton 条件 + 141 个黑箱条件）：参照 100% 建立、独立复算 696,816 个元素 0 违反；RQ2 工具 238/238、控制组 0/404、未触发 0/57；RQ3 通过容差而平均作用经 Holm 确认 9 个 | RQ1–RQ3 | 已完成且有报告 | `docs/external_eval_results_20261006.md`；`results/external/gpuemu/`（协议 `docs/external_eval_protocol_20261006.md`，偏离 1 条） |
| 19 | 外部基线 B1–B3（默认与扫描）、消融 A1、A2 | RQ2、RQ3 | 已完成且有报告 | 同上；B4 随机算术式尚未运行（协议写明本轮不做）；A3 不适用于该集合 |
| 20 | 陌生组合子集（Unsloth 14 个入口，bf16）：参照 31/31 完整；无规格差异；经 Holm 确认的平均作用全部为实现特有，RoPE、RMSNorm、SwiGLU 的机制已核实 | RQ1、RQ3 | 已完成且有报告 | `docs/unfamiliar_subset_results_20261006.md`；`results/external/unsloth*`（协议偏离 3 条）；GeGLU 扩张等机制待核实 |
| 21 | 等价轴两项校准；敏感性曲线（正参照与正负混合参照） | RQ3、RQ4 | 已完成且有报告 | `docs/statistics_calibration_20261006.md`；`results/reference_eval/calibration_equivalence.json`、`sensitivity_curves*.json`；δ 取值待确认 |
| 22 | 真实场景的逐层表（编译后的 HF RMSNorm：输出、梯度、SGD / AdamW 理想响应与实际写入） | RQ4 | 已完成且有报告 | `docs/layer_table_results_20261006.md`；`results/reference_eval/layer_table/`；L1/L2 作用机制待核实 |
| 23 | FPCore 交叉复算：56 个片段、9,120 个元素，0 违反（titanfp，MPFR 241 位） | RQ1 | 已完成且有报告 | `results/fpcore/`；`scripts/fpcore_crosscheck.py` |
| 24 | 诊断收益对照（B016、B012 上工具与直接差分） | RQ5 | 已完成且有报告 | `docs/diagnosis_comparison_20261006.md` |
| 25 | 普查统一报告 | RQ5 | 已完成且有报告 | `docs/census_unified_20261006.md`；`results/tool_spec/final/summary.{md,json}` |
| 26 | 成本：分段计时（含捕获开销修正）；同等预算下的比较 | RQ1、RQ5 | 分段计时已完成且有报告；同等预算比较尚未写成报告 | 各报告的 `timing_seconds`；`docs/external_eval_results_20261006.md` RQ1 |
| 27 | 外部接入试验（未参与开发的人按指南接入） | RQ5 | 尚未运行（需要外部人员） | `docs/reuse_trial_protocol.md` |
| 28 | 上游提交 B012、B016、B017、B018 | RQ5 | 尚未运行（草稿待用户提交） | `bugs/upstream_drafts/` |

注：盲测揭盲后，v1、v2 只作回归集；之后的泛化成绩只来自外部语料与陌生组合。
