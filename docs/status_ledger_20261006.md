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
| 18 | 外部受控集合、陌生组合子集上的建立率与对照 | RQ1–RQ3 | 尚未运行 | M2 |
| 19 | 外部基线（默认配置与扫描）与三项消融 | RQ2、RQ3 | 尚未运行 | M2 |
| 20 | 等价性两项校准；敏感性曲线 | RQ3、RQ4 | 尚未运行 | M1 |
| 21 | 真实场景的输出、梯度、实际写入逐层表 | RQ4 | 尚未运行 | M4 |
| 22 | 外部接入试验；诊断收益对照；普查统一报告；成本的同等预算比较 | RQ5 | 尚未运行 | M3 |
| 23 | FPCore 交叉复算 | RQ1 | 尚未运行 | M2 |

注：盲测揭盲后，v1、v2 只作回归集；之后的泛化成绩只来自外部语料与陌生组合。
