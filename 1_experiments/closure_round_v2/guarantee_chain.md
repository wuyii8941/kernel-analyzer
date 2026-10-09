# 保证链：规则 → 函数 → 测试 → 降级记录（工作项 A，2026-10-08）

由 `1_experiments/closure_round_v2/results/chain_table.json` 生成（`pre-reorg-20261009:scripts/closure/build_chain_table.py`，本文件由 `pre-reorg-20261009:scripts/closure/render_rules_doc.py` 渲染）；包内校验器 `1_experiments/specs/phase2/validate_chain_table.py`：**0 problems**（`1_experiments/closure_round_v2/results/chain_table_validation.txt`）；表中引用的测试（含参数化展开）：`48 passed in 7.17s`（`1_experiments/closure_round_v2/results/chain_table_tests_run.txt`）。检测器 2.3（`2_tool/docs/changelog.md`）。逐节点求值规则的细目沿用 `pre-reorg-20261009:docs/rules_to_code_20261006.md` 的 13 行，本表按保证链的五个箭头组织。

## 箭头 (0) 声明与执行绑定

前提：输入/初始内存/启动顺序/实例数与捕获一致；IR 与产物同一次编译；输出与写入缓冲区的配对按存储实例可证

规则：保存 IR/配置/产物哈希；同一输入重放两次逐位相同；配对按存储实例追踪

| 代码（函数 / 标识） | 文件 |
|---|---|
| `class TritonLaunchRecorder` | `2_tool/src/kernel_analyzer/reference_eval/capture.py` |
| `def storage_ids` | `2_tool/src/kernel_analyzer/reference_eval/capture.py` |
| `def check_replay` | `2_tool/src/kernel_analyzer/reference_eval/capture.py` |
| `cubin_sha256` | `2_tool/src/kernel_analyzer/reference_eval/capture.py` |
| `binding_unconfirmed` | `2_tool/src/kernel_analyzer/check.py` |
| `reused_address` | `2_tool/src/kernel_analyzer/check.py` |

| 测试 | 文件 |
|---|---|
| `test_aten_output_at_freed_triton_address_is_not_bound` | `2_tool/tests/test_output_binding.py` |
| `test_triton_written_output_is_still_bound` | `2_tool/tests/test_output_binding.py` |
| `test_capture_package_replays_bit_for_bit_from_the_saved_binary` | `2_tool/tests/test_reference_eval_ttir.py` |

| 降级条件 | 处理 | 实际触发的记录 |
|---|---|---|
| 配对不可证 | 输出未建立 | `1_experiments/closure_round_v2/results/degrade/arrow0_binding_not_established.json` |

## 箭头 (1) 组合包含性

前提：A1；A2；A3；A4；A5；kappa=complete

规则：规则表逐节点求值；kappa 单调传播

| 代码（函数 / 标识） | 文件 |
|---|---|
| `def evaluate_sequence` | `2_tool/src/kernel_analyzer/reference_eval/ttir_eval.py` |
| `def _op_load` | `2_tool/src/kernel_analyzer/reference_eval/ttir_eval.py` |
| `def _op_store` | `2_tool/src/kernel_analyzer/reference_eval/ttir_eval.py` |
| `def _op_if` | `2_tool/src/kernel_analyzer/reference_eval/ttir_eval.py` |
| `def _with_cond` | `2_tool/src/kernel_analyzer/reference_eval/ttir_eval.py` |
| `def _op_atomic_rmw` | `2_tool/src/kernel_analyzer/reference_eval/ttir_eval.py` |
| `def rule_for` | `2_tool/src/kernel_analyzer/reference_eval/ttir_mapping.py` |
| `def kernel_coverage` | `2_tool/src/kernel_analyzer/reference_eval/ttir_mapping.py` |
| `def iadd` | `2_tool/src/kernel_analyzer/reference_eval/intervals.py` |
| `def imul` | `2_tool/src/kernel_analyzer/reference_eval/intervals.py` |
| `def torch_intermediates` | `2_tool/src/kernel_analyzer/check.py` |

| 测试 | 文件 |
|---|---|
| `test_store_then_load_keeps_the_reference_value` | `2_tool/tests/test_reference_eval_counterexamples.py` |
| `test_undecided_branch_takes_the_union_of_both_branches` | `2_tool/tests/test_reference_eval_counterexamples.py` |
| `test_unknown_op_or_attribute_is_rejected_not_skipped` | `2_tool/tests/test_reference_eval_counterexamples.py` |
| `test_atomic_return_value_requires_an_order` | `2_tool/tests/test_reference_eval_counterexamples.py` |
| `test_control_dependence_on_a_pinned_load_downgrades_writes` | `2_tool/tests/test_reference_eval_ttir.py` |
| `test_stores_from_several_instances_to_one_address_are_a_race_unless_equal` | `2_tool/tests/test_reference_eval_ttir.py` |
| `test_registry_of_locked_build_is_covered_exhaustively` | `2_tool/tests/test_reference_eval_ttir.py` |
| `test_elementary_enclosure_contains_every_sampled_value` | `2_tool/tests/test_reference_eval_enclosure.py` |
| `test_composed_chain_contains_the_true_value_and_widens_monotonically` | `2_tool/tests/test_reference_eval_enclosure.py` |
| `test_output_with_identical_stale_bytes_still_depends_on_the_aten_intermediate` | `2_tool/tests/test_upstream_sources.py` |
| `test_output_reading_only_case_inputs_has_no_upstream_intermediate` | `2_tool/tests/test_upstream_sources.py` |
| `test_kernel_reference_records_the_storages_it_stored` | `2_tool/tests/test_upstream_sources.py` |

| 降级条件 | 处理 | 实际触发的记录 |
|---|---|---|
| kappa=conditional | 不进统计，报告比例 | `1_experiments/closure_round_v2/results/degrade/arrow1_kappa_conditional.json` |
| kappa=unestablished | 不进统计，报告比例 | `1_experiments/closure_round_v2/results/degrade/arrow1_kappa_unestablished.json` |
| 未识别语义 | 拒绝 | `1_experiments/closure_round_v2/results/degrade/arrow1_unrecognised_semantics_rejected.json` |

## 箭头 (2) 残差与投影

前提：箭头1成立；有向舍入向外；方向在确认前确定；D_m 有区间扩展

规则：e_num 区间；区间点积

| 代码（函数 / 标识） | 文件 |
|---|---|
| `def residual_interval` | `2_tool/src/kernel_analyzer/reference_eval/analysis.py` |
| `def isub` | `2_tool/src/kernel_analyzer/reference_eval/intervals.py` |
| `def idot` | `2_tool/src/kernel_analyzer/reference_eval/intervals.py` |
| `def project_bounds` | `2_tool/src/kernel_analyzer/reference_eval/intervals.py` |
| `def _project` | `2_tool/src/kernel_analyzer/reference_eval/analysis.py` |
| `def _direction` | `2_tool/src/kernel_analyzer/reference_eval/analysis.py` |
| `def apply_direction_rules` | `2_tool/src/kernel_analyzer/reference_eval/analysis.py` |
| `residual_interval(k, r_lo, r_hi)` | `2_tool/src/kernel_analyzer/check.py` |

| 测试 | 文件 |
|---|---|
| `test_residual_interval_encloses_what_plain_subtraction_rounds_away` | `2_tool/tests/test_reference_eval_guards.py` |
| `test_grouped_projection_is_in_original_units` | `2_tool/tests/test_reference_eval_rules.py` |
| `test_fixed_and_aligned_rules_match_previous_implementation` | `2_tool/tests/test_reference_eval_rules.py` |

| 降级条件 | 处理 | 实际触发的记录 |
|---|---|---|
| D_m 无区间扩展 | 未建立 | `pre-reorg-20261009:results/reference_eval/blind_test_v2/phase3/phase3_adamw_history_actual_write.csv` |
| 方向来源未登记 | 不出正式结论 | `1_experiments/closure_round_v2/results/degrade/arrow2_direction_not_registered.json` |

## 箭头 (3) 均值区间

前提：抽样单位独立或按 run 聚类；端点检验前提；开发/确认分离；剔除已声明；检验族事前固定

规则：单侧界并集；Holm；TOST；敏感性曲线

| 代码（函数 / 标识） | 文件 |
|---|---|
| `def _summarize` | `2_tool/src/kernel_analyzer/reference_eval/analysis.py` |
| `def _sample_guard` | `2_tool/src/kernel_analyzer/reference_eval/analysis.py` |
| `def apply_holm` | `2_tool/src/kernel_analyzer/reference_eval/analysis.py` |
| `def equivalence` | `2_tool/src/kernel_analyzer/reference_eval/analysis.py` |
| `def _t_approximation` | `2_tool/src/kernel_analyzer/reference_eval/analysis.py` |
| `def _robust_companion` | `2_tool/src/kernel_analyzer/reference_eval/analysis.py` |
| `def statistical_judgment` | `2_tool/scripts/essential/contract_v3.py` |
| `def main` | `pre-reorg-20261009:scripts/sensitivity_curves.py` |

| 测试 | 文件 |
|---|---|
| `test_summary_cannot_judge_with_one_unit_or_numerical_failure` | `2_tool/tests/test_reference_eval_guards.py` |
| `test_rules_with_one_confirmation_unit_stay_out_of_holm` | `2_tool/tests/test_reference_eval_guards.py` |
| `test_zero_variance_leaves_both_axes_unresolved` | `2_tool/tests/test_reference_eval_equivalence.py` |
| `test_skewed_units_are_flagged_and_get_a_bootstrap_companion` | `2_tool/tests/test_reference_eval_equivalence.py` |
| `test_zero_variance_is_degenerate_not_p0` | `2_tool/tests/test_contract_v3.py` |
| `test_strong_skew_small_n_cannot_judge_distribution` | `2_tool/tests/test_contract_v3.py` |
| `test_small_sample_cannot_judge` | `2_tool/tests/test_contract_v3.py` |

| 降级条件 | 处理 | 实际触发的记录 |
|---|---|---|
| 方差为零 | 无法判断（退化） | `1_experiments/closure_round_v2/results/degrade/arrow3_zero_variance.json` |
| 分布前提不成立 | 无法判断（分布） | `1_experiments/closure_round_v2/results/degrade/arrow3_distribution_premise.json` |
| 样本不足 | 无法判断（样本） | `1_experiments/closure_round_v2/results/degrade/arrow3_sample_size.json` |

## 箭头 (4) 结论分栏

前提：箭头3成立；契约已声明允许的数值行为

规则：四栏；精度不变性分类（黑箱）

| 代码（函数 / 标识） | 文件 |
|---|---|
| `def combine` | `pre-reorg-20261009:scripts/essential/classify.py` |
| `def fr_assess` | `pre-reorg-20261009:scripts/essential/classify.py` |
| `def precision_invariance` | `2_tool/scripts/essential/contract_v3.py` |
| `def four_column` | `2_tool/scripts/essential/contract_v3.py` |
| `def classify_condition` | `pre-reorg-20261009:scripts/closure/contract_v2.py` |

| 测试 | 文件 |
|---|---|
| `test_fr_with_no_usable_element_is_not_established` | `pre-reorg-20261009:scripts/essential/tests/test_classify_guards.py` |
| `test_shared_relation_needs_a_comparable_pair` | `pre-reorg-20261009:scripts/essential/tests/test_classify_guards.py` |
| `test_precision_invariance_classes` | `2_tool/tests/test_contract_v3.py` |
| `test_precision_invariance_nonfinite_goes_to_column4` | `2_tool/tests/test_contract_v3.py` |

| 降级条件 | 处理 | 实际触发的记录 |
|---|---|---|
| ok_elements=0 | 未建立 | `1_experiments/closure_round_v2/results/degrade/arrow4_ok_elements_zero.json` |
| 无可比配对 | 共有关系未建立 | `1_experiments/closure_round_v2/results/degrade/arrow4_no_comparable_pair.json` |
| 契约未声明 | 只记录 | `1_experiments/closure_round_v2/results/degrade/arrow4_contract_not_declared.json` |

