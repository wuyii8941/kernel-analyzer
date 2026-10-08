# 保证链：规则 → 函数 → 测试 → 降级记录（工作项 A，2026-10-08）

由 `results/closure/chain_table.json` 生成（`scripts/closure/build_chain_table.py`）；包内校验器 `specs/phase2/validate_chain_table.py`：**0 problems**（`results/closure/chain_table_validation.txt`）；表中引用的 45 个测试（含参数化展开）全部通过（`results/closure/chain_table_tests_run.txt`）。逐节点求值规则的细目（参照内存、atomic、路径、控制依赖、跨实例冲突、可观测性、三类输出、未知操作、接口、统计、等价轴）沿用 `docs/rules_to_code_20261006.md` 的 13 行，本表按保证链的五个箭头组织。

## 箭头 (0) 声明与执行绑定

前提：输入/初始内存/启动顺序/实例数与捕获一致；IR 与产物同一次编译；输出与写入缓冲区的配对按存储实例可证

规则：保存 IR/配置/产物哈希；同一输入重放两次逐位相同；配对按存储实例追踪

| 代码（函数 / 标识） | 文件 |
|---|---|
| `class TritonLaunchRecorder` | `src/kernel_analyzer/reference_eval/capture.py` |
| `def storage_ids` | `src/kernel_analyzer/reference_eval/capture.py` |
| `def check_replay` | `src/kernel_analyzer/reference_eval/capture.py` |
| `cubin_sha256` | `src/kernel_analyzer/reference_eval/capture.py` |
| `binding_unconfirmed` | `src/kernel_analyzer/check.py` |
| `reused_address` | `src/kernel_analyzer/check.py` |

| 测试 | 文件 |
|---|---|
| `test_aten_output_at_freed_triton_address_is_not_bound` | `tests/test_output_binding.py` |
| `test_triton_written_output_is_still_bound` | `tests/test_output_binding.py` |
| `test_capture_package_replays_bit_for_bit_from_the_saved_binary` | `tests/test_reference_eval_ttir.py` |

| 降级条件 | 处理 | 实际触发的记录 |
|---|---|---|
| 配对不可证 | 输出未建立 | `results/closure/degrade/arrow0_binding_not_established.json` |

## 箭头 (1) 组合包含性

前提：A1；A2；A3；A4；A5；kappa=complete

规则：规则表逐节点求值；kappa 单调传播

| 代码（函数 / 标识） | 文件 |
|---|---|
| `def evaluate_sequence` | `src/kernel_analyzer/reference_eval/ttir_eval.py` |
| `def _op_load` | `src/kernel_analyzer/reference_eval/ttir_eval.py` |
| `def _op_store` | `src/kernel_analyzer/reference_eval/ttir_eval.py` |
| `def _op_if` | `src/kernel_analyzer/reference_eval/ttir_eval.py` |
| `def _with_cond` | `src/kernel_analyzer/reference_eval/ttir_eval.py` |
| `def _op_atomic_rmw` | `src/kernel_analyzer/reference_eval/ttir_eval.py` |
| `def rule_for` | `src/kernel_analyzer/reference_eval/ttir_mapping.py` |
| `def kernel_coverage` | `src/kernel_analyzer/reference_eval/ttir_mapping.py` |
| `def iadd` | `src/kernel_analyzer/reference_eval/intervals.py` |
| `def imul` | `src/kernel_analyzer/reference_eval/intervals.py` |

| 测试 | 文件 |
|---|---|
| `test_store_then_load_keeps_the_reference_value` | `tests/test_reference_eval_counterexamples.py` |
| `test_undecided_branch_takes_the_union_of_both_branches` | `tests/test_reference_eval_counterexamples.py` |
| `test_unknown_op_or_attribute_is_rejected_not_skipped` | `tests/test_reference_eval_counterexamples.py` |
| `test_atomic_return_value_requires_an_order` | `tests/test_reference_eval_counterexamples.py` |
| `test_control_dependence_on_a_pinned_load_downgrades_writes` | `tests/test_reference_eval_ttir.py` |
| `test_stores_from_several_instances_to_one_address_are_a_race_unless_equal` | `tests/test_reference_eval_ttir.py` |
| `test_registry_of_locked_build_is_covered_exhaustively` | `tests/test_reference_eval_ttir.py` |
| `test_elementary_enclosure_contains_every_sampled_value` | `tests/test_reference_eval_enclosure.py` |
| `test_composed_chain_contains_the_true_value_and_widens_monotonically` | `tests/test_reference_eval_enclosure.py` |

| 降级条件 | 处理 | 实际触发的记录 |
|---|---|---|
| kappa=conditional | 不进统计，报告比例 | `results/closure/degrade/arrow1_kappa_conditional.json` |
| kappa=unestablished | 不进统计，报告比例 | `results/closure/degrade/arrow1_kappa_unestablished.json` |
| 未识别语义 | 拒绝 | `results/closure/degrade/arrow1_unrecognised_semantics_rejected.json` |

## 箭头 (2) 残差与投影

前提：箭头1成立；有向舍入向外；方向在确认前确定；D_m 有区间扩展

规则：e_num 区间；区间点积

| 代码（函数 / 标识） | 文件 |
|---|---|
| `def residual_interval` | `src/kernel_analyzer/reference_eval/analysis.py` |
| `def isub` | `src/kernel_analyzer/reference_eval/intervals.py` |
| `def idot` | `src/kernel_analyzer/reference_eval/intervals.py` |
| `def project_bounds` | `src/kernel_analyzer/reference_eval/intervals.py` |
| `def _project` | `src/kernel_analyzer/reference_eval/analysis.py` |
| `def _direction` | `src/kernel_analyzer/reference_eval/analysis.py` |
| `def apply_direction_rules` | `src/kernel_analyzer/reference_eval/analysis.py` |
| `residual_interval(k, r_lo, r_hi)` | `src/kernel_analyzer/check.py` |

| 测试 | 文件 |
|---|---|
| `test_residual_interval_encloses_what_plain_subtraction_rounds_away` | `tests/test_reference_eval_guards.py` |
| `test_grouped_projection_is_in_original_units` | `tests/test_reference_eval_rules.py` |
| `test_fixed_and_aligned_rules_match_previous_implementation` | `tests/test_reference_eval_rules.py` |

| 降级条件 | 处理 | 实际触发的记录 |
|---|---|---|
| D_m 无区间扩展 | 未建立 | `results/reference_eval/blind_test_v2/phase3/phase3_adamw_history_actual_write.csv` |
| 方向来源未登记 | 不出正式结论 | `results/closure/degrade/arrow2_direction_not_registered.json` |

## 箭头 (3) 均值区间

前提：抽样单位独立或按 run 聚类；端点检验前提；开发/确认分离；剔除已声明；检验族事前固定

规则：单侧界并集；Holm；TOST；敏感性曲线

| 代码（函数 / 标识） | 文件 |
|---|---|
| `def _summarize` | `src/kernel_analyzer/reference_eval/analysis.py` |
| `def _sample_guard` | `src/kernel_analyzer/reference_eval/analysis.py` |
| `def apply_holm` | `src/kernel_analyzer/reference_eval/analysis.py` |
| `def equivalence` | `src/kernel_analyzer/reference_eval/analysis.py` |
| `def _t_approximation` | `src/kernel_analyzer/reference_eval/analysis.py` |
| `def _robust_companion` | `src/kernel_analyzer/reference_eval/analysis.py` |
| `def statistical_judgment` | `scripts/essential/contract_v3.py` |
| `def main` | `scripts/sensitivity_curves.py` |

| 测试 | 文件 |
|---|---|
| `test_summary_cannot_judge_with_one_unit_or_numerical_failure` | `tests/test_reference_eval_guards.py` |
| `test_rules_with_one_confirmation_unit_stay_out_of_holm` | `tests/test_reference_eval_guards.py` |
| `test_zero_variance_leaves_both_axes_unresolved` | `tests/test_reference_eval_equivalence.py` |
| `test_skewed_units_are_flagged_and_get_a_bootstrap_companion` | `tests/test_reference_eval_equivalence.py` |
| `test_zero_variance_is_degenerate_not_p0` | `scripts/essential/tests/test_contract_v3.py` |
| `test_strong_skew_small_n_cannot_judge_distribution` | `scripts/essential/tests/test_contract_v3.py` |
| `test_small_sample_cannot_judge` | `scripts/essential/tests/test_contract_v3.py` |

| 降级条件 | 处理 | 实际触发的记录 |
|---|---|---|
| 方差为零 | 无法判断（退化） | `results/closure/degrade/arrow3_zero_variance.json` |
| 分布前提不成立 | 无法判断（分布） | `results/closure/degrade/arrow3_distribution_premise.json` |
| 样本不足 | 无法判断（样本） | `results/closure/degrade/arrow3_sample_size.json` |

## 箭头 (4) 结论分栏

前提：箭头3成立；契约已声明允许的数值行为

规则：四栏；精度不变性分类（黑箱）

| 代码（函数 / 标识） | 文件 |
|---|---|
| `def combine` | `scripts/essential/classify.py` |
| `def fr_assess` | `scripts/essential/classify.py` |
| `def precision_invariance` | `scripts/essential/contract_v3.py` |
| `def four_column` | `scripts/essential/contract_v3.py` |
| `def classify_condition` | `scripts/closure/contract_v2.py` |

| 测试 | 文件 |
|---|---|
| `test_fr_with_no_usable_element_is_not_established` | `scripts/essential/tests/test_classify_guards.py` |
| `test_shared_relation_needs_a_comparable_pair` | `scripts/essential/tests/test_classify_guards.py` |
| `test_precision_invariance_classes` | `scripts/essential/tests/test_contract_v3.py` |
| `test_precision_invariance_nonfinite_goes_to_column4` | `scripts/essential/tests/test_contract_v3.py` |

| 降级条件 | 处理 | 实际触发的记录 |
|---|---|---|
| ok_elements=0 | 未建立 | `results/closure/degrade/arrow4_ok_elements_zero.json` |
| 无可比配对 | 共有关系未建立 | `results/closure/degrade/arrow4_no_comparable_pair.json` |
| 契约未声明 | 只记录 | `results/closure/degrade/arrow4_contract_not_declared.json` |

说明：箭头 (2) 的「D_m 无区间扩展」用的是已有的真实记录（盲测 v2 阶段 3 的实际 FP32 写入，prog_01 等在 20 个坐标 × seed 上参照未建立，全部规则 UNRESOLVED_REFERENCE）；其余降级由 `scripts/closure/trigger_degrades.py` 用最小构造输入经生产代码触发一次。「方向来源未登记」在规则注册表查找时即拒绝（KeyError），不进入投影与检验，所以不出正式结论。
