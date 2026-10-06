# 核心规则与代码、测试的对应（评价计划工作项 A，2026-10-06）

每条规则写明：声明在哪里（阶段总结第 4、5 节）、实现在哪里、哪个测试检验它（正例与负对照）。路径相对
`src/kernel_analyzer/reference_eval/`，测试在 `tests/`。不要求一条规则只对应一处代码。

| # | 规则 | 实现 | 测试（正例 / 负对照） |
|---|---|---|---|
| 1 | 数值差异模式：名称带舍入后缀的函数与 PTX 指令同样取实数运算；舍入核验模式另行声明 | `ttir_eval.NumericMode`；各 `_op_*` 按 `self.mode` 分支 | `test_reference_eval_counterexamples::test_rounding_suffix_does_not_change_numerical_difference_reference`；`test_reference_eval_ttir::test_rounding_check_mode_reproduces_ieee_kernels_bit_for_bit` |
| 2 | 包含性：点输入的基本运算精确，初等函数用定向舍入得包围，区间按区间算术传播（非单调函数处理内部极值） | `intervals.py`（`iadd`、`imul`、`isum`、`project_bounds`…）；`numbers.py` | `test_reference_eval_enclosure`（三项）；`counterexamples::test_sqrt2_is_an_interval…`、`::test_exp_then_sum_keeps_a_nonzero_enclosing_width`、`::test_sin_enclosure_includes_an_interior_maximum` |
| 3 | 参照内存：区域内 store 写参照值，之后的 load 读参照值，不读捕获值 | `ttir_eval._op_store`（941）、`_op_load`（840）、`Buffer` | 正例 `counterexamples::test_store_then_load_keeps_the_reference_value`；负对照 `ttir::test_pinning_a_load_to_its_captured_value_downgrades_the_output` |
| 4 | atomic 返回旧值并更新内存；只有返回值未用、贡献与次序无关时才折成一次精确和 | `ttir_eval._op_atomic_rmw`（1032） | 正例 `counterexamples::test_unused_atomic_return_folds_into_an_exact_sum`；负对照 `counterexamples::test_atomic_return_value_requires_an_order`、`ttir::test_atomic_return_value_is_not_established_when_used` |
| 5 | 路径：条件用参照值判定；未判定时两支取并集 | `ttir_eval._op_if`（1144）、`_op_while`、`_op_for`、控制流 | `ttir::test_branch_is_decided_by_the_reference_value`；`counterexamples::test_branch_is_decided_by_reference_value_not_actual_path`、`::test_undecided_branch_takes_the_union_of_both_branches` |
| 6 | 控制依赖：由固定了实际值的量决定的写入降为条件参照 | `ttir_eval.ProgramState.ctrl`、`_with_cond`（1180） | 负对照 `ttir::test_control_dependence_on_a_pinned_load_downgrades_writes`（参数化） |
| 7 | 跨实例冲突：不同 program 写同一地址且值不同、或读到另一 program 的写入，次序未声明，结果未建立 | `ttir_eval._op_store`（`writer`，1005 起）、`_op_load`（888 起） | 负对照 `ttir::test_stores_from_several_instances_to_one_address_are_a_race_unless_equal`（2026-10-06 新增：值不同为未建立，值相同仍建立） |
| 8 | 可观测性：fma 合并后不存在的中间值不取；被融合的节点在区域边界比较 | `emulate.py`（下降规则、区域）；`ttir_eval` 不插桩 | `counterexamples::test_fma_fusion_moves_the_local_residual_to_the_region_boundary`；`ttir::test_fp_fusion_is_the_source_of_the_rounding_check_mismatch` |
| 9 | 三类输出：完整组合参照、条件局部参照、参照未建立；中止的 program 不算干净 | `TV.st` / `TV.cond`、`ProgramAbort`、`evaluate_sequence` | `ttir::test_aborted_programs_are_reported_and_unwritten_outputs_are_not_clean`、`::test_values_changed_by_an_aborted_program_are_not_established_downstream`、`::test_masked_load_without_other_is_undefined_only_where_used` |
| 10 | 未知操作或属性拒绝而不跳过；注册表穷举 | `ttir_mapping.rule_for`、`kernel_coverage` | `counterexamples::test_unknown_op_or_attribute_is_rejected_not_skipped`；`ttir::test_registry_of_locked_build_is_covered_exhaustively` |
| 11 | 接口：运行时标量按 kernel 接收的值，编译期常数单列 | `interface.py`（`scalar_interface`、`float_constants`） | `test_reference_eval_interface`（三项） |
| 12 | 统计：参照误差只让检验更保守（端点保守 t 推断），不制造假的偏差；单位不足不判 | `analysis._summarize`、`assess_units`、`apply_holm` | `test_reference_eval_guards`（五项）；`test_reference_eval_rules`；校准 `detector_calibration_v2.json` |
| 13 | 等价轴（2026-10-06 新增）：与非零判定分开报告；|偏度| ≥ 1 时附 bootstrap-t | `analysis.equivalence`、`add_equivalence`、`_t_approximation`、`_robust_companion` | 校准 `scripts/calibrate_equivalence.py`（`results/reference_eval/calibration_equivalence.json`）；单元测试 `test_reference_eval_equivalence` |

贯穿例子（方法章节 §1.2）由统一入口执行：`kernel-analyzer check examples/bindings/softmax.py`（模式 B）与
`softmax_no_spec.py`（模式 A）。
