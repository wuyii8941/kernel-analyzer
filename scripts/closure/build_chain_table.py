#!/usr/bin/env python3
"""Row 1: fill specs/phase2/chain_table_template.json into results/closure/chain_table.json (code locations, tests,
triggered records for every degrade path) and run the package's validator.

    python scripts/closure/build_chain_table.py
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
T = json.loads((ROOT / "specs/phase2/chain_table_template.json").read_text())
SRC = "src/kernel_analyzer/reference_eval/"
CODE = {
    0: [("class TritonLaunchRecorder", SRC + "capture.py"), ("def storage_ids", SRC + "capture.py"),
        ("def check_replay", SRC + "capture.py"), ("cubin_sha256", SRC + "capture.py"),
        ("binding_unconfirmed", "src/kernel_analyzer/check.py"), ("reused_address", "src/kernel_analyzer/check.py")],
    1: [("def evaluate_sequence", SRC + "ttir_eval.py"), ("def _op_load", SRC + "ttir_eval.py"),
        ("def _op_store", SRC + "ttir_eval.py"), ("def _op_if", SRC + "ttir_eval.py"), ("def _with_cond", SRC + "ttir_eval.py"),
        ("def _op_atomic_rmw", SRC + "ttir_eval.py"), ("def rule_for", SRC + "ttir_mapping.py"),
        ("def kernel_coverage", SRC + "ttir_mapping.py"), ("def iadd", SRC + "intervals.py"), ("def imul", SRC + "intervals.py")],
    2: [("def residual_interval", SRC + "analysis.py"), ("def isub", SRC + "intervals.py"), ("def idot", SRC + "intervals.py"),
        ("def project_bounds", SRC + "intervals.py"), ("def _project", SRC + "analysis.py"), ("def _direction", SRC + "analysis.py"),
        ("def apply_direction_rules", SRC + "analysis.py"), ("residual_interval(k, r_lo, r_hi)", "src/kernel_analyzer/check.py")],
    3: [("def _summarize", SRC + "analysis.py"), ("def _sample_guard", SRC + "analysis.py"), ("def apply_holm", SRC + "analysis.py"),
        ("def equivalence", SRC + "analysis.py"), ("def _t_approximation", SRC + "analysis.py"),
        ("def _robust_companion", SRC + "analysis.py"), ("def statistical_judgment", "scripts/essential/contract_v3.py"),
        ("def main", "scripts/sensitivity_curves.py")],
    4: [("def combine", "scripts/essential/classify.py"), ("def fr_assess", "scripts/essential/classify.py"),
        ("def precision_invariance", "scripts/essential/contract_v3.py"), ("def four_column", "scripts/essential/contract_v3.py"),
        ("def classify_condition", "scripts/closure/contract_v2.py")],
}
TESTS = {
    0: [("test_aten_output_at_freed_triton_address_is_not_bound", "tests/test_output_binding.py"),
        ("test_triton_written_output_is_still_bound", "tests/test_output_binding.py"),
        ("test_capture_package_replays_bit_for_bit_from_the_saved_binary", "tests/test_reference_eval_ttir.py")],
    1: [("test_store_then_load_keeps_the_reference_value", "tests/test_reference_eval_counterexamples.py"),
        ("test_undecided_branch_takes_the_union_of_both_branches", "tests/test_reference_eval_counterexamples.py"),
        ("test_unknown_op_or_attribute_is_rejected_not_skipped", "tests/test_reference_eval_counterexamples.py"),
        ("test_atomic_return_value_requires_an_order", "tests/test_reference_eval_counterexamples.py"),
        ("test_control_dependence_on_a_pinned_load_downgrades_writes", "tests/test_reference_eval_ttir.py"),
        ("test_stores_from_several_instances_to_one_address_are_a_race_unless_equal", "tests/test_reference_eval_ttir.py"),
        ("test_registry_of_locked_build_is_covered_exhaustively", "tests/test_reference_eval_ttir.py"),
        ("test_elementary_enclosure_contains_every_sampled_value", "tests/test_reference_eval_enclosure.py"),
        ("test_composed_chain_contains_the_true_value_and_widens_monotonically", "tests/test_reference_eval_enclosure.py")],
    2: [("test_residual_interval_encloses_what_plain_subtraction_rounds_away", "tests/test_reference_eval_guards.py"),
        ("test_grouped_projection_is_in_original_units", "tests/test_reference_eval_rules.py"),
        ("test_fixed_and_aligned_rules_match_previous_implementation", "tests/test_reference_eval_rules.py")],
    3: [("test_summary_cannot_judge_with_one_unit_or_numerical_failure", "tests/test_reference_eval_guards.py"),
        ("test_rules_with_one_confirmation_unit_stay_out_of_holm", "tests/test_reference_eval_guards.py"),
        ("test_zero_variance_leaves_both_axes_unresolved", "tests/test_reference_eval_equivalence.py"),
        ("test_skewed_units_are_flagged_and_get_a_bootstrap_companion", "tests/test_reference_eval_equivalence.py"),
        ("test_zero_variance_is_degenerate_not_p0", "scripts/essential/tests/test_contract_v3.py"),
        ("test_strong_skew_small_n_cannot_judge_distribution", "scripts/essential/tests/test_contract_v3.py"),
        ("test_small_sample_cannot_judge", "scripts/essential/tests/test_contract_v3.py")],
    4: [("test_fr_with_no_usable_element_is_not_established", "scripts/essential/tests/test_classify_guards.py"),
        ("test_shared_relation_needs_a_comparable_pair", "scripts/essential/tests/test_classify_guards.py"),
        ("test_precision_invariance_classes", "scripts/essential/tests/test_contract_v3.py"),
        ("test_precision_invariance_nonfinite_goes_to_column4", "scripts/essential/tests/test_contract_v3.py")],
}
D = "results/closure/degrade/"
TRIGGERS = {
    (0, "配对不可证"): D + "arrow0_binding_not_established.json",
    (1, "kappa=conditional"): D + "arrow1_kappa_conditional.json",
    (1, "kappa=unestablished"): D + "arrow1_kappa_unestablished.json",
    (1, "未识别语义"): D + "arrow1_unrecognised_semantics_rejected.json",
    # real record from blind test v2 phase 3 (actual FP32 write, prog_01: not established on 20 coordinate-seed pairs,
    # every rule UNRESOLVED_REFERENCE)
    (2, "D_m 无区间扩展"): "results/reference_eval/blind_test_v2/phase3/phase3_adamw_history_actual_write.csv",
    (2, "方向来源未登记"): D + "arrow2_direction_not_registered.json",
    (3, "方差为零"): D + "arrow3_zero_variance.json",
    (3, "分布前提不成立"): D + "arrow3_distribution_premise.json",
    (3, "样本不足"): D + "arrow3_sample_size.json",
    (4, "ok_elements=0"): D + "arrow4_ok_elements_zero.json",
    (4, "无可比配对"): D + "arrow4_no_comparable_pair.json",
    (4, "契约未声明"): D + "arrow4_contract_not_declared.json",
}


def main():
    for a in T["arrows"]:
        i = a["id"]
        a["code"] = [{"function": f, "file": p} for f, p in CODE[i]]
        a["tests"] = [{"name": n, "file": p} for n, p in TESTS[i]]
        for d in a["degrade_paths"]:
            d["triggered_record"] = TRIGGERS[(i, d["condition"])]
    T["filled"] = "2026-10-08, scripts/closure/build_chain_table.py; degrade triggers scripts/closure/trigger_degrades.py"
    out = ROOT / "results/closure/chain_table.json"
    out.write_text(json.dumps(T, indent=1, ensure_ascii=False) + "\n")
    r = subprocess.run([sys.executable, "-I", str(ROOT / "specs/phase2/validate_chain_table.py"), str(out), str(ROOT)],
                       capture_output=True, text=True)
    print(r.stdout, r.stderr)
    (ROOT / "results/closure/chain_table_validation.txt").write_text(r.stdout + r.stderr)
    sys.exit(r.returncode)


if __name__ == "__main__":
    main()
