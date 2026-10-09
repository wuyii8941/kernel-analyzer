"""Rule registry of the reference semantics (general-capability round, protocol section 2).

One entry per key (operation, attributes, types, sub-region): status, coverage category, internal operation, the
containment argument, a counterexample test and a negative control (``tests/<file>.py::<function>``), or the reason of
a rejection.  Built from the version-locked mapping (``ttir_mapping.MAPPING``, the libdevice and inline-asm tables,
the registered combiners); ``tests/test_rule_registry.py`` checks that every operation of the locked build has an
entry, every referenced test exists, and the committed registry of this version (``REGISTRY_OUT``) is current.
DSL v2 writes ``results/dsl_v2/rule_registry.json``; ``results/general/rule_registry.json`` stays the frozen record of
general-v3.1.

Status values: SUPPORTED; DECLARED_PREMISE (supported under a premise the code does not check -- listed for the
reviewer); NOT_ESTABLISHED (recognized, result reported as not established); REJECTED (semantics missing).
"""

from __future__ import annotations

import json
from pathlib import Path

from . import ttir_mapping as M

T_CE = "tests/test_reference_eval_counterexamples.py"
T_TT = "tests/test_reference_eval_ttir.py"
T_EN = "tests/test_reference_eval_enclosure.py"
T_GU = "tests/test_reference_eval_guards.py"
T_WF = "tests/test_welford_reduction.py"
T_GR = "tests/test_general_rules.py"
T_UP = "tests/test_upstream_sources.py"
T_EV = "tests/test_execution_validity.py"
T_AT = "tests/test_signatures_atomics.py"
T_LK = "tests/test_signatures_locks.py"
REGISTRY_OUT = Path(__file__).resolve().parents[3] / "data" / "rule_registry.json"  # 2_tool/data

CATEGORY = {
    "A": ("Integer, boolean, address and layout operations are exact in two's-complement width; control flow follows "
          "the reference value of its predicate, and an undecided predicate joins the effects of both branches (an "
          "enclosure of the true branch, possibly wide); nested loops are unrolled in program order.",
          f"{T_CE}::test_branch_is_decided_by_reference_value_not_actual_path",
          f"{T_CE}::test_undecided_branch_takes_the_union_of_both_branches"),
    "B": ("+, -, x, /, fma: point inputs are evaluated exactly by error-free transformations, intervals by directed "
          "rounding, so the result is a superset of the real image.",
          f"{T_GU}::test_residual_interval_encloses_what_plain_subtraction_rounds_away",
          f"{T_CE}::test_fma_fusion_moves_the_local_residual_to_the_region_boundary"),
    "C": ("Comparisons, selection, min / max, rounding to integers: decided on the reference interval; an undecided "
          "outcome joins both results (still an enclosure) or, for integer results, is not established.",
          f"{T_TT}::test_comparison_of_a_value_with_itself_is_decided_by_nan_alone",
          f"{T_GR}::test_undecided_select_takes_the_union_which_still_encloses_the_true_branch"),
    "D": ("Elementary functions: MPFR-rounded (or proven-error) enclosures of the real function over the input "
          "interval, interior extrema included; outside the domain -> not established.",
          f"{T_CE}::test_sin_enclosure_includes_an_interior_maximum",
          f"{T_CE}::test_log1p_domain_is_open_at_minus_one"),
    "E": ("Reductions, scans and dot products: sums and dot products by SumK / DotK with the Ogita-Rump-Oishi bound "
          "(tool 3.0); registered order-free combiners as fast paths, any other combine region interpreted along the "
          "lowering's combination order (sub-entries).",
          f"{T_CE}::test_exp_then_sum_keeps_a_nonzero_enclosing_width",
          f"{T_WF}::test_three_independent_sums_get_a_complete_reference"),
    "F": ("Conversions: in numerical-difference mode a floating conversion is the identity on the reals (its rounding "
          "belongs to e_num); integer <-> float conversions are exact or enclose the lost units.",
          f"{T_TT}::test_large_integer_to_float_keeps_the_lost_unit_inside_the_interval",
          f"{T_CE}::test_rounding_suffix_does_not_change_numerical_difference_reference"),
    "G": ("Bit-level reinterpretation: exact on point values; a non-point interval cannot be reinterpreted -> not "
          "established.",
          f"{T_GR}::test_bitcast_round_trip_of_point_values_is_exact",
          f"{T_TT}::test_copysign_and_signbit_read_the_sign_of_zero"),
    "H": ("Memory, atomics, program ids: a load reads the reference value of the last store (composed across "
          "launches and aliases); races and stores through addresses that are not established invalidate the "
          "target; values made by non-Triton ops inside the capture enter as exact inputs and mark the output mixed.  "
          "Execution validity (DSL v2): an address one program reads and another program of the launch writes is an "
          "execution race in either order; a value stored and loaded back in one program without a barrier is ordered "
          "only when the same single thread holds it in both accesses (TTGIR layouts), an execution race when another "
          "thread reads it, not established when the thread mapping is unknown; gpu.barrier starts a new phase.",
          f"{T_EV}::test_scalar_roundtrip_without_barrier_is_an_execution_race",
          f"{T_EV}::test_tensor_roundtrip_in_one_layout_is_ordered"),
    "I": ("Calls are evaluated in place; extern libdevice symbols and inline assembly only by registered "
          "declarations (sub-entries); anything else is rejected.",
          f"{T_CE}::test_rounding_suffix_does_not_change_numerical_difference_reference",
          f"{T_CE}::test_unknown_op_or_attribute_is_rejected_not_skipped"),
}
REJECT_TEST = f"{T_CE}::test_unknown_op_or_attribute_is_rejected_not_skipped"


def _entry(op, status, category, internal=None, argument="", ce="", nc="", reason="", attrs="*", types="*",
           subregion="*"):
    return {"key": {"op": op, "attrs": attrs, "types": types, "subregion": subregion}, "status": status,
            "category": category, "internal": internal, "containment_argument": argument, "counterexample_test": ce,
            "negative_control": nc, "reason": reason}


def _op_rule(name):
    return M.rule_for(name)


def build() -> dict:
    reg = json.loads(Path(M.REGISTRY_PATH).read_text())
    entries = []
    for dialect, names in reg["operations"].items():
        for name in sorted(names):
            rule = _op_rule(name)
            if rule is None:
                entries.append(_entry(name, "REJECTED", "-", reason="no mapping entry (coverage failure)", ce=REJECT_TEST))
                continue
            if rule.status == "REJECTED":
                entries.append(_entry(name, "REJECTED", rule.category, reason=rule.reason, ce=REJECT_TEST))
                continue
            arg, ce, nc = CATEGORY[rule.category]
            entries.append(_entry(name, "SUPPORTED", rule.category, rule.internal, arg, ce, nc))
    # ---- sub-regions of tt.reduce
    simple = sorted(set(M._SIMPLE_COMBINERS.values()))
    for comb in simple + ["max_select", "min_select", "argmax", "argmin"]:
        arg = ("Registered associative-commutative combiner: the real result does not depend on the reduction tree; "
               "floating sums by SumK.")
        entries.append(_entry("tt.reduce", "SUPPORTED", "E", comb, arg, CATEGORY["E"][1], CATEGORY["E"][2],
                              subregion=comb))
    entries.append(_entry(
        "tt.reduce", "SUPPORTED", "E", "welford",
        "Welford merge of (mean, M2, weight): for weights >= 0 with W > 0 every merge tree equals the closed form "
        "W = sum w, mean = sum w m / W, M2 = sum s + sum w (m - mean)^2 (pairwise parallel-axis identity; zero-weight "
        "triples only add s); W = 0 (all input weights exactly 0) -> mean not established, M2 = sum s; possibly "
        "negative weights -> not established; unguarded ratio r = w_b / W with two weights that may be 0 -> 0 / 0 in "
        "some merge tree -> not established; rounding-check mode and the bit-exact emulator reject it (float Welford "
        "depends on the merge tree).  Matched on dataflow, never on names; a fast path (Phi certificate): rows the "
        "closed form leaves open because they depend on the merge tree are evaluated along the actual tree when the "
        "TTGIR gives it (generic combine region entry).",
        f"{T_WF}::test_all_zero_weights_follow_the_actual_merge_tree", f"{T_WF}::test_possibly_negative_weights_are_not_established",
        subregion="welford(mean, M2, weight)"))
    entries.append(_entry(
        "tt.reduce", "DECLARED_PREMISE", "E", "generic_region_lowering_order",
        "Any other combine region is interpreted step by step with interval semantics along the combination order of "
        "the locked lowering, read from the captured TTGIR layout (sequential within a thread in register order, "
        "butterfly over lanes, then over warps; a warp-synchronous result is the hull over its lanes).  The target is "
        "order-specific (DSL v2 rc3 02 6.2).  Premise for the reviewer: the order model (checked bit-exactly against "
        "the device for float sums by the emulator; the operand order inside a combine follows the lowering source).  "
        "No TTGIR layout -> not established, never guessed.",
        f"{T_WF}::test_merge_without_the_weight_factor_gets_an_order_specific_reference",
        f"{T_WF}::test_unguarded_ratio_with_zero_weights_is_not_established", subregion="other"))
    entries.append(_entry(
        "tt.atomic_cas", "DECLARED_PREMISE", "H", "cas_serialization",
        "Contended compare-and-swap (DSL v2 increment 7, rc3 02 6.9 declared order + evidence): the launch is "
        "evaluated in every program order when it has at most 4 program instances, otherwise in program order and "
        "reverse program order, each from the memory before the launch; only elements every evaluated order "
        "establishes and agrees on keep a value (hull).  For a spin lock (acquire loop + exchange release, one "
        "critical section per program) the commutativity certificate replays every program's critical section "
        "symbolically and z3 proves the sections commute pairwise: the result then holds for every serialization "
        "(unconditional).  Otherwise the values are complete under the premise (proof status axis, audit F03), never "
        "in the unconditional complete class.  Happens-before through release / acquire with vector clocks; relaxed "
        "locks keep racing.  Premise for the reviewer when no certificate: agreement of the evaluated orders stands "
        "for every serialization (evidence, not a proof).",
        f"{T_LK}::test_lock_protected_accumulation_is_established_under_the_order_premise",
        f"{T_LK}::test_first_writer_inside_a_lock_depends_on_the_order", attrs="contended"))
    # ---- sub-regions of tt.scan
    entries.append(_entry("tt.scan", "SUPPORTED", "E", "sum", "Prefix sums enclosed with the gamma bound per prefix.",
                          f"{T_TT}::test_scan_with_a_custom_combine_region_encloses_the_exact_prefix",
                          f"{T_TT}::test_product_scan_encloses_the_exact_prefix_products", subregion="sum"))
    entries.append(_entry("tt.scan", "SUPPORTED", "A", "sum_int", "Integer prefix sums are exact (wrap-around).",
                          CATEGORY["A"][1], CATEGORY["A"][2], subregion="sum_int"))
    for sub in ("generic fold (one operand)", "generic fold (several operands)"):
        entries.append(_entry(
            "tt.scan", "DECLARED_PREMISE", "E", "generic_fold",
            "The combine region is folded in scan order with interval semantics and checked against a second "
            "bracketing (Hillis-Steele doubling) on the same inputs; prefixes the two disagree on are not "
            "established.  Exact in the reals only if the combiner is associative (the tl.associative_scan "
            "precondition): when z3 proves the combine region associative for all arguments (reals / bit-vectors, "
            "finite scanned values) the results are unconditional; otherwise agreement on these inputs is evidence, "
            "not a proof, and the results are complete under the premise (proof status axis, audit F03; premise for "
            "the reviewer).",
            f"{T_TT}::test_scan_with_a_custom_combine_region_encloses_the_exact_prefix",
            f"{T_TT}::test_product_scan_encloses_the_exact_prefix_products", subregion=sub))
    # ---- extern libdevice and inline asm
    for sym, internal in sorted(M.LIBDEVICE.items()):
        entries.append(_entry("tt.extern_elementwise", "SUPPORTED", "D", internal, CATEGORY["D"][0], CATEGORY["D"][1],
                              CATEGORY["D"][2], attrs=f"symbol={sym}"))
    for sym, mode in sorted((M.LIBDEVICE_ROUNDING or {}).items()):
        entries.append(_entry("tt.extern_elementwise", "SUPPORTED", "F", f"rounding:{mode}", CATEGORY["F"][0],
                              CATEGORY["F"][1], CATEGORY["F"][2], attrs=f"symbol={sym}"))
    entries.append(_entry("tt.extern_elementwise", "REJECTED", "I", reason="libdevice symbol without declared semantics",
                          ce=REJECT_TEST, attrs="symbol=other"))
    for pat, internal in sorted(M.INLINE_ASM.items()):
        entries.append(_entry("tt.elementwise_inline_asm", "SUPPORTED", "D", internal, CATEGORY["D"][0],
                              CATEGORY["D"][1], CATEGORY["D"][2], attrs=f"asm~{pat}"))
    entries.append(_entry("tt.elementwise_inline_asm", "REJECTED", "I", reason="unregistered inline assembly",
                          ce=REJECT_TEST, attrs="asm=other"))
    # ---- attribute / use dependent cases
    entries.append(_entry("tt.atomic_rmw", "SUPPORTED", "H", "atomic_fold",
                          "Returned value unused: the updates of one address fold into an exact combination when they "
                          "are one commutative kind or a mixed pair with a checked joint order law (DSL v2 increment 4).",
                          f"{T_CE}::test_unused_atomic_return_folds_into_an_exact_sum",
                          f"{T_AT}::test_mixed_kinds_without_a_joint_law_are_not_established", attrs="return unused"))
    entries.append(_entry("tt.atomic_rmw", "SUPPORTED", "H", "atomic_return_set",
                          "Returned value used (DSL v2 increment 4, rc3 02 5.3 / 10): two passes; without contention "
                          "the value before the launch (a point), with contention the hull of the value before the "
                          "launch combined with every subset of the other updates (a set target, L_E); integers only "
                          "as points.",
                          f"{T_AT}::test_official_pattern_contended_float_add_returns_a_set",
                          f"{T_AT}::test_contended_integer_return_is_not_a_point_unless_every_order_agrees",
                          attrs="return used"))
    entries.append(_entry("tt.dot", "SUPPORTED", "E", "dot",
                          "G is the exact product of the operands as given; a reduced input precision (tf32) is part "
                          "of e_num and recorded (dot_input_precision).", CATEGORY["E"][1], CATEGORY["E"][2],
                          attrs="inputPrecision=ieee|tf32|tf32x3"))
    entries.append(_entry("scf.for/scf.if (nested)", "SUPPORTED", "A", "control", CATEGORY["A"][0],
                          f"{T_GR}::test_nested_loops_and_data_branch_are_exact_where_decided",
                          f"{T_GR}::test_undecided_control_flow_takes_the_union_of_both_branches", subregion="nested"))
    entries.append(_entry("tt.load/tt.store (alias)", "SUPPORTED", "H", "memory",
                          "One storage reached through several pointers is one reference buffer.",
                          f"{T_GR}::test_store_then_load_through_an_alias_carries_the_reference_value",
                          f"{T_TT}::test_store_through_a_not_established_address_invalidates_the_target", attrs="alias"))
    entries.append(_entry("launch sequence (cross-launch state)", "SUPPORTED", "H", "composed",
                          "Launches chain through one reference memory; buffers changed by non-Triton ops in between "
                          "re-enter as exact inputs and mark downstream outputs mixed.",
                          f"{T_TT}::test_composed_reference_across_launches_with_windowed_capture",
                          f"{T_UP}::test_output_with_identical_stale_bytes_still_depends_on_the_aten_intermediate"))
    entries.append(_entry("tt.load (pinned)", "NOT_ESTABLISHED", "H", reason="a load pinned to its captured value (the "
                          "pin_loads option) makes downstream elements conditional (kappa = conditional); check.run does "
                          "not pin", ce=f"{T_TT}::test_pinning_a_load_to_its_captured_value_downgrades_the_output"))
    counts = {}
    for e in entries:
        counts[e["status"]] = counts.get(e["status"], 0) + 1
    return {"schema": "kernel-analyzer-rule-registry-v1", "triton": M.LOCKED_TRITON,
            "source": "ttir_mapping (MAPPING, LIBDEVICE, LIBDEVICE_ROUNDING, INLINE_ASM, combiners)",
            "counts": counts, "entries": entries}
