"""Composition rules of the common semantic kernel (DSL v2 rc3 04 W2 / W4 / W6; batch 1).

A composition rule is a guarantee that holds for *any* program built from supported constructs once its obligations
are met -- it is not tied to an operator or kernel name.  Each entry states the chain the task asks for:

    obligation -> implementation entry -> checkable evidence / independent answer -> composition tests -> report

``check_registry`` verifies that every implementation entry resolves, every referenced test exists, and every report
field is one the unified entry writes (``tests/test_composition_rules.py``).  The expanded declaration of
``measure.run`` lists the rule ids and the registry digest, so a report names the rule set it was produced under.

Families: M (memory effects / provenance), T (trace non-interference), P (precision / region evaluation), Q (the bias
query), K (packaging).  A rule whose obligation is not met yields no guarantee and the report says which obligation
failed; it never falls back to a weaker reading silently.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import re
from pathlib import Path

T_SE = "tests/test_storage_effects.py"
T_B1 = "tests/test_batch1_guarantees.py"
T_AF = "tests/test_audit_findings.py"
T_Q = "tests/test_bias_query.py"

RULES = [
    {"id": "M1", "family": "memory effects", "name": "storage identity and lifetime",
     "obligation": "an address names one storage instance only while it lives: every provenance decision is made on "
                   "storages held alive by the analysis and matched by address and StorageImpl identity",
     "entry": ["kernel_analyzer.reference_eval.storage_effects:InputSnapshot",
               "kernel_analyzer.reference_eval.storage_effects:StorageMap.bind",
               "kernel_analyzer.reference_eval.storage_effects:StorageWatch.hold"],
     "independent_answer": "StorageImpl identity (_cdata) of the held storage versus the captured argument",
     "tests": [f"{T_SE}::test_declared_read_identity_and_coverage",
               f"{T_SE}::test_clean_storages_are_held_alive_and_released_when_not_clean"],
     "report": ["provenance_seed0.reads"]},
    {"id": "M2", "family": "memory effects", "name": "exact coverage",
     "obligation": "a write through a view covers exactly the bytes of its elements (offset, strides, overlap, "
                   "stride 0); a read covers the elements the reference loaded from initial values; a hull write joins",
     "entry": ["kernel_analyzer.reference_eval.storage_effects:byte_runs",
               "kernel_analyzer.reference_eval.storage_effects:read_runs_of",
               "kernel_analyzer.reference_eval.ttir_eval:KernelReferenceEvaluator._note_initial_read"],
     "independent_answer": "the bytes a real torch write of 0xFF through the same view changes",
     "tests": [f"{T_SE}::test_byte_runs_equal_the_bytes_a_real_write_changes",
               f"{T_SE}::test_overlapping_full_size_view_is_not_full_coverage",
               f"{T_B1}::test_partial_write_through_an_overlapping_full_size_view",
               f"{T_B1}::test_copies_and_exact_constants_keep_the_call_level_reference"],
     "report": ["outputs.*.reference.reference_scope"]},
    {"id": "M3", "family": "memory effects", "name": "content is not provenance (declared-input read)",
     "obligation": "a declared-input read needs: bytes read inside the declared input tensors, equal to that storage's "
                   "own bytes before the call, and no write evidence (unannounced version increments, ATen writes in "
                   "the aligned producer trace)",
     "entry": ["kernel_analyzer.reference_eval.storage_effects:InputSnapshot.declared_read",
               "kernel_analyzer.reference_eval.storage_effects:StorageWatch.count",
               "kernel_analyzer.check:torch_intermediates"],
     "independent_answer": "the bytes of the storage itself before the call; real version counters",
     "tests": [f"{T_SE}::test_declared_read_uses_the_storages_own_bytes",
               f"{T_SE}::test_storage_watch_counts_aten_writes_through_views_and_detach",
               f"{T_B1}::test_cross_input_digest_match_is_not_a_declared_input",
               f"{T_B1}::test_inplace_arithmetic_with_unchanged_bytes_on_an_input",
               f"{T_B1}::test_inplace_arithmetic_through_data_is_seen_by_the_producer_trace",
               f"{T_B1}::test_declared_inputs_and_legal_aliases_stay_declared"],
     "report": ["provenance_seed0.reads", "outputs.*.reference.reference_scope"]},
    {"id": "M4", "family": "memory effects", "name": "cross-launch carry-over",
     "obligation": "the reference memory of a recorded launch carries over only with unchanged bytes and no write "
                   "evidence in between; announced raw-pointer writes (AOTAutograd) are not ATen writes",
     "entry": ["kernel_analyzer.reference_eval.storage_effects:carry_over",
               "kernel_analyzer.reference_eval.ttir_eval:evaluate_sequence",
               "kernel_analyzer.reference_eval.storage_effects:StorageWatch.install"],
     "independent_answer": "real version counters around torch.autograd.graph.increment_version",
     "tests": [f"{T_SE}::test_carry_over_needs_bytes_and_agreeing_counts",
               f"{T_SE}::test_storage_watch_excludes_announced_raw_pointer_writes",
               f"{T_B1}::test_inplace_arithmetic_between_recorded_launches",
               f"{T_B1}::test_announced_compiled_inplace_write_keeps_the_composition"],
     "report": ["provenance_seed0.carry_overs", "provenance_seed0.external_reentries"]},
    {"id": "M5", "family": "memory effects", "name": "producer lattice",
     "obligation": "input < const < copy are clean, computed / triton / unlabeled are not; a read joins the labels of "
                   "the bytes it reads; an exact write replaces, a hull write joins",
     "entry": ["kernel_analyzer.reference_eval.storage_effects:join",
               "kernel_analyzer.reference_eval.storage_effects:StorageMap.write",
               "kernel_analyzer.provenance:producer_records"],
     "independent_answer": "a per-byte model of the same lattice",
     "tests": [f"{T_SE}::test_join_is_commutative_associative_idempotent_on_kinds",
               f"{T_SE}::test_storage_map_matches_a_per_byte_model",
               f"{T_SE}::test_partial_constant_write_over_uninitialised_bytes_stays_computed"],
     "report": ["outputs.*.upstream_with_producer_record", "outputs.*.mixed_non_triton_sources"]},
    {"id": "T1", "family": "trace non-interference", "name": "the traced run does not change the measured runs",
     "obligation": "the unmeasured producer trace leaves compiled code and its caches untouched (force_eager stance); "
                   "without that API no trace is made",
     "entry": ["kernel_analyzer.provenance:producer_records"],
     "independent_answer": "the recorded launches of every later seed (the compiled kernel is still launched)",
     "tests": [f"{T_B1}::test_announced_compiled_inplace_write_keeps_the_composition"],
     "report": ["producer_trace_per_seed", "notes.ir_kinds"]},
    {"id": "P1", "family": "precision", "name": "precision dependence is observed, not inferred",
     "obligation": "a higher working-precision level can change an enclosure only through a precision-dependent rule "
                   "(SumK / DotK with the level's K, prefix sums); the pass counts their calls",
     "entry": ["kernel_analyzer.reference_eval.intervals:PRECISION_DEPENDENT_CALLS",
               "kernel_analyzer.reference_eval.intervals:sum_k", "kernel_analyzer.reference_eval.intervals:icumsum"],
     "independent_answer": "exact rational row sums inside every level's enclosure",
     "tests": [f"{T_B1}::test_three_level_summation_enclosures_against_exact_row_sums"],
     "report": ["precision_dependent_calls", "refinement.levels"]},
    {"id": "P2", "family": "precision", "name": "refinement stop categories",
     "obligation": "the controller refines while an output can still improve and stops only as met, budget exhausted, "
                   "backend limit (float64 endpoints, highest level, no precision-dependent rule), intrinsic set "
                   "width, no numerical enclosure or pass failed; an unchanged pass fraction is not a stop reason",
     "entry": ["kernel_analyzer.measure:run_level", "kernel_analyzer.measure:REFINEMENT_CATEGORIES"],
     "independent_answer": "level-by-level widths of the three-level summation (levels 1 and 2 resolve nothing)",
     "tests": [f"{T_B1}::test_three_level_summation_reaches_level_3_from_the_unified_entry",
               f"{T_B1}::test_controller_does_not_stop_on_an_unchanged_pass_fraction",
               f"{T_B1}::test_controller_stop_categories", f"{T_B1}::test_controller_budget_between_levels",
               f"{T_AF}::test_refinement_reports_a_target_it_cannot_reach"],
     "report": ["refinement.category", "refinement.outcome", "refinement.per_output"]},
    {"id": "P3", "family": "precision", "name": "resolution counts numerical enclosures only",
     "obligation": "the requested resolution is measured on complete finite elements; set targets (L_E) are counted "
                   "apart with their own width and never reported as an enclosure that is too wide",
     "entry": ["kernel_analyzer.measure:reference_quality"],
     "independent_answer": "per-element set-target masks of the reference (Buffer.iset)",
     "tests": [f"{T_B1}::test_controller_stop_categories"],
     "report": ["outputs.*.reference.set_target_elements", "outputs.*.reference.unresolved_elements"]},
    {"id": "Q1", "family": "bias query", "name": "the query is fixed before the data",
     "obligation": "comparison target, input distribution, observable and sampling unit are written into the expanded "
                   "declaration (and its digest) before any unit is drawn",
     "entry": ["kernel_analyzer.measure:query_of", "kernel_analyzer.measure:expand"],
     "independent_answer": "the declaration digest changes with each of the four items",
     "tests": [f"{T_Q}::test_query_fixes_target_distribution_observable_and_unit"],
     "report": ["expanded_declaration.query"]},
    {"id": "Q2", "family": "bias query", "name": "strict bounded route with a family guarantee",
     "obligation": "with a declared per-element bound M, every rule gets Hoeffding's interval and p-value on endpoints "
                   "truncated to [-M, M]; Holm within each declared rule class gives FWER <= alpha per class for the "
                   "bounded route, apart from the approximate route",
     "entry": ["kernel_analyzer.reference_eval.sensitivity:bounded_route", "kernel_analyzer.measure:class_statistics"],
     "independent_answer": "Hoeffding's tail bound evaluated directly; simulated family-wise error under the null",
     "tests": [f"{T_Q}::test_bounded_p_value_matches_the_hoeffding_interval",
               f"{T_Q}::test_bounded_route_family_wise_error_under_the_null",
               f"{T_Q}::test_bounded_route_holm_within_class"],
     "report": ["outputs.*.statistics.*.axes.nonzero.bounded"]},
    {"id": "Q3", "family": "bias query", "name": "two decision axes",
     "obligation": "the nonzero axis and the equivalence axis are reported apart; an equivalence tolerance delta is "
                   "required only when the equivalence axis is requested, and is then part of the declaration",
     "entry": ["kernel_analyzer.measure:missing_items", "kernel_analyzer.measure:class_statistics"],
     "independent_answer": "declarations with and without the equivalence axis",
     "tests": [f"{T_Q}::test_equivalence_axis_requires_delta_only_when_requested",
               f"{T_Q}::test_axes_are_reported_apart"],
     "report": ["outputs.*.statistics.*.axes.nonzero", "outputs.*.statistics.*.axes.equivalence"]},
    {"id": "K1", "family": "packaging", "name": "no repository path at run time",
     "obligation": "the installed package imports and opens no path of the source repository; console entry points "
                   "run the unified entry and the captured-package analysis",
     "entry": ["kernel_analyzer.cli:measure_main", "kernel_analyzer.cli:analysis_main",
               "kernel_analyzer.contract_v3:statistical_judgment",
               "kernel_analyzer.reference_eval.emulate:default_cache_dir"],
     "independent_answer": "an audit hook on open / import in a process that runs a copy of the package",
     "tests": [f"{T_B1}::test_package_runs_from_a_copy_outside_the_repository"],
     "report": []},
]


def registry_digest() -> str:
    return hashlib.sha256(json.dumps(RULES, sort_keys=True).encode()).hexdigest()


def summary() -> dict:
    """Rule ids, names and the digest (for the expanded declaration)."""
    return {"rules": {r["id"]: r["name"] for r in RULES}, "registry_sha256": registry_digest(),
            "module": "kernel_analyzer.composition_rules"}


def _resolve(entry: str):
    mod, _, attr = entry.partition(":")
    obj = importlib.import_module(mod)
    for part in attr.split("."):
        obj = getattr(obj, part)
    return obj


def check_registry(tool_root: Path) -> list:
    """Problems with the registry: entries that do not resolve, tests that do not exist (file and function)."""
    problems = []
    ids = [r["id"] for r in RULES]
    if len(ids) != len(set(ids)):
        problems.append("duplicate rule ids")
    for r in RULES:
        for e in r["entry"]:
            try:
                _resolve(e)
            except Exception as exc:  # noqa: BLE001
                problems.append(f"{r['id']}: entry {e} does not resolve ({type(exc).__name__})")
        for t in r["tests"]:
            path, _, fn = t.partition("::")
            p = Path(tool_root) / path
            if not p.is_file():
                problems.append(f"{r['id']}: test file {path} missing")
            elif not re.search(rf"^def {re.escape(fn)}\(", p.read_text(), re.M):
                problems.append(f"{r['id']}: test {t} missing")
    return problems
