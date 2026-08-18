import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from kernel_analyzer import AnalysisSpec, AnalysisState, Analyzer, StepExecution
from kernel_analyzer.api import ReferenceAnalysis, ReferenceProvider
from kernel_analyzer.backends import CandidateRuntime, NumericalCandidateBackend
from kernel_analyzer.cli import verify_run
from kernel_analyzer.semantics import SemanticRegistry, SemanticRule
from kernel_analyzer.statistics import coherence_certificate


class Reference(ReferenceProvider):
    def analyze(self, spec, run_dir):
        return ReferenceAnalysis(
            subject=spec.subject,
            proof_units=[{"unit_id": "u-positive"}, {"unit_id": "u-negative"}],
            census={"execution_invocations": 4, "primary_fb_proof_units": 2},
        )


class Runtime(CandidateRuntime):
    def runtime_census(self, context):
        return [{"region_id": "r0"}, {"region_id": "r1"}]

    def local_observations(self, context, proof_units):
        contrast = {
            "kind": "OPTIMIZATION_SAME_DTYPE",
            "candidate_dtype": "bf16",
            "reference_dtype": "bf16",
            "semantic_boundary_exact": True,
            "candidate_program_sha256": "a" * 64,
            "reference_program_sha256": "b" * 64,
        }
        direction = {
            "full_coordinates": True,
            "independent_states": True,
            "cluster_bootstrap_lower_95": 0.1,
        }
        return {
            "u-positive": {"mapped": True, "finite": True, "repeat_stable": True,
                           "max_abs": 0.25, "natural": True,
                           "contrast": contrast, "local_direction": direction},
            "u-negative": {"mapped": True, "finite": True, "repeat_stable": True,
                           "max_abs": 0.25, "natural": True,
                           "contrast": contrast, "local_direction": direction},
        }

    def causal_replacements(self, context, proof_unit_ids):
        return {unit_id: {"replacement_exact": True, "sham_exact": True,
                          "parameter_reached": True, "delta_norm": 1.0,
                          "non_target_endpoints_exact": True,
                          "natural": True} for unit_id in proof_unit_ids}

    def confirmation_vectors(self, context, proof_unit_ids):
        rows = {}
        if "u-positive" in proof_unit_ids:
            rows["u-positive"] = {
                "vectors": [[1.0, 0.1], [0.9, 0.1], [1.1, 0.1], [1.0, 0.2],
                            [0.8, 0.1], [1.2, 0.1]],
                "complete_coordinates": True, "independent_states": True,
                "repeat_exact": True, "natural": True,
                "state_ids": ["c0", "c1", "c2", "c3", "c4", "c5"],
                "pilot_state_ids": ["p0"],
            }
        if "u-negative" in proof_unit_ids:
            rows["u-negative"] = {
                "vectors": [[1.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [-1.0, 0.0],
                            [1.0, 0.0], [-1.0, 0.0]],
                "complete_coordinates": True, "independent_states": True,
                "repeat_exact": True, "natural": True,
                "state_ids": ["c0", "c1", "c2", "c3", "c4", "c5"],
                "pilot_state_ids": ["p0"],
            }
        return rows

    def paired_trajectories(self, context, proof_unit_ids):
        return {unit_id: {
            "same_weight_contrast": True, "nonzero_gradient_contrast": True,
            "live_weight_divergence": True, "frozen_paired_trajectory": True,
            "natural": True, "state_ids": ["t0"],
        } for unit_id in proof_unit_ids}


class PrecisionRuntime(Runtime):
    def local_observations(self, context, proof_units):
        rows = super().local_observations(context, proof_units)
        for row in rows.values():
            row["contrast"] = {
                "kind": "PRECISION_SAME_SEMANTICS",
                "low_dtype": "bf16",
                "high_dtype": "float32",
                "semantic_boundary_exact": True,
                "semantic_program_sha256": "c" * 64,
                "low_arm_program_sha256": "d" * 64,
                "high_arm_program_sha256": "e" * 64,
            }
        return rows


class TotalOnlyRuntime(PrecisionRuntime):
    def local_observations(self, context, proof_units):
        rows = super().local_observations(context, proof_units)
        for row in rows.values():
            row["contrast"]["kind"] = "TOTAL_CANDIDATE_LOW_MINUS_REFERENCE_HIGH"
        return rows


def test_semantic_registry_has_no_name_or_family_fallback():
    registry = SemanticRegistry()
    registry.register(SemanticRule(
        overload="aten.neg.default", rule_id="neg", forward_map="y=-x",
        vjp_map="dx=-q", finite_arithmetic="sign-bit flip",
        error_relation="candidate-reference", assumptions=("floating tensor",),
    ))
    assert registry.resolve("aten.neg.default") is not None
    missing = registry.instantiate("aten.neg.out", [])
    assert missing["status"] == "UNRESOLVED_MISSING_EXACT_SEMANTIC_RULE"
    assert missing["name_or_family_fallback_used"] is False


def test_formula_witness_requires_concrete_forward_backward_program_proof():
    registry = SemanticRegistry()
    registry.register(SemanticRule(
        overload="aten.neg.default", rule_id="neg", forward_map="y=-x",
        vjp_map="dx=-q", finite_arithmetic="sign-bit flip",
        error_relation="candidate-reference", assumptions=("floating tensor",),
        witness=lambda: {"status": "PASS_EXECUTABLE_FORMULA"},
    ))
    formula_only = registry.instantiate("aten.neg.default", [])
    assert formula_only["status"] == "FORMULA_REGISTERED_EXECUTABLE_WITNESS_ONLY"
    assert formula_only["analytic_proof_status"] == "UNRESOLVED_NO_CONCRETE_BACKWARD_PROGRAM_PROOF"
    concrete = registry.instantiate("aten.neg.default", [], concrete_program_proof={
        "saved_tensor_origins_exact": True,
        "cotangent_edge_exact": True,
        "backward_program_matches_analytic_vjp": True,
        "non_tensor_arguments_exact": True,
        "output_edges_exact": True,
        "forward_program_sha256": "1" * 64,
        "backward_program_sha256": "2" * 64,
        "analytic_derivation_sha256": "3" * 64,
    })
    assert concrete["status"] == "INSTANTIATED_CERTIFIED_EXACT_SEMANTIC_RULE"
    assert concrete["analytic_proof_status"] == "ANALYTICALLY_PROVED"


def test_coherence_statistic_separates_common_and_sign_changing_directions():
    positive = coherence_certificate([[1.0], [0.9], [1.1], [0.8], [1.2], [1.0]],
                                     bootstrap_samples=400, seed=3)
    changing = coherence_certificate([[1.0], [-1.0], [1.0], [-1.0], [1.0], [-1.0]],
                                     bootstrap_samples=400, seed=3)
    assert positive["status"] == "PASS"
    assert changing["status"] == "FAIL_CAUSAL_NONCOHERENT"


def test_t1_accepts_attributable_precision_but_not_total_only(tmp_path):
    spec = AnalysisSpec(
        subject="contrast-types",
        reference=Reference(),
        candidates=[
            NumericalCandidateBackend("precision", PrecisionRuntime(), bootstrap_samples=100),
            NumericalCandidateBackend("total", TotalOnlyRuntime(), bootstrap_samples=100),
        ],
        states=[AnalysisState("p0", role="DISCOVERY")],
        output_dir=tmp_path,
    )
    report = Analyzer().analyze(spec)
    assert report.candidate_summaries["precision"]["T1_pass"] == 2
    assert report.candidate_summaries["total"]["T1_pass"] == 0
    assert report.candidate_summaries["total"]["T1_unresolved"] == 2
    import gzip
    with gzip.open(report.artifact_dir / "candidates/precision.json.gz", "rt") as handle:
        rows = json.load(handle)["certificates"]
    assert all(row["tiers"]["T1_LOCAL"]["evidence"]["cause_axis"] == "PRECISION" for row in rows)


def test_analyzer_enforces_tier_order_and_writes_verifiable_run(tmp_path):
    spec = AnalysisSpec(
        subject="toy",
        reference=Reference(),
        candidates=[NumericalCandidateBackend("candidate", Runtime(), bootstrap_samples=400)],
        states=[
            AnalysisState("p0", role="DISCOVERY"),
            *[AnalysisState("c%d" % index, role="CONFIRMATION") for index in range(6)],
            AnalysisState("t0", role="TRAJECTORY"),
        ],
        output_dir=tmp_path,
    )
    report = Analyzer().analyze(spec)
    assert len(report.case_certificates) == 1
    assert report.case_certificates[0].proof_unit_id == "u-positive"
    candidate = json.loads((report.artifact_dir / "report.json").read_text())[
        "candidate_summaries"]["candidate"]
    assert candidate["T1_accounted"] == candidate["T1_tested"] == 2
    assert candidate["T1_pass"] == candidate["T2_pass"] == 2
    assert candidate["T3_pass"] == candidate["T4_pass"] == 1
    assert candidate["complete_cases"] == 1
    assert candidate["pipeline_complete"] is True
    assert all(candidate[key] == 0 for key in (
        "T1_unresolved", "T2_unresolved", "T3_unresolved", "T4_unresolved",
    ))
    assert verify_run(report.artifact_dir)["status"] == "VALID"
    assert (report.artifact_dir / "mathematics.md").exists()
    assert len(list((report.artifact_dir / "cases").glob("*.md"))) == 1
    assert Analyzer().analyze(spec, resume=True).run_id == report.run_id


def test_qwen_vertical_regression_preserves_strict_positive_and_negative_controls(tmp_path):
    from examples.qwen_retained_spec import build_spec

    report = Analyzer().analyze(replace(build_spec(), output_dir=tmp_path, resources=None))
    assert report.status == "PARTIAL_FAIL_CLOSED"
    assert report.proof_unit_count == 3459
    assert {row.proof_unit_id for row in report.case_certificates} == {
        "retained-case::seq128_lm_head_input_vjp_mm",
        "retained-case::liger_fused_linear_ce_dw",
        "retained-case::phi4_seq64_lmhead_dx_mm",
            "retained-case::layer23_qproj_attention_state_region",
            "retained-case::mamba64_layer0_input_proj_output",
            "retained-case::qwen128_layer27_softmax_saved_state",
        }
    assert report.candidate_summaries["flash_paper_positive_control"]["positive_controls"] == 1
    screen = report.candidate_summaries["qwen_bf16_inductor_changed_regions"]
    assert screen["total_fb_units"] == 3459
    assert screen["T1_pass"] == 0
    assert screen["T1_unresolved"] == 3459
    assert screen["T2_pass"] == 0
    assert screen["T2_unresolved"] == 0
    assert screen["T3_pass"] == 0
    assert screen["T3_unresolved"] == 0
    assert screen["pipeline_complete"] is False
    candidate_path = report.artifact_dir / "candidates/retained_qwen_candidates.json.gz"
    import gzip
    with gzip.open(candidate_path, "rt") as handle:
        rows = json.load(handle)["certificates"]
    lmhead = next(row for row in rows if row["proof_unit_id"].endswith("seq128_lm_head_input_vjp_mm"))
    # The retained adapter uses the trajectory-local Flash-style track.  The
    # separate cross-state negative remains in the audit artifact.
    assert lmhead["classification"] == "COMPLETE_DIRECTIONAL_ACCUMULATION_CASE"
    assert lmhead["tiers"]["T3_COHERENT"]["status"] == "PASS"
    assert lmhead["tiers"]["T4_ACCUMULATION"]["status"] == "PASS"


def test_real_operator_capture_does_not_promote_formula_to_program_proof(tmp_path):
    from kernel_analyzer.providers import EagerReferenceProvider

    root = Path(__file__).resolve().parents[1]
    registry = SemanticRegistry()
    registry.load_catalog(
        root / "results/coverage/qwen_invocation_ledger.json.gz",
        root / "archive/round1_raw/training_semantic_oracle/qwen3_1p7b/full_step_inventory/eager_bf16_invocation_derivation_witness_v3.json",
    )

    class NegSum(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.arange(6.0).reshape(2, 3))

        def forward(self):
            return torch.sum(-self.weight, dim=[0, 1], keepdim=False)

    class UnusedCandidate:
        candidate_id = "unused"

    def build_step(model, state):
        return StepExecution(
            loss_closure=model,
            endpoint_closure=lambda: {"weight": model.weight.grad},
        )

    spec = AnalysisSpec(
        subject="neg-sum-operator",
        reference=EagerReferenceProvider(registry),
        candidates=[UnusedCandidate()],
        states=[AnalysisState("natural-0")],
        output_dir=tmp_path,
        model_factory=NegSum,
        step_builder=build_step,
    )
    result = spec.reference.analyze(spec, tmp_path)
    assert result.census["execution_invocations"] == 8
    assert result.census["primary_fb_proof_units"] == 2
    assert result.census["auxiliary_backward_invocations"] == 2
    assert len(result.unresolved) == result.census["primary_fb_proof_units"]
    assert all(
        unit["status"] == "ORIGIN_BOUND_FORMULA_REGISTERED_ONLY"
        for unit in result.proof_units
    )
