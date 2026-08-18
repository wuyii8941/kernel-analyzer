import json
import gzip
import math
from pathlib import Path
import tempfile

from kernel_analyzer import AnalysisSpec, AnalysisState, Analyzer
from kernel_analyzer.api import (
    FAIL,
    PASS,
    AnalysisContext,
    CandidateBackend,
    CandidateCensus,
    ReferenceAnalysis,
    ReferenceProvider,
    TierEvidence,
)
from kernel_analyzer.backends import CandidateRuntime, NumericalCandidateBackend

from kernel_analyzer.property import (
    CaseRole,
    EvidenceStage,
    EvidenceStatus,
    Hypothesis,
    HypothesisEvidence,
    PropertyCase,
    SignedTransportState,
    accumulation_decomposition,
    derive_signed_rounding_error,
    deterministic_conditional_rounding_mean,
    effective_rank,
    round_fp32_to_bf16,
    signed_event_transport,
    signed_transport_certificate,
    validate_predictor_features,
)


ROOT = Path(__file__).resolve().parents[1]


def assert_raises(error, function):
    try:
        function()
    except error:
        return
    raise AssertionError("expected %s" % error.__name__)


def test_deterministic_conditional_centering_is_not_discriminating():
    assert deterministic_conditional_rounding_mean(0.125) == 0.125
    assert deterministic_conditional_rounding_mean(-0.125) == -0.125


def test_zero_mean_scalar_error_can_have_nonzero_carrier_coupling():
    epsilon = (-1.0, 1.0)
    carrier = (-2.0, 2.0)
    assert math.fsum(epsilon) / 2.0 == 0.0
    assert math.fsum(e * r for e, r in zip(epsilon, carrier)) / 2.0 == 2.0


def test_biased_scalar_error_can_cancel_after_transport():
    epsilon = (1.0, 1.0)
    carrier = (-2.0, 2.0)
    assert math.fsum(epsilon) / 2.0 == 1.0
    assert math.fsum(e * r for e, r in zip(epsilon, carrier)) == 0.0


def test_accumulation_decomposition_is_an_identity_not_a_property():
    result = accumulation_decomposition(((1.0, 2.0), (-3.0, 4.0), (2.0, 1.0)))
    assert abs(result["residual"]) < 1e-12
    assert result["total_energy"] == 49.0


def test_effective_rank_handles_rank_one_and_isotropic_spectra():
    assert effective_rank((4.0, 0.0, 0.0)) == 1.0
    assert effective_rank((1.0, 1.0, 1.0)) == 3.0


def test_predictor_rejects_candidate_and_identity_leakage():
    assert_raises(
        ValueError,
        lambda: validate_predictor_features({"nested": {"candidate_output": [1.0]}}),
    )
    assert_raises(ValueError, lambda: validate_predictor_features({"model_name": "qwen"}))
    assert_raises(ValueError, lambda: validate_predictor_features({"candidate_id": "x"}))
    assert_raises(ValueError, lambda: validate_predictor_features({"t4_verdict": "PASS"}))
    assert_raises(ValueError, lambda: validate_predictor_features({"t3_carrier_l2": 1.0}))
    assert_raises(ValueError, lambda: validate_predictor_features({"operator_family": "mm"}))
    validate_predictor_features({
        "reduction_extent": 128,
        "reference_vjp_gain": 2.0,
        "declared_accumulation_dtype": "fp32",
    })


def test_semantic_region_cannot_count_as_root_positive():
    evidence = HypothesisEvidence(
        Hypothesis.CARRIER_GEOMETRY, EvidenceStatus.UNRESOLVED,
        stage=EvidenceStage.PREDICTOR,
    )
    assert_raises(
        ValueError,
        lambda: PropertyCase(
            case_id="region", role=CaseRole.COHERENT_FB_BIAS,
            mechanism_level="CLOSED_SEMANTIC_REGION",
            arithmetic_mechanism="unresolved", flash_style_verdict="PASS",
            evidence=(evidence,),
        ),
    )


def test_bf16_rounding_source_is_schedule_derived_and_ties_to_even():
    assert round_fp32_to_bf16(1.0 + 2.0 ** -8) == 1.0
    assert round_fp32_to_bf16(1.0 + 3.0 * 2.0 ** -8) == 1.0 + 2.0 ** -6
    assert derive_signed_rounding_error(1.001, "bf16") < 0.0
    assert derive_signed_rounding_error(1.001, "fp32") == 0.0
    assert_raises(ValueError, lambda: derive_signed_rounding_error(1.0, "fp8"))


def test_signed_event_transport_matches_flash_style_factor_sum():
    result = signed_event_transport((2.0, -1.0), ((1.0, 3.0), (4.0, 5.0)))
    assert result == (-2.0, 1.0)


def test_signed_transport_certificate_requires_direction_amplitude_and_remainder():
    coherent = tuple(
        SignedTransportState(str(index), (1.0, 0.0)) for index in range(4)
    )
    passed = signed_transport_certificate(
        coherent, reference_margin=0.5, bootstrap_samples=200, seed=7
    )
    assert passed["status"] == "PREDICTED_COHERENT_F_B_BIAS"
    assert passed["amplitude"] == 1.0
    assert passed["directional_energy"] == 1.0
    assert passed["concentration"] == 1.0
    assert passed["candidate_tensor_values_read"] is False

    bounded_away = tuple(
        SignedTransportState(str(index), (1.0, 0.0), 1.0) for index in range(4)
    )
    rejected = signed_transport_certificate(
        bounded_away, reference_margin=0.0, bootstrap_samples=200, seed=7
    )
    assert rejected["status"] == "NO_PREDICTED_COHERENT_F_B_BIAS"
    assert rejected["certified_mean_magnitude"] == 0.0

    insufficient = signed_transport_certificate(
        coherent[:3], reference_margin=0.0, bootstrap_samples=20
    )
    assert insufficient["status"] == "ABSTAIN_UNRESOLVED_COHERENCE"


def test_property_matrix_targets_t3_and_preserves_every_endpoint():
    result = json.loads((ROOT / "results/property/hypothesis_matrix.json").read_text())
    assert result["status"] == "NO_REFERENCE_ONLY_PROPERTY_CLAIM_YET"
    assert result["target"]["verdict_layer"] == \
        "T3_COHERENT_COMPLETE_PARAMETER_GRADIENT_CARRIER"
    assert result["target"]["t4_used_as_label_or_predictor"] is False
    assert result["population"]["endpoint_count"] == 1562
    assert result["population"]["role_counts"] == {
        "COHERENT_F_B_BIAS": 57,
        "NORMAL_REFERENCE": 588,
        "UNRESOLVED": 917,
    }
    assert result["population"]["representative_sampling"] is False
    assert len(result["rows"]) == 1562
    assert all("t4_artifact" not in row for row in result["rows"])
    assert result["hypothesis"]["id"] == "SIGNED_TRANSPORT_COHERENCE"
    assert result["hypothesis"]["claim_status"].startswith("UNRESOLVED_")

    with gzip.open(ROOT / "results/property/signed_transport_queue.json.gz", "rt") as handle:
        queue = json.load(handle)
    assert queue["endpoint_count"] == 1562
    assert queue["all_endpoints_queued"] is True
    assert queue["representative_sampling"] is False
    assert len(queue["rows"]) == 1562


class _PropertyReference(ReferenceProvider):
    def analyze(self, spec, run_dir):
        return ReferenceAnalysis(
            subject=spec.subject,
            proof_units=[{"unit_id": "u0"}],
            census={"primary_fb_proof_units": 1},
        )


class _PropertyBackend(CandidateBackend):
    candidate_id = "property-test"

    def census(self, context: AnalysisContext):
        return CandidateCensus(
            candidate_id=self.candidate_id, runtime_regions=(),
            status="CAPTURED_EXECUTION_DERIVED",
        )

    def predict_signed_transport(self, context, proof_units):
        return {"u0": TierEvidence(status=PASS, evidence={
            "predictor_inputs": {
                "declared_accumulation_dtype": "bf16",
                "reference_vjp_gain": 1.0,
            },
        })}

    def measure_local(self, context, proof_units):
        return {"u0": TierEvidence(status=FAIL)}

    def intervene_causally(self, context, proof_unit_ids):
        return {}

    def confirm_coherence(self, context, proof_unit_ids):
        return {}

    def run_trajectory(self, context, proof_unit_ids):
        return {}


def test_analyzer_runs_property_before_observed_tiers_and_keeps_full_denominator():
    with tempfile.TemporaryDirectory(dir="/tmp") as temporary:
        spec = AnalysisSpec(
            subject="property-runner",
            reference=_PropertyReference(), candidates=[_PropertyBackend()],
            states=[AnalysisState("s0")], output_dir=Path(temporary),
        )
        report = Analyzer().analyze(spec)
        summary = report.candidate_summaries["property-test"]
        assert summary["PROPERTY_accounted"] == 1
        assert summary["PROPERTY_predicted_coherent"] == 1
        assert summary["PROPERTY_unresolved"] == 0
        stage = json.loads((
            report.artifact_dir / "stages/property-test/PROPERTY_SIGNED_TRANSPORT.json"
        ).read_text())
        assert stage["input_unit_ids"] == ["u0"]
        assert stage["candidate_tensor_values_read"] is False
        assert stage["t4_used_as_label_or_predictor"] is False


class _FactorRuntime(CandidateRuntime):
    def runtime_census(self, context):
        return ()

    def signed_transport_factors(self, context, proof_units):
        return {"u0": {
            "predictor_inputs": {
                "declared_accumulation_dtype": "bf16",
                "reference_operand_digest": "a" * 64,
            },
            "reference_margin": 0.25,
            "states": [
                {
                    "state_id": str(index),
                    "event_errors": [1.0],
                    "reference_transport_directions": [[1.0, 0.0]],
                    "nonlinear_remainder_bound": 0.0,
                }
                for index in range(4)
            ],
        }}

    def local_observations(self, context, proof_units):
        return {}

    def causal_replacements(self, context, proof_unit_ids):
        return {}

    def confirmation_vectors(self, context, proof_unit_ids):
        return {}

    def paired_trajectories(self, context, proof_unit_ids):
        return {}


def test_generic_backend_builds_property_from_event_factors_without_candidate_values():
    backend = NumericalCandidateBackend(
        "factor", _FactorRuntime(), bootstrap_samples=200
    )
    result = backend.predict_signed_transport(None, [{"unit_id": "u0"}])
    assert result["u0"].status == PASS
    evidence = result["u0"].evidence
    assert evidence["candidate_tensor_values_read"] is False
    assert evidence["t4_used_as_label_or_predictor"] is False
    assert evidence["certificate"]["status"] == "PREDICTED_COHERENT_F_B_BIAS"
