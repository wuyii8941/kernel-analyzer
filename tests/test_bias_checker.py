import pytest
import torch

from kernel_analyzer import check_bias


def _inputs(index):
    # The input factory is deterministic and deliberately returns independent
    # values by index; check_bias still treats the declared generator as the
    # source of any population interpretation.
    return (torch.linspace(-1.0, 1.0, 8) + 0.03 * index,)


def test_check_bias_reports_not_confirmed_for_identity_without_using_allclose():
    report = check_bias(lambda x: x * x, lambda x: x * x, _inputs, samples=8)

    assert report["status"] == "SYSTEMATIC_BIAS_NOT_CONFIRMED"
    assert report["measurement_status"] == "VALID"
    output = report["stages"]["OUTPUT"]
    assert output["decision"] == "SYSTEMATIC_BIAS_NOT_CONFIRMED"
    assert output["total_rms"] == 0.0
    assert output["allclose_auxiliary"]["allclose"] is True


def test_check_bias_pairs_torch_randomness_between_candidate_and_reference():
    report = check_bias(
        lambda x: x + torch.rand_like(x),
        lambda x: x + torch.rand_like(x),
        _inputs,
        samples=8,
        preserve_rng=True,
    )

    assert report["status"] == "SYSTEMATIC_BIAS_NOT_CONFIRMED"
    assert report["output"]["total_rms"] == 0.0


def test_check_bias_detects_aligned_scaling_even_when_mean_direction_cancels():
    report = check_bias(
        lambda x: 1.02 * x,
        lambda x: x,
        lambda index: (torch.linspace(-1.0, 1.0, 16) + 0.01 * ((index % 2) - 0.5),),
        samples=12,
    )

    output = report["stages"]["OUTPUT"]
    assert report["status"] == "SYSTEMATIC_BIAS_CONFIRMED"
    assert output["aligned_ratio_of_sums"] == pytest.approx(0.02, rel=1e-5)
    assert "STATEWISE_ALIGNED_GAIN_NONZERO" in output["systematic_bias_reasons"]
    assert output["allclose_auxiliary"]["allclose"] is False


def test_check_bias_detects_a_directional_additive_effect():
    report = check_bias(
        lambda x: x + 0.25,
        lambda x: x,
        lambda index: (torch.linspace(-1.0, 1.0, 8) + 0.02 * index,),
        samples=12,
    )

    output = report["stages"]["OUTPUT"]
    assert report["status"] == "SYSTEMATIC_BIAS_CONFIRMED"
    assert "HELD_OUT_DIRECTIONAL_MEAN_NONZERO" in output["systematic_bias_reasons"]
    assert output["direction"]["status"] == "VALID"


def test_zero_directional_projections_are_not_counted_as_negative_bias():
    def candidate(x):
        # Calibration samples carry a positive effect; confirmation samples
        # are exact zeros.  Zeros must not become evidence for the opposite
        # direction.
        return x + (0.25 if float(x[0]) < -0.9 else 0.0)

    report = check_bias(candidate, lambda x: x, _inputs, samples=8)

    assert report["status"] == "SYSTEMATIC_BIAS_NOT_CONFIRMED"
    sign = report["output"]["direction"]["sign_prevalence"]
    assert sign["decision"] == "NOT_IDENTIFIABLE_ZERO_EFFECTS"
    assert sign["confirmation_zero_projection_count"] == 4


def test_check_bias_can_measure_backward_and_keeps_it_separate_from_output():
    report = check_bias(
        lambda x: 1.01 * x * x,
        lambda x: x * x,
        lambda index: (torch.tensor([-1.0, 0.5, 2.0]) + 0.01 * index,),
        samples=8,
        check_backward=True,
    )

    assert report["measurement_status"] == "VALID"
    assert report["stages"]["OUTPUT"]["measurement_status"] == "VALID"
    assert report["stages"]["BACKWARD"]["measurement_status"] == "VALID"
    assert report["stages"]["BACKWARD"]["decision"] == "SYSTEMATIC_BIAS_CONFIRMED"


def test_check_bias_does_not_turn_execution_failure_into_a_negative_result():
    def candidate(x):
        if float(x[0]) > -0.9:
            raise RuntimeError("deliberate input failure")
        return x

    report = check_bias(candidate, lambda x: x, _inputs, samples=8)

    assert report["status"] == "UNRESOLVED_MEASUREMENT"
    assert report["measurement_status"] == "PARTIAL"
    assert report["stages"]["OUTPUT"]["sample_count"] < 8
    assert report["stages"]["OUTPUT"]["errors"]


def test_check_bias_rejects_unusable_sample_configuration():
    with pytest.raises(ValueError):
        check_bias(lambda x: x, lambda x: x, _inputs, samples=3)
