import math

import torch

from kernel_analyzer.seup import (
    SEUPCalibrator,
    SEUPAccumulator,
    SymmetricSEUPEvaluator,
    adamw_effective_update_delta,
    balanced_sign_schedule,
    flip_carrier_component,
    sgd_effective_update_delta,
)


def assert_raises(error, function):
    try:
        function()
    except error:
        return
    raise AssertionError("expected %s" % error.__name__)


def _certificate(evaluation):
    accumulator = SEUPAccumulator(calibration_steps=2, evaluation_steps=len(evaluation))
    accumulator.add("c0", torch.tensor([1.0, 0.0]))
    accumulator.add("c1", torch.tensor([2.0, 0.0]))
    for index, value in enumerate(evaluation):
        accumulator.add("e%d" % index, torch.tensor(value))
    return accumulator.finalize()


def test_constant_signed_updates_have_unit_capture_and_persistence():
    result = _certificate(((1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0)))
    assert math.isclose(result["carrier_energy_capture"], 1.0)
    assert math.isclose(result["signed_persistence"], 1.0)
    assert math.isclose(result["residual_accumulation_l2"], 0.0)
    assert result["t4_verdict_used_to_fit_carrier"] is False


def test_alternating_updates_cancel_despite_nonzero_local_error():
    result = _certificate(((1.0, 0.0), (-1.0, 0.0), (1.0, 0.0), (-1.0, 0.0)))
    assert math.isclose(result["carrier_energy_capture"], 1.0)
    assert math.isclose(result["signed_persistence"], 0.0)
    assert all(row["update_l2"] > 0.0 for row in result["evaluation_records"])


def test_carrier_and_residual_are_measured_separately():
    result = _certificate(((1.0, 1.0), (1.0, -1.0)))
    assert math.isclose(result["carrier_energy_capture"], 0.5, rel_tol=1e-6)
    assert math.isclose(result["signed_persistence"], 1.0)
    assert math.isclose(result["residual_accumulation_l2"], 0.0, abs_tol=1e-7)


def test_calibration_and_evaluation_windows_are_enforced():
    accumulator = SEUPAccumulator(calibration_steps=2, evaluation_steps=1)
    accumulator.add("c0", torch.tensor([1.0]))
    assert_raises(RuntimeError, accumulator.finalize)
    accumulator.add("c1", torch.tensor([1.0]))
    accumulator.add("e0", torch.tensor([1.0]))
    assert_raises(RuntimeError, lambda: accumulator.add("extra", torch.tensor([1.0])))
    assert accumulator.finalize()["evaluation_steps"] == 1


def test_zero_calibration_mean_and_nonfinite_updates_fail_closed():
    accumulator = SEUPAccumulator(calibration_steps=2, evaluation_steps=1)
    accumulator.add("c0", torch.tensor([1.0]))
    assert_raises(RuntimeError, lambda: accumulator.add("c1", torch.tensor([-1.0])))
    finite = SEUPAccumulator(calibration_steps=1, evaluation_steps=1)
    assert_raises(ValueError, lambda: finite.add("c0", torch.tensor([float("nan")])))


def test_sgd_and_adamw_use_one_common_pre_step_state():
    sgd = sgd_effective_update_delta(
        torch.tensor([2.0]), torch.tensor([1.0]), learning_rate=0.1
    )
    assert torch.allclose(sgd["value"], torch.tensor([-0.1]))

    candidate = torch.tensor([2.0]); repair = torch.tensor([1.0])
    zero = torch.tensor([0.0]); parameter = torch.tensor([3.0])
    adam = adamw_effective_update_delta(
        candidate, repair, zero, zero, parameter,
        step=1, learning_rate=0.01, weight_decay=0.1,
    )
    # At the first scalar Adam step both positive gradients normalise to the
    # same update; the equal pre-step weight decay also cancels.
    assert torch.allclose(adam["value"], torch.zeros(1), atol=1e-7)


def test_carrier_flip_preserves_norm_and_changes_only_projection_sign():
    delta = torch.tensor([3.0, 4.0])
    basis = torch.tensor([1.0, 0.0])
    flipped = flip_carrier_component(delta, basis, -1)["value"]
    assert torch.allclose(flipped, torch.tensor([-3.0, 4.0]))
    assert math.isclose(float(torch.linalg.vector_norm(flipped)), 5.0)
    assert balanced_sign_schedule(16).count(1) == 8
    assert balanced_sign_schedule(16).count(-1) == 8


def test_cross_fit_gate_and_symmetric_recurrence_close_exactly():
    calibrator = SEUPCalibrator(4, gate_cosine=0.5)
    for index, value in enumerate((1.0, 2.0, 1.0, 2.0)):
        calibrator.add("c%d" % index, torch.tensor([value, 0.0]))
    carrier = calibrator.freeze()
    assert carrier.stable
    evaluator = SymmetricSEUPEvaluator(carrier, evaluation_steps=1)
    # Uc(sc)=1, Ur(sc)=0, Uc(sr)=2, Ur(sr)=1.  Hence L=1, B=-1.
    evaluator.add("e0", torch.tensor([1.0, 0.0]), torch.tensor([0.0, 0.0]),
                  torch.tensor([2.0, 0.0]), torch.tensor([1.0, 0.0]),
                  torch.tensor([0.0, 0.0]), torch.tensor([0.0, 0.0]))
    result = evaluator.finalize()
    assert math.isclose(result["max_recurrence_relative_residual"], 0.0)
    assert math.isclose(result["local_projection_accumulation_abs"], 1.0)
    assert math.isclose(result["feedback_projection_accumulation_abs"], 1.0)
    assert math.isclose(result["local_fraction_of_projected_accumulation"], 0.5)


def test_unstable_cross_fit_is_fail_closed_without_a_basis():
    calibrator = SEUPCalibrator(4, gate_cosine=0.5)
    for index, value in enumerate((1.0, -1.0, 1.0, -1.0)):
        calibrator.add("c%d" % index, torch.tensor([value, 0.0]))
    carrier = calibrator.freeze()
    assert not carrier.stable
    assert carrier.basis is None
