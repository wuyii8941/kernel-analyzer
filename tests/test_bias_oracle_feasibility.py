import itertools
import math

import pytest

from kernel_analyzer.bias_oracle_feasibility import (
    moment_response_sketch,
    paired_response_decomposition,
    quadratic_bias_decomposition,
    shared_block_hvp_sketch,
    subset_square_matrix,
)


def test_quadratic_bias_decomposition_separates_two_channels():
    # F(e) = [2 e0 - e1 + 3/2 e0^2 + e1^2].
    result = quadratic_bias_decomposition(
        source_mean=[0.25, -0.5],
        source_second_moment=[[2.0, 0.0], [0.0, 3.0]],
        jacobian=[[2.0, -1.0]],
        output_hessians=[[[3.0, 0.0], [0.0, 2.0]]],
    )
    assert result.transported_mean == pytest.approx((1.0,))
    assert result.curvature_rectification == pytest.approx((6.0,))
    assert result.predicted_bias == pytest.approx((7.0,))


def test_paired_response_identity_is_exact_for_mixed_response():
    def response(x):
        return [2.0 * x[0] + 3.0 * x[0] * x[0]]

    result = paired_response_decomposition([[1.0], [-0.5], [0.25]], response)
    assert result.closure_relative_error < 1e-14
    assert result.natural_mean_response == pytest.approx(
        tuple(a + b for a, b in zip(result.odd_mean_response, result.even_mean_response))
    )
    assert result.response_evaluations == 7


def test_moment_sketch_recovers_linear_and_quadratic_channels():
    def response(x):
        return [2.0 * x[0] + 3.0 * x[0] * x[0] + x[1] * x[1]]

    # L L^T = diag(4, 9). The raw second moment also includes mean^2=.25,
    # so 1/2 H:E[ee^T] = 3*(4+.25) + 1*9 = 21.75.
    result = moment_response_sketch(
        source_mean=[0.5, 0.0],
        covariance_factor=[[2.0, 0.0], [0.0, 3.0]],
        response=response,
        curvature_probes=4,
        scale=1.0,
        check_half_scale=True,
    )
    assert result.transported_mean == pytest.approx((1.0,))
    assert result.curvature_rectification == pytest.approx((21.75,))
    assert result.predicted_bias == pytest.approx((22.75,))
    assert result.amplitude_relative_difference < 1e-14
    assert result.status == "SCREEN"
    assert (
        result.baseline_evaluations
        + result.transported_evaluations
        + result.curvature_evaluations
    ) == 21


def test_moment_sketch_escalates_nonsmooth_support_switch():
    def response(x):
        return [max(0.0, abs(x[0]) - 0.6)]

    result = moment_response_sketch(
        source_mean=[0.0],
        covariance_factor=[[1.0]],
        response=response,
        curvature_probes=2,
        scale=1.0,
        check_half_scale=True,
        amplitude_tolerance=0.25,
    )
    assert result.status == "ESCALATE_NONLOCAL_OR_NONSMOOTH_RESPONSE"
    assert result.amplitude_relative_difference > 0.25


def test_subset_square_matrix_keeps_principal_gram():
    matrix = [[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]]
    assert subset_square_matrix(matrix, [2, 0]) == [[6.0, 3.0], [3.0, 1.0]]
    with pytest.raises(ValueError):
        subset_square_matrix(matrix, [0, 0])


def test_one_coded_hvp_population_estimates_all_blocks():
    torch = pytest.importorskip("torch")
    x = torch.zeros(2, dtype=torch.float64, requires_grad=True)
    y = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    # Hessian blocks are A=diag(2, 6), B=[10], plus cross-block coupling.
    response = (
        3.0 * x[0]
        - 2.0 * y[0]
        + x[0] * x[0]
        + 3.0 * x[1] * x[1]
        + 5.0 * y[0] * y[0]
        + 7.0 * x[0] * y[0]
    )
    factors = [torch.eye(2, dtype=torch.float64), torch.ones(1, 1, dtype=torch.float64)]
    codes = []
    for signs in itertools.product((-1.0, 1.0), repeat=3):
        codes.append([list(signs[:2]), [signs[2]]])
    result = shared_block_hvp_sketch(
        response,
        [x, y],
        [torch.tensor([0.5, 0.0]), torch.tensor([0.25])],
        factors,
        probes=len(codes),
        probe_signs=codes,
    )
    assert result.transported_mean_projections == pytest.approx((1.5, -0.5))
    # 1/2 trace(A)=4 and 1/2 trace(B)=5. Cross-block terms cancel over codes.
    assert result.curvature_projections == pytest.approx((4.0, 5.0))
    assert result.shared_forward_evaluations == 1
    assert result.shared_first_backward_passes == 1
    assert result.shared_hvp_passes == 8


def test_shared_hvp_treats_exactly_linear_block_as_zero_curvature():
    torch = pytest.importorskip("torch")
    nonlinear = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    linear = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    response = nonlinear.square().sum() + 3.0 * linear.sum()
    result = shared_block_hvp_sketch(
        response,
        [nonlinear, linear],
        [torch.zeros_like(nonlinear), torch.zeros_like(linear)],
        [torch.ones(1, 1, dtype=torch.float64), torch.ones(1, 1, dtype=torch.float64)],
        probes=2,
    )
    assert result.curvature_projections == pytest.approx((1.0, 0.0))


def test_shared_hvp_handles_an_entirely_linear_response():
    torch = pytest.importorskip("torch")
    local = torch.zeros(2, dtype=torch.float64, requires_grad=True)
    response = 2.0 * local[0] - 3.0 * local[1]
    result = shared_block_hvp_sketch(
        response,
        [local],
        [torch.tensor([0.5, 0.25], dtype=torch.float64)],
        [torch.eye(2, dtype=torch.float64)],
        probes=2,
    )
    assert result.transported_mean_projections == pytest.approx((0.25,))
    assert result.curvature_projections == pytest.approx((0.0,))
