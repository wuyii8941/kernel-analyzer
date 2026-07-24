import numpy as np

from forkcert.relational_oracle import compute_endpoint_oracle, compute_repeatability


def test_endpoint_oracle_is_operator_agnostic():
    row = compute_endpoint_oracle(np.array([1.0, 2.0]), np.array([1.0, 3.0]))
    assert not row.exact_match
    assert row.max_abs_delta == 1.0
    assert row.disagreement_count == 1


def test_endpoint_oracle_fails_closed_on_shape_change():
    row = compute_endpoint_oracle(np.zeros((2,)), np.zeros((3,)))
    assert not row.shape_match
    assert row.nonzero_delta_count == -1


def test_repeatability_separates_repeats_from_cross_implementation_delta():
    row = compute_repeatability([np.array([1.0]), np.array([1.0])])
    assert row["instantiated"]
    assert row["exact_all_repeats"]


def test_nonfinite_endpoint_fails_closed_instead_of_reporting_nan_magnitudes():
    row = compute_endpoint_oracle(np.array([1.0]), np.array([np.nan]))
    assert not row.finite_match
    assert row.max_abs_delta == float("inf")
    assert row.mean_signed_delta is None


def test_nonfinite_repeatability_is_explicitly_invalid():
    row = compute_repeatability([np.array([1.0]), np.array([np.nan])])
    assert row["instantiated"]
    assert not row["finite_all_repeats"]
    assert not row["valid_numeric_variance"]
