from kernel_analyzer.reference_relative_oracle import (
    ReferenceRelativeObservation,
    certify_reference_relative,
)


def row(index, dot, error=1.0, reference=1.0):
    return ReferenceRelativeObservation(str(index), dot, error, reference)


def test_rotating_absolute_directions_keep_same_reference_relative_bias():
    # These sufficient statistics can arise from unrelated orthogonal state
    # updates; the moving-frame coefficient remains -0.01 in every state.
    result = certify_reference_relative([row(i, -0.01) for i in range(8)])
    assert result.status == "REFERENCE_RELATIVE_DIRECTIONAL_RISK"
    assert result.negative == 8
    assert result.two_sided_sign_pvalue < 0.01


def test_sign_changing_relative_error_is_not_directional():
    result = certify_reference_relative([
        row(i, 0.01 if i % 2 else -0.01) for i in range(8)
    ])
    assert result.status == "REFERENCE_RELATIVE_CENTERED_OR_SIGN_CHANGING"


def test_heterogeneous_states_can_have_nonzero_reference_relative_mean():
    values = [-0.004, -0.003, -0.002, -0.002, -0.001, -0.001, 0.0001, 0.0002]
    result = certify_reference_relative([row(i, value) for i, value in enumerate(values)])
    assert result.status == "REFERENCE_RELATIVE_DIRECTIONAL_RISK"
    assert result.two_sided_sign_pvalue > 0.01


def test_orthogonal_error_is_not_directional():
    result = certify_reference_relative([row(i, 0.0) for i in range(8)])
    assert result.status == "REFERENCE_RELATIVE_CENTERED_OR_SIGN_CHANGING"
    assert result.tied == 8
