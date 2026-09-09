import pytest
from scripts.run_declared_reference_family import declaration, validate_capture, argument


def test_new_family_does_not_replace_frozen_registry():
    with pytest.raises(ValueError,match='replace'):
        declaration('SOFTPLUS_BIAS_BACKWARD','row_sum_reference','-row-sum')
    with pytest.raises(ValueError,match='identifiers'):
        declaration('NEW','../other','-row-sum')


def test_capture_requires_exact_entrypoint_and_adapter():
    d=declaration('ROW_SUM','row_sum_reference','-row-sum')
    m=dict(additional_reference_declaration=d,reference_family='ROW_SUM',adapter_sha256=d['adapter_sha256'])
    validate_capture(m,d)
    with pytest.raises(ValueError):
        validate_capture({},d)
    with pytest.raises(ValueError):
        validate_capture(m,dict(d,entrypoint_sha256='different'))
    with pytest.raises(ValueError):
        validate_capture(dict(m,adapter_sha256='different'),d)


def test_no_ambiguous_cli_path():
    with pytest.raises(ValueError):
        argument(['--output','a','--output','b'],'--output')
