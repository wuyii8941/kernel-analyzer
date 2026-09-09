import pytest
from scripts.finalize_residual_rms_forward import verify_translation


def fixture():
    case = dict(case_id='one', task_id='forward:1:out_ptr0', carrier='weight',
                expected_symbol='triton_example', reference_output_pointer='out_ptr0',
                reference_method='RESIDUAL_RMS_FORWARD_COMMON_INPUT')
    original = dict(schema='residual-rms-forward-bound-plan-v1', cases=[case],
                    contracts={'triton_example': dict(output_pointers=['out_ptr0'])})
    translated = dict(cases=[dict(case, reference_method='PARTIAL_REDUCTION_FROM_BOUND_INPUT',
                                 declared_reference_method=case['reference_method'])])
    return original, translated


def test_only_dispatch_change_is_accepted():
    original, translated = fixture()
    assert verify_translation(original, translated) == original['cases']


@pytest.mark.parametrize('field,value', [('carrier','other'), ('task_id','other'),
    ('expected_symbol','other'), ('reference_output_pointer','other'),
    ('declared_reference_method','other'), ('reference_method','other')])
def test_changed_scientific_identity_rejected(field, value):
    original, translated = fixture()
    translated['cases'][0][field] = value
    with pytest.raises(ValueError): verify_translation(original, translated)


def test_duplicate_and_empty_selection_rejected():
    original, translated = fixture()
    translated['cases'] *= 2
    with pytest.raises(ValueError): verify_translation(original, translated)
    with pytest.raises(ValueError): verify_translation(original, dict(cases=[]))
