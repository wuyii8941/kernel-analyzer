from copy import deepcopy
import pytest
from scripts.run_depthwise_convolution_capture import select


def test_selection_preserves_boundary_and_rejects_changes():
    case = dict(case_id='conv', task_id='forward:3:output_0', carrier='weight',
        reference_method='DEPTHWISE_CONV1D_COMMON_INPUT', implementation_kind='EXTERN',
        expected_symbol='convolution',
        replacement_boundary='EXTERNAL_CONVOLUTION_OUTPUT_BEFORE_SEPARATE_BIAS',
        source_contract=dict(convolution_line_sha256='hash', convolution=dict(groups=1536)))
    plan = dict(schema='depthwise-convolution-bound-plan-v1', cases=[case])
    assert select(plan, [case]) == {'hash': case['source_contract']}
    changed = deepcopy(case)
    changed['carrier'] = 'other'
    with pytest.raises(ValueError, match='differs'):
        select(plan, [changed])
    with pytest.raises(ValueError, match='Duplicate'):
        select(plan, [case, case])
    with pytest.raises(ValueError):
        select(plan, [])
