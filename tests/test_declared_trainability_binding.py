import pytest

from scripts.bind_declared_trainability_reference import bind


def manifest():
    return {
        'reference_family': 'RMS_SIMPLE_BACKWARD',
        'sources': [{'rows': [{'status': 'SOURCE_CHECKED', 'symbol': 'kernel',
                              'contract': {'symbol': 'kernel',
                                           'output_pointer': 'in_out_ptr0',
                                           'function_ast_sha256': 'abc'}}]}],
    }


def task(**changes):
    row = dict(task_id='backward:1:in_out_ptr0', symbol='kernel',
               formal_pointer='in_out_ptr0', implementation_kind='TRITON',
               phase='BACKWARD',
               status='INTERNAL_IMPLEMENTATION_BUFFER_COVERED_BY_CLOSED_SEMANTIC_ENDPOINT',
               closure_uses_candidate_values=False,
               closed_by_semantic_endpoint_tasks=['backward:2:output_0'])
    row.update(changes)
    return row


def test_registered_family_binds_without_family_specific_dispatch():
    result = bind(manifest(), {'rows': [task()]},
                  {'trainable_parameters': ['model.weight']})
    assert len(result['cases']) == 1
    case = result['cases'][0]
    assert case['reference_method'] == 'PARTIAL_REDUCTION_FROM_BOUND_INPUT'
    assert case['reference_contract_symbol'] == 'kernel'
    assert case['carrier'] == 'model.weight'
    assert case['runtime_parameter_reach'] == 'NOT_YET_MEASURED'


def test_binding_rejects_result_dependent_closure_and_non_triton_task():
    result = bind(manifest(), {'rows': [task(closure_uses_candidate_values=True)]},
                  {'trainable_parameters': ['model.weight']})
    assert not result['cases']
    assert result['unresolved'][0]['reason'] == 'CLOSURE_SELECTION_NOT_SOURCE_ONLY'
    result = bind(manifest(), {'rows': [task(implementation_kind='DIRECT_ATEN')]},
                  {'trainable_parameters': ['model.weight']})
    assert result['unresolved'][0]['reason'] == 'IMPLEMENTATION_IS_NOT_TRITON'


def test_binding_requires_one_predeclared_parameter_and_unique_tasks():
    with pytest.raises(ValueError, match='Exactly one'):
        bind(manifest(), {'rows': [task()]}, {'trainable_parameters': []})
    with pytest.raises(ValueError, match='Duplicate task'):
        bind(manifest(), {'rows': [task(), task()]},
             {'trainable_parameters': ['model.weight']})
