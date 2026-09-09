import pytest

from scripts.build_kernel_operation_inventory import describe, reuse_inventory


def source(body, name='embedding_dense_backward'):
    definition = 'def ' + name + '(p):\n' + body
    return f'{name} = async_compile.triton("k", {definition!r})'


def test_name_does_not_turn_zero_initialization_into_embedding_gradient():
    result = describe(source('    tl.store(p, 0)\n'), 'embedding_dense_backward')
    assert result['status'] == 'BODY_INSPECTED'
    assert result['has_store'] and not result['has_load']
    assert result['structural_role'] == 'OUTPUT_INITIALIZATION_ONLY'
    assert not result['semantics_verified']


def test_no_load_arange_computation_is_not_called_initialization():
    result = describe(
        source('    x = tl.arange(0, 8) - 1\n    tl.store(p, x)\n'),
        'embedding_dense_backward',
    )
    assert result['structural_role'] == 'NUMERICAL_COMPUTATION_REQUIRES_SEMANTIC_REVIEW'


def test_real_calls_and_rebinding():
    text = source('    x = tl.load(p)\n    tl.atomic_add(p, x * x)\n')
    result = describe(text, 'embedding_dense_backward')
    assert result['has_atomic'] and result['has_load']
    assert result['structural_role'] == 'NUMERICAL_COMPUTATION_REQUIRES_SEMANTIC_REVIEW'
    assert result['arithmetic'] == {'Mult': 1}
    assert describe(text + '\n' + text, 'embedding_dense_backward')['status'] == 'AMBIGUOUS_OR_MISSING_DEFINITION'


def test_reduction_flag_is_preserved_for_downstream_prioritization():
    result = describe(
        source('    x = tl.load(p)\n    tl.store(p, tl.sum(x, 0))\n'),
        'embedding_dense_backward',
    )
    assert result['has_reduction'] is True


def test_copy_or_cast_is_not_counted_as_operator_semantics():
    result = describe(source('    x = tl.load(p)\n    tl.store(p, x.to(tl.float32))\n'),
                      'embedding_dense_backward')
    assert result['structural_role'] == 'DATA_MOVEMENT_OR_CAST_ONLY'


def test_transcendental_pointwise_remains_for_review():
    result = describe(source('    x = tl.load(p)\n    tl.store(p, libdevice.exp(x))\n', name='copy'), 'copy')
    assert result['structural_role'] == 'NUMERICAL_COMPUTATION_REQUIRES_SEMANTIC_REVIEW'


def test_inventory_reuse_changes_coverage_fields_but_not_body_analysis():
    operation = describe(source('    x = tl.load(p)\n    tl.store(p, x + 1)\n'),
                         'embedding_dense_backward')
    prior = [dict(source='/data1/tzh/a.py', symbol='kernel', source_sha256='abc',
                  status='REFERENCE_ADAPTER_REQUIRED', operation_inventory=operation)]
    current = [dict(source='/data1/tzh/a.py', symbol='kernel', source_sha256='abc',
                    status='REFERENCE_TEMPLATE_AVAILABLE',
                    matched_reference_families=['FAMILY'])]
    rows = reuse_inventory(current, prior)
    assert rows[0]['status'] == 'REFERENCE_TEMPLATE_AVAILABLE'
    assert rows[0]['operation_inventory'] == operation


def test_inventory_reuse_rejects_changed_or_incomplete_source_set():
    operation = {'status': 'BODY_INSPECTED'}
    prior = [dict(source='/data1/tzh/a.py', symbol='kernel', source_sha256='abc',
                  operation_inventory=operation)]
    changed = [dict(source='/data1/tzh/a.py', symbol='kernel', source_sha256='def')]
    with pytest.raises(ValueError, match='differ'):
        reuse_inventory(changed, prior)
    with pytest.raises(ValueError, match='incomplete'):
        reuse_inventory([dict(source='/data1/tzh/a.py', symbol='kernel', source_sha256='abc')],
                        [dict(source='/data1/tzh/a.py', symbol='kernel', source_sha256='abc',
                              operation_inventory={'status': 'FAILED'})])
