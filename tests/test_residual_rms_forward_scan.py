import pytest
from scripts.scan_residual_rms_forward_sources import scan
from kernel_analyzer.residual_rms_forward_source import check_source
from test_residual_rms_forward_source import literal_source


def module():
    return literal_source().replace('kernel', 'triton_example')


def test_full_source_identity_and_three_outputs():
    source = 'unrelated = 1\n' + module()
    rows = scan(source)
    assert len(rows) == 1
    assert rows[0]['contract'] == check_source(source, 'triton_example')
    assert len(rows[0]['contract']['output_pointers']) == 3
    assert not rows[0]['contract']['runtime_binding_complete']


def test_unsupported_and_duplicate_definitions_retained():
    rows = scan(module() + '\ntriton_unknown = factory()')
    assert [r['status'] for r in rows] == ['SOURCE_CHECKED', 'REJECTED']
    assert all(r['status'] == 'REJECTED' for r in scan(module() + '\n' + module()))


@pytest.mark.parametrize('old,new', [('*bf16', '*fp32'),
                                    ('tmp13 * tmp17', 'tmp13 + tmp17')])
def test_changed_computation_not_accepted(old, new):
    assert scan(module().replace(old, new))[0]['status'] == 'REJECTED'
