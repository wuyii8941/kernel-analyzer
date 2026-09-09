import pytest

from scripts.scan_recurrence_sources import scan
from kernel_analyzer.decayed_recurrence_source import check_source
from test_decayed_recurrence_source import fixture


def source():
    return fixture().replace('recurrence', 'triton_recurrence')


def test_isolated_parse_preserves_full_module_contract():
    module = 'unrelated = 17\n' + source() + '\nother = None\n'
    rows = scan(module)
    assert len(rows) == 1
    assert rows[0]['contract'] == check_source(module, 'triton_recurrence')
    assert not rows[0]['contract']['runtime_binding_complete']


def test_duplicate_definitions_are_not_silently_selected():
    rows = scan(source() + '\n' + source())
    assert len(rows) == 2
    assert all(row['status'] == 'REJECTED' for row in rows)


@pytest.mark.parametrize('old,new', [
    ('tmp7 = -tmp6', 'tmp7 = tmp6'),
    ('4608 + x1', '3072 + x1'),
    ('*bf16', '*fp32'),
])
def test_semantic_mutations_remain_rejected(old, new):
    module = source().replace(old, new)
    with pytest.raises(ValueError):
        check_source(module, 'triton_recurrence')
    assert scan(module)[0]['status'] == 'REJECTED'


def test_unsupported_definitions_remain_in_denominator():
    rows = scan(source() + '\ntriton_unknown = factory()\n')
    assert [row['status'] for row in rows] == ['SOURCE_CHECKED', 'REJECTED']


def test_invalid_module_does_not_become_empty_success():
    with pytest.raises(SyntaxError):
        scan('this is not valid python !')
