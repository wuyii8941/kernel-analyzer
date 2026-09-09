import copy
import pytest
from scripts.summarize_source_reference_coverage import summarize


def scan(family, match):
    return {'reference_family': family, 'sources': [
        {'source': 'generated.py', 'source_sha256': 'source-hash', 'rows': [
            {'symbol': name, 'status': 'SOURCE_CHECKED' if name == match else 'NOT_THIS_REFERENCE_FAMILY',
             'contract': {'source_sha256': 'source-hash'}} for name in ('one', 'two', 'three')]}]}


def test_union_not_sum_and_unmatched_not_negative():
    report = summarize([scan('A', 'one'), scan('B', 'two')])
    assert report['definition_records'] == 3
    assert report['counts'] == {'REFERENCE_TEMPLATE_AVAILABLE': 2, 'REFERENCE_ADAPTER_REQUIRED': 1}
    assert not report['all_kernel_support_established']
    assert not report['unmatched_means_unbiased']


def test_overlap_requires_selection_not_double_count():
    report = summarize([scan('A', 'one'), scan('B', 'one')])
    assert report['counts']['MULTIPLE_REFERENCES_REQUIRE_EXPLICIT_SELECTION'] == 1


@pytest.mark.parametrize('error', ['source_version', 'missing_definition', 'duplicate', 'unknown_status'])
def test_incompatible_scans_rejected(error):
    a, b = scan('A','one'), scan('B','two')
    if error == 'source_version':
        b['sources'][0]['source_sha256'] = 'other'
    elif error == 'missing_definition':
        b['sources'][0]['rows'].pop()
    elif error == 'duplicate':
        b['sources'][0]['rows'].append(copy.deepcopy(b['sources'][0]['rows'][0]))
    else:
        b['sources'][0]['rows'][0]['status'] = 'UNKNOWN'
    with pytest.raises(ValueError):
        summarize([a,b])
