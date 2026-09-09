import pytest
from scripts.build_row_reduction_contracts import (
    discover, discover_prepared, prepare_source, unique_contracts,
)
from kernel_analyzer.rms_backward_reference import check_source
from test_rms_backward_reference import source as rms_source


def source(digest, status='SOURCE_CHECKED'):
    return {'rows': [{'symbol': 'shared_name', 'status': status,
                      'contract': {'function_ast_sha256': digest}}]}


def test_repeated_identical_source_is_allowed():
    assert unique_contracts([source('same'), source('same')]) == {
        'shared_name': {'function_ast_sha256': 'same'}}


def test_same_name_different_source_cannot_silently_replace_contract():
    with pytest.raises(ValueError, match='Ambiguous same-name'):
        unique_contracts([source('model-a'), source('model-b')])


def test_unrecognized_source_is_not_bound():
    assert unique_contracts([source('unknown', 'NOT_THIS_REFERENCE_FAMILY')]) == {}


def test_scanning_extracted_definition_preserves_full_source_contract(tmp_path):
    text = 'unrelated = 123\n' + rms_source() + '\nother = 456\n'
    path = tmp_path / 'output_code.py'
    path.write_text(text)
    row = discover(path, 'RMS_BACKWARD')['rows'][0]
    assert row['status'] == 'SOURCE_CHECKED'
    assert row['contract'] == check_source(text, 'rms')


@pytest.mark.parametrize('extra', [rms_source(), 'rms = unrelated()'])
def test_extraction_does_not_hide_duplicate_or_rebound_definitions(tmp_path, extra):
    path = tmp_path / 'output_code.py'
    path.write_text(rms_source() + '\n' + extra)
    assert discover(path, 'RMS_BACKWARD')['rows'][0]['status'] == 'NOT_THIS_REFERENCE_FAMILY'


def test_prepared_source_reuses_parse_without_changing_discovery(tmp_path):
    path = tmp_path / 'output_code.py'
    path.write_text(rms_source())
    assert discover_prepared(prepare_source(path), 'RMS_BACKWARD') == discover(
        path, 'RMS_BACKWARD')
