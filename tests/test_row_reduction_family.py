import textwrap
import pytest
import torch
from kernel_analyzer.row_reduction_reference import PREFIX, VARIANTS, check_source, reference
from scripts.run_row_reduction_capture import selected_contracts, selected_symbol_census


def source(rows, width, symbol='arbitrary_kernel'):
    code = f'def {symbol}(in_ptr0, out_ptr0, XBLOCK, R0_BLOCK):\n' + textwrap.indent(
        PREFIX.format(rows=rows, width=width).strip(), '    ')
    return f'{symbol} = async_compile.triton({symbol!r}, {code!r})'


@pytest.mark.parametrize('rows,width', [(1, 3), (7, 35), (128, 1536)])
def test_shape_independent_source_and_reference(rows, width):
    contract = check_source(source(rows, width), 'arbitrary_kernel')
    assert (contract['rows'], contract['width']) == (rows, width)
    x = torch.arange(rows * width).reshape(rows, width).float() / (rows * width)
    output = torch.empty(rows)
    assert torch.equal(reference({'runtime_pointers': {'in_ptr0': x}}, output, contract), x.square().sum(-1))
    for variant in VARIANTS:
        assert reference({'runtime_pointers': {'in_ptr0': x}}, output, contract, variant=variant).shape == output.shape


@pytest.mark.parametrize('old,new', [('tmp1 * tmp1', 'tmp1 + tmp1'),
                                  ('other=0.0', 'other=1.0'),
                                  ('35*x0', '36*x0'),
                                  ('r0_index < r0_numel', 'r0_index <= r0_numel')])
def test_semantic_changes_rejected(old, new):
    with pytest.raises(ValueError):
        check_source(source(7, 35).replace(old, new), 'arbitrary_kernel')


def test_wrong_layout_or_variant_rejected():
    contract = check_source(source(7, 35), 'arbitrary_kernel')
    x = torch.zeros(35, 7).T
    with pytest.raises(ValueError):
        reference({'runtime_pointers': {'in_ptr0': x}}, torch.empty(7), contract)
    with pytest.raises(ValueError):
        reference({'runtime_pointers': {'in_ptr0': x.contiguous()}}, torch.empty(7), contract, variant='invented')


def test_capture_requires_explicit_checked_symbol_and_output():
    contract = check_source(source(7, 35), 'arbitrary_kernel')
    manifest = {'sources': [{'rows': [{'symbol': 'arbitrary_kernel', 'status': 'SOURCE_CHECKED', 'contract': contract}]}]}
    case = {'reference_contract_symbol': 'arbitrary_kernel',
            'reference_method': 'PARTIAL_REDUCTION_FROM_BOUND_INPUT', 'task_id': 'forward:1:out_ptr0'}
    assert selected_contracts(manifest, [case]) == {'arbitrary_kernel': contract}
    for change in ({'reference_contract_symbol': 'unproved'}, {'task_id': 'forward:1:out_ptr1'},
                   {'reference_method': 'AOT_REPLAY'}):
        with pytest.raises(ValueError):
            selected_contracts(manifest, [{**case, **change}])


def test_capture_requires_all_occurrences_of_selected_symbols_only():
    rows = [
        {'symbol': 'selected', 'region_id': 'r0'},
        {'symbol': 'unrelated_old_name', 'region_id': 'r1'},
        {'symbol': 'selected', 'region_id': 'r2'},
    ]
    assert selected_symbol_census(rows, {'selected': {}}) == [rows[0], rows[2]]
    with pytest.raises(ValueError, match='absent from frozen campaign'):
        selected_symbol_census(rows, {'missing': {}})
