import pytest
from kernel_analyzer.grouped_causal_softmax_launch import validate_launch


def check(rows=6, masked=True, xblock=4, grid=(2,), **kw):
    return validate_launch(dict(rows=rows, width=2, row_bounds_masked=masked),
                           xnumel=rows, r0_numel=2, xblock=xblock, grid=grid, **kw)


def test_masked_tail_and_exact_unmasked():
    assert check()['status'] == 'RESOLVED_GEOMETRY_CHECKED'
    assert check(rows=8, masked=False)['grid'] == [2, 1, 1]


def test_unmasked_tail_rejected():
    with pytest.raises(ValueError, match='invalid rows'):
        check(masked=False)


@pytest.mark.parametrize('grid', [(1,), (3,), (2, 2), (2, 1, 2), lambda: (2,), (True,)])
def test_under_over_or_unresolved_grid_rejected(grid):
    with pytest.raises(ValueError):
        check(grid=grid)


def test_dimension_contract_rejected():
    with pytest.raises(ValueError, match='dimensions differ'):
        validate_launch(dict(rows=8, width=2, row_bounds_masked=True),
                        xnumel=6, r0_numel=2, xblock=4, grid=(2,))
