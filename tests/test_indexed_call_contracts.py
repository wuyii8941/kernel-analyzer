import pytest
from scripts.build_indexed_call_contracts import check_call


def test_exact_call_semantics():
    c=check_call('aten.index_put_(buffer, [tokens], gradients, True)')
    assert c['initial_name']=='buffer' and c['index_name']=='tokens'


@pytest.mark.parametrize('expression',[
    'aten.index_put_(b,[i],v,False)', 'aten.index_put_(b,[i,j],v,True)',
    'aten.index_put_(b,[None],v,True)', 'aten.index_put_(b,[i],v,1)',
    'aten.index_put_(b,[i],v,accumulate=True)', 'other.index_put_(b,[i],v,True)'])
def test_other_calls_rejected(expression):
    with pytest.raises(ValueError): check_call(expression)
