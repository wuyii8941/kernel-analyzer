import pytest
from kernel_analyzer.source_check_cache import cached_source_checker


def test_exact_content_and_symbol_are_both_required():
    calls=[]
    def checker(source,symbol):
        calls.append((source,symbol)); return {'nested':[source,symbol]}
    check=cached_source_checker(checker)
    first=check('source','a'); first['nested'][0]='corrupted'
    assert check('source','a')=={'nested':['source','a']}
    check('source ','a'); check('source','b')
    assert len(calls)==3
    assert check.cache_info().currsize==2


def test_rejection_is_never_replaced_with_previous_success():
    calls=[]
    def checker(source,symbol):
        calls.append(source)
        if source=='bad': raise ValueError('changed source')
        return {'valid':True}
    check=cached_source_checker(checker)
    check('good','a')
    for _ in range(2):
        with pytest.raises(ValueError): check('bad','a')
    assert calls==['good','bad','bad']


@pytest.mark.parametrize('capacity',[0,-1,True,1.5])
def test_invalid_capacity(capacity):
    with pytest.raises(ValueError): cached_source_checker(lambda s,n:{},max_entries=capacity)
