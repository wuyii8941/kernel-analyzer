import copy
from scripts.compare_relocated_aot import relocated_compare


def capture(device):
    return {'graphs':[{'phase':'FORWARD','nodes':[{'name':'x','op':'call_function',
        'target':'allocate','arguments':{'kwargs':{'device':device,'dtype':'float32'}},'input_edges':[]}]}]}


def test_only_device_change_is_allowed_without_mutation():
    original=capture('cuda:3'); saved=copy.deepcopy(original)
    result=relocated_compare(capture('cuda:0'), original, 'cuda:3','cuda:0')
    assert not result['strict']['structure_identical']
    assert result['after_declared_device_change']['structure_identical']
    assert original==saved
    original['graphs'][0]['nodes'][0]['arguments']['kwargs']['dtype']='float64'
    result=relocated_compare(capture('cuda:0'),original,'cuda:3','cuda:0')
    assert not result['after_declared_device_change']['structure_identical']
