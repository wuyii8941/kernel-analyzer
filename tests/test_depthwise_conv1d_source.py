import pytest
from kernel_analyzer.depthwise_conv1d_source import check_call, scan

CALL = 'extern_kernels.convolution(x, w, stride=(1,), padding=(3,), dilation=(1,), transposed=False, output_padding=(0,), groups=1536, bias=None)'


def test_checked_call_not_runtime_proof():
    result = check_call(CALL)
    assert not result['runtime_binding_complete']
    assert result['required_weight_shape'] == [1536, 1, 4]
    assert len(scan('def call():\n    y = '+CALL)) == 1


@pytest.mark.parametrize('old,new', [('groups=1536','groups=1'), ('bias=None','bias=b'),
    ('stride=(1,)','stride=(True,)'), ('padding=(3,)','padding=(2,)'),
    ('transposed=False','transposed=True'), ('(x, w,','(x, w, b,')])
def test_reject_other_semantics(old, new):
    with pytest.raises(ValueError): check_call(CALL.replace(old, new))


def test_unknown_calls_retained_not_silently_dropped():
    rows = scan('y = '+CALL+'\nz = '+CALL.replace('groups=1536','groups=1'))
    assert [r['status'] for r in rows] == ['SOURCE_CHECKED', 'UNSUPPORTED']
