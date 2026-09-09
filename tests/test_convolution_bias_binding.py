import ast
import pytest
from kernel_analyzer.convolution_bias_binding import associate

SOURCE = '''buf6 = extern_kernels.convolution(x, w, stride=(1,), padding=(3,), dilation=(1,), transposed=False, output_padding=(0,), groups=1536, bias=None)
assert_size_stride(buf6, (1,1536,67), (102912,67,1), 'convolution')
buf7 = buf6
del buf6
raw_stream0 = get_raw_stream(0)
bias.run(buf7, b, 102912, stream=raw_stream0)
'''


def test_explicit_alias_chain():
    result = associate(ast.parse(SOURCE).body, 0, 'bias')
    assert result['output_name'] == 'buf7'
    assert not result['runtime_binding_complete']


@pytest.mark.parametrize('change', [
    SOURCE.replace('buf7 = buf6', 'buf7 = other'),
    SOURCE.replace('del buf6', 'buf7.zero_()'),
    SOURCE.replace('bias.run(buf7', 'bias.run(other'),
])
def test_reject_unproved_connection(change):
    with pytest.raises(ValueError): associate(ast.parse(change).body, 0, 'bias')
