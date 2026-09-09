import ast
import pytest
from kernel_analyzer.residual_rms_forward_source import expected_body, body_matches
from kernel_analyzer.residual_rms_forward_source import check_source


def fixture():
    fn=ast.parse('def kernel(): pass').body[0]
    fn.body=expected_body(64,4096,1e-6)
    return fn


def test_complete_expression():
    assert body_matches(fixture(),rows=64,width=4096,epsilon=1e-6)


@pytest.mark.parametrize('old,new', [('tmp3 = tmp2.to(tl.float32)','tmp3 = tmp2.to(tl.bfloat16)'),
    ('tmp14.to(tl.float32)','tmp0.to(tl.float32)'),
    ('tl.sum(_tmp6, 1)','tl.sum(_tmp6, 0)'),
    ('tmp13 * tmp17','tmp13 + tmp17'),
    ('4096 * x0','2048 * x0')])
def test_order_and_arithmetic_changes_rejected(old,new):
    text=ast.unparse(fixture())
    assert old in text
    fn=ast.parse(text.replace(old,new)).body[0]
    assert not body_matches(fn,rows=64,width=4096,epsilon=1e-6)


def test_extra_write_rejected():
    fn=fixture()
    fn.body.append(ast.parse('tl.store(in_ptr1, 0)').body[0])
    assert not body_matches(fn,rows=64,width=4096,epsilon=1e-6)


def literal_source():
    signature=dict(in_out_ptr0='*bf16',in_out_ptr1='*fp32',in_ptr0='*bf16',
                   in_ptr1='*bf16',out_ptr0='*bf16',xnumel='i32',r0_numel='i32')
    inner=f'@wrap(triton_meta={{"signature": {signature!r}}})\ndef kernel('+','.join([*signature,'XBLOCK','R0_BLOCK'])+'):\n'
    inner+='\n'.join('    '+line for statement in expected_body(64,2048,1e-6)
                     for line in ast.unparse(statement).splitlines())
    return f'kernel=async_compile.triton("kernel",{inner!r})'


def test_automatic_dimensions_and_types():
    c=check_source(literal_source(),'kernel')
    assert (c['rows'],c['width'],c['epsilon'])==(64,2048,1e-6)
    assert 'in_out_ptr0' in c['required_pre_call_inputs']
    assert not c['runtime_binding_complete']


@pytest.mark.parametrize('old,new',[('*bf16','*fp32'),('tmp13 * tmp17','tmp13 + tmp17')])
def test_invalid_literal_source(old,new):
    with pytest.raises(ValueError): check_source(literal_source().replace(old,new),'kernel')
