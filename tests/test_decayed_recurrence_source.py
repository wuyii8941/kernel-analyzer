import ast
import pytest
from kernel_analyzer.decayed_recurrence_source import expected_body,body_matches,check_source


def fixture(outputs=3):
    signature={f'in_ptr{i}':'*fp32' if i==2 else '*bf16' for i in range(5+2*outputs)}
    signature.update({f'out_ptr{i}':'*fp32' for i in range(outputs)})
    signature['xnumel']='i32'
    inner=f'@heuristic(triton_meta={{"signature": {signature!r}}})\ndef recurrence('+','.join([*signature,'XBLOCK'])+'):\n'
    inner+='\n'.join('    '+line for line in expected_body(outputs=outputs).splitlines())
    return f'recurrence = async_compile.triton("recurrence", {inner!r})'


@pytest.mark.parametrize('outputs',[1,3,63])
def test_complete_template(outputs):
    result=check_source(fixture(outputs),'recurrence')
    assert result['outputs']==outputs
    assert result['time_offsets_descending']==list(range(outputs,0,-1))
    assert not result['runtime_binding_complete']


@pytest.mark.parametrize('old,new', [('tmp7 = -tmp6','tmp7 = tmp6'),
    ('4608 + x1','3072 + x1'), ('tmp45, None','tmp27, None'), ('*bf16','*fp32')])
def test_changed_semantics_rejected(old,new):
    source=fixture()
    assert old in source
    with pytest.raises(ValueError): check_source(source.replace(old,new),'recurrence')
