import ast
import pytest
from kernel_analyzer.decayed_recurrence_source import expected_body
from kernel_analyzer.segmented_recurrence_source import first_segment_body,first_segment_matches


def test_original_time_origin_is_unchanged():
    assert [ast.dump(x) for x in first_segment_body(channels=3,width=2,outputs=4,time_start=4)]==[
        ast.dump(x) for x in ast.parse(expected_body(3,2,4)).body]


def test_only_time_offsets_shift():
    original=expected_body(3,2,4)
    shifted=original
    for old,new in [(12,30),(9,27),(6,24),(3,21)]:
        shifted=shifted.replace(f'in_ptr3 + ({old} + x1)',f'in_ptr3 + ({new} + x1)')
    fn=ast.parse('def f():\n'+''.join('    '+line+'\n' for line in shifted.splitlines())).body[0]
    assert first_segment_matches(fn,channels=3,width=2,outputs=4,time_start=10)
    fn.body[-1].value.args[1]=ast.Name(id='wrong',ctx=ast.Load())
    assert not first_segment_matches(fn,channels=3,width=2,outputs=4,time_start=10)


@pytest.mark.parametrize('start',[0,3,-1,1.5,True])
def test_invalid_time_origin_rejected(start):
    with pytest.raises(ValueError): first_segment_body(channels=3,width=2,outputs=4,time_start=start)


def test_signature_is_checked_after_body_validation():
    from kernel_analyzer.segmented_recurrence_source import check_first_segment
    signature={f'in_ptr{i}':'*fp32' if i==2 else '*bf16' for i in range(7)}
    signature.update(out_ptr0='*fp32',xnumel='i32')
    body=first_segment_body(channels=1536,width=16,outputs=1,time_start=127)
    inner=f'@heuristic(triton_meta={{"signature":{signature!r}}})\ndef recurrence('+','.join([*signature,'XBLOCK'])+'):\n'
    inner+='\n'.join('    '+ast.unparse(n) for n in body)
    source=f'recurrence=async_compile.triton("recurrence",{inner!r})'
    result=check_first_segment(source,'recurrence',time_start=127)
    assert result['time_offsets_descending']==[127]
    assert not result['runtime_binding_complete']
    with pytest.raises(ValueError,match='storage types'):
        check_first_segment(source.replace('*bf16','*fp32'),'recurrence',time_start=127)
