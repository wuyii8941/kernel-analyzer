import ast
import pytest
from kernel_analyzer.continued_recurrence_source import continued_body, continued_matches
from kernel_analyzer.continued_recurrence_source import check_source


def function():
    fn = ast.parse('def kernel(): pass').body[0]
    fn.body = continued_body(outputs=3, time_start=8)
    return fn


def test_full_body():
    assert continued_matches(function(), outputs=3, time_start=8)


@pytest.mark.parametrize('old,new', [
    ('tmp3 = -tmp2', 'tmp3 = tmp2'),
    ('tmp0 * tmp16', 'tmp1 * tmp16'),
    ('12288 + x1', '10752 + x1'),
    ('tmp59, None', 'tmp41, None'),
])
def test_changed_recurrence_rejected(old, new):
    text = ast.unparse(function())
    assert old in text
    mutated = ast.parse(text.replace(old, new)).body[0]
    assert not continued_matches(mutated, outputs=3, time_start=8)


@pytest.mark.parametrize('start', [True, 2, -1, 8.5])
def test_invalid_time(start):
    with pytest.raises(ValueError):
        continued_body(outputs=3, time_start=start)


def literal_source():
    signature = {f'in_ptr{i}': '*fp32' if i < 2 else '*bf16' for i in range(10)}
    signature.update({f'out_ptr{i}': '*fp32' for i in range(3)})
    signature['xnumel'] = 'i32'
    inner = f'@heuristic(triton_meta={{"signature": {signature!r}}})\n'
    inner += 'def kernel(' + ','.join([*signature, 'XBLOCK']) + '):\n'
    inner += '\n'.join('    '+ast.unparse(n) for n in continued_body(outputs=3, time_start=8))
    return f'kernel = async_compile.triton("kernel", {inner!r})'


def test_full_contract():
    contract = check_source(literal_source(), 'kernel', time_start=8)
    assert contract['time_offsets_descending'] == [8, 7, 6]
    assert contract['segment_input_kind'] == 'FP32_PREVIOUS_STATE'
    assert not contract['runtime_binding_complete']


@pytest.mark.parametrize('old,new', [('*fp32', '*bf16'), ('*bf16', '*fp32'),
                                    ('async_compile.triton', 'async_compile.other'),
                                    ('tmp0 * tmp16', 'tmp1 * tmp16')])
def test_invalid_contract(old, new):
    with pytest.raises(ValueError):
        check_source(literal_source().replace(old, new), 'kernel', time_start=8)


def test_duplicate_contract():
    with pytest.raises(ValueError):
        check_source(literal_source()+'\n'+literal_source(), 'kernel', time_start=8)
