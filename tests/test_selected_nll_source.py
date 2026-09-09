import ast
from pathlib import Path
import pytest
from kernel_analyzer.selected_nll_source import check_source

SOURCE = Path('results/coverage/runtime_releases/deepseek8b_seq128_r1/trace/model__1_backward_segment0_executed/output_code.py')


def definition():
    tree = ast.parse(SOURCE.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and 'nll_loss_backward' in t.id for t in n.targets))
    return ast.unparse(node), node.targets[0].id


def test_reviewed_actual_body_and_storage():
    source, symbol = definition()
    result = check_source(source, symbol)
    assert result['clone_input_before_execution']
    assert result['output_pointer'] == 'in_out_ptr0'
    assert not result['runtime_binding_complete']


def test_modified_arithmetic_and_storage_rejected():
    source, symbol = definition()
    with pytest.raises(ValueError, match='arithmetic'):
        check_source(source.replace('tmp35 - tmp43', 'tmp35 + tmp43'), symbol)
    with pytest.raises(ValueError, match='Storage'):
        check_source(source.replace('*bf16', '*fp32'), symbol)
    with pytest.raises(ValueError, match='Unique'):
        check_source(source+'\n'+source, symbol)
