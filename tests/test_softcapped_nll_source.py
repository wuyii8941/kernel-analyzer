import ast
from pathlib import Path
import pytest
from kernel_analyzer.softcapped_nll_source import check_source


def reviewed():
    path = Path('results/property/three_mechanism_profiles_v1/runs/gemma4_text128_scan_0037/runtime_release/trace/model__1_backward_segment0_executed/output_code.py')
    if not path.exists():
        pytest.skip('Historical generated-source fixture unavailable')
    nodes = [n for n in ast.parse(path.read_text()).body if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and 'tanh_backward' in t.id for t in n.targets)]
    assert len(nodes) == 1
    return ast.unparse(nodes[0]), nodes[0].targets[0].id


def test_reviewed_source_and_renaming():
    text, name = reviewed()
    contract = check_source(text, name)
    assert contract['cap'] == 30 and not contract['runtime_binding_complete']
    assert contract['vocabulary'] == 262144
    assert check_source(text.replace(name, 'renamed'), 'renamed')['body_sha256'] == contract['body_sha256']


@pytest.mark.parametrize('old,new', [('30.0', '31.0'), ('262144', '262143'),
                                  ('*bf16', '*fp32'), ('tmp55 - tmp54', 'tmp55 + tmp54')])
def test_rejects_changed_semantics_or_storage(old, new):
    text, name = reviewed()
    assert old in text
    with pytest.raises(ValueError):
        check_source(text.replace(old, new), name)


def test_rejects_rebinding():
    text, name = reviewed()
    with pytest.raises(ValueError, match='Unique source'):
        check_source(text+'\n'+text, name)
