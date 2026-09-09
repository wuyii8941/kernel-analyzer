import ast
from pathlib import Path

import pytest
import torch

from kernel_analyzer.exponential_weighted_reduction_reference import check_source, reference


SOURCE = Path(
    'results/coverage/runtime_releases/mamba_seq128_r1/trace/'
    'model__1_backward_segment0_executed/output_code.py')
SYMBOL = 'triton_per_fused_exp_mul_neg_sum_unsqueeze_153'


def _single_assignment_source(path=SOURCE, symbol=SYMBOL):
    tree = ast.parse(path.read_text())
    matches = [node for node in tree.body if isinstance(node, ast.Assign)
               and any(isinstance(target, ast.Name) and target.id == symbol for target in node.targets)]
    assert len(matches) == 1
    return ast.unparse(matches[0])


def test_checks_real_generated_source_and_dimensions():
    contract = check_source(_single_assignment_source(), SYMBOL)
    assert contract['rows'] == 196608
    assert contract['width'] == 16
    assert contract['rows_per_group'] == 128
    assert contract['group_count'] == 1536


@pytest.mark.parametrize(('length', 'suffix', 'rows'), (
    (64, '87', 98304),
    (128, '153', 196608),
    (256, '287', 393216),
))
def test_one_adapter_checks_all_saved_sequence_lengths(length, suffix, rows):
    path = Path(
        f'results/coverage/runtime_releases/mamba_seq{length}_r1/trace/'
        'model__1_backward_segment0_executed/output_code.py')
    symbol = f'triton_per_fused_exp_mul_neg_sum_unsqueeze_{suffix}'
    contract = check_source(_single_assignment_source(path, symbol), symbol)
    assert contract['rows'] == rows
    assert contract['rows_per_group'] == length
    assert contract['group_count'] == 1536


def test_rejects_changed_expression():
    source = _single_assignment_source().replace('tmp3 = -tmp2', 'tmp3 = tmp2')
    with pytest.raises(ValueError, match='expression or indexing differs'):
        check_source(source, SYMBOL)


def test_semantic_identity_ignores_only_generated_device_metadata():
    source = _single_assignment_source()
    original = check_source(source, SYMBOL)
    moved = check_source(source.replace("index=0", "index=3", 1), SYMBOL)
    assert original['function_ast_sha256'] != moved['function_ast_sha256']
    assert (original['function_semantic_ast_sha256']
            == moved['function_semantic_ast_sha256'])


def test_semantic_identity_still_changes_with_signature():
    source = _single_assignment_source()
    changed = source.replace('xnumel, r0_numel, XBLOCK',
                             'xnumel, r0_numel, EXTRA, XBLOCK', 1)
    assert (check_source(source, SYMBOL)['function_semantic_ast_sha256']
            != check_source(changed, SYMBOL)['function_semantic_ast_sha256'])


def test_reference_matches_direct_expression_and_variants():
    contract = {'rows': 6, 'width': 3, 'rows_per_group': 2, 'group_count': 3,
                'input_pointer': 'in_ptr0', 'log_weight_pointer': 'in_ptr1'}
    source = torch.arange(18, dtype=torch.float32).reshape(6, 3) / 10
    log_weight = torch.tensor([[0.1, -0.2, 0.3], [0.4, 0.2, -0.1], [-0.3, 0.5, 0.0]])
    candidate = torch.empty(6, dtype=torch.float32)
    metadata = {'runtime_pointers': {'in_ptr0': source, 'in_ptr1': log_weight}}
    expected = (source * -log_weight.repeat_interleave(2, dim=0).exp()).sum(-1)
    actual = reference(metadata, candidate, contract, variant='FP32_NATIVE')
    assert torch.equal(actual, expected)
    reverse = reference(metadata, candidate, contract, variant='FP32_REVERSE_COMPONENT_ORDER')
    expected_reverse = (source * -log_weight.repeat_interleave(2, dim=0).exp()).flip(-1).sum(-1)
    assert torch.equal(reverse, expected_reverse)
    high = reference(metadata, candidate, contract, variant='FP64_EVALUATION')
    assert high.dtype == torch.float32


def test_reference_rejects_wrong_dtype():
    contract = {'rows': 2, 'width': 2, 'rows_per_group': 1, 'group_count': 2,
                'input_pointer': 'in_ptr0', 'log_weight_pointer': 'in_ptr1'}
    metadata = {'runtime_pointers': {'in_ptr0': torch.ones(2, 2, dtype=torch.float64),
                                     'in_ptr1': torch.ones(2, 2)}}
    with pytest.raises(ValueError, match='Source input'):
        reference(metadata, torch.empty(2), contract)
