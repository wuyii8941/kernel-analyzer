import pytest
from kernel_analyzer.numerical_batching import capture_batches


def fixture():
    rows = [{'task_id': f't{i}', 'case': {'case_id': f'c{i}', 'reference_method': 'AOT_REPLAY'}} for i in range(4)]
    tasks = {f't{i}': {'exact_aot_endpoint_id': f'node{i}'} for i in range(4)}
    return rows, tasks


def test_grouping_preserves_order_and_all_cases():
    rows, tasks = fixture()
    groups = capture_batches(rows, tasks, 3)
    assert list(map(len, groups)) == [3, 1]
    assert [row for group in groups for row in group] == rows


def test_shared_aot_cut_is_not_ambiguously_batched():
    rows, tasks = fixture(); tasks['t1']['exact_aot_endpoint_id'] = 'node0'
    assert list(map(len, capture_batches(rows, tasks, 4))) == [1, 3]


def test_common_input_reference_needs_no_aot_cut():
    rows, tasks = fixture()
    for row in rows:
        row['case']['reference_method'] = 'EXTERNAL_FP32_RECOMPUTE'
    assert len(capture_batches(rows, {row['task_id']: {} for row in rows}, 4)) == 1


def test_duplicate_case_cannot_overwrite_raw_output():
    rows, tasks = fixture(); rows[1]['case']['case_id'] = rows[0]['case']['case_id']
    with pytest.raises(ValueError):
        capture_batches(rows, tasks, 2)


def test_invalid_size_or_missing_task_rejected():
    rows, tasks = fixture()
    with pytest.raises(ValueError): capture_batches(rows, tasks, 0)
    with pytest.raises(ValueError): capture_batches(rows, {}, 2)
