"""Group capture work without pooling effects or replacing multiple endpoints."""


def capture_batches(rows, tasks_by_id, size):
    if not isinstance(size, int) or size < 1:
        raise ValueError('Batch size must be positive')
    groups, current, cuts = [], [], set()
    seen_tasks, seen_cases = set(), set()
    for row in rows:
        task_id = row['task_id']; case = row['case']
        if task_id in seen_tasks or case['case_id'] in seen_cases:
            raise ValueError('Duplicate task or case identity')
        seen_tasks.add(task_id); seen_cases.add(case['case_id'])
        if task_id not in tasks_by_id:
            raise ValueError('Task absent from frozen release')
        cut = tasks_by_id[task_id].get('exact_aot_endpoint_id') if case.get('reference_method') == 'AOT_REPLAY' else None
        if case.get('reference_method') == 'AOT_REPLAY' and not cut:
            raise ValueError('AOT reference endpoint missing')
        if current and (len(current) == size or cut is not None and cut in cuts):
            groups.append(current); current, cuts = [], set()
        current.append(row)
        if cut is not None:
            cuts.add(cut)
    if current:
        groups.append(current)
    return groups
