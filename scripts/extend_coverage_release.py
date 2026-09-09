"""Register a new execution release without transferring old measurement labels."""
import argparse
from collections import Counter
from pathlib import Path
from scripts.build_reference_reach_inventory import classify
from scripts.join_coverage_measurements import validate_retained_measurements
from scripts.run_numerical_coverage import read, save, sha


def extend(inventory, release, tasks):
    release = str(Path(release).resolve())
    old = inventory['records']
    if any(str(Path(row['release']).resolve()) == release for row in old):
        raise ValueError('Release already registered; do not duplicate or replace it')
    ids = [row['task_id'] for row in tasks]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError('Empty or duplicate task inventory')
    added = []
    for task in tasks:
        status, references = classify(task, [], None)
        added.append(dict(release=release, task_id=task['task_id'],
            implementation_kind=task.get('implementation_kind'), phase=task.get('phase'),
            symbol=task.get('symbol'), formal_pointer=task.get('formal_pointer'),
            carrier=None, eligibility=status, reference_candidates=references,
            runtime_measurement_status='NOT_ASSESSED_BY_THIS_INVENTORY'))
    rows = [dict(row) for row in old] + added
    result = dict(inventory, records=rows, positions=len(rows),
        runtime_counts=dict(Counter(r['runtime_measurement_status'] for r in rows)),
        position_count_scope='RELEASE_QUALIFIED_OUTPUT_POSITIONS_NOT_UNIQUE_KERNEL_IMPLEMENTATIONS',
        release_extension=dict(release=release, added_positions=len(added),
            previous_measurement_labels_transferred=False),
        all_kernel_support_established=False)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(Path('/data1/tzh')):
        parser.error('Choose a new output under /data1/tzh')
    inventory = read(args.inventory)
    validate_retained_measurements(inventory)
    tasks = args.release/'same_dtype_tasks.json.gz'
    result = extend(inventory, args.release, read(tasks)['rows'])
    result['extension_input_sha256'] = {str(p.resolve()): sha(p) for p in (args.inventory, tasks)}
    save(args.output, result)
    print(result['release_extension'])
