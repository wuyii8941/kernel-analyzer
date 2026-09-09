#!/usr/bin/env python3
"""Join freshly verified family measurements to the complete position inventory."""
import argparse
from collections import Counter
import json
from pathlib import Path
from scripts.audit_family_plan_execution import audit
from scripts.run_numerical_coverage import read, sha


PLAN_REFERENCE_FAMILIES = {
    'grouped-causal-softmax-forward-bound-plan-v1': 'GROUPED_CAUSAL_SOFTMAX',
}


def validate_retained_measurements(inventory):
    """Do not carry a previous VERIFIED label after its raw evidence changes.

    This integrity check is not a new execution verification. The report retains
    that distinction; newly supplied families undergo the full audit below.
    """
    checked = 0
    for row in inventory['records']:
        status = row['runtime_measurement_status']
        if status not in ('VERIFIED', 'RECORDED_MEASUREMENT_CHECKED'):
            continue
        evidence = row.get('measurement_evidence', {})
        if (evidence.get('status') != status
                or evidence.get('task_id') != row['task_id']
                or not evidence.get('raw_artifact') or not evidence.get('raw_sha256')):
            raise ValueError('Missing retained measurement evidence')
        if sha(Path(evidence['raw_artifact'])) != evidence['raw_sha256']:
            raise ValueError('Retained measurement data changed')
        checked += 1
    return checked


def join(inventory, release, report, measurement_bindings=None):
    rows = [dict(r) for r in inventory['records']]
    index = {(str(Path(r['release']).resolve()), r['task_id']): r for r in rows}
    if len(index) != len(rows):
        raise ValueError('Duplicate inventory identity')
    seen = set()
    for measured in report['records']:
        key = (str(Path(release).resolve()), measured['task_id'])
        if key not in index or key in seen:
            raise ValueError('Unknown or duplicate measurement position')
        seen.add(key)
        row = index[key]
        row['runtime_measurement_status'] = measured['status']
        row['measurement_evidence'] = measured
        if measurement_bindings is not None:
            binding=measurement_bindings.get(measured['task_id'])
            if binding is None:
                raise ValueError('Measurement lacks audited plan binding')
            if row.get('carrier') not in (None,binding['carrier']):
                raise ValueError('Plan carrier differs from retained inventory carrier')
            row['carrier']=binding['carrier']
            references=list(row.get('reference_candidates',[]))
            family=binding.get('reference_family')
            if family and not any(ref.get('family')==family for ref in references):
                references.append(dict(family=family,source='AUDITED_BOUND_MEASUREMENT_PLAN'))
            row['reference_candidates']=references
    return dict(schema='coverage-measurement-join-v1', records=rows,
        positions=len(rows), runtime_counts=dict(Counter(
            r['runtime_measurement_status'] for r in rows)),
        verification_errors=inventory.get('verification_errors', []) + report['errors'],
        all_kernel_support_established=False,
        position_count_scope='RELEASE_QUALIFIED_OUTPUT_POSITIONS_NOT_UNIQUE_KERNEL_IMPLEMENTATIONS',
        scope='Observed output positions; verified fixed-suite measurement is not a bias or training-quality claim')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inventory', type=Path, required=True)
    p.add_argument('--release', type=Path, required=True)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--root', type=Path, nargs='*', default=[])
    p.add_argument('--queue', type=Path, nargs='*', default=[])
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists() or not a.output.resolve().is_relative_to(Path('/data1/tzh')):
        p.error('Use a new output under /data1/tzh')
    # Bind the release to the actual source plan, not a user-selected model label.
    task_path = (a.release/'same_dtype_tasks.json.gz').resolve()
    plan = read(a.plan)
    hashes = {str(Path(k).resolve()): v for k, v in plan.get('source_sha256', {}).items()}
    expected = hashes.get(str(task_path), plan.get('tasks_sha256'))
    if expected != sha(task_path):
        raise ValueError('Plan is not bound to this release')
    inventory = read(a.inventory)
    retained = validate_retained_measurements(inventory)
    family=PLAN_REFERENCE_FAMILIES.get(plan.get('schema'))
    bindings={case['task_id']:dict(carrier=case['carrier'],reference_family=family)
              for case in plan.get('cases',[])} if family else None
    result = join(inventory, a.release, audit(a.plan, a.root, a.queue),bindings)
    result['retained_measurement_integrity'] = dict(checked=retained,
        scope='Raw content hashes rechecked; prior execution verification is retained, not rerun')
    result['input_sha256'] = {str(x.resolve()): sha(x) for x in (a.inventory, a.plan, task_path)}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps(result['runtime_counts']))


if __name__ == '__main__':
    main()
