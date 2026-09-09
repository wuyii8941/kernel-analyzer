"""Extend coverage with omitted task packages, retaining unverified status.

Identical task files in different releases do not establish identical execution
protocols. Keep release-qualified identities and report duplication separately.
"""
import argparse
from collections import Counter
from pathlib import Path
from scripts.run_numerical_coverage import read, save, sha


def extend(inventory, audit):
    records = list(inventory['records'])
    seen = {(r['release'], r['task_id']) for r in records}
    if len(seen) != len(records):
        raise ValueError('Duplicate input identity')
    sources, additions = {}, []
    for package in audit['rows']:
        if package['included_in_current_inventory']:
            continue
        if package['status'] != 'TASK_PACKAGE_READ':
            additions.append(dict(release=package['release'], status='PACKAGE_INVALID_RETAINED', added=0))
            continue
        path = Path(package['task_file'])
        if sha(path) != package['task_sha256']:
            raise ValueError('Historical task package changed')
        sources[str(path)] = package['task_sha256']
        tasks = read(path)['rows']
        missing = package['missing_package_files']
        for task in tasks:
            identity = (package['release'], task['task_id'])
            if identity in seen:
                raise ValueError('Duplicate historical identity')
            seen.add(identity)
            records.append(dict(release=identity[0], task_id=identity[1],
                implementation_kind=task.get('implementation_kind'), phase=task.get('phase'),
                symbol=task.get('symbol'), formal_pointer=task.get('formal_pointer'),
                carrier=None, reference_candidates=[],
                eligibility='HISTORICAL_PACKAGE_INCOMPLETE' if missing else 'HISTORICAL_REFERENCE_AND_BINDING_NOT_REAUDITED',
                runtime_measurement_status='NOT_ASSESSED_BY_THIS_INVENTORY',
                historical_task_status=task.get('status'), missing_package_files=missing,
                historical_task_sha256=package['task_sha256']))
        additions.append(dict(release=package['release'], status='ADDED_UNASSESSED', added=len(tasks)))
    return dict(schema='historical-extended-reference-inventory-v1', records=records,
        positions=len(records), releases=len({r['release'] for r in records}),
        eligibility_counts=dict(Counter(r['eligibility'] for r in records)),
        runtime_counts=dict(Counter(r['runtime_measurement_status'] for r in records)),
        additions=additions, historical_task_sources=sources,
        duplicate_file_groups=audit['duplicate_file_groups'],
        scope='Release-qualified output positions, including historical duplicates; not unique kernels or operator families',
        new_measurement_count=0, all_kernel_support_established=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('inventory','audit','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists() or not a.output.resolve().is_relative_to(Path('/data1/tzh')):
        p.error('New output under /data1/tzh required')
    audit=read(a.audit)
    if audit['source_sha256'].get(str(a.inventory.resolve())) != sha(a.inventory):
        raise ValueError('Audit belongs to another input inventory')
    report=extend(read(a.inventory),audit)
    report['source_sha256']={str(path.resolve()):sha(path) for path in (a.inventory,a.audit,Path(__file__))}
    save(a.output,report)
    print(dict(positions=report['positions'],releases=report['releases'],new_measurements=0))


if __name__=='__main__': main()
