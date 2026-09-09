"""Join reviewed NLL bodies to existing exact AOT parameter mappings."""
import argparse
from pathlib import Path
from scripts.run_numerical_coverage import read, save, sha


def bind(scan, mappings, tasks, release):
    if mappings.get('schema') != 'release-parameter-mappings-v1':
        raise ValueError('Existing parameter mapping required')
    if Path(mappings['release_path']).resolve() != release.resolve():
        raise ValueError('Mapping release differs')
    mapped = {r['task_id']: r for r in mappings['cases']}
    task_map = {r['task_id']: r for r in tasks['rows']}
    if len(mapped) != len(mappings['cases']) or len(task_map) != len(tasks['rows']):
        raise ValueError('Duplicate task IDs')
    contracts = {}
    for row in scan['records']:
        if row['status'] != 'SOURCE_CHECKED':
            continue
        path = Path(row['source']).resolve()
        if not path.is_relative_to(release.resolve()/'trace'):
            continue
        if sha(path) != row['source_sha256']:
            raise ValueError('Reviewed source changed')
        if row['symbol'] in contracts:
            raise ValueError('Ambiguous source symbol')
        contracts[row['symbol']] = row['contract']
    cases, unresolved = [], []
    for task in tasks['rows']:
        contract = contracts.get(task.get('symbol'))
        if contract is None or task.get('formal_pointer') != contract['output_pointer']:
            continue
        row = mapped.get(task['task_id'])
        if row is None:
            unresolved.append(dict(task_id=task['task_id'], reason='NO_PARAMETER_MAPPING'))
            continue
        if (task.get('implementation_kind') != 'TRITON' or task.get('phase') != 'BACKWARD'
                or row['expected_symbol'] != task['symbol']
                or row['exact_aot_endpoint_id'] != task['exact_aot_endpoint_id']):
            raise ValueError('Source/task/mapping identity differs')
        cases.append(dict(row, reference_method='SELECTED_NLL_COMMON_INPUT',
                          reference_output_pointer=contract['output_pointer']))
    return dict(schema='selected-nll-task-plan-v1', contracts=contracts, cases=cases,
                unresolved=unresolved, runtime_measurement_complete=False,
                parameter_scope='EXISTING_AOT_MAPPED_PARAMETER',
                numerical_results_used_for_selection=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('scan', 'mappings', 'release', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    if a.output.exists() or not a.output.resolve().is_relative_to(Path('/data1/tzh')):
        raise ValueError('New output under /data1/tzh required')
    scan, mappings = read(a.scan), read(a.mappings)
    for record in (scan, mappings):
        for name, digest in record['source_sha256'].items():
            if sha(Path(name)) != digest:
                raise ValueError('Frozen dependency changed: '+name)
    tasks_path = a.release/'same_dtype_tasks.json.gz'
    result = bind(scan, mappings, read(tasks_path), a.release)
    result['source_sha256'] = {str(path.resolve()): sha(path) for path in
                              (a.scan, a.mappings, tasks_path, Path(__file__))}
    save(a.output, result)
    print(dict(cases=len(result['cases']), unresolved=len(result['unresolved'])))


if __name__ == '__main__':
    main()
