#!/usr/bin/env python3
"""Bind every checked residual RMS output to recorded parameter paths."""
import argparse
from pathlib import Path
from scripts.scan_residual_rms_forward_sources import scan
from scripts.run_numerical_coverage import read, save, sha
from kernel_analyzer import residual_rms_forward_source, residual_rms_forward_reference


def bind(tasks, contracts, mapping):
    by_endpoint = {}
    for row in mapping['rows']:
        if row['endpoint'] in by_endpoint:
            raise ValueError('Duplicate endpoint mapping')
        by_endpoint[row['endpoint']] = row
    seen, cases, unresolved = set(), [], []
    for task in tasks:
        if task['task_id'] in seen:
            raise ValueError('Duplicate task identity')
        seen.add(task['task_id'])
        contract = contracts.get(task.get('symbol'))
        if contract is None:
            continue
        endpoint = task.get('exact_aot_endpoint_id')
        parameters = by_endpoint.get(endpoint, {}).get('parameters', [])
        if (task.get('status') != 'EXACT_CANDIDATE_BUFFER_TO_AOT_SEMANTIC_ENDPOINT'
                or task.get('implementation_kind') != 'TRITON'
                or task.get('formal_pointer') not in contract['output_pointers']
                or not str(endpoint).startswith('forward:') or not parameters):
            unresolved.append(dict(task_id=task['task_id'], reason='SOURCE_OUTPUT_OR_PARAMETER_PATH_UNAVAILABLE'))
            continue
        selected = min(parameters, key=lambda p: (p['aot_distance'], p['name']))
        cases.append(dict(case_id='mapped_' + task['task_id'].replace(':', '_') + '-residual-rms-forward',
            task_id=task['task_id'], carrier=selected['name'], mapping_evidence=selected,
            parameter_scope='SELECTED_PARAMETER_ONLY', expected_symbol=task['symbol'],
            exact_aot_endpoint_id=endpoint, reference_output_pointer=task['formal_pointer'],
            reference_method='RESIDUAL_RMS_FORWARD_COMMON_INPUT',
            runtime_parameter_reach='NOT_YET_MEASURED'))
    return dict(cases=cases, unresolved=unresolved, contracts=contracts,
                runtime_measurement_complete=False, numerical_results_read=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'tasks', 'mapping', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(Path('/data1/tzh')):
        parser.error('Choose a new output under /data1/tzh')
    mapping = read(args.mapping)
    if mapping.get('schema') != 'forward-parameter-paths-v1':
        raise ValueError('Audited forward path mapping required')
    if mapping.get('source_sha256', {}).get(str(args.tasks.resolve())) != sha(args.tasks):
        raise ValueError('Task source identity differs')
    for path, expected in mapping['source_sha256'].items():
        if sha(Path(path)) != expected:
            raise ValueError('Mapping dependency changed: ' + path)
    rows = scan(args.source.read_text())
    contracts = {r['symbol']: r['contract'] for r in rows if r['status'] == 'SOURCE_CHECKED'}
    result = bind(read(args.tasks)['rows'], contracts, mapping)
    result['schema'] = 'residual-rms-forward-bound-plan-v1'
    result['source_scan'] = rows
    paths = [args.source, args.tasks, args.mapping, Path(__file__),
             Path(residual_rms_forward_source.__file__), Path(residual_rms_forward_reference.__file__)]
    result['source_sha256'] = {str(p.resolve()): sha(p) for p in paths}
    save(args.output, result)
    print(dict(bound=len(result['cases']), unresolved=len(result['unresolved'])))


if __name__ == '__main__':
    main()
