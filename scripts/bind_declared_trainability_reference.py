#!/usr/bin/env python3
"""Bind any registered source reference under a frozen trainability scope.

This path is for historical or deliberately restricted experiments where the
trainable parameter set is already declared but a complete graph-distance map
is unavailable. Runtime capture must still demonstrate a nonzero parameter
response; this binder never claims graph reach from source matching alone.
"""
import argparse
import json
from pathlib import Path

from kernel_analyzer.source_reference_registry import get_reference
from scripts.build_row_reduction_contracts import unique_contracts
from scripts.run_numerical_coverage import read, save, sha


ALLOWED_TASK_STATUS = {
    'EXACT_CANDIDATE_BUFFER_TO_AOT_SEMANTIC_ENDPOINT',
    'INTERNAL_IMPLEMENTATION_BUFFER_COVERED_BY_CLOSED_SEMANTIC_ENDPOINT',
}


def bind(manifest, tasks, trainability):
    family = manifest.get('reference_family')
    specification = get_reference(family)
    parameters = trainability.get('trainable_parameters', [])
    if len(parameters) != 1 or not isinstance(parameters[0], str):
        raise ValueError('Exactly one previously declared trainable parameter is required')
    contracts = unique_contracts(manifest['sources'])
    cases, unresolved, seen = [], [], set()
    for task in tasks['rows']:
        task_id = task.get('task_id')
        if task_id in seen:
            raise ValueError('Duplicate task identity')
        seen.add(task_id)
        contract = contracts.get(task.get('symbol'))
        if contract is None or task.get('formal_pointer') != contract['output_pointer']:
            continue
        if task.get('implementation_kind') != 'TRITON':
            unresolved.append(dict(task_id=task_id, reason='IMPLEMENTATION_IS_NOT_TRITON'))
            continue
        if task.get('status') not in ALLOWED_TASK_STATUS:
            unresolved.append(dict(task_id=task_id, reason='NO_DECLARED_SEMANTIC_BOUNDARY'))
            continue
        if (task.get('status') == 'INTERNAL_IMPLEMENTATION_BUFFER_COVERED_BY_CLOSED_SEMANTIC_ENDPOINT'
                and task.get('closure_uses_candidate_values') is not False):
            unresolved.append(dict(task_id=task_id, reason='CLOSURE_SELECTION_NOT_SOURCE_ONLY'))
            continue
        cases.append(dict(
            case_id=(family.lower() + '_' + task_id.replace(':', '_')
                     + specification.case_suffix),
            task_id=task_id,
            carrier=parameters[0],
            reference_method='PARTIAL_REDUCTION_FROM_BOUND_INPUT',
            reference_contract_symbol=task['symbol'],
            implementation_kind='TRITON',
            phase=task.get('phase'),
            exact_aot_endpoint_id=task.get('exact_aot_endpoint_id'),
            historical_boundary_status=task.get('status'),
            downstream_closure_tasks=task.get('closed_by_semantic_endpoint_tasks', []),
            parameter_scope='DECLARED_SINGLE_TRAINABLE_PARAMETER',
            parameter_selection='FROZEN_TRAINABILITY_PROTOCOL_NOT_INFERRED_GRAPH_REACH',
            runtime_parameter_reach='NOT_YET_MEASURED',
        ))
    return dict(
        schema='registered-reference-declared-trainability-plan-v1',
        reference_family=family,
        cases=cases,
        unresolved=unresolved,
        contracts=contracts,
        trainable_parameters=parameters,
        runtime_measurement_complete=False,
        source_selection_uses_numerical_results=False,
        scope=('Source and historical trainability binding only; runtime execution and '
               'parameter response remain required'),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('reference-manifest', 'tasks', 'trainability', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(Path('/data1/tzh')):
        parser.error('Choose a new output under /data1/tzh')
    manifest = read(args.reference_manifest)
    specification = get_reference(manifest.get('reference_family'))
    adapter = (Path(__file__).resolve().parents[1] / 'src/kernel_analyzer'
               / specification.source_filename)
    if sha(adapter) != manifest.get('adapter_sha256'):
        raise ValueError('Reference adapter differs from source scan')
    for source in manifest['sources']:
        if sha(Path(source['source'])) != source['source_sha256']:
            raise ValueError('Scanned source changed')
    report = bind(manifest, read(args.tasks), read(args.trainability))
    dependencies = (Path(__file__), args.reference_manifest, args.tasks,
                    args.trainability, adapter,
                    Path(__file__).resolve().parents[1]
                    / 'src/kernel_analyzer/source_reference_registry.py')
    report['source_sha256'] = {str(path.resolve()): sha(path) for path in dependencies}
    save(args.output, report)
    print(json.dumps(dict(bound=len(report['cases']), unresolved=len(report['unresolved']))))


if __name__ == '__main__':
    main()
