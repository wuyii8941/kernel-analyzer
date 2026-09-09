#!/usr/bin/env python3
"""Bind source-checked outputs to declared downstream parameters.

An internal output need not equal an AOT tensor to use its own same-input
reference. Existing semantic-closure mappings select a parameter to measure;
they are not substituted for proof of its actual runtime response.

Exact AOT endpoints use their direct parameter mapping.  This keeps newly
reviewed reference families on one binding path instead of requiring a
family-specific case selector.
"""
import argparse
import json
from pathlib import Path

from scripts.build_row_reduction_contracts import unique_contracts
from scripts.finalize_numerical_family import read, sha
from scripts.run_numerical_coverage import read as read_compressed
from kernel_analyzer.source_reference_registry import get_reference


def normalized_task_mappings(tasks, mappings):
    """Accept historical case maps or the generic forward-path artifact.

    Forward families used to require a family-specific adapter merely to turn
    an exact endpoint path into the same shortest-distance carrier record used
    by backward families.  Normalize that source-only graph result here so new
    reviewed forward Triton families can share the existing binder.
    """
    if 'cases' in mappings:
        return list(mappings['cases'])
    if mappings.get('schema') != 'forward-parameter-paths-v1':
        raise ValueError('Unsupported parameter mapping schema')
    by_endpoint = {str(row['endpoint']): row for row in mappings.get('rows', [])}
    if len(by_endpoint) != len(mappings.get('rows', [])):
        raise ValueError('Duplicate forward endpoint mapping')
    result = []
    for task in tasks.get('rows', []):
        endpoint = task.get('exact_aot_endpoint_id')
        if not endpoint:
            continue
        path = by_endpoint.get(str(endpoint))
        if not path or not path.get('parameters'):
            continue
        parameter = min(
            path['parameters'],
            key=lambda row: (int(row['aot_distance']), str(row['name'])),
        )
        result.append({
            'task_id': str(task['task_id']),
            'carrier': str(parameter['name']),
            'mapping_evidence': {
                **parameter,
                'exact_aot_endpoint_id': str(endpoint),
                'selection_rule': 'SHORTEST_AOT_DISTANCE_THEN_PARAMETER_NAME',
            },
        })
    return result


def bind(manifest, tasks, mappings):
    specification = get_reference(manifest['reference_family'])
    contracts = unique_contracts(manifest['sources'])
    mapping_rows = normalized_task_mappings(tasks, mappings)
    mapped = {r['task_id']: r for r in mapping_rows}
    if len(mapped) != len(mapping_rows):
        raise ValueError('Duplicate mapped task')
    cases, unresolved = [], []
    selected = [t for t in tasks['rows'] if t.get('symbol') in contracts
                and t.get('formal_pointer') == contracts[t['symbol']]['output_pointer']]
    if len({t['task_id'] for t in selected}) != len(selected):
        raise ValueError('Duplicate internal task')
    for task in selected:
        if task.get('status') == 'EXACT_CANDIDATE_BUFFER_TO_AOT_SEMANTIC_ENDPOINT':
            chosen = mapped.get(task['task_id'])
            if not chosen or not chosen.get('carrier'):
                unresolved.append(dict(task_id=task['task_id'], reason='NO_DIRECT_PARAMETER_MAPPING'))
                continue
            selection_rule = 'EXACT_AOT_ENDPOINT_DIRECT_PARAMETER_MAPPING'
        elif task.get('status') == 'INTERNAL_IMPLEMENTATION_BUFFER_COVERED_BY_CLOSED_SEMANTIC_ENDPOINT':
            if task.get('closure_uses_candidate_values') is not False:
                unresolved.append(dict(task_id=task['task_id'], reason='CLOSURE_SELECTION_NOT_SOURCE_ONLY'))
                continue
            downstream = [mapped[t] for t in task.get('closed_by_semantic_endpoint_tasks', [])
                          if t in mapped and mapped[t].get('carrier')]
            if not downstream:
                unresolved.append(dict(task_id=task['task_id'], reason='NO_MAPPED_DOWNSTREAM_PARAMETER'))
                continue
            chosen = min(downstream, key=lambda r: (r['mapping_evidence']['aot_distance'],
                                                   r['carrier'], r['task_id']))
            selection_rule = 'DECLARED_CLOSURE_SHORTEST_AOT_DISTANCE_THEN_NAME_AND_TASK'
        else:
            unresolved.append(dict(task_id=task['task_id'], reason='NO_DECLARED_PARAMETER_REACH_PATH'))
            continue
        cases.append(dict(case_id='internal_'+task['task_id'].replace(':','_')+specification.case_suffix,
            task_id=task['task_id'], carrier=chosen['carrier'],
            reference_method='REGISTERED_SAME_INPUT_REFERENCE',
            reference_family=manifest['reference_family'],
            reference_variant=specification.variants[0],
            reference_contract=contracts[task['symbol']],
            reference_contract_symbol=task['symbol'], implementation_kind=task['implementation_kind'],
            exact_aot_endpoint_id=task.get('exact_aot_endpoint_id'),
            parameter_selection_rule=selection_rule,
            parameter_selection_evidence=dict(downstream_task_id=chosen['task_id'],
                mapping_evidence=chosen['mapping_evidence']),
            runtime_parameter_reach='NOT_YET_MEASURED',
            comparison_scope='INTERNAL_BUFFER_COMMON_INPUT_SUBSTITUTION_NOT_AOT_TENSOR_EQUALITY'))
    return dict(schema='source-checked-reference-plan-v2', cases=cases,
        unresolved=unresolved, source_checked_internal_positions=len(selected),
        source_total_positions=len(tasks['rows']),
        positions_outside_selected_reference=len(tasks['rows'])-len(selected),
        source_selection_uses_numerical_results=False,
        all_source_positions_supported=False,
        scope='Declared parameter selected from existing closure; runtime identity and response require capture')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-manifest',type=Path,required=True)
    parser.add_argument('--tasks',type=Path,required=True)
    parser.add_argument('--parameter-map',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args()
    if a.output.exists() or not a.output.resolve().is_relative_to(Path('/data1/tzh')):
        parser.error('Choose a new output under /data1/tzh')
    report=bind(read(a.reference_manifest),read_compressed(a.tasks),read(a.parameter_map))
    report['source_sha256']={str(p.resolve()):sha(p) for p in
        (a.reference_manifest,a.tasks,a.parameter_map,Path(__file__))}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as stream: json.dump(report,stream,indent=2,allow_nan=False)
    print(json.dumps(dict(bound=len(report['cases']), unresolved=len(report['unresolved']))))


if __name__=='__main__': main()
