#!/usr/bin/env python3
"""Bind all observed outputs of an explicitly declared continued segment."""
import argparse
from pathlib import Path
from scripts.run_numerical_coverage import read, save, sha
from scripts.build_reference_reach_inventory import merge_carriers
from kernel_analyzer import continued_recurrence_source, continued_recurrence_reference
from kernel_analyzer import decayed_recurrence_reference


def bind(tasks, mapping, contract):
    if len({c['task_id'] for c in mapping}) != len(mapping):
        raise ValueError('Ambiguous parameter mapping')
    if len({t['task_id'] for t in tasks}) != len(tasks):
        raise ValueError('Duplicate observed task')
    carriers = {}
    merge_carriers(carriers, mapping, tasks)
    by_task = {c['task_id']: c for c in mapping}
    cases, unresolved = [], []
    for task in tasks:
        if task.get('symbol') != contract['symbol']:
            continue
        pointer = task.get('formal_pointer')
        if (pointer not in contract['output_pointers'] or task['task_id'] not in carriers
                or task['task_id'] not in by_task):
            unresolved.append(dict(task_id=task['task_id'], reason='OUTPUT_OR_PARAMETER_BINDING_UNAVAILABLE'))
            continue
        original = by_task[task['task_id']]
        cases.append(dict(original, case_id=original['case_id']+'-continued-recurrence',
            source_case_id=original['case_id'], reference_method='CONTINUED_RECURRENCE_COMMON_INPUT',
            reference_output_pointer=pointer,
            reference_output_index=contract['output_pointers'].index(pointer),
            runtime_parameter_reach='NOT_YET_MEASURED'))
    if not cases and not unresolved:
        raise ValueError('Declared symbol has no observed tasks')
    return cases, unresolved


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'mapping', 'tasks', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--symbol', required=True)
    p.add_argument('--sequence-length', type=int, required=True)
    p.add_argument('--time-start', type=int, required=True)
    a = p.parse_args()
    if a.output.exists() or not a.output.resolve().is_relative_to(Path('/data1/tzh')):
        p.error('Choose new output under /data1/tzh')
    if not 0 < a.time_start < a.sequence_length:
        p.error('Time origin outside sequence')
    contract = continued_recurrence_source.check_source(a.source.read_text(), a.symbol,
                                                      time_start=a.time_start)
    contract['sequence_length'] = a.sequence_length
    contract['reference_dependencies_sha256'] = continued_recurrence_reference.dependency_hashes()
    cases, unresolved = bind(read(a.tasks)['rows'], read(a.mapping)['cases'], contract)
    paths = [a.source, a.mapping, a.tasks, Path(__file__),
             Path(continued_recurrence_source.__file__), Path(continued_recurrence_reference.__file__),
             Path(decayed_recurrence_reference.__file__)]
    save(a.output, dict(schema='continued-recurrence-bound-plan-v1', contract=contract,
        cases=cases, unresolved=unresolved, tasks_sha256=sha(a.tasks), numerical_results_read=False,
        runtime_measurement_complete=False, source_sha256={str(x.resolve()): sha(x) for x in paths}))
    print(dict(bound_positions=len(cases), unresolved_positions=len(unresolved)))


if __name__ == '__main__':
    main()
