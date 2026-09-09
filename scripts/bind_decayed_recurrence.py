#!/usr/bin/env python3
"""Bind all checked recurrence outputs without reading numerical outcomes."""
import argparse
from pathlib import Path
from scripts.run_numerical_coverage import read,save,sha
from scripts.build_reference_reach_inventory import merge_carriers
from kernel_analyzer.decayed_recurrence_source import check_source


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','mapping','tasks','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--symbol',required=True)
    a=p.parse_args()
    if a.output.exists() or not a.output.resolve().is_relative_to(Path('/data1/tzh')):
        p.error('Use a new output under /data1/tzh')
    contract=check_source(a.source.read_text(),a.symbol)
    tasks=read(a.tasks)['rows']; mapping=read(a.mapping)['cases']
    carriers={}; merge_carriers(carriers,mapping,tasks)
    mapped={c['task_id']:c for c in mapping}
    cases=[]; unresolved=[]
    for task in tasks:
        if task.get('symbol')!=a.symbol: continue
        pointer=task.get('formal_pointer')
        if pointer not in contract['output_pointers'] or task['task_id'] not in carriers:
            unresolved.append(dict(task_id=task['task_id'],reason='OUTPUT_OR_PARAMETER_BINDING_UNAVAILABLE'))
            continue
        original=mapped[task['task_id']]
        cases.append(dict(original,case_id=original['case_id']+'-decayed-recurrence-common-input',
            source_case_id=original['case_id'],reference_method='DECAYED_RECURRENCE_COMMON_INPUT',
            reference_output_pointer=pointer,reference_output_index=contract['output_pointers'].index(pointer),
            runtime_parameter_reach='NOT_YET_MEASURED'))
    save(a.output,dict(schema='decayed-recurrence-bound-plan-v1',contract=contract,cases=cases,
        unresolved=unresolved,tasks_sha256=sha(a.tasks),numerical_results_read=False,
        runtime_adapter_complete=False,
        source_sha256={str(x.resolve()):sha(x) for x in (a.source,a.mapping,a.tasks,Path(__file__),
            Path('src/kernel_analyzer/decayed_recurrence_source.py'),Path('src/kernel_analyzer/decayed_recurrence_reference.py'))}))
    print(dict(bound_positions=len(cases),unresolved_positions=len(unresolved)))


if __name__=='__main__':main()
