#!/usr/bin/env python3
"""Bind a checked first recurrence segment without selecting numerical results."""
import argparse
from pathlib import Path
from scripts.run_numerical_coverage import read,save,sha
from scripts.build_reference_reach_inventory import merge_carriers
from kernel_analyzer import segmented_recurrence_source,decayed_recurrence_source
from kernel_analyzer import segmented_recurrence_reference,decayed_recurrence_reference


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','mapping','tasks','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--symbol',required=True)
    p.add_argument('--sequence-length',type=int,required=True)
    a=p.parse_args()
    if a.output.exists() or not a.output.resolve().is_relative_to(Path('/data1/tzh')):
        p.error('Use new output under /data1/tzh')
    contract=segmented_recurrence_source.check_first_segment(a.source.read_text(),a.symbol,
                                                           time_start=a.sequence_length-1)
    contract['sequence_length']=a.sequence_length
    checkers=[Path(m.__file__).resolve() for m in (segmented_recurrence_source,decayed_recurrence_source)]
    contract['checker_dependencies_sha256']={str(path):sha(path) for path in checkers}
    tasks=read(a.tasks)['rows']; mapping=read(a.mapping)['cases']; carriers={}
    merge_carriers(carriers,mapping,tasks)
    by_task={c['task_id']:c for c in mapping}; cases=[]; unresolved=[]
    for task in tasks:
        if task.get('symbol')!=a.symbol: continue
        pointer=task.get('formal_pointer')
        if pointer not in contract['output_pointers'] or task['task_id'] not in carriers:
            unresolved.append(dict(task_id=task['task_id'],reason='OUTPUT_OR_PARAMETER_BINDING_UNAVAILABLE'))
            continue
        original=by_task[task['task_id']]
        cases.append(dict(original,case_id=original['case_id']+'-segmented-recurrence-first',
            source_case_id=original['case_id'],reference_method='SEGMENTED_RECURRENCE_FIRST_COMMON_INPUT',
            reference_output_pointer=pointer,reference_output_index=contract['output_pointers'].index(pointer),
            runtime_parameter_reach='NOT_YET_MEASURED'))
    paths=[a.source,a.mapping,a.tasks,Path(__file__),*checkers,
           Path(segmented_recurrence_reference.__file__),Path(decayed_recurrence_reference.__file__)]
    save(a.output,dict(schema='segmented-recurrence-first-bound-plan-v1',contract=contract,cases=cases,
         unresolved=unresolved,tasks_sha256=sha(a.tasks),numerical_results_read=False,
         runtime_measurement_complete=False,source_sha256={str(x.resolve()):sha(x) for x in paths}))
    print(dict(bound_positions=len(cases),unresolved_positions=len(unresolved)))


if __name__=='__main__': main()
