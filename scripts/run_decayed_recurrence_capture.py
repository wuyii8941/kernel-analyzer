#!/usr/bin/env python3
"""Multi-output recurrence adapter; reuse common-state capture and statistics."""
import argparse
from pathlib import Path
import sys
from scripts.run_numerical_coverage import read,save,sha,ROOT


PLAN_METHODS={
    'decayed-recurrence-bound-plan-v1':('DECAYED_RECURRENCE_COMMON_INPUT','DECAYED_RECURRENCE'),
    'segmented-recurrence-first-bound-plan-v1':('SEGMENTED_RECURRENCE_FIRST_COMMON_INPUT','SEGMENTED_RECURRENCE_FIRST'),
    'continued-recurrence-bound-plan-v1':('CONTINUED_RECURRENCE_COMMON_INPUT','CONTINUED_RECURRENCE'),
}


def plan_method(plan):
    if plan.get('schema') not in PLAN_METHODS:
        raise ValueError('Unknown recurrence plan schema')
    method,family=PLAN_METHODS[plan['schema']]
    if any(c.get('reference_method')!=method for c in plan['cases']):
        raise ValueError('Reference method differs from declared plan')
    return method,family


def select_contracts(plan, cases):
    original={c['case_id']:c for c in plan['cases']}
    if len(original)!=len(plan['cases']):
        raise ValueError('Duplicate case IDs in frozen recurrence plan')
    if len({c['case_id'] for c in cases})!=len(cases):
        raise ValueError('Duplicate selected recurrence cases')
    selected={}
    for case in cases:
        if case!=original.get(case['case_id']):
            raise ValueError('Selected case differs from frozen recurrence plan')
        pointer=case['reference_output_pointer']
        if pointer not in plan['contract']['output_pointers']:
            raise ValueError('Unknown recurrence output')
        if case['expected_symbol']!=plan['contract']['symbol']:
            raise ValueError('Recurrence symbol differs from checked source')
        if case['reference_output_index']!=plan['contract']['output_pointers'].index(pointer):
            raise ValueError('Recurrence output index differs from pointer')
        key=(case['expected_symbol'],pointer)
        contract=dict(plan['contract'],output_pointer=pointer)
        if key in selected and selected[key]!=contract:
            raise ValueError('Conflicting recurrence output contract')
        selected[key]=contract
    if not selected: raise ValueError('Empty recurrence selection')
    return selected


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('recurrence-plan','case-plan','training-bias-profile-v2-output-dir',
                 'output-dir','spool-dir','input-bank','release-dir','model'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--device',required=True)
    a,remaining=p.parse_known_args()
    output=a.training_bias_profile_v2_output_dir.resolve()
    for path in (output,a.output_dir,a.spool_dir):
        if path.exists() or not path.resolve().is_relative_to(Path('/data1/tzh')):
            raise ValueError('New output and spool paths on data disk required')
    plan=read(a.recurrence_plan)
    method,family=plan_method(plan)
    for path,digest in plan['source_sha256'].items():
        if sha(Path(path))!=digest: raise ValueError('Frozen recurrence input changed: '+path)
    cases=read(a.case_plan)['cases']
    selected=select_contracts(plan,cases)
    from kernel_analyzer import decayed_recurrence_reference as adapter
    from kernel_analyzer import decayed_recurrence_source as checker_module
    dependency_paths=[]
    if family=='CONTINUED_RECURRENCE':
        from kernel_analyzer import continued_recurrence_reference as adapter
        from kernel_analyzer import continued_recurrence_source as checker_module
        from kernel_analyzer import decayed_recurrence_reference as base_reference
        dependency_paths=[Path(base_reference.__file__)]
        def checked_source(source,symbol):
            return checker_module.check_source(source,symbol,time_start=plan['contract']['time_start'])
    elif family=='SEGMENTED_RECURRENCE_FIRST':
        from kernel_analyzer import segmented_recurrence_reference as adapter
        from kernel_analyzer import segmented_recurrence_source as segment_checker
        from kernel_analyzer import decayed_recurrence_reference as base_reference
        dependency_paths=[Path(segment_checker.__file__),Path(base_reference.__file__)]
        def checked_source(source,symbol):
            return segment_checker.check_first_segment(source,symbol,time_start=plan['contract']['time_start'])
    else:
        checked_source=adapter.check_source
    # The checker dependency is declared separately, not hidden in an import.
    for contract in selected.values():
        contract['checker_dependency_sha256']=sha(Path(checker_module.__file__))
    translated=output.parent/'recurrence_capture_plan.json'
    if translated.exists(): raise ValueError('Do not overwrite translated plan')
    save(translated,dict(cases=[dict(c,reference_method='PARTIAL_REDUCTION_FROM_BOUND_INPUT',
        declared_reference_method=c['reference_method']) for c in cases]))
    arguments=list(remaining)
    for name,value in [('case-plan',translated),('training-bias-profile-v2-output-dir',output),
        ('output-dir',a.output_dir),('spool-dir',a.spool_dir),('input-bank',a.input_bank),
        ('release-dir',a.release_dir),('model',a.model),('device',a.device)]:
        arguments.extend(['--'+name,str(value)])
    from scripts import capture_bound_endpoint_bias_formation_v21 as capture
    original=capture.SameDtypeSemanticCandidateObserver
    class CheckedObserver(original):
        def __init__(self,**kwargs):
            for symbol in {key[0] for key in selected}:
                expected=next(c['function_ast_sha256'] for (s,_),c in selected.items() if s==symbol)
                found=[checked_source(Path(m.__file__).read_text(),symbol)['function_ast_sha256']
                       for m in kwargs['modules'] if hasattr(m,symbol)]
                if found!=[expected]: raise ValueError('Live recurrence source differs')
            super().__init__(**kwargs)
    def reference(metadata,candidate,**unused):
        key=(metadata.get('symbol'),metadata.get('formal_pointer'))
        if key not in selected: raise ValueError('Unselected recurrence output')
        return adapter.reference(metadata,candidate,selected[key])
    capture.SameDtypeSemanticCandidateObserver=CheckedObserver
    capture.partial_reduction_reference=reference
    old_scope=capture.reference_scope
    capture.reference_scope=lambda method: dict(comparison='COMMON_OPERAND_SOURCE_CHECKED_'+family,
        same_local_operands=True,includes_possible_upstream_differences=False,reference_variant='FP32_NATIVE',
        single_kernel_source_attribution='CHECKED_FUNCTION_AST_AND_PRE_CALL_INPUTS') if method=='PARTIAL_REDUCTION_FROM_BOUND_INPUT' else old_scope(method)
    paths=[Path(__file__),Path(adapter.__file__),Path(checker_module.__file__),a.recurrence_plan,a.case_plan,
        translated,a.input_bank,a.model/'config.json',*(a.release_dir/n for n in ('same_dtype_tasks.json.gz','campaign.json.gz','inventory.json.gz'))]
    paths+=dependency_paths
    paths += [ROOT/'scripts'/n for n in ('capture_bound_endpoint_bias_formation_v21.py','same_dtype_semantic_observer.py',
        'run_parallel_bound_capture.py','run_training_bias_profile_v2_empirical.py')]
    paths += [ROOT/'src/kernel_analyzer'/n for n in ('parallel_measurement.py','training_numerical_analysis.py',
        'training_equivalence.py','training_bias_profile.py','update_write.py','capture_cost.py')]
    save(output/'family_execution_protocol.json',dict(schema='decayed-recurrence-capture-v1',
        reference_family=family,variant='FP32_NATIVE',parallel_measurement=True,
        contracts=[dict(c) for c in selected.values()],capture_arguments=arguments,
        carriers=sorted({c['carrier'] for c in cases}),primary_stage='PARAMETER_WRITE',
        claim_scope='FIXED_SUITE_UPDATE',fixed_suite_margins=dict(full_update_rms=.01),
        source_sha256={str(x.resolve()):sha(x) for x in paths},statistical_method_changed=False))
    from kernel_analyzer.capture_cost import measured_capture
    from scripts.run_parallel_bound_capture import main as run
    sys.argv=[sys.argv[0],*arguments]
    measured_capture(run,device=a.device,emit=lambda m:save(output/'capture_cost.json',m))


if __name__=='__main__':main()
