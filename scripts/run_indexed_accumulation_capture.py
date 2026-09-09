#!/usr/bin/env python3
"""Adapt indexed accumulation to the existing common-state capture/statistics.

The translated reference_method is only an existing callback slot. The frozen
protocol preserves INDEXED_INPUT_ORDER_FP32 as the actual reference semantics.
"""
import argparse
import os
from pathlib import Path
import sys
from scripts.run_numerical_coverage import read,save,sha,ROOT


def translate_plan(plan):
    if not plan.get('cases'): raise ValueError('No exactly bound indexed cases')
    cases=[]
    for case in plan['cases']:
        if (case.get('reference_method')!='INDEXED_INPUT_ORDER_FP32'
                or case.get('implementation_kind')!='DIRECT_ATEN'
                or case.get('expected_symbol')!='index_put_' or not case.get('exact_aot_endpoint_id')):
            raise ValueError('Plan is not an exact indexed-accumulation binding')
        cases.append(dict(case,reference_method='PARTIAL_REDUCTION_FROM_BOUND_INPUT',
                          declared_reference_method='INDEXED_INPUT_ORDER_FP32'))
    return dict(cases=cases,scope='Compatibility callback translation; not a reduction semantic claim')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--indexed-contracts',type=Path,required=True)
    p.add_argument('--case-plan',type=Path,required=True)
    p.add_argument('--training-bias-profile-v2-output-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--spool-dir',type=Path,required=True)
    p.add_argument('--input-bank',type=Path,required=True)
    p.add_argument('--release-dir',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--device',required=True)
    a,remaining=p.parse_known_args()
    output=a.training_bias_profile_v2_output_dir.resolve()
    for path in (output,a.output_dir,a.spool_dir):
        if path.exists() or not path.resolve().is_relative_to(Path('/data1/tzh')):
            raise ValueError('Use new output/spool directories under /data1/tzh')
    original=read(a.case_plan); translated=translate_plan(original)
    declaration=read(a.indexed_contracts)
    for name,digest in declaration['source_sha256'].items():
        if sha(Path(name))!=digest: raise ValueError('Indexed declaration input changed: '+name)
    contracts={c['task_id']:c for c in declaration['contracts']}
    if set(contracts)!={c['task_id'] for c in translated['cases']}:
        raise ValueError('Source contracts and selected cases differ')
    plan_path=output.parent/'indexed_capture_plan.json'
    if plan_path.exists(): raise ValueError('Do not overwrite translated plan')
    save(plan_path,translated)
    arguments=[*remaining]
    for name,value in [('--case-plan',plan_path),('--training-bias-profile-v2-output-dir',output),
        ('--output-dir',a.output_dir),('--spool-dir',a.spool_dir),('--input-bank',a.input_bank),
        ('--release-dir',a.release_dir),('--model',a.model),('--device',a.device)]:
        arguments.extend([name,str(value)])
    os.environ.update(HF_HOME='/data1/tzh/cache/huggingface',XDG_CACHE_HOME='/data1/tzh/cache/xdg',
        TRITON_CACHE_DIR='/data1/tzh/cache/triton',TORCHINDUCTOR_CACHE_DIR='/data1/tzh/cache/torchinductor',
        PYTHONDONTWRITEBYTECODE='1')
    from scripts import capture_bound_endpoint_bias_formation_v21 as capture
    from scripts.indexed_accumulation_observer import IndexedAccumulationObserver
    from scripts.indexed_runtime_binding import bind_live_call
    from kernel_analyzer.indexed_row_accumulation_reference import reference
    observed=[]

    class BoundObserver(IndexedAccumulationObserver):
        def __init__(self,**kwargs):
            super().__init__(**kwargs)
            live={task:bind_live_call(c,self.modules) for task,c in contracts.items()}
            sink=self.sink
            def checked_sink(task,tensor,metadata):
                c=live[task]
                for key in ('executing_filename','executing_line','source_line_sha256'):
                    if metadata.get(key)!=c[key]: raise RuntimeError('Executed indexed call differs: '+key)
                sink(task,tensor,dict(metadata,indexed_live_contract=c))
            self.sink=checked_sink
            observed.append(live)

    capture.SameDtypeSemanticCandidateObserver=BoundObserver
    capture.partial_reduction_reference=lambda metadata,candidate,**unused:reference(
        metadata,candidate,metadata['indexed_live_contract'])
    old_scope=capture.reference_scope
    family='INDEXED_ROW_ACCUMULATION'
    capture.reference_scope=lambda method: dict(comparison='COMMON_OPERAND_SOURCE_CHECKED_'+family,
        same_local_operands=True,includes_possible_upstream_differences=False,
        reference_variant='FP32_INPUT_ORDER',single_kernel_source_attribution='DIRECT_ATEN_SELECTED_CALL_ONLY',
        reference_is_absolute_truth=False) if method=='PARTIAL_REDUCTION_FROM_BOUND_INPUT' else old_scope(method)
    sources=[Path(__file__),a.case_plan,plan_path,a.indexed_contracts,a.input_bank,a.model/'config.json',
        *(a.release_dir/f for f in ('same_dtype_tasks.json.gz','campaign.json.gz','inventory.json.gz'))]
    sources += [ROOT/'scripts'/name for name in ('indexed_accumulation_observer.py','indexed_runtime_binding.py',
        'build_indexed_call_contracts.py','same_dtype_semantic_observer.py','capture_bound_endpoint_bias_formation_v21.py',
        'run_parallel_bound_capture.py','run_training_bias_profile_v2_empirical.py')]
    sources += [ROOT/'src/kernel_analyzer'/name for name in ('indexed_row_accumulation_reference.py',
        'parallel_measurement.py','training_numerical_analysis.py','training_equivalence.py',
        'training_bias_profile.py','update_write.py','capture_cost.py')]
    sources += [Path(c['source_path']) for c in contracts.values()]
    save(output/'family_execution_protocol.json',dict(schema='indexed-accumulation-capture-v1',
        reference_family=family,variant='FP32_INPUT_ORDER',contracts=contracts,
        capture_arguments=arguments,carriers=sorted({c['carrier'] for c in translated['cases']}),
        primary_stage='PARAMETER_WRITE',claim_scope='FIXED_SUITE_UPDATE',
        fixed_suite_margins=dict(full_update_rms=.01),parallel_measurement=True,
        source_sha256={str(path.resolve()):sha(path) for path in sources},statistical_method_changed=False,
        data_use='DECLARED_VARIANT_NOT_BLIND_IMPLEMENTATION_CONFIRMATION'))
    from kernel_analyzer.capture_cost import measured_capture
    from scripts.run_parallel_bound_capture import main as run
    sys.argv=[sys.argv[0],*arguments]
    measured_capture(run,device=a.device,emit=lambda m:save(output/'capture_cost.json',m))
    save(output/'indexed_runtime_bindings.json',dict(observations=observed,
        scope='Selected generated calls; full-model execution equivalence not inferred'))


if __name__=='__main__':main()
