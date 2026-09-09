"""Run checked GELU outputs through the shared three-stage training analyzer."""
import argparse
import sys
from pathlib import Path
from scripts.run_numerical_coverage import read,save,sha,ROOT


def select(plan,cases):
    if plan.get('schema')!='gelu-product-task-plan-v1' or not cases:
        raise ValueError('Nonempty GELU task plan required')
    originals={c['task_id']:c for c in plan['cases']}
    if len(originals)!=len(plan['cases']) or len({c['task_id'] for c in cases})!=len(cases):
        raise ValueError('Duplicate task')
    for case in cases:
        if case!=originals.get(case['task_id']):raise ValueError('Selected case changed')
    return {c['expected_symbol']:plan['contracts'][c['expected_symbol']] for c in cases}


def check_state_count(bank, arguments):
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--states',type=int,required=True)
    parser.add_argument('--warmup-steps',type=int,default=0)
    options,_=parser.parse_known_args(arguments)
    states=bank.get('states',bank.get('records'))
    if (options.states<1 or options.warmup_steps<0 or not isinstance(states,list)
            or len(states)<options.states+options.warmup_steps):
        raise ValueError('Insufficient state-bank inputs before model loading')
    ids=[s['state_id'] for s in states[:options.states+options.warmup_steps]]
    if len(set(ids))!=len(ids):raise ValueError('Duplicate state IDs')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('family-plan','case-plan','output-dir','spool-dir',
                 'training-bias-profile-v2-output-dir','input-bank','release-dir','model'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--device',required=True)
    parser.add_argument('--state-bank',type=Path)
    a,arguments=parser.parse_known_args()
    check_state_count(read(a.state_bank or a.input_bank),arguments)
    if a.state_bank:arguments.extend(['--state-bank',str(a.state_bank)])
    output=a.training_bias_profile_v2_output_dir.resolve()
    for path in (output,a.output_dir,a.spool_dir):
        if path.exists() or not path.resolve().is_relative_to(Path('/data1/tzh')):
            raise ValueError('New output directories under /data1/tzh required')
    plan=read(a.family_plan)
    dependencies=[]
    visited=set()
    def verify(record):
        for name,digest in record.get('source_sha256',{}).items():
            path=Path(name)
            if sha(path)!=digest:raise ValueError('Frozen input changed: '+name)
            dependencies.append(path)
            if path.suffix=='.json' and name not in visited:
                visited.add(name)
                nested=read(path)
                if isinstance(nested,dict) and isinstance(nested.get('source_sha256'),dict):verify(nested)
    verify(plan)
    cases=read(a.case_plan)['cases'];contracts=select(plan,cases)
    translated=output.parent/'gelu_capture_plan.json'
    save(translated,dict(cases=[dict(c,reference_method='PARTIAL_REDUCTION_FROM_BOUND_INPUT',
        declared_reference_method=c['reference_method']) for c in cases]))
    for name,value in [('case-plan',translated),('output-dir',a.output_dir),('spool-dir',a.spool_dir),
        ('training-bias-profile-v2-output-dir',output),('input-bank',a.input_bank),
        ('release-dir',a.release_dir),('model',a.model),('device',a.device)]:
        arguments.extend(['--'+name,str(value)])
    from scripts import capture_bound_endpoint_bias_formation_v21 as capture
    from scripts.same_dtype_semantic_observer import runtime_signature
    from kernel_analyzer.gelu_product_observer import observer_class,validate_pointers
    from kernel_analyzer.gelu_product_reference import evaluate
    original_load=capture.load_model
    def load(*args,**kwargs):
        model=original_load(*args,**kwargs)
        selected=set(plan['trainable_parameters'])
        for name,p in model.named_parameters():p.requires_grad_(name in selected)
        if {name for name,p in model.named_parameters() if p.requires_grad}!=selected:
            raise ValueError('Declared training parameter absent')
        return model
    capture.load_model=load
    capture.SameDtypeSemanticCandidateObserver=observer_class(
        capture.SameDtypeSemanticCandidateObserver,contracts,runtime_signature)
    def reference(metadata,candidate,**unused):
        contract=contracts.get(metadata.get('symbol'))
        if contract is None or metadata.get('formal_pointer')!='out_ptr0':
            raise ValueError('Undeclared GELU output')
        decoded=validate_pointers(metadata['runtime_pointers'],contract)
        return evaluate(*decoded,output_dtype=candidate.dtype).reshape(candidate.shape)
    capture.partial_reduction_reference=reference
    old_scope=capture.reference_scope
    capture.reference_scope=lambda method:dict(comparison='INTERNAL_GELU_OUTPUT_REPLACEMENT',
        same_local_operands=True,includes_possible_upstream_differences=False,
        parameter_scope='DECLARED_SINGLE_TRAINABLE_PARAMETER',reference_is_absolute_truth=False,
        reference_variant='TANH_GELU_PRODUCT_FP32_BF16_WRITE') if method=='PARTIAL_REDUCTION_FROM_BOUND_INPUT' else old_scope(method)
    dependencies += [Path(__file__),a.family_plan,a.case_plan,translated,a.input_bank,a.model/'config.json']
    if a.state_bank:dependencies.append(a.state_bank)
    dependencies += [ROOT/'src/kernel_analyzer'/n for n in ('gelu_product_source.py','gelu_product_reference.py',
        'gelu_product_observer.py','parallel_measurement.py','training_numerical_analysis.py',
        'training_equivalence.py','training_bias_profile.py','update_write.py','capture_cost.py')]
    dependencies += [ROOT/'scripts'/n for n in ('capture_bound_endpoint_bias_formation_v21.py',
        'same_dtype_semantic_observer.py','run_parallel_bound_capture.py','run_training_bias_profile_v2_empirical.py')]
    save(output/'family_execution_protocol.json',dict(schema='gelu-product-capture-v1',
        contracts=contracts,trainable_parameters=plan['trainable_parameters'],capture_arguments=arguments,
        statistical_method_changed=False,claim_scope='FIXED_SUITE_UPDATE',primary_stage='PARAMETER_WRITE',
        data_use='HISTORICAL_FAMILY_NEW_REFERENCE_CAPTURE_NOT_UNSEEN_DISCOVERY',
        source_sha256={str(p.resolve()):sha(p) for p in dependencies}))
    from scripts.run_parallel_bound_capture import main as run
    from kernel_analyzer.capture_cost import measured_capture
    sys.argv=[sys.argv[0],*arguments]
    measured_capture(run,device=a.device,emit=lambda v:save(output/'capture_cost.json',v))


if __name__=='__main__':main()
