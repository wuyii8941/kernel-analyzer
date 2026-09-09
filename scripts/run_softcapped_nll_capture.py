"""Measure reviewed softcap/NLL outputs using the unchanged training analyzer."""
import argparse
import sys
from pathlib import Path
from scripts.run_numerical_coverage import read, save, sha, ROOT
from scripts.run_selected_nll_capture import preflight_state_bank


def validate_plan(plan):
    if plan.get('schema') != 'softcapped-nll-task-plan-v1' or not plan.get('cases'):
        raise ValueError('Nonempty softcap task plan required')
    if len(plan.get('trainable_parameters', [])) != 1:
        raise ValueError('Existing single-parameter scope required')
    seen = set()
    for case in plan['cases']:
        if case['task_id'] in seen:
            raise ValueError('Duplicate task')
        seen.add(case['task_id'])
        if (case['carrier'] != plan['trainable_parameters'][0]
                or case['reference_method'] != 'SOFTCAPPED_NLL_COMMON_INPUT'
                or case['expected_symbol'] not in plan['contracts']):
            raise ValueError('Task contract differs')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('family-plan', 'output-dir', 'spool-dir', 'training-bias-profile-v2-output-dir',
                 'input-bank', 'release-dir', 'model'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--state-bank', type=Path)
    p.add_argument('--device', required=True)
    a, arguments = p.parse_known_args()
    preflight_state_bank(read(a.state_bank or a.input_bank), arguments)
    if a.state_bank:
        arguments.extend(['--state-bank', str(a.state_bank)])
    plan = read(a.family_plan)
    validate_plan(plan)
    dependencies = []
    for name, digest in plan['source_sha256'].items():
        path = Path(name)
        if sha(path) != digest:
            raise ValueError('Frozen dependency changed: '+name)
        dependencies.append(path)
    output = a.training_bias_profile_v2_output_dir.resolve()
    for path in (output, a.output_dir, a.spool_dir):
        if path.exists() or not path.resolve().is_relative_to(Path('/data1/tzh')):
            raise ValueError('New output directories under /data1/tzh required')
    translated = output.parent/'softcap_capture_plan.json'
    save(translated, dict(cases=[dict(c, reference_method='PARTIAL_REDUCTION_FROM_BOUND_INPUT',
        declared_reference_method=c['reference_method']) for c in plan['cases']]))
    for name, value in (('case-plan', translated), ('output-dir', a.output_dir),
        ('spool-dir', a.spool_dir), ('training-bias-profile-v2-output-dir', output),
        ('input-bank', a.input_bank), ('release-dir', a.release_dir), ('model', a.model), ('device', a.device)):
        arguments.extend(['--'+name, str(value)])
    from scripts import capture_bound_endpoint_bias_formation_v21 as capture
    from scripts.same_dtype_semantic_observer import runtime_signature
    from kernel_analyzer.checked_inplace_observer import observer_class
    from kernel_analyzer.softcapped_nll_source import check_source, SIGNATURE
    from kernel_analyzer.softcapped_nll_runtime import snapshot_inputs, reference_callback
    original_load = capture.load_model
    def load(*args, **kwargs):
        model = original_load(*args, **kwargs)
        selected = set(plan['trainable_parameters'])
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name in selected)
        if {n for n, v in model.named_parameters() if v.requires_grad} != selected:
            raise ValueError('Declared training parameter absent')
        return model
    capture.load_model = load
    capture.SameDtypeSemanticCandidateObserver = observer_class(
        capture.SameDtypeSemanticCandidateObserver, plan['contracts'], runtime_signature,
        check_source=check_source, signature=SIGNATURE, snapshot_inputs=snapshot_inputs)
    capture.partial_reduction_reference = reference_callback(plan['contracts'])
    old_scope = capture.reference_scope
    capture.reference_scope = lambda method: dict(comparison='INTERNAL_SOFTCAPPED_NLL_OUTPUT_REPLACEMENT',
        same_local_operands=True, includes_possible_upstream_differences=False,
        reference_is_absolute_truth=False, reference_variant='SOFTCAPPED_NLL_FP64_BF16_WRITE'
        ) if method == 'PARTIAL_REDUCTION_FROM_BOUND_INPUT' else old_scope(method)
    dependencies += [Path(__file__), a.family_plan, translated, a.input_bank, a.model/'config.json']
    if a.state_bank:
        dependencies.append(a.state_bank)
    dependencies += [ROOT/'src/kernel_analyzer'/name for name in (
        'softcapped_nll_source.py', 'softcapped_nll_reference.py', 'softcapped_nll_runtime.py',
        'selected_nll_source.py', 'selected_nll_reference.py', 'checked_inplace_observer.py',
        'parallel_measurement.py', 'training_numerical_analysis.py', 'training_equivalence.py',
        'training_bias_profile.py', 'update_write.py', 'capture_cost.py')]
    dependencies += [ROOT/'scripts'/name for name in ('run_selected_nll_capture.py',
        'run_gelu_product_capture.py', 'capture_bound_endpoint_bias_formation_v21.py',
        'same_dtype_semantic_observer.py', 'run_parallel_bound_capture.py', 'run_training_bias_profile_v2_empirical.py')]
    save(output/'family_execution_protocol.json', dict(schema='softcapped-nll-capture-v1',
        contracts=plan['contracts'], trainable_parameters=plan['trainable_parameters'],
        capture_arguments=arguments, statistical_method_changed=False, claim_scope='FIXED_SUITE_UPDATE',
        primary_stage='PARAMETER_WRITE', data_use='REFERENCE_DEVELOPMENT_NOT_UNSEEN_CONFIRMATION',
        source_sha256={str(path.resolve()):sha(path) for path in dependencies}))
    from scripts.run_parallel_bound_capture import main as run
    from kernel_analyzer.capture_cost import measured_capture
    sys.argv = [sys.argv[0], *arguments]
    measured_capture(run, device=a.device, emit=lambda value:save(output/'capture_cost.json', value))


if __name__ == '__main__': main()
