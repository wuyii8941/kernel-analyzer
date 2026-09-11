#!/usr/bin/env python3
"""Run frozen forward-output replacements with the common training analyzer."""
import argparse
import sys
from pathlib import Path
from scripts.run_numerical_coverage import read, save, sha, ROOT


def select(plan, cases):
    if plan.get('schema') != 'residual-rms-forward-bound-plan-v1':
        raise ValueError('Unknown plan schema')
    originals = {c['case_id']: c for c in plan['cases']}
    if len(originals) != len(plan['cases']) or not cases:
        raise ValueError('Nonempty unique plan required')
    if len({c['case_id'] for c in cases}) != len(cases):
        raise ValueError('Duplicate selected case')
    selected = {}
    for case in cases:
        if case != originals.get(case['case_id']):
            raise ValueError('Selected case differs from plan')
        symbol = case['expected_symbol']
        contract = plan['contracts'][symbol]
        if (case['reference_method'] != 'RESIDUAL_RMS_FORWARD_COMMON_INPUT'
                or case['reference_output_pointer'] not in contract['output_pointers']):
            raise ValueError('Output or method differs')
        selected[symbol] = contract
    return selected


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('family-plan', 'case-plan', 'training-bias-profile-v2-output-dir',
                 'output-dir', 'spool-dir', 'input-bank', 'release-dir', 'model'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--device', required=True)
    a, arguments = p.parse_known_args()
    output = a.training_bias_profile_v2_output_dir.resolve()
    for path in (output, a.output_dir, a.spool_dir):
        if path.exists() or not path.resolve().is_relative_to(Path('/data1/tzh')):
            raise ValueError('New output paths under /data1/tzh required')
    plan = read(a.family_plan)
    for path, expected in plan['source_sha256'].items():
        if sha(Path(path)) != expected:
            raise ValueError('Frozen family dependency changed: ' + path)
    cases = read(a.case_plan)['cases']
    contracts = select(plan, cases)
    translated = output.parent / 'forward_capture_plan.json'
    save(translated, dict(cases=[dict(c, reference_method='PARTIAL_REDUCTION_FROM_BOUND_INPUT',
        declared_reference_method=c['reference_method']) for c in cases]))
    for name, value in [('case-plan', translated), ('training-bias-profile-v2-output-dir', output),
        ('output-dir', a.output_dir), ('spool-dir', a.spool_dir), ('input-bank', a.input_bank),
        ('release-dir', a.release_dir), ('model', a.model), ('device', a.device)]:
        arguments.extend(['--' + name, str(value)])
    from scripts import capture_bound_endpoint_bias_formation_v21 as capture
    from scripts.same_dtype_semantic_observer import runtime_signature
    from kernel_analyzer import residual_rms_forward_observer as observer
    from kernel_analyzer import residual_rms_forward_reference as reference
    capture.SameDtypeSemanticCandidateObserver = observer.observer_class(
        capture.SameDtypeSemanticCandidateObserver, contracts, runtime_signature)
    selected_outputs = {(c['expected_symbol'], c['reference_output_pointer']) for c in cases}
    def local_reference(metadata, candidate, **unused):
        key = (metadata.get('symbol'), metadata.get('formal_pointer'))
        if key not in selected_outputs or metadata.get('input_output_storage_aliases') != []:
            raise ValueError('Unexpected output or storage alias')
        contract = contracts[key[0]]
        return reference.select_output(metadata['runtime_pointers'], candidate,
            formal_pointer=key[1], rows=contract['rows'], width=contract['width'], epsilon=contract['epsilon'])
    capture.partial_reduction_reference = local_reference
    old_scope = capture.reference_scope
    capture.reference_scope = lambda method: dict(
        comparison='SINGLE_FORWARD_OUTPUT_REPLACEMENT', same_local_operands=True,
        includes_possible_upstream_differences=False,
        complete_multi_output_implementation_replacement=False,
        reference_variant='DECLARED_FP32_EXPRESSION_BF16_WRITE') if method == 'PARTIAL_REDUCTION_FROM_BOUND_INPUT' else old_scope(method)
    paths = [Path(__file__), Path(observer.__file__), Path(reference.__file__), a.family_plan,
             a.case_plan, translated, a.input_bank, a.model / 'config.json']
    paths += [ROOT / 'scripts' / n for n in ('capture_bound_endpoint_bias_formation_v21.py',
        'same_dtype_semantic_observer.py', 'run_parallel_bound_capture.py', 'run_training_bias_profile_v2_empirical.py')]
    paths += [ROOT / 'src/kernel_analyzer' / n for n in ('residual_rms_forward_source.py',
        'parallel_measurement.py', 'training_numerical_analysis.py', 'training_equivalence.py',
        'training_bias_profile.py', 'update_write.py', 'capture_cost.py')]
    paths += [a.release_dir / n for n in ('same_dtype_tasks.json.gz', 'campaign.json.gz', 'inventory.json.gz')]
    save(output / 'family_execution_protocol.json', dict(schema='residual-rms-forward-capture-v1',
        capture_arguments=arguments, contracts=contracts, primary_stage='PARAMETER_WRITE',
        contrast_id='SINGLE_FORWARD_OUTPUT_REPLACEMENT', claim_scope='FIXED_SUITE_UPDATE',
        fixed_suite_margins=dict(full_update_rms=0.01),
        source_sha256={str(path.resolve()): sha(path) for path in paths}, statistical_method_changed=False))
    from scripts.run_parallel_bound_capture import main as run
    from kernel_analyzer.capture_cost import measured_capture
    sys.argv = [sys.argv[0], *arguments]
    measured_capture(run, device=a.device, emit=lambda value: save(output / 'capture_cost.json', value))


if __name__ == '__main__':
    main()
