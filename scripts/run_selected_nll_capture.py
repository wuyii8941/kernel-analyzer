"""Run reviewed NLL replacements through the existing three-stage capture."""
import argparse
import sys
from pathlib import Path
from scripts.run_numerical_coverage import read, save, sha, ROOT
from scripts.run_gelu_product_capture import check_state_count


def preflight_state_bank(bank, arguments):
    # Match the existing capture loader's ID policy without rewriting the bank.
    states = bank.get('states', bank.get('records'))
    if not isinstance(states, list):
        raise ValueError('Missing state list')
    normalized = dict(bank, states=[dict(s, state_id=str(s.get('state_id', s.get('sequence_id', i))))
                                    for i, s in enumerate(states)])
    check_state_count(normalized, arguments)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('family-plan', 'output-dir', 'spool-dir', 'training-bias-profile-v2-output-dir',
                 'input-bank', 'release-dir', 'model'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--state-bank', type=Path)
    parser.add_argument('--device', required=True)
    args, rest = parser.parse_known_args()
    preflight_state_bank(read(args.state_bank or args.input_bank), rest)
    if args.state_bank:
        rest.extend(['--state-bank', str(args.state_bank)])
    plan = read(args.family_plan)
    if plan.get('schema') != 'selected-nll-task-plan-v1' or not plan['cases']:
        raise ValueError('Nonempty reviewed NLL plan required')
    dependencies = []
    visited = set()
    def verify(record):
        for name, digest in record.get('source_sha256', {}).items():
            path = Path(name)
            if sha(path) != digest:
                raise ValueError('Frozen dependency changed: '+name)
            dependencies.append(path)
            if path.suffix == '.json' and name not in visited:
                visited.add(name)
                child = read(path)
                if isinstance(child, dict) and isinstance(child.get('source_sha256'), dict):
                    verify(child)
    verify(plan)
    output = args.training_bias_profile_v2_output_dir.resolve()
    for path in (output, args.output_dir, args.spool_dir):
        if path.exists() or not path.resolve().is_relative_to(Path('/data1/tzh')):
            raise ValueError('New output directories under /data1/tzh required')
    cases = plan['cases']
    if len({c['task_id'] for c in cases}) != len(cases):
        raise ValueError('Duplicate NLL task')
    translated = output.parent/'nll_capture_plan.json'
    save(translated, dict(cases=[dict(c, reference_method='PARTIAL_REDUCTION_FROM_BOUND_INPUT',
                                     declared_reference_method=c['reference_method']) for c in cases]))
    for name, value in (('case-plan', translated), ('output-dir', args.output_dir),
                        ('spool-dir', args.spool_dir), ('training-bias-profile-v2-output-dir', output),
                        ('input-bank', args.input_bank), ('release-dir', args.release_dir),
                        ('model', args.model), ('device', args.device)):
        rest.extend(['--'+name, str(value)])
    from scripts import capture_bound_endpoint_bias_formation_v21 as capture
    from scripts.same_dtype_semantic_observer import runtime_signature
    from kernel_analyzer.selected_nll_observer import observer_class
    from kernel_analyzer.selected_nll_runtime import reference_callback
    capture.SameDtypeSemanticCandidateObserver = observer_class(
        capture.SameDtypeSemanticCandidateObserver, plan['contracts'], runtime_signature)
    capture.partial_reduction_reference = reference_callback(plan['contracts'])
    original_scope = capture.reference_scope
    capture.reference_scope = lambda method: dict(
        comparison='INTERNAL_NLL_BACKWARD_OUTPUT_REPLACEMENT', same_local_operands=True,
        includes_possible_upstream_differences=False, reference_is_absolute_truth=False,
        reference_variant='SAVED_INPUT_NLL_FP64_BF16_WRITE') if method == 'PARTIAL_REDUCTION_FROM_BOUND_INPUT' else original_scope(method)
    dependencies += [Path(__file__), args.family_plan, translated, args.input_bank, args.model/'config.json']
    if args.state_bank:
        dependencies.append(args.state_bank)
    dependencies += [ROOT/'src/kernel_analyzer'/name for name in (
        'selected_nll_source.py', 'selected_nll_reference.py', 'selected_nll_runtime.py',
        'selected_nll_observer.py', 'update_write.py', 'training_bias_profile.py',
        'training_numerical_analysis.py', 'training_equivalence.py', 'parallel_measurement.py')]
    dependencies += [ROOT/'scripts'/name for name in (
        'capture_bound_endpoint_bias_formation_v21.py', 'same_dtype_semantic_observer.py',
        'run_parallel_bound_capture.py', 'run_training_bias_profile_v2_empirical.py', 'run_gelu_product_capture.py')]
    save(output/'family_execution_protocol.json', dict(schema='selected-nll-capture-v1',
        capture_arguments=rest, contracts=plan['contracts'], claim_scope='FIXED_SUITE_UPDATE',
        primary_stage='PARAMETER_WRITE', statistical_method_changed=False,
        reference_variant='SAVED_INPUT_NLL_FP64_BF16_WRITE',
        source_sha256={str(p.resolve()):sha(p) for p in dependencies}))
    from scripts.run_parallel_bound_capture import main as run
    from kernel_analyzer.capture_cost import measured_capture
    sys.argv = [sys.argv[0], *rest]
    measured_capture(run, device=args.device, emit=lambda value:save(output/'capture_cost.json', value))


if __name__ == '__main__':
    main()
