"""Source-bound depthwise convolution through the existing training analyzer."""
import argparse
import os
import sys
from pathlib import Path
from scripts.run_numerical_coverage import read, save, sha, ROOT


def select(plan, cases):
    if plan.get('schema') != 'depthwise-convolution-bound-plan-v1':
        raise ValueError('Unknown convolution plan')
    originals = {c['case_id']: c for c in plan['cases']}
    if len(originals) != len(plan['cases']) or not cases:
        raise ValueError('Nonempty unique plan required')
    if len({c['task_id'] for c in cases}) != len(cases):
        raise ValueError('Duplicate selected task')
    contracts = {}
    for case in cases:
        if case != originals.get(case['case_id']):
            raise ValueError('Selected case differs from bound plan')
        if (case['reference_method'] != 'DEPTHWISE_CONV1D_COMMON_INPUT'
                or case['implementation_kind'] != 'EXTERN'
                or case['expected_symbol'] != 'convolution'
                or case['replacement_boundary'] != 'EXTERNAL_CONVOLUTION_OUTPUT_BEFORE_SEPARATE_BIAS'
                or not case.get('carrier')):
            raise ValueError('Unsupported convolution boundary')
        pair = case['source_contract']
        digest = pair['convolution_line_sha256']
        if digest in contracts and contracts[digest] != pair:
            raise ValueError('Ambiguous convolution call identity')
        contracts[digest] = pair
    return contracts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('family-plan', 'case-plan', 'training-bias-profile-v2-output-dir',
                 'output-dir', 'spool-dir', 'input-bank', 'release-dir', 'model'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--device', required=True)
    a, arguments = p.parse_known_args()
    output = a.training_bias_profile_v2_output_dir.resolve()
    for path in (output, a.output_dir, a.spool_dir):
        if path.exists() or not path.resolve().is_relative_to(Path('/data1/tzh')):
            raise ValueError('New output directories under /data1/tzh required')
    plan = read(a.family_plan)
    checked = set()
    def verify_dependencies(record):
        for name, expected in record['source_sha256'].items():
            path = Path(name)
            if sha(path) != expected:
                raise ValueError('Frozen dependency changed: '+name)
            if name not in checked and path.suffix == '.json':
                checked.add(name)
                nested = read(path)
                if isinstance(nested, dict) and 'source_sha256' in nested:
                    verify_dependencies(nested)
    verify_dependencies(plan)
    cases = read(a.case_plan)['cases']
    contracts = select(plan, cases)
    translated = output.parent/'convolution_capture_plan.json'
    if translated.exists():
        raise ValueError('Translated plan already exists')
    save(translated, dict(cases=[dict(c,
        reference_method='PARTIAL_REDUCTION_FROM_BOUND_INPUT',
        declared_reference_method=c['reference_method']) for c in cases]))
    for name, value in [('case-plan', translated), ('training-bias-profile-v2-output-dir', output),
        ('output-dir', a.output_dir), ('spool-dir', a.spool_dir), ('input-bank', a.input_bank),
        ('release-dir', a.release_dir), ('model', a.model), ('device', a.device)]:
        arguments.extend(['--'+name, str(value)])
    os.environ.update(HF_HOME='/data1/tzh/cache/huggingface', XDG_CACHE_HOME='/data1/tzh/cache/xdg',
        TRITON_CACHE_DIR='/data1/tzh/cache/triton', TORCHINDUCTOR_CACHE_DIR='/data1/tzh/cache/torchinductor',
        PYTHONDONTWRITEBYTECODE='1')
    from scripts import capture_bound_endpoint_bias_formation_v21 as capture
    from scripts.convolution_capture_observer import observer_class
    from kernel_analyzer.depthwise_conv1d_reference import reference
    capture.SameDtypeSemanticCandidateObserver = observer_class(contracts)
    def local_reference(metadata, candidate, **unused):
        contract = contracts.get(metadata.get('source_line_sha256'))
        if contract is None or metadata.get('convolution_contract') != contract['convolution']:
            raise ValueError('Actual convolution is outside declared source contracts')
        return reference(metadata, candidate, contract['convolution'])
    capture.partial_reduction_reference = local_reference
    old_scope = capture.reference_scope
    capture.reference_scope = lambda method: dict(
        comparison='EXTERNAL_CONVOLUTION_OUTPUT_BEFORE_SEPARATE_BIAS',
        same_local_operands=True, includes_possible_upstream_differences=False,
        reference_variant='INPUT_ORDER_FP32_ACCUMULATION_BF16_WRITE',
        reference_is_absolute_truth=False, parameter_scope='SELECTED_PARAMETER_ONLY',
        implementation_attribution='EXTERNAL_LIBRARY_NOT_TRITON') if method == 'PARTIAL_REDUCTION_FROM_BOUND_INPUT' else old_scope(method)
    paths = [Path(__file__), a.family_plan, a.case_plan, translated, a.input_bank, a.model/'config.json']
    paths += [ROOT/'scripts'/n for n in ('convolution_capture_observer.py',
        'capture_bound_endpoint_bias_formation_v21.py', 'same_dtype_semantic_observer.py',
        'generated_nontriton_fp32_observer.py', 'run_parallel_bound_capture.py',
        'run_training_bias_profile_v2_empirical.py')]
    paths += [ROOT/'src/kernel_analyzer'/n for n in ('depthwise_conv1d_source.py',
        'depthwise_conv1d_reference.py', 'parallel_measurement.py', 'training_numerical_analysis.py',
        'training_equivalence.py', 'training_bias_profile.py', 'update_write.py', 'capture_cost.py')]
    paths += [a.release_dir/n for n in ('capture.json', 'same_dtype_tasks.json.gz', 'campaign.json.gz', 'inventory.json.gz')]
    save(output/'family_execution_protocol.json', dict(schema='depthwise-convolution-capture-v1',
        contracts=contracts, capture_arguments=arguments, primary_stage='PARAMETER_WRITE',
        claim_scope='FIXED_SUITE_UPDATE', statistical_method_changed=False,
        data_use='DECLARED_REFERENCE_NOT_BLIND_IMPLEMENTATION_CONFIRMATION',
        source_sha256={str(path.resolve()): sha(path) for path in paths}))
    from scripts.run_parallel_bound_capture import main as run
    from kernel_analyzer.capture_cost import measured_capture
    sys.argv = [sys.argv[0], *arguments]
    measured_capture(run, device=a.device, emit=lambda v:save(output/'capture_cost.json', v))


if __name__ == '__main__':
    main()
