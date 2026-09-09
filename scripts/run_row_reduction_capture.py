#!/usr/bin/env python3
"""Use source-checked row-reduction references with the existing capture engine.

The case plan declares reference_contract_symbol per endpoint. There is no
model/case dispatch and no change to the shared statistical analysis.
"""
import argparse
import json
import math
import os
from pathlib import Path
import sys

from kernel_analyzer.row_reduction_reference import VARIANTS
from kernel_analyzer.source_reference_registry import get_reference
from scripts.run_numerical_coverage import ROOT, read, save, sha


def selected_contracts(manifest, cases):
    available = {}
    for source in manifest['sources']:
        for row in source['rows']:
            if row['status'] == 'SOURCE_CHECKED':
                contract = row['contract']
                previous = available.setdefault(row['symbol'], contract)
                identity_key = ('function_semantic_ast_sha256'
                                if 'function_semantic_ast_sha256' in contract
                                else 'function_ast_sha256')
                if previous[identity_key] != contract[identity_key]:
                    raise ValueError('Ambiguous same-name reference contracts')
    selected = {}
    for case in cases:
        symbol = case.get('reference_contract_symbol')
        if symbol not in available or case.get('reference_method') != 'PARTIAL_REDUCTION_FROM_BOUND_INPUT':
            raise ValueError('Case lacks an explicit source-checked reference')
        if not case['task_id'].endswith(':' + available[symbol]['output_pointer']):
            raise ValueError('Row reference requires the checked output pointer')
        selected[symbol] = available[symbol]
    if not selected:
        raise ValueError('Empty reference selection')
    return selected


def selected_symbol_census(campaign_rows, contracts):
    """Keep the complete invocation census for the reviewed source symbols.

    The saved release may contain unrelated generated kernels whose names vary
    across PyTorch/Inductor releases.  A family capture must still observe the
    complete frozen census of every *selected* symbol so ordinal endpoint
    binding cannot drift, but it must not require unrelated symbols to remain
    byte-for-byte present in the newly compiled graph.
    """
    symbols = set(contracts)
    rows = [row for row in campaign_rows if str(row.get('symbol')) in symbols]
    observed = {str(row.get('symbol')) for row in rows}
    if observed != symbols:
        raise ValueError('Selected source symbol is absent from frozen campaign')
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     epilog='Append the existing bound-capture arguments for model, input banks, release, case plan, output and spool directories.')
    parser.add_argument('--reference-manifest', type=Path, required=True)
    parser.add_argument('--reference-variant', choices=VARIANTS, default='FP32_NATIVE')
    parser.add_argument('--train-only-declared-parameters', action='store_true')
    parser.add_argument('--parallel-measurement', action='store_true')
    parser.add_argument('--retain-small-update-vectors', action='store_true',
                        help='Post-discovery diagnostics for exactly one parameter with at most 4096 elements.')
    parser.add_argument('--local-rtol', type=float)
    parser.add_argument('--local-atol', type=float)
    args, remaining = parser.parse_known_args()
    if (args.local_rtol is None) != (args.local_atol is None):
        raise ValueError('Declare both local tolerances or neither')
    local_enabled = args.local_rtol is not None
    if local_enabled and not all(math.isfinite(v) and v >= 0 for v in (args.local_rtol, args.local_atol)):
        raise ValueError('Local tolerances must be finite and nonnegative')
    capture_args = argparse.ArgumentParser(add_help=False)
    capture_args.add_argument('--case-plan', type=Path, required=True)
    capture_args.add_argument('--training-bias-profile-v2-output-dir', type=Path, required=True)
    capture_args.add_argument('--spool-dir', type=Path, required=True)
    capture_args.add_argument('--output-dir', type=Path, required=True)
    capture_args.add_argument('--input-bank', type=Path, required=True)
    capture_args.add_argument('--state-bank', type=Path)
    capture_args.add_argument('--release-dir', type=Path, required=True)
    capture_args.add_argument('--model', type=Path, required=True)
    capture_args.add_argument('--device', required=True)
    options, _ = capture_args.parse_known_args(remaining)
    output = options.training_bias_profile_v2_output_dir.resolve()
    if (output / 'family_execution_protocol.json').exists():
        raise ValueError('Use a new capture directory; do not overwrite a frozen protocol')
    if not all(path.resolve().is_relative_to(Path('/data1/tzh')) for path in (output, options.spool_dir, options.output_dir)):
        raise ValueError('Capture and spool outputs must remain under /data1/tzh')
    manifest = read(args.reference_manifest)
    family = manifest.get('reference_family', 'ROW_SQUARE_SUM')
    specification = get_reference(family)
    checker = specification.check_source
    adapter = ROOT / 'src/kernel_analyzer' / specification.source_filename
    if args.reference_variant not in manifest['legal_variants']:
        raise ValueError('Variant is not declared for this reference family')
    if sha(adapter) != manifest['adapter_sha256']:
        raise ValueError('Reference adapter changed after source checking')
    cases = read(options.case_plan)['cases']
    contracts = selected_contracts(manifest, cases)
    if args.retain_small_update_vectors and len(cases) != 1:
        raise ValueError('Vector diagnostics require exactly one declared case')
    os.environ.update(HF_HOME='/data1/tzh/cache/huggingface', XDG_CACHE_HOME='/data1/tzh/cache/xdg',
                      TRITON_CACHE_DIR='/data1/tzh/cache/triton', TORCHINDUCTOR_CACHE_DIR='/data1/tzh/cache/torchinductor',
                      PYTHONDONTWRITEBYTECODE='1')
    from scripts import capture_bound_endpoint_bias_formation_v21 as capture
    original_observer = capture.SameDtypeSemanticCandidateObserver
    local_observations = {case['task_id']: [] for case in cases}
    recorder = None
    if args.retain_small_update_vectors:
        from kernel_analyzer.update_write_diagnostics import SmallUpdateRecorder
        recorder = SmallUpdateRecorder(capture.adamw_parameter_write)
        capture.adamw_parameter_write = recorder

    class CheckedObserver(original_observer):
        def __init__(self, **kwargs):
            if local_enabled:
                sink = kwargs['sink']
                def identified_sink(task_id, tensor, metadata):
                    return sink(task_id, tensor, {**metadata, '_analysis_observed_task_id': task_id})
                kwargs['sink'] = identified_sink
            for symbol, contract in contracts.items():
                matched = []
                runtime_rows = []
                identity_key = ('function_semantic_ast_sha256'
                                if 'function_semantic_ast_sha256' in contract
                                else 'function_ast_sha256')
                for module in kwargs['modules']:
                    module_path = Path(module.__file__)
                    candidate_symbols = sorted(
                        name for name in vars(module)
                        if name.startswith('triton_') and 'exp' in name and 'sum' in name)
                    if hasattr(module, symbol):
                        try:
                            current = checker(module_path.read_text(), symbol)
                        except Exception as exc:
                            runtime_rows.append(dict(module=str(module_path.resolve()),
                                has_expected_symbol=True, checker_status='REJECTED',
                                reason=type(exc).__name__ + ': ' + str(exc),
                                related_symbols=candidate_symbols))
                            continue
                        matched.append(current[identity_key])
                        runtime_rows.append(dict(module=str(module_path.resolve()),
                            has_expected_symbol=True, checker_status='SOURCE_CHECKED',
                            function_ast_sha256=current['function_ast_sha256'],
                            identity_key=identity_key, identity_sha256=current[identity_key],
                            related_symbols=candidate_symbols))
                    else:
                        runtime_rows.append(dict(module=str(module_path.resolve()),
                            has_expected_symbol=False, checker_status='NOT_APPLICABLE',
                            related_symbols=candidate_symbols))
                if matched != [contract[identity_key]]:
                    audit_path = output / 'runtime_source_identity_failure.json'
                    if not audit_path.exists():
                        save(audit_path, dict(schema='runtime-source-identity-failure-v1',
                            symbol=symbol, expected_identity_key=identity_key,
                            expected_identity_sha256=contract[identity_key],
                            checked_match_count=len(matched), modules=runtime_rows,
                            conclusion=('EXPECTED_SYMBOL_ABSENT_FROM_COMPILED_GRAPH'
                                if not matched and not any(r['has_expected_symbol'] for r in runtime_rows)
                                else 'EXPECTED_SYMBOL_PRESENT_BUT_IDENTITY_NOT_UNIQUE_OR_DIFFERENT')))
                    raise RuntimeError('Executed source identity failed for checked reduction: '
                                       + symbol + '; see ' + str(audit_path))
            kwargs['campaign_rows'] = selected_symbol_census(
                kwargs['campaign_rows'], contracts)
            super().__init__(**kwargs)

    def bound_reference(metadata, candidate, **unused):
        symbol = metadata.get('symbol')
        if symbol not in contracts or metadata.get('formal_pointer') != contracts[symbol]['output_pointer']:
            raise RuntimeError('Reference runtime boundary differs')
        result = specification.evaluate(metadata, candidate, contracts[symbol], variant=args.reference_variant)
        if local_enabled:
            from kernel_analyzer.local_tolerance import compare_outputs
            task_id = metadata['_analysis_observed_task_id']
            local_observations[task_id].append(compare_outputs(candidate, result,
                                                              rtol=args.local_rtol, atol=args.local_atol))
        return result

    capture.SameDtypeSemanticCandidateObserver = CheckedObserver
    capture.partial_reduction_reference = bound_reference
    old_scope = capture.reference_scope
    capture.reference_scope = lambda method: ({
        'comparison': 'COMMON_OPERAND_SOURCE_CHECKED_' + family,
        'same_local_operands': True, 'includes_possible_upstream_differences': False,
        'reference_variant': args.reference_variant,
        'single_kernel_source_attribution': 'CHECKED_FUNCTION_AST_AND_PRE_CALL_INPUTS',
    } if method == 'PARTIAL_REDUCTION_FROM_BOUND_INPUT' else old_scope(method))
    carriers = sorted({case['carrier'] for case in cases})
    if args.train_only_declared_parameters:
        original_load = capture.load_model

        def load(*positional, **keyword):
            model = original_load(*positional, **keyword)
            for name, parameter in model.named_parameters():
                parameter.requires_grad_(name in carriers)
            if sorted(name for name, p in model.named_parameters() if p.requires_grad) != carriers:
                raise ValueError('Declared trainability scope differs from model')
            return model
        capture.load_model = load
    source_paths = [Path(__file__), adapter, ROOT / 'src/kernel_analyzer/source_reference_registry.py',
                    Path(capture.__file__), args.reference_manifest, options.case_plan,
                    ROOT / 'scripts/same_dtype_semantic_observer.py',
                    ROOT / 'scripts/run_training_bias_profile_v2_empirical.py',
                    ROOT / 'src/kernel_analyzer/update_write.py',
                    ROOT / 'src/kernel_analyzer/training_numerical_analysis.py',
                    ROOT / 'src/kernel_analyzer/training_equivalence.py',
                    ROOT / 'src/kernel_analyzer/training_bias_profile.py',
                    ROOT / 'src/kernel_analyzer/capture_cost.py',
                    options.input_bank, options.model / 'config.json',
                    options.release_dir / 'same_dtype_tasks.json.gz',
                    options.release_dir / 'campaign.json.gz', options.release_dir / 'inventory.json.gz']
    if options.state_bank:
        source_paths.append(options.state_bank)
    if args.parallel_measurement:
        source_paths.extend([ROOT / 'scripts/run_parallel_bound_capture.py',
                             ROOT / 'src/kernel_analyzer/parallel_measurement.py'])
    if recorder is not None:
        source_paths.append(ROOT / 'src/kernel_analyzer/update_write_diagnostics.py')
    if local_enabled:
        source_paths.append(ROOT / 'src/kernel_analyzer/local_tolerance.py')
    protocol_path = output / 'family_execution_protocol.json'
    save(protocol_path, {
        'schema': 'row-reduction-family-capture-v1', 'contracts': contracts, 'reference_family': family,
        'reference_dispatch': 'EXPLICIT_FAMILY_REGISTRY_V1',
        'candidate_census_scope': 'COMPLETE_FROZEN_INVOCATION_CENSUS_FOR_SELECTED_SOURCE_SYMBOLS',
        'primary_stage': 'PARAMETER_WRITE', 'claim_scope': 'FIXED_SUITE_UPDATE',
        'fixed_suite_margins': {'full_update_rms': .01},
        'variant': args.reference_variant, 'train_only_declared_parameters': args.train_only_declared_parameters,
        'parallel_measurement': args.parallel_measurement,
        'retain_small_update_vectors': args.retain_small_update_vectors,
        'diagnostic_data_use': 'POST_DISCOVERY_MECHANISM_REPLAY' if recorder is not None else 'NOT_REQUESTED',
        'carriers': carriers, 'capture_arguments': remaining,
        'source_sha256': {str(path.resolve()): sha(path) for path in source_paths},
        'data_use': 'DECLARED_VARIANT_MEASUREMENT_NOT_AUTOMATICALLY_UNSEEN_CONFIRMATION',
        'statistical_method_changed': False,
        'capture_cost_measurement': 'WALL_CPU_AND_PYTORCH_CUDA_MEMORY_NOT_TRAINING_THROUGHPUT',
        'local_tolerance_baseline': {'rtol': args.local_rtol, 'atol': args.local_atol,
                                    'kernel_author_policy': False} if local_enabled else None,
    })
    # A direct invocation must be as reproducible as a queued invocation.  Save
    # the exact Python texts before model loading so later edits cannot make a
    # completed capture impossible to recompute.  The protocol hashes above
    # remain the authority; the finalizer checks every snapshot entry against
    # them.
    snapshot_path = output.parent / 'source_snapshot.json'
    save(snapshot_path, {
        str(path.resolve()): path.read_text()
        for path in source_paths if path.suffix == '.py'
    })
    sys.argv = [sys.argv[0], *remaining]
    def execute():
        if args.parallel_measurement:
            from scripts.run_parallel_bound_capture import main as parallel_main
            parallel_main()
        else:
            capture.main()
        if recorder is not None:
            raw_path = output / (cases[0]['case_id'] + '.json')
            save(output / 'update_vector_diagnostics.json', recorder.finish(read(raw_path)))
        if local_enabled:
            rows = []
            for case in cases:
                raw_path = output / (case['case_id'] + '.json')
                raw = read(raw_path)
                observations = local_observations[case['task_id']]
                if len(observations) != len(raw['state_ids']):
                    raise ValueError('Local comparison count differs from captured states')
                rows.append({'case_id': case['case_id'], 'task_id': case['task_id'],
                             'raw_sha256': sha(raw_path),
                             'observations': [dict(record, state_id=state) for state, record in zip(raw['state_ids'], observations)]})
            save(output / 'local_tolerance_comparisons.json', {
                'schema': 'local-tolerance-comparisons-v1', 'records': rows,
                'scope': 'Declared elementwise baseline, not author acceptance test or training verdict'})
    from kernel_analyzer.capture_cost import measured_capture
    measured_capture(execute, device=options.device,
                     emit=lambda metrics: save(output / 'capture_cost.json', metrics))


if __name__ == '__main__':
    main()
