"""Recompute saved-state diagnostics from complete same-call observations."""
import argparse
from pathlib import Path
from scripts.run_numerical_coverage import read, save, sha


def summarize_record(record, *, report_recompute_differences=False):
    import torch
    from kernel_analyzer.softmax_saved_state_diagnostic import evaluate
    output = record['post_call_outputs']
    old = record['diagnostic']
    # Scale is supplied explicitly by the caller from the checked source record.
    fresh = evaluate(output['in_out_ptr0'], output['out_ptr0'], output['out_ptr1'],
                     scale=record['checked_scale'])
    differences = {}
    for name in ('row_mass', 'normalization_defect', 'constant_cotangent_response'):
        if old[name].shape != fresh[name].shape or not torch.isfinite(old[name]).all():
            raise ValueError('Invalid stored diagnostic')
        differences[name] = (old[name]-fresh[name]).abs().max().item()
        if not report_recompute_differences and not torch.equal(old[name], fresh[name]):
            raise ValueError('Stored diagnostic differs from same-call outputs')
    defect = fresh['normalization_defect']
    response = fresh['constant_cotangent_response']
    return dict(symbol=record['symbol'], invocation_index=record['invocation_index'],
        rows=defect.numel(), normalization_defect_mean=defect.mean().item(),
        normalization_defect_rms=defect.square().mean().sqrt().item(),
        normalization_defect_max_abs=defect.abs().max().item(),
        constant_cotangent_response_rms=response.square().mean().sqrt().item(),
        recompute_max_absolute_differences=differences,
        recompute_bitwise_identical=all(v == 0 for v in differences.values()),
        recompute_policy='CPU_RECOMPUTATION_DIFFERENCES_REPORTED_NO_TOLERANCE_PASS' if report_recompute_differences else 'STRICT_BITWISE',
        scope='WITHIN_CALL_DESCRIPTIVE_NOT_INDEPENDENT_STATE_INFERENCE',
        mechanism_novelty_established=False, training_consequence_measured=False)


def main():
    import torch
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--report-recompute-differences', action='store_true')
    a = p.parse_args()
    if a.output.exists() or not a.output.resolve().is_relative_to(Path('/data1/tzh')):
        p.error('Choose a new output under /data1/tzh')
    protocol = read(a.root/'protocol.json')
    completion = read(a.root/'completion.json')
    if completion.get('status') != 'SAME_CALL_DIAGNOSTIC_CAPTURED':
        raise ValueError('Capture not complete')
    loaded = read(a.root/'loaded_sources.json')
    records = []
    for i in range(len(protocol['selected_states'])):
        state = read(a.root/f'state_{i:03d}.json')
        for j in range(state['calls']):
            path = a.root/f'state_{i:03d}_call_{j:03d}.pt'
            before = sha(path)
            record = torch.load(path, map_location='cpu', weights_only=True)
            contract = loaded['contracts'][record['symbol']]
            if record['function_ast_sha256'] != contract['function_ast_sha256']:
                raise ValueError('Function identity differs')
            record['checked_scale'] = contract['scale']
            outputs = dict(record['post_call_outputs'])
            outputs['in_out_ptr0'] = outputs['in_out_ptr0'].reshape(contract['rows'], contract['width'])
            for name in ('out_ptr0', 'out_ptr1'):
                outputs[name] = outputs[name].reshape(contract['rows'])
            record['post_call_outputs'] = outputs
            summary = summarize_record(record, report_recompute_differences=a.report_recompute_differences)
            if sha(path) != before:
                raise ValueError('Input changed during read')
            records.append(dict(summary, state_index=i, raw_path=str(path), raw_sha256=before))
    save(a.output, dict(schema='softmax-mechanism-summary-v1', records=records,
         protocol_sha256=sha(a.root/'protocol.json'),
         reporting_script_sha256=sha(Path(__file__)), new_mechanism_confirmed=False))
    print(dict(calls=len(records), new_mechanism_confirmed=False))


if __name__ == '__main__':
    main()
