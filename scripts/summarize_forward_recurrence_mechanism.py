"""CPU recomputation of observed forward recurrence states, without bias claims."""
import argparse
from pathlib import Path
from scripts.run_numerical_coverage import read, save, sha
from kernel_analyzer.forward_recurrence_diagnostic import analyze


def summarize_record(record, contract):
    import torch
    dimensions = {k: contract[k] for k in ('steps', 'channels', 'state_width',
                                          'packed_width', 'state_offset')}
    if (record['dimensions'] != dimensions
            or record['function_ast_sha256'] != contract['function_ast_sha256']):
        raise ValueError('Record dimensions or function identity differ')
    fresh = analyze(record['pre_call_inputs'], record['post_call_outputs'], **dimensions)
    differences = {}
    for name in ('state_difference', 'reference_states', 'combined_local_remainder'):
        old = record['diagnostic'][name]
        if old.shape != fresh[name].shape or not torch.isfinite(old).all():
            raise ValueError('Invalid stored diagnostic')
        differences[name] = (old-fresh[name]).abs().max().item()
    reference_energy = fresh['reference_states'].square().sum(dim=(1, 2))
    effect_energy = fresh['state_difference'].square().sum(dim=(1, 2))
    ratios = [float((e/r).sqrt()) if r > 0 else None
              for e, r in zip(effect_energy, reference_energy)]
    return dict(symbol=record['symbol'], invocation_index=record['invocation_index'],
                step_relative_rms=ratios, step_signed_coordinate_mean=fresh['per_step_mean'].tolist(),
                final_reference_cast_rms=fresh['final_reference_cast_difference'].square().mean().sqrt().item(),
                final_difference_from_rounded_reference_rms=fresh['final_difference_from_rounded_reference'].square().mean().sqrt().item(),
                reconstruction_max_abs=fresh['reconstruction_max_abs'].item(),
                recompute_max_absolute_differences=differences,
                recompute_policy='CPU_DIFFERENCES_REPORTED_NO_TOLERANCE_PASS',
                actual_final_candidate_cast_error_identified=False,
                scope='WITHIN_CALL_DESCRIPTIVE_NOT_STATE_POPULATION_INFERENCE',
                new_mechanism_confirmed=False)


def main():
    import torch
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists() or not a.output.resolve().is_relative_to(Path('/data1/tzh')):
        p.error('New output under /data1/tzh required')
    protocol = read(a.root/'protocol.json')
    completion = read(a.root/'completion.json')
    if (protocol.get('mechanism_family') != 'forward_recurrence'
            or completion.get('status') != 'SAME_CALL_DIAGNOSTIC_CAPTURED'
            or completion['states'] != len(protocol['selected_states'])):
        raise ValueError('Incomplete or different-family capture')
    loaded = read(a.root/'loaded_sources.json')
    records = []
    for i in range(completion['states']):
        state = read(a.root/f'state_{i:03d}.json')
        if state['calls'] <= 0:
            raise ValueError('Missing observed calls')
        for j in range(state['calls']):
            path = a.root/f'state_{i:03d}_call_{j:03d}.pt'
            digest = sha(path)
            record = torch.load(path, map_location='cpu', weights_only=True)
            summary = summarize_record(record, loaded['contracts'][record['symbol']])
            if sha(path) != digest:
                raise ValueError('Raw capture changed during read')
            records.append(dict(summary, state_index=i, raw_path=str(path), raw_sha256=digest))
    save(a.output, dict(schema='forward-recurrence-mechanism-summary-v1', records=records,
                        protocol_sha256=sha(a.root/'protocol.json'),
                        reporting_script_sha256=sha(Path(__file__)),
                        new_mechanism_confirmed=False, training_consequence_measured=False))
    print(dict(calls=len(records), new_mechanism_confirmed=False))


if __name__ == '__main__':
    main()
