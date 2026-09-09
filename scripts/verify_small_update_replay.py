#!/usr/bin/env python3
"""Verify a small-vector mechanism replay against original capture and AdamW."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from kernel_analyzer.update_write import adamw_parameter_write


def verify(original, repeated, diagnostics):
    fields = ('case_id', 'carrier', 'state_ids', 'calibration_state_ids', 'confirmation_state_ids',
              'optimizer', 'runtime_boundary', 'reference_comparison_scope', 'parameter_write_protocol',
              'original_coordinate_statistics')
    if any(k not in raw for raw in (original, repeated) for k in fields):
        raise ValueError('Required capture identity or statistics are absent')
    if any(raw.get('status') != 'COMPLETE' for raw in (original, repeated)):
        raise ValueError('Capture is not complete')
    if any(original.get(k) != repeated.get(k) for k in fields):
        raise ValueError('Repeated measurement differs from original data or declared setup')
    if (diagnostics['case_id'] != repeated['case_id'] or diagnostics['parameter_scope'] != repeated['carrier']
            or [row['state_id'] for row in diagnostics['rows']] != repeated['state_ids']):
        raise ValueError('Diagnostic identity or state order differs')
    differences = []
    for i, row in enumerate(diagnostics['rows']):
        for name in ('candidate', 'reference'):
            saved = row[name]
            dtype = {'torch.bfloat16': torch.bfloat16, 'torch.float16': torch.float16,
                     'torch.float32': torch.float32}[saved['base_dtype']]
            shape = saved['shape']
            base = torch.tensor(saved['base'], dtype=dtype).reshape(shape)
            gradient = torch.tensor(saved['gradient'], dtype=torch.float32).reshape(shape)
            moments = {k: None if saved[k] is None else torch.tensor(saved[k], dtype=torch.float32).reshape(shape)
                       for k in ('first', 'second')}
            observed = adamw_parameter_write(base, gradient, **moments, **saved['optimizer_arguments'])
            if not torch.equal(observed.reshape(-1), torch.tensor(saved['actual_write'], dtype=torch.float32)):
                raise ValueError('Actual optimizer replay does not reproduce saved write')
        c, r = row['candidate'], row['reference']
        for field in ('base', 'base_dtype', 'shape', 'first', 'second', 'optimizer_arguments'):
            if c[field] != r[field]:
                raise ValueError('Comparison does not start from the same optimizer inputs')
        delta = torch.tensor(c['actual_write']) - torch.tensor(r['actual_write'])
        coordinates = torch.nonzero(delta).reshape(-1).tolist()
        if coordinates != row['nonzero_update_coordinates']:
            raise ValueError('Changed-coordinate list differs from vectors')
        stats = repeated['original_coordinate_statistics']['PARAMETER_WRITE'][i]
        energy = float(delta.double().square().sum())
        if abs(energy-stats['effect_energy']) > 1e-10*max(energy,stats['effect_energy'],1e-300):
            raise ValueError('Vector energy differs from primary capture')
        for coordinate in coordinates:
            gc, gr = c['gradient'][coordinate], r['gradient'][coordinate]
            differences.append({'state_id': row['state_id'], 'coordinate': coordinate,
                                'candidate_gradient': gc, 'reference_gradient': gr,
                                'gradient_sign_changed': gc*gr < 0,
                                'candidate_write': c['actual_write'][coordinate],
                                'reference_write': r['actual_write'][coordinate],
                                'first': None if c['first'] is None else c['first'][coordinate],
                                'second': None if c['second'] is None else c['second'][coordinate],
                                'optimizer_arguments': c['optimizer_arguments']})
    return {'schema': 'small-update-mechanism-verification-v1', 'status': 'VERIFIED_REPLAY',
            'case_id': repeated['case_id'], 'states': len(diagnostics['rows']),
            'original_statistics_identical': True, 'optimizer_writes_reproduced': True,
            'changed_coordinates': differences,
            'scope': 'Post-discovery same-state replay; no persistence, population-bias, or loss claim'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('original','repeated','diagnostics','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists() or not a.output.resolve().is_relative_to(Path('/data1/tzh')):
        p.error('Choose a new output under /data1/tzh')
    paths=[a.original,a.repeated,a.diagnostics]
    data=[path.read_bytes() for path in paths]
    result=verify(*[json.loads(value) for value in data])
    result['source_sha256']={str(path.resolve()):hashlib.sha256(value).hexdigest() for path,value in zip(paths,data)}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f: json.dump(result,f,indent=2,allow_nan=False); f.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
