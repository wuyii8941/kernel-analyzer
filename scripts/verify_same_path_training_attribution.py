"""Independent saved-record and checkpoint integrity check for attribution replay."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

from scripts.verify_adamw8bit_error_compensation_training import interval

ROOT=Path(__file__).resolve().parents[1]
HISTORICAL=ROOT/'results/property/numerical_coverage_v1/adamw8bit_error_compensation_training_v1'


def read(path):return json.loads(path.read_text())


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def endpoint(record,steps):
    values=record['evaluation_loss_by_step'][str(steps)]
    return math.fsum(values)/len(values)


def verify(directory):
    protocol=read(directory/'protocol.json');protocol_sha=sha(directory/'protocol.json')
    errors=[];missing=[];rows=[];sources={}
    if protocol.get('schema')!='same-path-training-attribution-v1' or protocol.get('conditions')!=['OFF','ON'] or not protocol.get('source_sha256'):
        errors.append('incomplete attribution protocol')
    for name,h in protocol['source_sha256'].items():
        p=Path(name)
        if not p.is_file() or sha(p)!=h:errors.append('source changed: '+name)
    for i in range(protocol['stream_count']):
        paths={mode:directory/'runs'/f'stream_{i:02d}_{mode}.json' for mode in ['OFF','ON']}
        absent=[str(p) for p in paths.values() if not p.is_file()]
        if absent:missing.extend(absent);continue
        pair={mode:read(p) for mode,p in paths.items()}
        for mode,record in pair.items():
            label=f'{i}:{mode}'
            sources[str(paths[mode])]=sha(paths[mode])
            if record.get('status')!='COMPLETE' or record.get('condition')!=mode or record.get('stream_index')!=i or record.get('protocol_sha256')!=protocol_sha:
                errors.append('record identity: '+label)
            if record.get('training_steps')!=protocol['steps'] or len(record.get('training_loss',[]))!=protocol['steps'] or not all(math.isfinite(v) for v in record.get('training_loss',[])):
                errors.append('incomplete or nonfinite training: '+label)
            for step in protocol['evaluation_steps']:
                values=record.get('evaluation_loss_by_step',{}).get(str(step),[])
                if len(values)!=protocol['evaluation_states'] or not all(math.isfinite(v) for v in values):
                    errors.append('invalid evaluation: '+label+':'+str(step))
            checkpoint=Path(record['checkpoint'])
            if not checkpoint.is_relative_to(directory) or not checkpoint.is_file() or sha(checkpoint)!=record['checkpoint_sha256']:
                errors.append('checkpoint identity: '+label)
            if record.get('exact_endpoint_reproduction') is not True or record.get('reevaluated_loss')!=record['evaluation_loss_by_step'][str(protocol['steps'])]:
                errors.append('checkpoint evaluation differs: '+label)
        if pair['OFF']['evaluation_loss_by_step']['0']!=pair['ON']['evaluation_loss_by_step']['0']:
            errors.append('initial evaluation differs: '+str(i))
        old_path=HISTORICAL/'streams'/f'stream_{i:02d}.json'
        old={r['condition']:r for r in read(old_path)['records']}
        default=endpoint(old['ADAMW8BIT_BLOCK256'],protocol['steps'])
        off=endpoint(pair['OFF'],protocol['steps']);on=endpoint(pair['ON'],protocol['steps'])
        prior_on=old['ADAMW8BIT_COMPENSATED_BLOCK256']
        bridge=default-off;toggle=off-on
        rows.append(dict(stream=i,default_loss=default,off_loss=off,on_loss=on,
            default_minus_off=bridge,off_minus_on=toggle,default_minus_on=default-on,
            decomposition_residual=(default-on)-(bridge+toggle),
            old_on_loss_minus_on=endpoint(prior_on,protocol['steps'])-on,
            on_parameter_hash_matches_history=prior_on['final_parameter_sha256']==pair['ON']['final_parameter_sha256'],
            on_training_losses_match_history=prior_on['training_loss']==pair['ON']['training_loss'],
            off_steps_per_second=pair['OFF']['training_steps_per_second_with_evaluation_overhead'],
            on_steps_per_second=pair['ON']['training_steps_per_second_with_evaluation_overhead']))
    result=dict(schema='same-path-attribution-independent-record-check-v1',
        status='INVALID' if errors else 'INCOMPLETE' if missing else 'VERIFIED',
        errors=errors,missing=missing,rows=rows,record_sha256=sources,protocol_sha256=protocol_sha,
        limits=['same previously observed data streams, not external generalization',
                'paired t intervals depend on stream sampling assumptions',
                'historical compiled baseline is reused rather than retrained',
                'checkpoint evaluation was rerun by executor; this verifier checks its saved record and file hash'])
    if not errors and not missing:
        gains=[r['off_minus_on'] for r in rows];ci=interval(gains)
        result['primary']=dict(mean=math.fsum(gains)/len(gains),interval_95=ci,
            decision='MATERIAL_IMPROVEMENT' if ci[0]>protocol['material_margin'] else 'DETECTABLE_IMPROVEMENT' if ci[0]>0 else 'NOT_CONFIRMED')
        result['excluded_pilot_stream_sensitivity']=dict(mean=math.fsum(gains[1:])/len(gains[1:]),interval_95=interval(gains[1:]))
        result['historical_on_reproduced']=all(r['on_parameter_hash_matches_history'] and r['on_training_losses_match_history'] for r in rows)
        result['additional_implementation_effect_mean']=math.fsum(r['default_minus_off'] for r in rows)/len(rows)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output.resolve();directory=a.root.resolve()
    if out.exists() or not out.is_relative_to(ROOT):p.error('Use new repository output')
    result=verify(directory);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2,allow_nan=False))
    if result['rows']:
        with out.with_suffix('.csv').open('x',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(result['rows'][0]));w.writeheader();w.writerows(result['rows'])
    print(json.dumps({k:v for k,v in result.items() if k not in ['rows','record_sha256']}))
    if result['status']!='VERIFIED':raise SystemExit(1)
