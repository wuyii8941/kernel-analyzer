#!/usr/bin/env python3
"""Recompute comparable summaries from recorded original statistics, not scores."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from kernel_analyzer.numerical_baselines import stage_baselines


def local_comparison_summary(payload, data, record):
    """Join directly measured tolerance results only to their exact capture."""
    if record['case_id'] != payload['case_id'] or record['raw_sha256'] != hashlib.sha256(data).hexdigest():
        raise ValueError('Local comparison belongs to different raw data')
    observations = record['observations']
    ids = payload['state_ids']
    confirmation = payload['confirmation_state_ids']
    if len(set(ids)) != len(ids) or [r['state_id'] for r in observations] != ids:
        raise ValueError('Local comparison state order differs')
    if not confirmation or len(set(confirmation)) != len(confirmation) or not set(confirmation) <= set(ids):
        raise ValueError('Invalid confirmation states')
    selected = [r for r in observations if r['state_id'] in confirmation]
    policies = {(r['rtol'], r['atol']) for r in observations}
    if len(policies) != 1:
        raise ValueError('Mixed local tolerance policies')
    import math
    rtol, atol = next(iter(policies))
    if not all(math.isfinite(v) and v >= 0 for v in (rtol, atol)):
        raise ValueError('Invalid local tolerance policy')
    for r in observations:
        if r['status'] not in ('FINITE_COMPARISON', 'NONFINITE_OUTPUT'):
            raise ValueError('Unknown local comparison status')
        if (r['status'] == 'FINITE_COMPARISON' and type(r['allclose']) is not bool
                or r['status'] == 'NONFINITE_OUTPUT' and r['allclose'] is not None):
            raise ValueError('Inconsistent local comparison result')
    finite = all(r['status'] == 'FINITE_COMPARISON' for r in selected)
    return dict(local_allclose=all(r['allclose'] for r in selected) if finite else 'NONFINITE_NOT_ASSESSED',
                local_rtol=rtol, local_atol=atol,
                local_tolerance_scope='DECLARED_BASELINE_NOT_KERNEL_AUTHOR_POLICY',
                local_failed_states=sum(r['allclose'] is False for r in selected),
                local_nonfinite_states=sum(r['status'] != 'FINITE_COMPARISON' for r in selected))


def full_mean_summary(payload, data, record, stage):
    import math
    if (record.get('status') != 'VERIFIED_IDENTICAL_RECAPTURE'
            or record.get('raw_sha256') != hashlib.sha256(data).hexdigest()
            or record.get('case_id') != payload['case_id']):
        raise ValueError('Full-coordinate means belong to different or unverified capture')
    for key in ('state_ids','calibration_state_ids','confirmation_state_ids'):
        if record.get(key) != payload[key]: raise ValueError('Mean state partitions differ')
    result=record['stages'][stage]
    if result['population_guarantee'] is not False or result['decision_role']!='DESCRIPTIVE_NOT_USED_FOR_EQUIVALENCE':
        raise ValueError('Unexpected scope for full-coordinate means')
    stats=payload['original_coordinate_statistics'][stage]
    if (result['calibration_count']!=len(payload['calibration_state_ids'])
            or result['confirmation_count']!=len(payload['confirmation_state_ids'])
            or len(result['rows'])!=len(stats)):
        raise ValueError('Mean observation count differs')
    for measured,original in zip(result['rows'],stats):
        for key in ('effect_energy','repair_energy','effect_repair_inner_product'):
            x,y=measured[key],original[key]
            if not math.isfinite(x) or abs(x-y)>1e-10*max(abs(x),abs(y),1e-300):
                raise ValueError('Mean vectors do not match original coordinate statistics')
    if result['status']!='VALID':
        return dict(mean_status='FULL_COORDINATE_MEAN_UNAVAILABLE_'+result['status'])
    conf=[payload['state_ids'].index(i) for i in payload['confirmation_state_ids']]
    energy=math.fsum(stats[i]['effect_energy'] for i in conf)
    repair=math.fsum(stats[i]['repair_energy'] for i in conf)
    if repair<=0: raise ValueError('Mean normalization has no repair energy')
    bound=math.sqrt(energy/repair)
    summary=dict(mean_status='FULL_COORDINATE_STREAMING_FIXED_SUITE',
                 residual_mean_status=result['residual_status'])
    for source,target in (('confirmation_mean_relative_magnitude','mean_relative_magnitude'),
                          ('confirmation_residual_mean_relative_magnitude','residual_mean_relative_magnitude')):
        value=result[source]
        if value is None and (source=='confirmation_mean_relative_magnitude' or result['residual_status']=='VALID'):
            raise ValueError('Valid full-coordinate mean is missing')
        if value is not None:
            if not math.isfinite(value) or value<0 or value>bound+1e-10*max(bound,1e-300):
                raise ValueError('Mean magnitude exceeds full-coordinate energy bound')
            summary[target]=value
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--raw', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--local-comparisons', type=Path)
    p.add_argument('--full-means', type=Path, nargs='+')
    a = p.parse_args()
    if a.output.exists() or not a.output.resolve().is_relative_to(Path('/data1/tzh')):
        p.error('Use a new output under /data1/tzh')
    rows = []
    comparisons = {}
    means = {}
    for path in a.full_means or []:
        record=json.loads(path.read_text())
        if record['case_id'] in means: raise ValueError('Duplicate full-mean case')
        means[record['case_id']]=record
    if a.local_comparisons:
        for record in json.loads(a.local_comparisons.read_text())['records']:
            if record['case_id'] in comparisons:
                raise ValueError('Duplicate local comparison case')
            comparisons[record['case_id']] = record
    for path in a.raw:
        data = path.read_bytes(); payload = json.loads(data)
        for row in stage_baselines(payload):
            if row['stage'] == 'LOCAL' and a.local_comparisons:
                record = comparisons[payload['case_id']]
                row.update(local_comparison_summary(payload, data, record))
            if payload['case_id'] in means:
                row.update(full_mean_summary(payload,data,means[payload['case_id']],row['stage']))
            rows.append({'case_id': payload['case_id'], 'raw_path': str(path.resolve()),
                         'raw_sha256': hashlib.sha256(data).hexdigest(), **row})
    a.output.mkdir(parents=True)
    source_paths = (Path(__file__), Path(__file__).resolve().parents[1] / 'src/kernel_analyzer/numerical_baselines.py')
    with (a.output/'baselines.json').open('x') as f:
        json.dump({'schema':'same-data-baselines-v1','rows':rows,
                   'local_comparisons_sha256': hashlib.sha256(a.local_comparisons.read_bytes()).hexdigest() if a.local_comparisons else None,
                   'full_means_sha256': {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in a.full_means or []},
                   'source_sha256': {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths},
                   'claim':'Descriptive reanalysis; no retrospective detection thresholds or missing allclose values invented'},f,indent=2,allow_nan=False)
    with (a.output/'baselines.csv').open('x') as f:
        writer=csv.DictWriter(f,fieldnames=sorted({k for r in rows for k in r}))
        writer.writeheader(); writer.writerows(rows)
    # Plot precisely the recorded quantities, including missing entries.
    import math
    from html import escape
    cases=list(dict.fromkeys(r['case_id'] for r in rows))
    stages=['LOCAL','PARAMETER_GRADIENT','PARAMETER_WRITE']
    width=1710; height=155+len(cases)*42
    svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
         '<rect width="100%" height="100%" fill="white"/>',
         '<g font-family="sans-serif" font-size="12" fill="#172435">',
         '<text x="20" y="26">Same recorded confirmation states; descriptive, protocol-specific comparisons</text>']
    for i,case in enumerate(cases):
        svg.append(f'<text x="15" y="{116+i*42}">{escape(case)}</text>')
    for panel,(field,title) in enumerate(zip(['relative_rms','aligned_ratio_of_sums','mean_relative_magnitude'],
                                           ['Total difference / reference RMS','Aligned scaling','Mean / reference RMS (full coordinates)'])):
        left=490+panel*400
        svg.append(f'<text x="{left}" y="55">{title}</text>')
        values={(r['case_id'],r['stage']):r[field]*100 for r in rows if field in r}
        maximum=max([abs(v) for v in values.values()]+[1e-30])
        for j,stage in enumerate(stages):
            x=left+j*128
            svg.append(f'<text x="{x+4}" y="79">{["Local","Gradient","Parameter write"][j]}</text>')
            for i,case in enumerate(cases):
                y=91+i*42; value=values.get((case,stage))
                if value is None:
                    color='#ededed'; text='N/A'
                else:
                    strength=math.sqrt(abs(value)/maximum)
                    base=(228,170,128) if value<0 else (145,184,223)
                    channels=[round(255+(c-255)*strength) for c in base]
                    color='#'+''.join(f'{c:02x}' for c in channels); text=f'{value:.4g}%'
                svg.append(f'<rect x="{x}" y="{y}" width="124" height="38" fill="{color}"/>')
                svg.append(f'<text x="{x+62}" y="{y+25}" text-anchor="middle">{text}</text>')
    svg+=['<text x="20" y="'+str(height-12)+'">N/A means missing original statistics or full-coordinate means; it is not a zero measurement.</text>','</g></svg>']
    with (a.output/'profiles.svg').open('x') as f: f.write('\n'.join(svg))
    print(json.dumps({'cases': len(a.raw), 'stage_rows':len(rows)}))


if __name__=='__main__': main()
