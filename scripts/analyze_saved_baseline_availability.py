"""Inventory recorded same-data comparisons; never infer missing allclose."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/property/numerical_coverage_v1'


def build():
    rows = {}
    conflicts = []
    files = list(BASE.rglob('baselines.json'))
    for path in sorted(files):
        payload = json.loads(path.read_text())
        if payload.get('schema') != 'same-data-baselines-v1':
            continue
        for row in payload['rows']:
            key = (row['case_id'], row['raw_sha256'], row['stage'])
            item = rows.setdefault(key, dict(case_id=key[0],raw_sha256=key[1],stage=key[2],
                sources=[], local_allclose='NOT_RECORDED', measurements={}))
            item['sources'].append(str(path.relative_to(ROOT)))
            ac = row.get('local_allclose')
            if isinstance(ac, bool):
                if isinstance(item['local_allclose'], bool) and item['local_allclose'] != ac:
                    conflicts.append(dict(key=key,field='local_allclose'))
                item['local_allclose'] = ac
            for name in ['relative_rms','aligned_ratio_of_sums','mean_relative_magnitude','residual_mean_relative_magnitude']:
                if name in row:
                    previous = item['measurements'].get(name)
                    if previous is not None and previous != row[name]:
                        conflicts.append(dict(key=key,field=name))
                    item['measurements'][name] = row[name]
    selected=list(rows.values())
    return dict(schema='saved-baseline-availability-v1',
                scope='AVAILABILITY_AND_CONSISTENCY_NOT_DETECTION_ACCURACY',
                sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                distinct_case_capture_count=len({(r['case_id'],r['raw_sha256']) for r in selected}),
                recorded_local_allclose= dict(Counter(str(r['local_allclose']) for r in selected if r['stage']=='LOCAL')),
                conflicts=conflicts, rows=selected,
                limits=['Missing local comparisons cannot be recovered from energy',
                        'Different reference constructions cannot be merged as identical tasks',
                        'No baseline is treated as independent ground truth',
                        'Repeated reports of the same capture are counted once'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT):p.error('Use new output inside repository')
    result=build();out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps({k:v for k,v in result.items() if k not in ['rows','sources']}))
