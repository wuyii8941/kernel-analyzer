"""Find historical release packages omitted from the current coverage input.

This audits source coverage, not runtime support or semantic family novelty.
Byte-identical task packages are grouped, without merging distinct protocols.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
from collections import Counter


def audit(root, inventory):
    included = {str(Path(r['release']).resolve()) for r in inventory['records']}
    listing = subprocess.check_output(['rg', '--files', '--no-ignore', str(root/'results')], text=True)
    paths = sorted(Path(p) for p in listing.splitlines() if Path(p).name == 'same_dtype_tasks.json.gz')
    rows = []
    for path in paths:
        row = dict(release=str(path.parent.resolve()), task_file=str(path.resolve()),
                   included_in_current_inventory=str(path.parent.resolve()) in included)
        try:
            raw = path.read_bytes()
            data = json.loads(gzip.decompress(raw))
            tasks = data['rows']
            ids = [r['task_id'] for r in tasks]
            if len(ids) != len(set(ids)):
                raise ValueError('Duplicate task IDs')
            row.update(status='TASK_PACKAGE_READ', task_sha256=hashlib.sha256(raw).hexdigest(),
                task_count=len(tasks),
                symbols=sorted({str(t.get('symbol')) for t in tasks if t.get('symbol')}),
                missing_package_files=[n for n in ('capture.json', 'inventory.json.gz', 'campaign.json.gz')
                                       if not (path.parent/n).exists()])
        except (OSError, ValueError, KeyError, TypeError) as exc:
            row.update(status='UNREADABLE_OR_INVALID', reason=str(exc))
        rows.append(row)
    digests = Counter(r['task_sha256'] for r in rows if 'task_sha256' in r)
    return dict(schema='release-inventory-coverage-audit-v1', rows=rows,
        discovered_packages=len(rows), included_packages=sum(r['included_in_current_inventory'] for r in rows),
        omitted_packages=sum(not r['included_in_current_inventory'] for r in rows),
        distinct_task_file_digests=len(digests),
        duplicate_file_groups=[dict(sha256=k, copies=v) for k,v in digests.items() if v>1],
        scope='All same_dtype_tasks.json.gz under results; other formats and cache-only packages not enumerated',
        runtime_support_inferred=False, new_family_count=None)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inventory',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists() or not a.output.resolve().is_relative_to(Path('/data1/tzh')):
        p.error('New output under /data1/tzh required')
    raw=a.inventory.read_bytes()
    report=audit(Path(__file__).resolve().parents[1], json.loads(raw))
    report['source_sha256']={str(a.inventory.resolve()):hashlib.sha256(raw).hexdigest(),
        str(Path(__file__).resolve()):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    with a.output.open('x') as stream:
        json.dump(report,stream,indent=2,allow_nan=False)
    print({k:report[k] for k in ('discovered_packages','included_packages','omitted_packages','distinct_task_file_digests')})


if __name__=='__main__': main()
