#!/usr/bin/env python3
"""Parse a generated module once, checking every literal Triton definition."""
import argparse
import ast
from collections import Counter
from pathlib import Path
from scripts.run_numerical_coverage import save,sha
from kernel_analyzer import decayed_recurrence_source


def scan(source, *, checker=decayed_recurrence_source.check_source):
    assignments=[n for n in ast.parse(source).body if isinstance(n,ast.Assign)]
    counts=Counter(t.id for n in assignments for t in n.targets if isinstance(t,ast.Name))
    rows=[]
    for node in assignments:
        for target in node.targets:
            if not isinstance(target,ast.Name) or not target.id.startswith('triton_'): continue
            symbol=target.id
            if counts[symbol]!=1:
                rows.append(dict(symbol=symbol,status='REJECTED',reason='Duplicate top-level assignment'))
                continue
            try:
                contract=checker(ast.unparse(node),symbol)
                # The isolated assignment is only a parsing optimization, not
                # an executable substitution or a different source identity.
                import hashlib
                contract['source_sha256']=hashlib.sha256(source.encode()).hexdigest()
                rows.append(dict(symbol=symbol,status='SOURCE_CHECKED',contract=contract))
            except ValueError as exc:
                rows.append(dict(symbol=symbol,status='REJECTED',reason=str(exc)))
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists() or not a.output.resolve().is_relative_to(Path('/data1/tzh')):
        p.error('Choose new output under /data1/tzh')
    sources=[]
    for path in a.source:
        rows=scan(path.read_text()); sources.append(dict(path=str(path.resolve()),sha256=sha(path),rows=rows))
        print(dict(source=str(path),definitions=len(rows),checked=sum(r['status']=='SOURCE_CHECKED' for r in rows)),flush=True)
    save(a.output,dict(schema='recurrence-source-scan-v1',sources=sources,numerical_results_read=False,
        runtime_measurement_complete=False,source_sha256={str(p.resolve()):sha(p) for p in
        [Path(__file__),Path(decayed_recurrence_source.__file__)]}))


if __name__=='__main__': main()
