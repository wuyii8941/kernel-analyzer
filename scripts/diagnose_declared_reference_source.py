#!/usr/bin/env python3
"""Record the actual checked module and stop before numerical measurement."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import runpy
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--diagnostic-output',type=Path,required=True)
    a,remaining=p.parse_known_args()
    if a.diagnostic_output.exists() or not a.diagnostic_output.resolve().is_relative_to(Path('/data1/tzh')):
        p.error('Use new diagnostic output under /data1/tzh')
    if remaining.count('--module')!=1:
        p.error('An explicit reference module is required')
    name=remaining[remaining.index('--module')+1]
    if not name.isidentifier():
        p.error('Local module identifier required')
    module=importlib.import_module('kernel_analyzer.'+name)
    original=module.check_source
    def observe(text,symbol):
        from torch._inductor.codecache import PyCodeCache
        digest=hashlib.sha256(text.encode()).hexdigest()
        paths=[]
        for loaded in PyCodeCache.modules:
            path=Path(loaded.__file__)
            if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest()==digest:
                paths.append(str(path.resolve()))
        try:
            contract=original(text,symbol)
            error=None
        except ValueError as exc:
            contract=None
            error=str(exc)
        record=dict(status='ACTUAL_SOURCE_DIAGNOSTIC_ONLY',symbol=symbol,
                    actual_source_sha256=digest,actual_module_paths=paths,
                    source_text=text,checked_contract=contract,source_check_error=error,
                    numerical_measurement_performed=False,
                    original_guard_relaxed=False)
        a.diagnostic_output.parent.mkdir(parents=True,exist_ok=True)
        with a.diagnostic_output.open('x') as f:
            json.dump(record,f,indent=2,allow_nan=False)
        raise RuntimeError('Diagnostic captured actual source; stop before measurement')
    module.check_source=observe
    sys.argv=['run_declared_reference_family',*remaining]
    runpy.run_module('scripts.run_declared_reference_family',run_name='__main__')


if __name__=='__main__':
    main()
